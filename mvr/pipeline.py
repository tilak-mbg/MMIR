"""End-to-end pipeline: data -> features -> temporal heads -> index -> evaluation.

Each step caches its output on disk, so the dashboard can resume and the
search page loads instantly once the pipeline has run once.
"""
from __future__ import annotations

import json
import re
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .config import LEARNED_METHODS, PARAM_FREE_METHODS, Config
from .data import load_subset, prepare_subset, read_video_frames, segment_indices
from .index import VideoIndex
from .metrics import holm, paired_bootstrap, summarize_ranks, t2v_ranks, v2t_metrics, wilcoxon_p
from .temporal import (head_embed, load_head, pool_video, query_attn_scores, save_head,
                       select_tau, train_head)

ProgressFn = Optional[Callable[[float, str], None]]

TEMPORAL_WORDS = {"then", "after", "before", "while", "begins", "begin", "starts", "start", "started",
                  "finally", "first", "again", "continues", "continue", "until", "during", "later",
                  "next", "followed", "afterwards", "meanwhile", "eventually", "once", "now"}


def _p(progress: ProgressFn, frac, msg):
    if progress:
        progress(min(max(frac, 0.0), 1.0), msg)
    else:
        print(f"[{frac*100:5.1f}%] {msg}", flush=True)


_ENCODER_CACHE: dict = {}


def get_encoder(cfg: Config):
    from .encoders import ClipEncoder
    key = (cfg.clip_model, cfg.resolved_device())
    if key not in _ENCODER_CACHE:
        _ENCODER_CACHE.clear()
        _ENCODER_CACHE[key] = ClipEncoder(cfg.clip_model, cfg.resolved_device())
    return _ENCODER_CACHE[key]


# ---------------------------------------------------------------------------
# Step 2: feature extraction
# ---------------------------------------------------------------------------
def step_features(cfg: Config, progress: ProgressFn = None, force: bool = False) -> dict:
    if cfg.features_path.exists() and not force:
        _p(progress, 1.0, "Features already extracted (cached)")
        return dict(np.load(cfg.features_path, allow_pickle=True))
    subset = load_subset(cfg)
    if subset is None:
        raise RuntimeError("Run the data step first.")
    enc = get_encoder(cfg)
    T = cfg.num_frames

    def encode_videos(items, lo, hi, label):
        feats, ok_ids, t_dec, t_enc = [], [], 0.0, 0.0
        for i, v in enumerate(items):
            path = cfg.videos_dir / f"{v['video_id']}.mp4"
            try:
                t0 = time.perf_counter()
                frames, _ = read_video_frames(path, T)
                t1 = time.perf_counter()
                feats.append(enc.encode_frames(frames, cfg.batch_size))
                t_enc += time.perf_counter() - t1
                t_dec += t1 - t0
                ok_ids.append(i)
            except Exception:
                continue
            if i % 5 == 0:
                _p(progress, lo + (hi - lo) * (i + 1) / len(items), f"{label}: {i+1}/{len(items)} clips")
        return np.stack(feats) if feats else np.zeros((0, T, enc.dim), np.float32), ok_ids, t_dec, t_enc

    index_frames, ok, t_dec, t_enc = encode_videos(subset["index"], 0.0, 0.35, "Encoding index clip frames")
    if len(ok) != len(subset["index"]):
        raise RuntimeError("Some index videos could not be decoded; re-run the data step with force.")
    _p(progress, 0.36, "Encoding the 200 text queries")
    t0 = time.perf_counter()
    query_emb = enc.encode_text([q["text"] for q in subset["queries"]])
    t_text = time.perf_counter() - t0

    train_frames, ok_tr, _, _ = encode_videos(subset["train"], 0.38, 0.85, "Encoding training clip frames")
    train_items = [subset["train"][i] for i in ok_tr]
    C = 20
    train_text = np.zeros((len(train_items), C, enc.dim), np.float32)
    text_mask = np.zeros((len(train_items), C), bool)
    flat, owner = [], []
    for i, t in enumerate(train_items):
        for j, c in enumerate(t["captions"][:C]):
            flat.append(c)
            owner.append((i, j))
    for b in range(0, len(flat), 2048):
        _p(progress, 0.86 + 0.13 * b / max(len(flat), 1), f"Encoding training captions {b}/{len(flat)}")
        E = enc.encode_text(flat[b:b + 2048])
        for (i, j), e in zip(owner[b:b + 2048], E):
            train_text[i, j] = e
            text_mask[i, j] = True

    out = {"index_frames": index_frames.astype(np.float32), "query_emb": query_emb.astype(np.float32),
           "train_frames": train_frames.astype(np.float32), "train_text": train_text, "text_mask": text_mask,
           "train_ids": np.array([t["video_id"] for t in train_items]),
           "timing": np.array(json.dumps({
               "decode_s_per_video": t_dec / len(ok), "encode_s_per_video": t_enc / len(ok),
               "text_encode_ms_per_query": 1000 * t_text / len(subset["queries"]),
               "device": cfg.resolved_device(), "model": cfg.clip_model, "frames_per_video": T}))}
    cfg.features_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cfg.features_path, **out)
    _p(progress, 1.0, f"Saved features to {cfg.features_path.name}")
    return out


def load_features(cfg: Config) -> dict | None:
    if not cfg.features_path.exists():
        return None
    return dict(np.load(cfg.features_path, allow_pickle=True))


# ---------------------------------------------------------------------------
# Step 3: learned temporal heads
# ---------------------------------------------------------------------------
def head_path(cfg: Config, arch: str, seed: int) -> Path:
    return cfg.models_dir / f"{arch}_T{cfg.num_frames}_seed{seed}.pt"


def step_train(cfg: Config, progress: ProgressFn = None, force: bool = False) -> dict:
    F = load_features(cfg)
    if F is None:
        raise RuntimeError("Run the feature step first.")
    if len(F["train_frames"]) < 40:
        raise RuntimeError("Training pool too small (need >= 40 clips). Increase 'train videos per category'.")
    dev = cfg.resolved_device()
    tau, tau_curve = select_tau(F["train_frames"], F["train_text"], F["text_mask"])
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    (cfg.models_dir / "tau.json").write_text(json.dumps({"tau": tau, "grid_mrr": tau_curve}))
    jobs = [(a, s) for a in LEARNED_METHODS for s in cfg.train_seeds]
    hists = {}
    for j, (arch, seed) in enumerate(jobs):
        p = head_path(cfg, arch, seed)
        if p.exists() and not force:
            hists[f"{arch}_seed{seed}"] = json.loads(p.with_suffix(".json").read_text())
            continue
        lo, hi = j / len(jobs), (j + 1) / len(jobs)
        head, hist = train_head(arch, F["train_frames"], F["train_text"], F["text_mask"], seed=seed,
                                epochs=cfg.eff_epochs(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                                batch=cfg.train_batch, layers=cfg.tfm_layers, heads=cfg.tfm_heads, device=dev,
                                progress=lambda f, m: _p(progress, lo + (hi - lo) * f, m))
        save_head(head, hist, p)
        hists[f"{arch}_seed{seed}"] = hist
    _p(progress, 1.0, f"Trained {len(jobs)} temporal heads (query-attention tau={tau})")
    return hists


def load_heads(cfg: Config, dim: int) -> dict:
    heads = {}
    for arch in LEARNED_METHODS:
        for s in cfg.train_seeds:
            p = head_path(cfg, arch, s)
            if p.exists():
                heads[(arch, s)] = load_head(arch, dim, p, cfg.num_frames, cfg.tfm_layers, cfg.tfm_heads, "cpu")
    return heads


def get_tau(cfg: Config) -> float:
    p = cfg.models_dir / "tau.json"
    return json.loads(p.read_text())["tau"] if p.exists() else 0.01


# ---------------------------------------------------------------------------
# Step 4: index
# ---------------------------------------------------------------------------
def step_index(cfg: Config, progress: ProgressFn = None) -> VideoIndex:
    F = load_features(cfg)
    subset = load_subset(cfg)
    fr = F["index_frames"]
    t0 = time.perf_counter()
    pooled = {m: pool_video(fr, m) for m in ("single_frame", "mean_pool", "max_pool")}
    heads = load_heads(cfg, fr.shape[-1])
    for arch in LEARNED_METHODS:
        if (arch, cfg.train_seeds[0]) in heads:
            pooled[arch] = head_embed(heads[(arch, cfg.train_seeds[0])], fr)
    idx = VideoIndex([v["video_id"] for v in subset["index"]], fr, pooled, tau=get_tau(cfg))
    build_ms = (time.perf_counter() - t0) * 1000
    idx.save(cfg.index_dir)
    (cfg.index_dir / "build.json").write_text(json.dumps({"build_ms": build_ms, "backend": idx.backend}))
    _p(progress, 1.0, f"Index built with {len(pooled)} representations ({idx.backend}, {build_ms:.1f} ms)")
    return idx


def load_index(cfg: Config) -> VideoIndex | None:
    if (cfg.index_dir / "meta.json").exists():
        return VideoIndex.load(cfg.index_dir)
    return None


# ---------------------------------------------------------------------------
# Step 5 (optional): X-CLIP external spatio-temporal reference
# ---------------------------------------------------------------------------
def step_xclip(cfg: Config, progress: ProgressFn = None, force: bool = False) -> str:
    out = Path(cfg.artifacts_dir) / "xclip_scores.npy"
    if out.exists() and not force:
        return "cached"
    if not cfg.use_xclip:
        return "disabled"
    try:
        from .encoders import XClipScorer
        subset = load_subset(cfg)
        _p(progress, 0.05, f"Loading {cfg.xclip_model}")
        xs = XClipScorer(cfg.xclip_model, cfg.resolved_device())
        pix = []
        for i, v in enumerate(subset["index"]):
            frames, _ = read_video_frames(cfg.videos_dir / f"{v['video_id']}.mp4", xs.num_frames)
            pix.append(xs.video_pixels(frames))
            if i % 10 == 0:
                _p(progress, 0.1 + 0.6 * i / len(subset["index"]), f"X-CLIP decoding {i}/{len(subset['index'])}")
        _p(progress, 0.75, "X-CLIP scoring 200 queries x 100 videos")
        S = xs.score_matrix([q["text"] for q in subset["queries"]], pix)
        np.save(out, S)
        (Path(cfg.artifacts_dir) / "xclip_error.txt").unlink(missing_ok=True)
        _p(progress, 1.0, "X-CLIP done")
        return "ok"
    except Exception as e:
        (Path(cfg.artifacts_dir) / "xclip_error.txt").write_text(traceback.format_exc())
        _p(progress, 1.0, f"X-CLIP skipped: {e}")
        return f"failed: {e}"


# ---------------------------------------------------------------------------
# Step 6: evaluation & research investigation
# ---------------------------------------------------------------------------
def _scores_for(method: str, q: np.ndarray, frames: np.ndarray, heads: dict, seeds, tau: float,
                pos_idx=None) -> list[np.ndarray]:
    """Return a list of (Q,N) score matrices (one per seed for learned methods)."""
    if method in ("single_frame", "mean_pool", "max_pool"):
        return [q @ pool_video(frames, method).T]
    if method == "query_attn":
        return [query_attn_scores(q, frames, tau)[0]]
    return [q @ head_embed(heads[(method, s)], frames, pos_idx).T for s in seeds if (method, s) in heads]


def step_eval(cfg: Config, progress: ProgressFn = None) -> dict:
    subset = load_subset(cfg)
    F = load_features(cfg)
    fr, Q = F["index_frames"], F["query_emb"]
    gt = np.array([q["video_idx"] for q in subset["queries"]])
    cats = np.array([q["category"] for q in subset["queries"]])
    heads = load_heads(cfg, fr.shape[-1])
    seeds = list(cfg.train_seeds)
    tau = get_tau(cfg)
    methods = PARAM_FREE_METHODS + [m for m in LEARNED_METHODS if any((m, s) in heads for s in seeds)]
    ks = tuple(cfg.ks)
    R: dict = {"config": {**cfg.to_dict(), "train_videos_per_category": cfg.eff_train_per_cat(),
                          "epochs": cfg.eff_epochs()}, "methods": methods, "tau": tau,
               "n_queries": int(len(Q)), "n_videos": int(len(fr))}
    rel = [set() for _ in range(len(fr))]
    for qi, g in enumerate(gt):
        rel[g].add(qi)

    # ---------------- main T2V / V2T tables ----------------
    _p(progress, 0.05, "Computing T2V and V2T metrics for all representations")
    t2v, t2v_std, v2t, v2t_std, per_query = {}, {}, {}, {}, {}
    for m in methods:
        Ss = _scores_for(m, Q, fr, heads, seeds, tau)
        tm = [summarize_ranks(t2v_ranks(S, gt), ks) for S in Ss]
        vm = [v2t_metrics(S.T, rel, ks)[0] for S in Ss]
        t2v[m] = {k: float(np.mean([d[k] for d in tm])) for k in tm[0]}
        t2v_std[m] = {k: float(np.std([d[k] for d in tm])) for k in tm[0]}
        v2t[m] = {k: float(np.mean([d[k] for d in vm])) for k in vm[0]}
        v2t_std[m] = {k: float(np.std([d[k] for d in vm])) for k in vm[0]}
        ranks = np.stack([t2v_ranks(S, gt) for S in Ss])        # (seeds,Q)
        per_query[m] = {"rank_seed0": ranks[0].tolist(), "rr_mean": (1 / ranks).mean(0).tolist(),
                        **{f"hit@{k}": (ranks <= k).mean(0).tolist() for k in ks}}
    xs_path = Path(cfg.artifacts_dir) / "xclip_scores.npy"
    if xs_path.exists():
        S = np.load(xs_path)
        if S.shape == (len(Q), len(fr)):
            t2v["xclip"] = summarize_ranks(t2v_ranks(S, gt), ks)
            v2t["xclip"] = v2t_metrics(S.T, rel, ks)[0]
            t2v_std["xclip"] = {k: 0.0 for k in t2v["xclip"]}
            v2t_std["xclip"] = {k: 0.0 for k in v2t["xclip"]}
            r = t2v_ranks(S, gt)
            per_query["xclip"] = {"rank_seed0": r.tolist(), "rr_mean": (1 / r).tolist(),
                                  **{f"hit@{k}": (r <= k).astype(float).tolist() for k in ks}}
    R.update({"t2v": t2v, "t2v_std": t2v_std, "v2t": v2t, "v2t_std": v2t_std, "per_query": per_query})

    # ---------------- significance tests ----------------
    _p(progress, 0.25, "Paired bootstrap + Wilcoxon significance tests")
    comparisons = [(m, "mean_pool") for m in per_query if m != "mean_pool"]
    comparisons += [("temporal_tfm", "set_tfm"), ("set_tfm", "mlp_adapter"),
                    ("temporal_tfm", "mlp_adapter"), ("seq_lstm", "mlp_adapter")]
    sig, pw = [], {}
    for a, b in comparisons:   # effect = a - b
        if a not in per_query or b not in per_query:
            continue
        pa, pb = per_query[a], per_query[b]
        row = {"method": a, "reference": b}
        row["MRR"] = paired_bootstrap(np.array(pb["rr_mean"]), np.array(pa["rr_mean"]), cfg.bootstrap_samples)
        for k in (1, 5):
            row[f"R@{k}"] = paired_bootstrap(np.array(pb[f"hit@{k}"]) * 100, np.array(pa[f"hit@{k}"]) * 100,
                                             cfg.bootstrap_samples)
        row["wilcoxon_p"] = wilcoxon_p(np.array(pb["rr_mean"]), np.array(pa["rr_mean"]))
        d = np.array(pa["rr_mean"]) - np.array(pb["rr_mean"])
        row["n_better"], row["n_worse"], row["n_tie"] = int((d > 1e-9).sum()), int((d < -1e-9).sum()), int((np.abs(d) <= 1e-9).sum())
        pw[f"{a}|{b}"] = row["wilcoxon_p"]
        sig.append(row)
    adj = holm(pw)
    for row in sig:
        row["wilcoxon_p_holm"] = adj.get(f"{row['method']}|{row['reference']}", float("nan"))
    R["significance"] = sig

    # ---------------- number-of-frames sweep ----------------
    _p(progress, 0.40, "Frame-count sweep")
    T = fr.shape[1]
    sweep = {}
    ks_frames = [k for k in (1, 2, 4, 8, 16, 32) if k <= T]
    for m in methods:
        sweep[m] = {}
        for k in ks_frames:
            sel = segment_indices(T, k)
            Ss = _scores_for(m, Q, fr[:, sel], heads, seeds, tau, pos_idx=sel)
            ms = [summarize_ranks(t2v_ranks(S, gt), ks) for S in Ss]
            sweep[m][str(k)] = {x: float(np.mean([d[x] for d in ms])) for x in ("R@1", "R@5", "MRR")}
    R["frame_sweep"] = sweep

    # ---------------- temporal order ablation (shuffle / reverse) ----------------
    _p(progress, 0.55, "Temporal-order ablation (shuffled / reversed frames)")
    rng = np.random.default_rng(123)
    perm = np.stack([rng.permutation(T) for _ in range(len(fr))])
    fr_shuf = np.take_along_axis(fr, perm[:, :, None], axis=1)
    fr_rev = fr[:, ::-1]
    order = {}
    for m in methods:
        if m == "single_frame":   # its "middle frame" changes under permutation - not meaningful here
            continue
        res = {}
        for name, X in (("original", fr), ("shuffled", fr_shuf), ("reversed", fr_rev)):
            Ss = _scores_for(m, Q, X, heads, seeds, tau)
            ms = [summarize_ranks(t2v_ranks(S, gt), ks) for S in Ss]
            res[name] = {x: float(np.mean([d[x] for d in ms])) for x in ("R@1", "R@5", "MRR")}
        order[m] = res
    R["order_ablation"] = order

    # ---------------- per-category & temporal-language subsets ----------------
    _p(progress, 0.65, "Per-category and temporal-language breakdown")
    names = subset["category_names"]
    per_cat = {}
    for m in per_query:
        rr = np.array(per_query[m]["rr_mean"])
        h5 = np.array(per_query[m]["hit@5"])
        per_cat[m] = {names[c]: {"MRR": float(rr[cats == c].mean()), "R@5": float(h5[cats == c].mean() * 100)}
                      for c in range(len(names)) if (cats == c).any()}
    R["per_category"] = per_cat
    is_temp = np.array([bool(TEMPORAL_WORDS & set(re.findall(r"[a-z]+", q["text"].lower())))
                        for q in subset["queries"]])
    R["temporal_language"] = {
        "n_temporal": int(is_temp.sum()), "n_other": int((~is_temp).sum()),
        "examples": [q["text"] for q, t in zip(subset["queries"], is_temp) if t][:8],
        "by_method": {m: {"temporal_MRR": float(np.array(per_query[m]["rr_mean"])[is_temp].mean()) if is_temp.any() else None,
                          "other_MRR": float(np.array(per_query[m]["rr_mean"])[~is_temp].mean())}
                      for m in per_query}}

    # ---------------- edge-case robustness (query side) ----------------
    _p(progress, 0.72, "Query edge-case robustness experiments")
    try:
        R["robustness"] = robustness_study(cfg, subset, fr, gt)
    except Exception as e:
        R["robustness"] = {"error": str(e)}

    # ---------------- positive control (can the setup detect temporal benefit?) ----------------
    _p(progress, 0.82, "Positive-control experiment (synthetic order-only task)")
    try:
        from .control import run_positive_control
        R["positive_control"] = run_positive_control()
    except Exception as e:
        R["positive_control"] = {"error": str(e)}

    # ---------------- efficiency & training curves ----------------
    _p(progress, 0.90, "Efficiency statistics")
    R["timing"] = json.loads(str(F["timing"]))
    bpath = cfg.index_dir / "build.json"
    if bpath.exists():
        R["timing"].update(json.loads(bpath.read_text()))
    idx = load_index(cfg)
    if idx is not None:
        lat = {}
        for m in idx.methods:
            t0 = time.perf_counter()
            for q in Q:
                idx.search(q, m, 10)
            lat[m] = 1000 * (time.perf_counter() - t0) / len(Q)
        R["timing"]["search_ms_per_query"] = lat
    curves = {}
    for arch in LEARNED_METHODS:
        for s in seeds:
            p = head_path(cfg, arch, s).with_suffix(".json")
            if p.exists():
                curves[f"{arch}_seed{s}"] = json.loads(p.read_text())
    R["training"] = curves
    tp = cfg.models_dir / "tau.json"
    if tp.exists():
        R["tau_selection"] = json.loads(tp.read_text())
    R["dataset"] = subset["protocol"]
    R["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    from .analysis import research_conclusions
    R["conclusions"] = research_conclusions(R)
    cfg.results_path.write_text(json.dumps(R, indent=1, default=float))
    try:
        from .figures import make_all_figures
        make_all_figures(R, cfg.figures_dir)
    except Exception as e:
        print("figure generation failed:", e)
    _p(progress, 1.0, "Evaluation complete")
    return R


def robustness_study(cfg: Config, subset: dict, fr: np.ndarray, gt: np.ndarray) -> dict:
    """Evaluate query-side edge-case handling with the mean-pooling index."""
    from .query import QueryProcessor, inject_typos, shorten
    enc = get_encoder(cfg)
    lex = load_lexicon(cfg)
    qp = QueryProcessor(enc, lex)
    V = pool_video(fr, "mean_pool")
    texts = [q["text"] for q in subset["queries"]]
    rng = np.random.default_rng(7)
    typo = [inject_typos(t, rng) for t in texts]
    short = [shorten(t, 2) for t in texts]
    # "detailed" queries: the query caption + up to 9 other captions of the same clip (often > 77 tokens)
    detailed = []
    for q in subset["queries"]:
        caps = [c for c in subset["index"][q["video_idx"]]["all_captions"] if c != q["text"]][:9]
        detailed.append(". ".join([q["text"]] + caps) + ".")

    def run(ts, **kw):
        E = qp.encode_many(ts, **kw)
        return summarize_ranks(t2v_ranks(E @ V.T, gt), tuple(cfg.ks))

    off = dict(auto_correct=False, use_templates=False, use_chunking=False)
    rows = [
        ("Original captions", "none", run(texts, **off)),
        ("Short (2 content words)", "raw", run(short, **off)),
        ("Short (2 content words)", "prompt-template ensemble", run(short, auto_correct=False, use_templates=True)),
        ("Typo-corrupted", "raw", run(typo, **off)),
        ("Typo-corrupted", "spelling correction", run(typo, auto_correct=True, use_templates=False)),
        ("Detailed (multi-sentence)", "truncate at 77 tokens", run(detailed, **off)),
        ("Detailed (multi-sentence)", "chunk + average", run(detailed, auto_correct=False, use_templates=False, use_chunking=True)),
    ]
    n_long = sum(enc.token_count(d) > enc.max_len for d in detailed)
    n_unknown = sum(1 for t in typo if qp.process(t, auto_correct=False).unknown_words)
    return {"rows": [{"condition": a, "handling": b, **m} for a, b, m in rows],
            "n_long_queries": int(n_long), "n_typo_queries_with_unknown_words": int(n_unknown),
            "examples": {"short": short[:3], "typo": typo[:3], "detailed": detailed[:1]}}


def load_lexicon(cfg: Config) -> Counter:
    p = Path(cfg.data_dir) / "lexicon.json"
    return Counter(json.loads(p.read_text())) if p.exists() else Counter()


def load_results(cfg: Config) -> dict | None:
    return json.loads(cfg.results_path.read_text()) if cfg.results_path.exists() else None


def run_all(cfg: Config, progress: ProgressFn = None, force: bool = False) -> dict:
    steps = [("Data ingestion", lambda p: prepare_subset(cfg, p, force)),
             ("Feature extraction", lambda p: step_features(cfg, p, force)),
             ("Temporal head training", lambda p: step_train(cfg, p, force)),
             ("Index build", lambda p: step_index(cfg, p)),
             ("X-CLIP reference", lambda p: step_xclip(cfg, p, force)),
             ("Evaluation", lambda p: step_eval(cfg, p))]
    out = None
    for i, (name, fn) in enumerate(steps):
        lo, hi = i / len(steps), (i + 1) / len(steps)
        out = fn(lambda f, m, lo=lo, hi=hi, name=name: _p(progress, lo + (hi - lo) * f, f"[{name}] {m}"))
    return out
