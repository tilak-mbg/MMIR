"""Streamlit dashboard - Natural-Language Video Retrieval on MSR-VTT.

Run:   streamlit run app.py
Everything (download, feature extraction, training, indexing, evaluation,
report generation) can be triggered from the "Pipeline Runner" page - no
manual script execution is needed.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from mvr.config import (ALL_SEARCH_METHODS, LEARNED_METHODS, METHOD_INFO, PARAM_FREE_METHODS, Config,
                        method_label)

st.set_page_config(page_title="MSR-VTT Text-to-Video Retrieval", page_icon="🎬", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1400px;}
.res-card {border: 1px solid rgba(128,128,128,.25); border-radius: 10px; padding: .55rem .7rem .3rem; margin-bottom: .6rem;}
.rank-badge {display:inline-block; background:#2a78d6; color:#fff; border-radius: 6px; padding: 0 .45rem; font-weight:600; margin-right:.35rem;}
.gt-badge {display:inline-block; background:#1baf7a; color:#fff; border-radius: 6px; padding: 0 .45rem; font-weight:600; margin-left:.35rem;}
.small-muted {color: rgba(128,128,128,1); font-size: .82rem;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# cached loaders
# ---------------------------------------------------------------------------
def get_cfg() -> Config:
    if "cfg" not in st.session_state:
        st.session_state.cfg = Config.load()
    return st.session_state.cfg


def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_resource(show_spinner="Loading CLIP model...")
def cached_encoder(model: str, device: str):
    from mvr.encoders import ClipEncoder
    return ClipEncoder(model, device)


@st.cache_resource(show_spinner=False)
def cached_index(path: str, mtime: float):
    from mvr.index import VideoIndex
    return VideoIndex.load(Path(path))


@st.cache_data(show_spinner=False)
def cached_json(path: str, mtime: float):
    return json.loads(Path(path).read_text())


@st.cache_data(show_spinner=False)
def cached_features(path: str, mtime: float):
    d = np.load(path, allow_pickle=True)
    return {"query_emb": d["query_emb"], "index_frames": d["index_frames"]}


@st.cache_resource(show_spinner=False)
def cached_lexicon(path: str, mtime: float):
    from collections import Counter
    p = Path(path)
    return Counter(json.loads(p.read_text())) if p.exists() else Counter()


def state(cfg: Config) -> dict:
    return {
        "subset": cfg.subset_path.exists(),
        "features": cfg.features_path.exists(),
        "heads": any(cfg.models_dir.glob(f"*_T{cfg.num_frames}_seed*.pt")) if cfg.models_dir.exists() else False,
        "index": (cfg.index_dir / "meta.json").exists(),
        "results": cfg.results_path.exists(),
    }


def load_subset(cfg):
    return cached_json(str(cfg.subset_path), _mtime(cfg.subset_path)) if cfg.subset_path.exists() else None


def load_results(cfg):
    return cached_json(str(cfg.results_path), _mtime(cfg.results_path)) if cfg.results_path.exists() else None


def load_index(cfg):
    m = cfg.index_dir / "meta.json"
    return cached_index(str(cfg.index_dir), _mtime(m)) if m.exists() else None


def need(cfg, *keys) -> bool:
    s = state(cfg)
    missing = [k for k in keys if not s[k]]
    if missing:
        st.info("This page needs the pipeline outputs: **" + ", ".join(missing) + "**. "
                "Open **⚙️ Pipeline Runner** in the sidebar and click **Run full pipeline**.")
        return False
    return True


def fmt_metrics_df(table: dict, std: dict | None = None, cols=None) -> pd.DataFrame:
    cols = cols or ["R@1", "R@5", "R@10", "MRR", "mAP", "nDCG@10", "MdR", "MnR"]
    rows = []
    for m, d in table.items():
        r = {"Method": method_label(m)}
        for c in cols:
            v = d.get(c)
            s = (std or {}).get(m, {}).get(c, 0.0)
            if v is None:
                r[c] = ""
            elif c.startswith("R@"):
                r[c] = f"{v:.1f}" + (f" ± {s:.1f}" if s else "")
            elif c in ("MdR", "MnR"):
                r[c] = f"{v:.1f}" + (f" ± {s:.1f}" if s else "")
            else:
                r[c] = f"{v:.3f}" + (f" ± {s:.3f}" if s else "")
        rows.append(r)
    return pd.DataFrame(rows).set_index("Method")


FAMILY_COLORS = {"Parameter-free": "#2a78d6", "Learned head": "#eb6834", "External video model": "#1baf7a"}


def fam(m):
    return "Parameter-free" if m in PARAM_FREE_METHODS else ("Learned head" if m in LEARNED_METHODS else "External video model")


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------
cfg = get_cfg()
PAGES = ["🏠 Overview", "⚙️ Pipeline Runner", "🔎 Text → Video Search", "🎞️ Video → Text",
         "📊 Evaluation", "🔬 Research Investigation", "📚 Dataset Browser", "📄 Report"]
with st.sidebar:
    st.markdown("### 🎬 MSR-VTT Video Retrieval")
    page = st.radio("Navigate", PAGES, key="page", label_visibility="collapsed")
    st.divider()
    s = state(cfg)
    st.caption("Pipeline status")
    for k, label in (("subset", "Data (100 clips / 200 queries)"), ("features", "CLIP frame features"),
                     ("heads", "Temporal heads"), ("index", "Vector index"), ("results", "Evaluation results")):
        st.markdown(("✅ " if s[k] else "⬜ ") + label)
    st.divider()
    st.caption(f"Model: `{cfg.clip_model}`  \nFrames/clip: {cfg.num_frames}  \nDevice: `{cfg.resolved_device()}`")


# ===========================================================================
# PAGE: Overview
# ===========================================================================
def page_overview():
    st.title("Text-to-Video Retrieval Using Semantic and Temporal Video Representations")
    st.markdown("**Dataset:** MSR-VTT · **Index:** 100 clips (5 × 20 categories) · **Queries:** 200 (2 captions/clip) "
                "· **Search corpus matrix:** 200 × 100")
    R = load_results(cfg)
    if R:
        t = R["t2v"]
        best = max(t, key=lambda m: t[m]["MRR"])
        c = st.columns(5)
        c[0].metric("Mean-pool R@1", f"{t['mean_pool']['R@1']:.1f}%")
        c[1].metric("Mean-pool R@5", f"{t['mean_pool']['R@5']:.1f}%")
        c[2].metric("Mean-pool MRR", f"{t['mean_pool']['MRR']:.3f}")
        c[3].metric("Best method (MRR)", method_label(best).split(" (")[0], f"{t[best]['MRR']-t['mean_pool']['MRR']:+.3f}")
        c[4].metric("Median rank (mean-pool)", f"{t['mean_pool']['MdR']:.0f}")
        st.success("**Research answer:** " + R["conclusions"]["answer"])
    else:
        st.info("No results yet - go to **⚙️ Pipeline Runner** and click **Run full pipeline**.")

    st.subheader("System architecture")
    st.graphviz_chart("""
digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10];
 subgraph cluster_off { label="Offline indexing"; style=dashed; color=gray;
   V [label="MSR-VTT clips\\n100 = 5 x 20 categories", fillcolor="#dbe8f8"];
   S [label="Frame sampling\\nT uniform segment mid-points", fillcolor="#dbe8f8"];
   E [label="CLIP ViT image encoder\\n(frozen) -> 512-d", fillcolor="#dbe8f8"];
   A [label="Temporal aggregation\\nmean | max | single | query-attn\\nSet-TFM | Temporal-TFM | LSTM | MLP", fillcolor="#dbe8f8"];
   I [label="FAISS IndexFlatIP\\n(L2-normalised vectors)", fillcolor="#d3f1e6"]; V->S->E->A->I; }
 subgraph cluster_tr { label="Training (disjoint train clips)"; style=dashed; color=gray;
   TR [label="Train clips + 20 captions\\nfrozen CLIP features", fillcolor="#fde3d8"];
   L [label="Symmetric InfoNCE\\n3 seeds, early stop on val", fillcolor="#fde3d8"]; TR->L; }
 L -> A [style=dashed, label="weights"];
 subgraph cluster_on { label="Online query"; style=dashed; color=gray;
   Q [label="Streamlit query box", fillcolor="#fbefcf"];
   P [label="Query processing\\nnormalise | spell-fix | templates | chunking", fillcolor="#fbefcf"];
   T [label="CLIP text encoder\\n(frozen) -> 512-d", fillcolor="#fbefcf"];
   K [label="Cosine similarity\\nTop-K + per-frame scores", fillcolor="#d3f1e6"];
   O [label="Playable clips + scores", fillcolor="#fbefcf"]; Q->P->T->K->O; }
 I -> K;
}""", use_container_width=True)

    st.subheader("Representations compared (Research Investigation 1)")
    rows = []
    for m in ALL_SEARCH_METHODS + ["xclip"]:
        rows.append({"Method": method_label(m), "Description": METHOD_INFO[m][1],
                     "Uses frame order": "yes" if m in ("temporal_tfm", "seq_lstm", "xclip") else "no",
                     "Trained": "yes" if m in LEARNED_METHODS else ("pre-trained only" if m == "xclip" else "no (zero-shot)")})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ===========================================================================
# PAGE: Pipeline runner
# ===========================================================================
def page_runner():
    st.title("⚙️ Pipeline Runner")
    st.caption("Runs the complete pipeline from the front-end. Each step caches its output; re-running skips finished steps "
               "unless *Force* is ticked. Keep this page open while it runs.")
    with st.form("settings"):
        c1, c2, c3 = st.columns(3)
        models = ["openai/clip-vit-base-patch32", "openai/clip-vit-base-patch16", "openai/clip-vit-large-patch14"]
        cur = cfg.clip_model if cfg.clip_model in models else None
        clip_model = c1.selectbox("CLIP backbone", models + ([cfg.clip_model] if cur is None else []),
                                  index=(models + [cfg.clip_model]).index(cfg.clip_model))
        num_frames = c1.select_slider("Frames sampled per clip", [4, 8, 12, 16, 24, 32], value=cfg.num_frames)
        device = c1.selectbox("Device", ["auto", "cuda", "cpu"], index=["auto", "cuda", "cpu"].index(cfg.device))
        tv = c2.slider("Training clips per category (temporal heads)", 0, 150, cfg.train_videos_per_category, 5,
                       help="Disjoint from the 100 index clips. 0 = skip learned heads.")
        epochs = c2.slider("Training epochs", 5, 100, cfg.epochs, 5)
        n_seeds = c2.slider("Random seeds per head", 1, 5, len(cfg.train_seeds))
        cpu_fast = c3.checkbox("CPU-fast mode (if no GPU: ≤25 training clips/category, ≤20 epochs)", cfg.cpu_fast)
        use_x = c3.checkbox("Also evaluate X-CLIP (pretrained spatio-temporal video model)", cfg.use_xclip)
        seed = c3.number_input("Sampling seed", 0, 10_000, cfg.seed)
        local = c3.text_input("Offline mirror of the HF dataset repo (optional)", cfg.local_repo_dir,
                              help="Folder containing msrvtt_test_1k.json, msrvtt_train_9k.json, raw_data/, MSRVTT_Videos.zip")
        if st.form_submit_button("💾 Save settings"):
            cfg.clip_model, cfg.num_frames, cfg.device = clip_model, int(num_frames), device
            cfg.train_videos_per_category, cfg.epochs = int(tv), int(epochs)
            cfg.train_seeds = tuple(range(n_seeds))
            cfg.cpu_fast = cpu_fast
            cfg.use_xclip, cfg.seed, cfg.local_repo_dir = use_x, int(seed), local.strip()
            cfg.save()
            st.success("Settings saved.")

    force = st.checkbox("Force re-run (ignore cached outputs)")
    c = st.columns(7)
    run_all = c[0].button("▶️ Run full pipeline", type="primary", use_container_width=True)
    steps = {"1 · Data": c[1].button("1 · Data", use_container_width=True),
             "2 · Features": c[2].button("2 · Features", use_container_width=True),
             "3 · Train heads": c[3].button("3 · Train heads", use_container_width=True),
             "4 · Index": c[4].button("4 · Index", use_container_width=True),
             "5 · X-CLIP": c[5].button("5 · X-CLIP", use_container_width=True),
             "6 · Evaluate": c[6].button("6 · Evaluate", use_container_width=True)}
    chosen = [k for k, v in steps.items() if v]
    if run_all or chosen:
        from mvr import pipeline as P
        from mvr.data import prepare_subset
        bar = st.progress(0.0, text="Starting...")
        log = st.empty()
        lines: list[str] = []

        def prog(f, msg):
            bar.progress(float(min(max(f, 0), 1)), text=msg[:150])
            if not lines or lines[-1] != msg:
                lines.append(f"{time.strftime('%H:%M:%S')}  {msg}")
                log.code("\n".join(lines[-14:]))

        t0 = time.time()
        try:
            if run_all:
                P.run_all(cfg, prog, force)
            else:
                fns = {"1 · Data": lambda: prepare_subset(cfg, prog, force),
                       "2 · Features": lambda: P.step_features(cfg, prog, force),
                       "3 · Train heads": lambda: P.step_train(cfg, prog, force),
                       "4 · Index": lambda: P.step_index(cfg, prog),
                       "5 · X-CLIP": lambda: P.step_xclip(cfg, prog, force),
                       "6 · Evaluate": lambda: P.step_eval(cfg, prog)}
                for k in chosen:
                    fns[k]()
            st.cache_data.clear()
            bar.progress(1.0, text="Done")
            st.success(f"Finished in {time.time()-t0:.0f} s. Open **🔎 Text → Video Search** or **📊 Evaluation**.")
        except Exception as e:
            import traceback
            st.error(f"Pipeline failed: {e}")
            st.code(traceback.format_exc())

    st.divider()
    st.subheader("Cached outputs")
    files = [cfg.subset_path, cfg.features_path, cfg.models_dir, cfg.index_dir, cfg.results_path,
             Path(cfg.artifacts_dir) / "xclip_scores.npy"]
    st.dataframe(pd.DataFrame([{"Output": str(Path(f).relative_to(Path(cfg.artifacts_dir).parent)) if str(f).startswith(str(Path(cfg.artifacts_dir).parent)) else str(f),
                                "Exists": "✅" if Path(f).exists() else "—",
                                "Modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(_mtime(Path(f)))) if Path(f).exists() else ""}
                               for f in files]), hide_index=True, use_container_width=True)
    xerr = Path(cfg.artifacts_dir) / "xclip_error.txt"
    if xerr.exists() and not (Path(cfg.artifacts_dir) / "xclip_scores.npy").exists():
        with st.expander("X-CLIP was skipped (optional) - details"):
            st.code(xerr.read_text()[-3000:])


# ===========================================================================
# PAGE: Text -> Video search
# ===========================================================================
EXAMPLES = ["a person is playing guitar on a stage", "a man is cooking food in a kitchen",
            "cars are racing on a track", "a cartoon character is talking", "a news anchor reports the news",
            "people are dancing at a party", "a dog is playing with a ball", "basketball",
            "a woman playng gutiar on stge",
            "a group of young people stand on a stage and sing a song while the crowd cheers, then the camera "
            "pans across the audience who are clapping and waving their hands in the air, and finally the lead "
            "singer bows and walks off the stage as the lights go down and the music slowly fades out"]


def _query_processor():
    from mvr.query import QueryProcessor
    enc = cached_encoder(cfg.clip_model, cfg.resolved_device())
    lexp = Path(cfg.data_dir) / "lexicon.json"
    return QueryProcessor(enc, cached_lexicon(str(lexp), _mtime(lexp)))


@st.cache_data(show_spinner=False)
def low_conf_threshold(index_mtime: float, method: str) -> float:
    """5th percentile of the ground-truth match score over the 200 eval queries."""
    idx = load_index(cfg)
    F = cached_features(str(cfg.features_path), _mtime(cfg.features_path))
    sub = load_subset(cfg)
    S = idx.scores(F["query_emb"], method)
    gt = np.array([q["video_idx"] for q in sub["queries"]])
    return float(np.percentile(S[np.arange(len(gt)), gt], 5))


def page_search():
    st.title("🔎 Text → Video Search")
    if not need(cfg, "subset", "features", "index"):
        return
    idx, sub = load_index(cfg), load_subset(cfg)
    eval_queries = [q["text"] for q in sub["queries"]]

    c1, c2 = st.columns([3, 1])
    with c2:
        ex = st.selectbox("Example queries", ["(type your own)"] + EXAMPLES + ["— from evaluation set —"] + eval_queries[:60])
    default = "" if ex.startswith("(") or ex.startswith("—") else ex
    with c1:
        with st.form("q"):
            query = st.text_input("Describe the video you are looking for", value=default,
                                  placeholder="e.g. A person is playing guitar on a stage.")
            a, b, c = st.columns([2, 1, 1])
            methods = [m for m in ALL_SEARCH_METHODS if m in idx.methods]
            method = a.selectbox("Video representation", methods, index=methods.index("mean_pool"),
                                 format_func=method_label)
            k = b.slider("Top-K", 1, 20, 5)
            compare = c.checkbox("Compare with mean-pool", value=False)
            o1, o2, o3 = st.columns(3)
            ac = o1.checkbox("Spelling correction", True)
            tp = o2.checkbox("Templates for short queries", True)
            ch = o3.checkbox("Chunk long queries", True)
            go = st.form_submit_button("Search", type="primary")
    if not go and not query:
        st.caption(f"**{method_label(method)}** – {METHOD_INFO[method][1]}")
        return

    from mvr.query import QueryError
    qp = _query_processor()
    try:
        t0 = time.perf_counter()
        pq = qp.process(query, auto_correct=ac, use_templates=tp, use_chunking=ch)
        q = qp.encode(pq)
        enc_ms = (time.perf_counter() - t0) * 1000
    except QueryError as e:
        st.error(str(e))
        return

    res = idx.search(q, method, k)
    thr = low_conf_threshold(_mtime(cfg.index_dir / "meta.json"), method)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Query encoding", f"{enc_ms:.0f} ms")
    m2.metric("Index search", f"{res['latency_ms']:.2f} ms")
    m3.metric("Top-1 cosine score", f"{res['scores'][0]:.3f}")
    m4.metric("Query mode", pq.mode)
    with st.expander("Query processing diagnostics (edge-case handling)", expanded=bool(pq.warnings or pq.corrections or pq.mode != "normal")):
        st.write({"normalised": pq.normalized, "final text": pq.final_text, "BPE tokens": pq.n_tokens,
                  "texts encoded": pq.texts_to_encode, "steps": pq.steps or ["none"],
                  "unknown words": pq.unknown_words or ["none"]})
    for w in pq.warnings:
        st.warning(w)
    if res["scores"][0] < thr:
        st.warning(f"Low confidence: the best score ({res['scores'][0]:.3f}) is below the 5th percentile of correct-match "
                   f"scores on the evaluation set ({thr:.3f}). The collection may not contain a matching clip.")

    gt_vid = None
    if query.strip() in eval_queries:
        qi = eval_queries.index(query.strip())
        gt_vid = sub["queries"][qi]["video_idx"]
        S_all = idx.scores(q, method)[0]
        gt_rank = int(1 + (S_all > S_all[gt_vid]).sum())
        st.info(f"This is evaluation query #{qi}. Ground-truth clip **{sub['index'][gt_vid]['video_id']}** is ranked "
                f"**{gt_rank}** / {len(sub['index'])} by {method_label(method)}.")

    tbl = pd.DataFrame([{"Rank": r + 1, "Video": sub["index"][i]["video_id"],
                         "Category": sub["index"][i]["category_name"], "Cosine score": round(s, 4),
                         "Ground truth": "✅" if i == gt_vid else ""}
                        for r, (i, s) in enumerate(zip(res["idx"], res["scores"]))])
    st.dataframe(tbl, hide_index=True, use_container_width=True)

    st.subheader(f"Top-{k} clips · {method_label(method)}")
    ncol = 3
    for row in range(0, len(res["idx"]), ncol):
        cols = st.columns(ncol)
        for j, col in enumerate(cols):
            r = row + j
            if r >= len(res["idx"]):
                break
            vi, score = res["idx"][r], res["scores"][r]
            v = sub["index"][vi]
            with col:
                st.markdown(f"<div class='res-card'><span class='rank-badge'>#{r+1}</span>"
                            f"<b>{v['video_id']}</b> · {v['category_name']} · score <b>{score:.3f}</b>"
                            + ("<span class='gt-badge'>GT</span>" if vi == gt_vid else "") + "</div>",
                            unsafe_allow_html=True)
                vp = cfg.videos_dir / f"{v['video_id']}.mp4"
                if vp.exists():
                    st.video(str(vp))
                fs = res["frame_sims"][r]
                st.bar_chart(pd.DataFrame({"frame similarity": fs}, index=[f"f{t:02d}" for t in range(len(fs))]),
                             height=120, color="#2a78d6")
                with st.expander("Captions & frames"):
                    th = Path(cfg.thumbs_dir) / f"{v['video_id']}.jpg"
                    if th.exists():
                        st.image(str(th), use_container_width=True)
                    for cap in v["all_captions"][:5]:
                        st.markdown(f"- {cap}")

    if compare and method != "mean_pool":
        st.subheader("Mean-pool baseline for the same query")
        r2 = idx.search(q, "mean_pool", k)
        st.dataframe(pd.DataFrame([{"Rank": r + 1, "Video": sub["index"][i]["video_id"],
                                    "Category": sub["index"][i]["category_name"], "Cosine score": round(s, 4)}
                                   for r, (i, s) in enumerate(zip(r2["idx"], r2["scores"]))]),
                     hide_index=True, use_container_width=True)


# ===========================================================================
# PAGE: Video -> Text
# ===========================================================================
def page_v2t():
    st.title("🎞️ Video → Text Retrieval")
    st.caption("The reverse modality: given a clip, rank the 200 caption queries. Each clip has 2 relevant captions.")
    if not need(cfg, "subset", "features", "index"):
        return
    idx, sub = load_index(cfg), load_subset(cfg)
    F = cached_features(str(cfg.features_path), _mtime(cfg.features_path))
    opts = list(range(len(sub["index"])))
    c1, c2, c3 = st.columns([2, 2, 1])
    vi = c1.selectbox("Clip", opts, format_func=lambda i: f"{sub['index'][i]['video_id']} · {sub['index'][i]['category_name']}")
    methods = [m for m in ALL_SEARCH_METHODS if m in idx.methods]
    method = c2.selectbox("Video representation", methods, index=methods.index("mean_pool"), format_func=method_label)
    k = c3.slider("Top-K captions", 1, 20, 10)
    S = idx.scores(F["query_emb"], method)[:, vi]
    order = np.argsort(-S)[:k]
    left, right = st.columns([1, 1])
    with left:
        st.video(str(cfg.videos_dir / f"{sub['index'][vi]['video_id']}.mp4"))
        th = Path(cfg.thumbs_dir) / f"{sub['index'][vi]['video_id']}.jpg"
        if th.exists():
            st.image(str(th), caption="Sampled frames", use_container_width=True)
    with right:
        st.dataframe(pd.DataFrame([{"Rank": r + 1, "Caption": sub["queries"][qi]["text"], "Score": round(float(S[qi]), 4),
                                    "Relevant": "✅" if sub["queries"][qi]["video_idx"] == vi else ""}
                                   for r, qi in enumerate(order)]), hide_index=True, use_container_width=True)


# ===========================================================================
# PAGE: Evaluation
# ===========================================================================
def page_eval():
    import altair as alt
    st.title("📊 Quantitative Performance Evaluation")
    if not need(cfg, "results"):
        return
    R = load_results(cfg)
    st.caption(f"{R['n_queries']} queries × {R['n_videos']} clips · backbone `{R['config']['clip_model']}` · "
               f"{R['config']['num_frames']} frames/clip · learned heads: mean ± sd over seeds {R['config']['train_seeds']} · "
               f"generated {R['generated_at']}")
    t1, t2, t3, t4, t5, t6 = st.tabs(["Text→Video", "Video→Text", "Charts", "Frames & per-category",
                                      "Edge-case robustness", "Efficiency"])
    with t1:
        st.dataframe(fmt_metrics_df(R["t2v"], R["t2v_std"]), use_container_width=True)
        st.caption("R@K in %, higher is better · MdR/MnR = median/mean rank of the correct clip, lower is better · "
                   "with one relevant clip per query, mAP = MRR.")
    with t2:
        st.dataframe(fmt_metrics_df(R["v2t"], R["v2t_std"]), use_container_width=True)
        st.caption("Video→Text: 2 relevant captions per clip; R@K = at least one relevant caption in top-K; mAP uses both.")
    with t3:
        rows = []
        for m, d in R["t2v"].items():
            for kk in ("R@1", "R@5", "R@10"):
                rows.append({"Method": method_label(m), "Metric": kk, "Value": d[kk], "Family": fam(m)})
        df = pd.DataFrame(rows)
        order = [method_label(m) for m in R["t2v"]]
        mets = ["R@1", "R@5", "R@10"]
        ch = alt.Chart(df).mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3).encode(
            y=alt.Y("Method:N", sort=order, title=None, axis=alt.Axis(labelLimit=260, labelOverlap=False)),
            x=alt.X("Value:Q", title="Recall (%)"),
            yOffset=alt.YOffset("Metric:N", sort=mets),
            color=alt.Color("Metric:N", sort=mets, scale=alt.Scale(domain=mets, range=["#2a78d6", "#eb6834", "#1baf7a"])),
            tooltip=["Method", "Metric", alt.Tooltip("Value:Q", format=".1f")]).properties(height=60 * len(order),
                                                                                           title="Text→Video Recall@K")
        st.altair_chart(ch, use_container_width=True)
        mdf = pd.DataFrame([{"Method": method_label(m), "MRR": d["MRR"], "sd": R["t2v_std"].get(m, {}).get("MRR", 0),
                             "Family": fam(m)} for m, d in R["t2v"].items()])
        mdf["lo"], mdf["hi"] = mdf.MRR - mdf.sd, mdf.MRR + mdf.sd
        base = alt.Chart(mdf).encode(y=alt.Y("Method:N", sort=order, title=None, axis=alt.Axis(labelLimit=260, labelOverlap=False)))
        bars = base.mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3).encode(
            x=alt.X("MRR:Q", title="MRR"),
            color=alt.Color("Family:N", scale=alt.Scale(domain=list(FAMILY_COLORS), range=list(FAMILY_COLORS.values()))),
            tooltip=["Method", alt.Tooltip("MRR:Q", format=".3f"), alt.Tooltip("sd:Q", format=".3f")])
        err = base.mark_errorbar(ticks=True).encode(x=alt.X("lo:Q", title="MRR"), x2="hi:Q")
        rule = alt.Chart(pd.DataFrame({"x": [R["t2v"]["mean_pool"]["MRR"]]})).mark_rule(strokeDash=[4, 4], color="gray").encode(x="x:Q")
        st.altair_chart((bars + err + rule).properties(height=44 * len(order), title="MRR ± sd (dashed = mean-pool baseline)"),
                        use_container_width=True)
    with t4:
        sw = R["frame_sweep"]
        df = pd.DataFrame([{"Method": method_label(m), "Frames": int(k), "MRR": v["MRR"], "R@1": v["R@1"]}
                           for m in sw for k, v in sw[m].items()])
        sel = st.multiselect("Methods", sorted(df.Method.unique()),
                             default=[method_label(m) for m in ("mean_pool", "max_pool", "query_attn", "temporal_tfm") if m in sw])
        ch = alt.Chart(df[df.Method.isin(sel)]).mark_line(point=alt.OverlayMarkDef(size=60), strokeWidth=2).encode(
            x=alt.X("Frames:Q", scale=alt.Scale(type="log", base=2)), y="MRR:Q", color="Method:N",
            tooltip=["Method", "Frames", alt.Tooltip("MRR:Q", format=".3f"), alt.Tooltip("R@1:Q", format=".1f")]
        ).properties(height=320, title="MRR vs frames sampled per clip")
        st.altair_chart(ch, use_container_width=True)
        pc = R["per_category"]
        metric = st.radio("Per-category metric", ["R@5", "MRR"], horizontal=True)
        hd = pd.DataFrame([{"Category": c, "Method": method_label(m), metric: v[metric]} for m in pc for c, v in pc[m].items()])
        hm = alt.Chart(hd).mark_rect().encode(
            x=alt.X("Method:N", sort=[method_label(m) for m in pc], axis=alt.Axis(labelAngle=-35, labelLimit=160)),
            y="Category:N", color=alt.Color(f"{metric}:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["Category", "Method", alt.Tooltip(f"{metric}:Q", format=".3f")]).properties(height=520)
        st.altair_chart(hm, use_container_width=True)
    with t5:
        rb = R.get("robustness", {})
        if "rows" in rb:
            st.dataframe(pd.DataFrame([{"Query condition": r["condition"], "Handling": r["handling"],
                                        "R@1": round(r["R@1"], 1), "R@5": round(r["R@5"], 1), "MRR": round(r["MRR"], 3),
                                        "MdR": r["MdR"]} for r in rb["rows"]]), hide_index=True, use_container_width=True)
            st.caption(f"Mean-pool index. Typo queries containing unknown words: {rb['n_typo_queries_with_unknown_words']} / 200; "
                       f"detailed queries exceeding 77 tokens: {rb['n_long_queries']} / 200.")
            with st.expander("Examples of perturbed queries"):
                st.json(rb["examples"])
        else:
            st.warning(f"Robustness study unavailable: {rb.get('error')}")
    with t6:
        tm = R.get("timing", {})
        c = st.columns(4)
        c[0].metric("Decode / clip", f"{tm.get('decode_s_per_video', 0)*1000:.0f} ms")
        c[1].metric("CLIP encode / clip", f"{tm.get('encode_s_per_video', 0)*1000:.0f} ms")
        c[2].metric("Text encode / query", f"{tm.get('text_encode_ms_per_query', 0):.1f} ms")
        c[3].metric("Index build", f"{tm.get('build_ms', 0):.1f} ms")
        if "search_ms_per_query" in tm:
            st.dataframe(pd.DataFrame([{"Method": method_label(m), "Search latency (ms/query, top-10)": round(v, 3)}
                                       for m, v in tm["search_ms_per_query"].items()]), hide_index=True)
        st.caption(f"Device: {tm.get('device')} · index backend: {tm.get('backend', 'n/a')}")


# ===========================================================================
# PAGE: Research investigation
# ===========================================================================
def page_research():
    import altair as alt
    st.title("🔬 Research Investigation 1")
    st.markdown("> *Is mean-pooling frame embeddings sufficient, or does explicitly modelling temporal relationships "
                "provide superior retrieval performance?*")
    if not need(cfg, "results"):
        return
    R = load_results(cfg)
    C = R["conclusions"]
    st.subheader("Answer")
    {"temporal_better": st.success, "adaptation_not_order": st.info, "mean_pool_sufficient": st.info,
     "mixed": st.warning}.get(C["verdict"], st.info)(C["answer"])
    st.subheader("Evidence")
    for p in C["points"]:
        st.markdown(f"- {p}")

    st.subheader("Experimental design (controlled ablation ladder)")
    st.markdown("""
| Step | Representation | What it adds | Isolates |
|---|---|---|---|
| 0 | Single middle frame | – | value of using many frames |
| 1 | **Mean pooling** (baseline) | average of all frames | – |
| 1b | Max / query-attentive pooling | non-uniform weighting (zero-shot) | smarter order-free pooling |
| 2 | MLP adapter (trained) | in-domain fine-tuning, **no** frame interaction | domain adaptation |
| 3 | Set Transformer (trained) | frame–frame attention, **no** order | inter-frame interaction |
| 4 | Temporal Transformer (trained) | + temporal position embeddings | **temporal order** |
| 4b | Temporal LSTM (trained) | recurrent ordered model | temporal order (alt. architecture) |
| ext | X-CLIP (pre-trained) | native spatio-temporal video encoder | different model family |

All learned heads share the frozen CLIP features, data (disjoint MSR-VTT training clips), loss (symmetric InfoNCE),
optimiser and early-stopping rule, are zero-initialised residuals of mean pooling, and are trained with 3 seeds.
Significance: paired bootstrap (10k resamples) 95% CIs on ΔMRR/ΔR@K and Wilcoxon signed-rank tests on per-query
reciprocal ranks with Holm–Bonferroni correction.""")

    st.subheader("Paired significance tests")
    rows = []
    for r in R["significance"]:
        rows.append({"Comparison": f"{method_label(r['method'])}  vs  {method_label(r['reference'])}",
                     "ΔMRR": f"{r['MRR']['delta']:+.3f}", "95% CI (MRR)": f"[{r['MRR']['ci_low']:+.3f}, {r['MRR']['ci_high']:+.3f}]",
                     "ΔR@1 (pts)": f"{r['R@1']['delta']:+.1f}", "ΔR@5 (pts)": f"{r['R@5']['delta']:+.1f}",
                     "Wilcoxon p": f"{r['wilcoxon_p']:.3g}", "Holm p": f"{r['wilcoxon_p_holm']:.3g}",
                     "better / worse / tie": f"{r['n_better']} / {r['n_worse']} / {r['n_tie']}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        oa = R["order_ablation"]
        df = pd.DataFrame([{"Method": method_label(m), "Frame order": c, "MRR": oa[m][c]["MRR"]}
                           for m in oa for c in ("original", "shuffled", "reversed")])
        conds = ["original", "shuffled", "reversed"]
        ch = alt.Chart(df).mark_bar().encode(
            y=alt.Y("Method:N", sort=[method_label(m) for m in oa], axis=alt.Axis(labelLimit=220, labelOverlap=False), title=None),
            yOffset=alt.YOffset("Frame order:N", sort=conds), x=alt.X("MRR:Q", scale=alt.Scale(zero=False)),
            color=alt.Color("Frame order:N", sort=conds, scale=alt.Scale(domain=conds, range=["#2a78d6", "#eb6834", "#1baf7a"])),
            tooltip=["Method", "Frame order", alt.Tooltip("MRR:Q", format=".4f")]).properties(
            height=48 * len(oa), title="Test-time temporal-order ablation")
        st.altair_chart(ch, use_container_width=True)
    with c2:
        tl = R["temporal_language"]
        st.markdown(f"**Queries with temporal language:** {tl['n_temporal']} of {tl['n_temporal'] + tl['n_other']}")
        st.dataframe(pd.DataFrame([{"Method": method_label(m), "MRR (temporal-language queries)": v["temporal_MRR"],
                                    "MRR (other queries)": v["other_MRR"]} for m, v in tl["by_method"].items()]),
                     hide_index=True, use_container_width=True)
        if tl["examples"]:
            with st.expander("Examples"):
                for e in tl["examples"]:
                    st.markdown(f"- {e}")
    pc = R.get("positive_control", {}).get("mrr")
    if pc:
        st.subheader("Positive control – synthetic task where only frame order matters")
        st.caption("Clips show state A then B (or B then A); captions encode the ordered pair. Confirms that the "
                   "temporal heads/evaluation CAN detect a temporal benefit when it exists.")
        st.dataframe(pd.DataFrame([{"Method": method_label(m), "MRR": round(v, 3)} for m, v in pc.items()]),
                     hide_index=True)
    tr = R.get("training", {})
    if tr:
        st.subheader("Training curves (validation MRR on held-out TRAIN clips)")
        df = pd.DataFrame([{"Run": k, "Head": method_label(k.rsplit("_seed", 1)[0]), "Epoch": e, "Val MRR": v}
                           for k, h in tr.items() for e, v in zip(h["epoch"], h["val_mrr"])])
        ch = alt.Chart(df).mark_line(strokeWidth=1.6).encode(x="Epoch:Q", y=alt.Y("Val MRR:Q", scale=alt.Scale(zero=False)),
                                                             color=alt.Color("Head:N", scale=alt.Scale(range=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"])), detail="Run:N",
                                                             tooltip=["Run", "Epoch", alt.Tooltip("Val MRR:Q", format=".3f")])
        st.altair_chart(ch.properties(height=300), use_container_width=True)
        st.dataframe(pd.DataFrame([{"Run": k, "Best epoch": h["best_epoch"], "Best val MRR": round(h["best_val_mrr"], 3),
                                    "Params": h["n_params"], "Train clips": h["n_train_videos"], "Seconds": h["train_seconds"]}
                                   for k, h in tr.items()]), hide_index=True, use_container_width=True)
    if "tau_selection" in R:
        st.caption(f"Query-attentive pooling temperature τ = {R['tau']} selected on the training pool: "
                   + ", ".join(f"τ={k}: MRR {v:.3f}" for k, v in R["tau_selection"]["grid_mrr"].items()))


# ===========================================================================
# PAGE: Dataset browser
# ===========================================================================
def page_dataset():
    st.title("📚 Dataset Browser")
    if not need(cfg, "subset"):
        return
    sub = load_subset(cfg)
    p = sub["protocol"]
    c = st.columns(4)
    c[0].metric("Index clips", p["n_index"])
    c[1].metric("Queries", p["n_queries"])
    c[2].metric("Categories", len(sub["category_names"]))
    c[3].metric("Training clips (disjoint)", p["n_train"])
    t1, t2 = st.tabs(["Clips", "Queries (200)"])
    with t1:
        cat = st.selectbox("Category", sub["category_names"])
        vids = [v for v in sub["index"] if v["category_name"] == cat]
        cols = st.columns(len(vids) if vids else 1)
        for col, v in zip(cols, vids):
            with col:
                st.video(str(cfg.videos_dir / f"{v['video_id']}.mp4"))
                st.markdown(f"**{v['video_id']}** · {v.get('duration', 0)} s · {v.get('width', '?')}×{v.get('height', '?')}")
                for qq in v["queries"]:
                    st.caption("🔍 " + qq)
    with t2:
        f = st.text_input("Filter queries")
        df = pd.DataFrame([{"qid": q["qid"], "Query": q["text"], "Video": q["video_id"],
                            "Category": sub["category_names"][q["category"]]} for q in sub["queries"]])
        if f:
            df = df[df.Query.str.contains(f, case=False)]
        st.dataframe(df, hide_index=True, use_container_width=True, height=520)


# ===========================================================================
# PAGE: Report
# ===========================================================================
def page_report():
    st.title("📄 Technical Report")
    if not need(cfg, "results"):
        return
    out = Path(cfg.artifacts_dir) / "MIR_Assignment2_Technical_Report.docx"
    if st.button("Generate / refresh report (.docx)", type="primary"):
        from mvr.report import build_report
        with st.spinner("Building report..."):
            build_report(cfg, out)
        st.success("Report generated.")
    if out.exists():
        st.download_button("⬇️ Download report", out.read_bytes(), file_name=out.name,
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    figs = sorted(Path(cfg.figures_dir).glob("*.png")) if Path(cfg.figures_dir).exists() else []
    for f in figs:
        st.image(str(f), caption=f.stem, use_container_width=True)


{"🏠 Overview": page_overview, "⚙️ Pipeline Runner": page_runner, "🔎 Text → Video Search": page_search,
 "🎞️ Video → Text": page_v2t, "📊 Evaluation": page_eval, "🔬 Research Investigation": page_research,
 "📚 Dataset Browser": page_dataset, "📄 Report": page_report}[page]()
