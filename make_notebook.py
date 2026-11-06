"""Builds MIR_Assignment2.ipynb - the graded Jupyter notebook.

The notebook is self-contained: every source module is embedded with
%%writefile (so all code is visible and graded inside the notebook), then each
computation is executed step by step with its output, and finally the
Streamlit dashboard is launched from the notebook.

Re-run this script whenever a module changes:   python make_notebook.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
nb = nbf.v4.new_notebook()
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


# ---------------------------------------------------------------------------
md(r"""
# Assignment 2 – Natural-Language Video Retrieval Using MSR-VTT
## Text-to-Video Retrieval Using Semantic and Temporal Video Representations

**Course:** Multimodal Information Retrieval (AIMLZG543) · **Environment:** BITS Virtual Lab · **Total marks:** 10

| Group No. | Member | BITS ID | Contribution |
|---|---|---|---|
| 10 | TILAK | 2025AE05975 | System design, data pipeline, CLIP features, FAISS retrieval, lab execution |
| | OM PRAKASH UNIYAL | 2025AE05324 | Temporal models (pooling, Transformer, LSTM) and research ablation study |
| | KIRAN P | 2025AE05409 | Evaluation metrics, significance tests and results analysis |
| | GAURISHA MATHUR | 2025AE05642 | Streamlit dashboard and query edge-case handling |
| | KARAN RUPANI | 2025AE05813 | Experiments write-up and technical report |

**Task.** Build an end-to-end text→video retrieval system over MSR-VTT: index of **N = 100 clips** (5 clips × 20 categories),
**Q = 200 queries** (2 captions per clip) → **200 × 100 search matrix**, served through a Streamlit dashboard, evaluated with
standard IR metrics, and used to answer **Research Investigation 1**: *is mean-pooling frame embeddings sufficient, or does
explicit temporal modelling give superior retrieval?*

### How this notebook is organised
| Section | Rubric component |
|---|---|
| 0. Environment check | – |
| 1. Source code (all modules, written to disk with `%%writefile`) | all |
| 2–4. Configuration, data ingestion, frame feature extraction | **1. Data ingestion & feature engineering (2.0)** |
| 5–6. Learned temporal heads, vector index | 1 & 2 |
| 7–8. X-CLIP reference, evaluation | **4. Quantitative evaluation (1.5)** |
| 9. Research Investigation 1 | **5. Experimental analysis (2.0)** |
| 10–12. Edge cases, live text→video and video→text retrieval demos | **2. Cross-modal retrieval engine (2.5)** |
| 13–14. Efficiency, technical report | 4 & 5 |
| 15. Streamlit dashboard launched from the notebook | **3. Streamlit dashboard (2.0)** |

> **Run:** `Kernel → Restart & Run All`. All steps cache their outputs (`data/`, `artifacts/`), and frame features are
> saved after every clip, so if the lab session is interrupted, simply `Restart & Run All` again – it resumes where it stopped.
> Use the kernel **"Python (mmir-venv)"** – see the README for the one-time kernel registration.
""")

md("## 0. Environment check")
code(r"""
# Verify the interpreter and that every required library is importable (versions are printed for reproducibility).
import sys, os, glob, platform, importlib
# If this kernel is the lab's default Python, also use the packages installed in ~/workspace/mmir-venv.
for sp in glob.glob(os.path.expanduser("~/workspace/mmir-venv/lib/python3*/site-packages")):
    if sp not in sys.path:
        sys.path.insert(0, sp)
        os.environ["PYTHONPATH"] = sp + os.pathsep + os.environ.get("PYTHONPATH", "")
print("Python", sys.version.split()[0], "|", platform.platform())
for pkg in ["torch", "transformers", "huggingface_hub", "numpy", "pandas", "scipy", "cv2", "faiss",
            "streamlit", "altair", "matplotlib", "docx"]:
    try:
        m = importlib.import_module(pkg)
        print(f"  {pkg:16s} {getattr(m, '__version__', 'ok')}")
    except Exception as e:
        print(f"  {pkg:16s} MISSING -> {e}")
import torch
print("CUDA available:", torch.cuda.is_available())
""")

# ---------------------------------------------------------------------------
md(r"""
## 1. Source code

All application code is embedded below and written to disk with `%%writefile`, so the notebook is the single source of
truth. The package `mvr/` holds the retrieval engine; `app.py` is the Streamlit dashboard.

| Module | Responsibility |
|---|---|
| `mvr/config.py` | All settings (dataset, model, frames, training) – also editable from the dashboard |
| `mvr/data.py` | MSR-VTT download, stratified 5×20 sampling, OpenCV frame decoding, thumbnails |
| `mvr/encoders.py` | Frozen CLIP image/text encoders → shared 512-d space; optional X-CLIP |
| `mvr/temporal.py` | Mean/max/single/query-attentive pooling + learned temporal heads and their training |
| `mvr/index.py` | FAISS `IndexFlatIP` vector index (cosine similarity on unit vectors) |
| `mvr/query.py` | Query normalisation and edge cases (empty, typos, short, long, non-English) |
| `mvr/metrics.py` | R@K, MRR, mAP, nDCG@10, MdR, MnR + bootstrap / Wilcoxon / Holm |
| `mvr/control.py` | Positive-control experiment (synthetic order-only task) |
| `mvr/analysis.py` | Data-driven conclusions for Research Investigation 1 |
| `mvr/figures.py` | Report figures |
| `mvr/pipeline.py` | Orchestration with caching + all experiments |
| `mvr/report.py` | Generates the technical report (.docx) from the results |
| `app.py` | Streamlit dashboard |
""")
code("import os\nos.makedirs('mvr', exist_ok=True)\nos.makedirs('tests', exist_ok=True)\nprint('working directory:', os.getcwd())")

MODULES = [
    ("mvr/__init__.py", "Package marker."),
    ("mvr/config.py", "**Configuration** – dataset protocol (5 clips × 20 categories, 2 captions/clip), backbone, frame count, training hyper-parameters, cached-output paths."),
    ("mvr/data.py", "**Data ingestion** – downloads annotations + videos from Hugging Face, merges caption sources, samples the stratified index, decodes TSN-style uniformly spaced frames with OpenCV, validates clips and saves thumbnails."),
    ("mvr/encoders.py", "**Encoders** – CLIP pre-processing, frame and text embedding into the shared space (L2-normalised), and the optional X-CLIP scorer."),
    ("mvr/temporal.py", "**Temporal representations** – parameter-free aggregators and learned temporal heads (MLP control, Set Transformer, Temporal Transformer, LSTM) trained with symmetric InfoNCE."),
    ("mvr/index.py", "**Vector index** – one FAISS `IndexFlatIP` per representation + the frame tensor for query-conditioned pooling."),
    ("mvr/query.py", "**Query processing / edge cases** – normalisation, unknown-word detection, frequency-aware spelling correction, prompt templates for short queries, chunking for long queries."),
    ("mvr/metrics.py", "**Metrics & statistics** – standard IR metrics for both directions and paired significance tests."),
    ("mvr/control.py", "**Positive control** – synthetic task in which only frame order separates clips."),
    ("mvr/analysis.py", "**Conclusions** – turns the measured numbers into the answer to Research Investigation 1."),
    ("mvr/figures.py", "**Figures** – matplotlib charts used in this notebook and the report."),
    ("mvr/pipeline.py", "**Pipeline** – data → features → heads → index → X-CLIP → evaluation, each step cached on disk."),
    ("mvr/report.py", "**Report** – builds the technical report (.docx) from the results."),
    ("app.py", "**Streamlit dashboard** – pipeline runner, text→video search with playable clips, video→text, evaluation, research and report pages."),
]
for path, desc in MODULES:
    md(f"#### `{path}`\n{desc}")
    code(f"%%writefile {path}\n" + (ROOT / path).read_text())

# ---------------------------------------------------------------------------
md("## 2. Configuration")
code(r"""
# Import the freshly written package and load the configuration (dashboard-saved settings are honoured).
import json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from IPython.display import display, Markdown, Image, Video, HTML
warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", 120)

from mvr.config import Config, method_label, METHOD_INFO
from mvr import pipeline as P
from mvr.data import prepare_subset

cfg = Config.load()
print("device:", cfg.resolved_device(), "| effective training clips/category:", cfg.eff_train_per_cat(),
      "| effective epochs:", cfg.eff_epochs())
pd.DataFrame(cfg.to_dict().items(), columns=["setting", "value"])
""")
code(r"""
# Compact progress printer for long-running steps (prints at most every 10% to keep outputs readable).
def progress_printer(label):
    state = {"last": -1.0}
    def cb(frac, msg):
        if frac >= 1.0 or frac - state["last"] >= 0.10:
            state["last"] = frac
            print(f"[{label} {frac*100:5.1f}%] {msg}", flush=True)
    return cb
""")

md(r"""
## 3. Data ingestion
* **Index:** 5 clips from each of the 20 MSR-VTT categories, drawn from the official 1k-A *test* split (seeded, stratified).
  Every candidate clip is decoded before acceptance; corrupt clips are replaced automatically.
* **Queries:** 2 human captions per clip (the official 1k-A caption + one other) → 200 queries.
* **Training pool:** clips from the 9k *train* split, **disjoint** from the index (asserted in code), used only by the learned temporal heads.
* **Content filter:** MSR-VTT is crowd-captioned web video; clips whose captions use sexual/explicit wording are excluded
  from both the index and the training pool (`is_sensitive()` in `mvr/data.py`), and replaced by the next sampled clip.
""")
code(r"""
t0 = time.time()
subset = prepare_subset(cfg, progress_printer("data"))
print(f"done in {time.time()-t0:.1f}s")
pd.DataFrame([subset["protocol"]])
""")
code(r"""
# Index composition: 5 clips per category, with clip metadata from the decoder.
idx_df = pd.DataFrame([{"video_id": v["video_id"], "category": v["category_name"], "duration_s": v.get("duration"),
                        "fps": round(v.get("fps", 0), 1), "frames": v.get("n_frames"),
                        "resolution": f"{v.get('width')}x{v.get('height')}"} for v in subset["index"]])
display(idx_df.groupby("category").size().rename("clips").to_frame().T)
idx_df.describe(include="all").loc[["count", "mean", "min", "max"], ["duration_s", "fps", "frames"]]
""")
code(r"""
# Sample of the 200 queries (2 per clip).
pd.DataFrame(subset["queries"])[["qid", "text", "video_id"]].head(10)
""")
code(r"""
# Raw frames sampled from two clips (TSN-style segment mid-points) and one embedded, playable clip.
for v in subset["index"][:2]:
    print(v["video_id"], "-", v["category_name"], "-", v["queries"][0])
    display(Image(filename=str(Path(cfg.thumbs_dir) / f"{v['video_id']}.jpg")))
v0 = subset["index"][0]
Video(str(cfg.videos_dir / f"{v0['video_id']}.mp4"), embed=True, width=360)
""")

md(r"""
## 4. Visual feature extraction & cross-modal embedding
Each clip → **T = 16** uniformly spaced frames → frozen **CLIP ViT** image tower + projection → 512-d L2-normalised vectors.
Queries → CLIP text Transformer + projection → the **same** 512-d space, so cosine similarity = dot product.
""")
code(r"""
t0 = time.time()
F = P.step_features(cfg, progress_printer("features"))
print(f"done in {time.time()-t0:.1f}s")
for k in ("index_frames", "query_emb", "train_frames", "train_text"):
    print(f"{k:13s} {F[k].shape}  dtype={F[k].dtype}")
json.loads(str(F["timing"]))
""")
code(r"""
# Sanity check of the shared space: matching (query, clip) pairs must be far more similar than random pairs.
fr, Q = F["index_frames"], F["query_emb"]
gt = np.array([q["video_idx"] for q in subset["queries"]])
from mvr.temporal import pool_video
V = pool_video(fr, "mean_pool")
S = Q @ V.T
pos = S[np.arange(len(gt)), gt]
neg = S[np.ones_like(S, bool) & (np.arange(S.shape[1])[None, :] != gt[:, None])]
print(f"cosine(query, matching clip)  mean={pos.mean():.3f} sd={pos.std():.3f}")
print(f"cosine(query, other clips)    mean={neg.mean():.3f} sd={neg.std():.3f}")
print(f"norms of frame embeddings ~ {np.linalg.norm(fr, axis=-1).mean():.3f} (L2-normalised)")
""")

md(r"""
## 5. Learned temporal heads (trained on the disjoint training pool)
Four heads share the frozen CLIP features, data, loss (symmetric InfoNCE), optimiser, augmentation and early stopping:
**MLP adapter** (control, no frame interaction), **Set Transformer** (frame interaction, no order),
**Temporal Transformer** (+ temporal position embeddings) and **LSTM** (ordered recurrence).
All are zero-initialised residuals of mean pooling and trained with 3 seeds. The query-attention temperature τ is chosen on
the training pool (never on the test queries).
""")
code(r"""
t0 = time.time()
hists = P.step_train(cfg, progress_printer("train"))
print(f"done in {time.time()-t0:.1f}s")
pd.DataFrame([{"run": k, "params": h["n_params"], "train clips": h["n_train_videos"], "val clips": h["n_val_videos"],
               "best epoch": h["best_epoch"], "best val MRR": round(h["best_val_mrr"], 4), "seconds": h["train_seconds"]}
              for k, h in hists.items()])
""")
code(r"""
json.loads((cfg.models_dir / "tau.json").read_text())
""")

md("## 6. Vector index (FAISS)")
code(r"""
index = P.step_index(cfg, progress_printer("index"))
print("backend:", index.backend, "| representations:", index.methods)
print("index matrix per representation:", {m: M.shape for m, M in index.pooled.items()})
""")

md(r"""
## 7. External reference: X-CLIP (pre-trained spatio-temporal video model)
Optional; a failure here does not affect the rest of the pipeline.
""")
code(r"""
print("X-CLIP:", P.step_xclip(cfg, progress_printer("xclip")))
""")

md(r"""
## 8. Quantitative evaluation
Metrics: **R@1/5/10** (%), **MRR**, **mAP**, **nDCG@10**, **MdR** / **MnR** (median / mean rank of the correct item).
Text→Video has one relevant clip per query (so mAP = MRR); Video→Text has two relevant captions per clip.
Learned heads are reported as mean ± sd over 3 seeds.
""")
code(r"""
t0 = time.time()
R = P.step_eval(cfg, progress_printer("eval"))
print(f"done in {time.time()-t0:.1f}s")

def metric_table(tab, std):
    cols = ["R@1", "R@5", "R@10", "MRR", "mAP", "nDCG@10", "MdR", "MnR"]
    rows = []
    for m, d in tab.items():
        r = {"Method": method_label(m)}
        for c in cols:
            s = std.get(m, {}).get(c, 0)
            f = "{:.1f}" if (c.startswith("R@") or c in ("MdR", "MnR")) else "{:.3f}"
            r[c] = f.format(d[c]) + ((" ± " + f.format(s)) if s else "")
        rows.append(r)
    return pd.DataFrame(rows).set_index("Method")

display(Markdown("**Table 1 – Text → Video (200 queries × 100 clips)**"))
display(metric_table(R["t2v"], R["t2v_std"]))
display(Markdown("**Table 2 – Video → Text (100 clips × 200 captions)**"))
display(metric_table(R["v2t"], R["v2t_std"]))
""")
code(r"""
Image(filename=str(cfg.figures_dir / "main_bars.png"))
""")

md(r"""
## 9. Research Investigation 1 – is mean pooling sufficient?
**Design (ablation ladder):** single frame → **mean pooling** → zero-shot max / query-attentive pooling → MLP adapter
(domain-adaptation control) → Set Transformer (interaction, no order) → Temporal Transformer / LSTM (explicit order),
plus X-CLIP. **Tests:** paired bootstrap 95% CIs, Wilcoxon signed-rank on per-query reciprocal ranks, Holm correction;
test-time frame shuffling/reversal; frame-count sweep; temporal-language query subset; synthetic positive control.
""")
code(r"""
# Paired significance tests (A vs B; positive delta = A better).
pd.DataFrame([{"A": method_label(r["method"]), "B": method_label(r["reference"]),
               "ΔMRR": round(r["MRR"]["delta"], 4),
               "95% CI": f"[{r['MRR']['ci_low']:+.3f}, {r['MRR']['ci_high']:+.3f}]",
               "ΔR@1": round(r["R@1"]["delta"], 1), "ΔR@5": round(r["R@5"]["delta"], 1),
               "Wilcoxon p": f"{r['wilcoxon_p']:.3g}", "Holm p": f"{r['wilcoxon_p_holm']:.3g}",
               "better/worse/tie": f"{r['n_better']}/{r['n_worse']}/{r['n_tie']}"} for r in R["significance"]])
""")
code(r"""
# Test-time temporal-order ablation: order-agnostic methods must be unchanged; order-aware ones reveal order use.
oa = R["order_ablation"]
display(pd.DataFrame({method_label(m): {c: round(oa[m][c]["MRR"], 4) for c in ("original", "shuffled", "reversed")}
                      for m in oa}).T)
Image(filename=str(cfg.figures_dir / "order.png"))
""")
code(r"""
# Number of sampled frames vs MRR.
sw = R["frame_sweep"]
display(pd.DataFrame({method_label(m): {int(k): round(v["MRR"], 4) for k, v in sw[m].items()} for m in sw}).T)
Image(filename=str(cfg.figures_dir / "frame_sweep.png"))
""")
code(r"""
# Queries with explicit temporal language vs the rest.
tl = R["temporal_language"]
print(f"{tl['n_temporal']} temporal-language queries, e.g.:", tl["examples"][:3])
pd.DataFrame({method_label(m): v for m, v in tl["by_method"].items()}).T.round(4)
""")
code(r"""
Image(filename=str(cfg.figures_dir / "per_category.png"))
""")
code(r"""
# Positive control: synthetic clips where ONLY frame order distinguishes them.
pd.DataFrame({method_label(m): {"MRR (order-only task)": round(v, 3)} for m, v in R["positive_control"]["mrr"].items()}).T
""")
code(r"""
Image(filename=str(cfg.figures_dir / "training.png"))
""")
code(r"""
# Data-driven conclusions (generated from the numbers above).
C = R["conclusions"]
display(Markdown("### Answer\n> " + C["answer"]))
display(Markdown("### Evidence\n" + "\n".join("- " + p for p in C["points"])))
""")

md(r"""
## 10. Query edge-case handling
The engine handles: empty / punctuation-only input, unicode noise and URLs, misspelled / unknown vocabulary
(frequency-aware edit-distance correction), very short queries (prompt-template ensemble), very long queries
(> 77 CLIP tokens → sentence-aware chunking + averaging), and non-English text (warning).
""")
code(r"""
from mvr.query import QueryProcessor, QueryError
enc = P.get_encoder(cfg)
qp = QueryProcessor(enc, P.load_lexicon(cfg))
edge_queries = ["", "  ?!...  ", "a woman playng gutiar on stge", "basketball",
                "A  man\tis   cooking   in a kitchen 😀 https://example.com",
                "un hombre toca la guitarra en el escenario",
                "a group of young people stand on a stage and sing a song while the crowd cheers, then the camera pans "
                "across the audience who are clapping and waving their hands in the air, and finally the lead singer "
                "bows and walks off the stage as the lights go down and the music slowly fades out"]
rows = []
for q in edge_queries:
    try:
        pq = qp.process(q)
        rows.append({"input": q[:60], "mode": pq.mode, "tokens": pq.n_tokens, "final text": pq.final_text[:60],
                     "#texts encoded": len(pq.texts_to_encode), "steps": "; ".join(pq.steps)[:90],
                     "warnings": "; ".join(pq.warnings)[:90]})
    except QueryError as e:
        rows.append({"input": repr(q), "mode": "rejected", "warnings": str(e)[:90]})
pd.DataFrame(rows)
""")
code(r"""
# Measured effect of the edge-case handling on all 200 (perturbed) queries, mean-pool index.
rb = R["robustness"]
display(pd.DataFrame(rb["rows"])[["condition", "handling", "R@1", "R@5", "MRR", "MdR"]].round(3))
print("examples:", json.dumps(rb["examples"], indent=1)[:600])
""")

md(r"""
## 11. Live text → video retrieval (Top-K with scores)
`search()` runs the full online path: query processing → CLIP text embedding → cosine similarity against the index →
Top-K clips with scores, category, per-frame similarity and the ground truth (when the query is from the evaluation set).
""")
code(r"""
eval_texts = [q["text"] for q in subset["queries"]]

def search(query, method="mean_pool", k=5, play_top=True):
    pq = qp.process(query)
    t0 = time.perf_counter(); qv = qp.encode(pq); enc_ms = (time.perf_counter() - t0) * 1000
    res = index.search(qv, method, k)
    gt_idx = subset["queries"][eval_texts.index(query)]["video_idx"] if query in eval_texts else None
    display(Markdown(f"**Query:** `{query[:120]}` → *{pq.final_text[:120]}*  |  representation: **{method_label(method)}**  |  "
                     f"encode {enc_ms:.0f} ms, search {res['latency_ms']:.2f} ms"))
    display(pd.DataFrame([{"rank": r + 1, "video": subset["index"][i]["video_id"],
                           "category": subset["index"][i]["category_name"], "cosine score": round(s, 4),
                           "peak frame": int(np.argmax(res["frame_sims"][r])),
                           "ground truth": "✔" if i == gt_idx else "",
                           "a caption of this clip": subset["index"][i]["all_captions"][0][:70]}
                          for r, (i, s) in enumerate(zip(res["idx"], res["scores"]))]))
    top = subset["index"][res["idx"][0]]
    display(Image(filename=str(Path(cfg.thumbs_dir) / f"{top['video_id']}.jpg")))
    if play_top:
        display(Video(str(cfg.videos_dir / f"{top['video_id']}.mp4"), embed=True, width=360))

search("a person is playing guitar on a stage")
""")
code(r"""
search("a woman playng gutiar on stge", play_top=False)       # typo-corrected query
""")
code(r"""
search(subset["queries"][10]["text"], method="temporal_tfm", play_top=False)   # evaluation query, temporal model
""")
code(r"""
search("basketball", method="query_attn", play_top=False)     # very short query -> template ensemble
""")

md("## 12. Video → text retrieval (the reverse modality)")
code(r"""
def video_to_text(vi=0, method="mean_pool", k=5):
    S = index.scores(F["query_emb"], method)[:, vi]
    order = np.argsort(-S)[:k]
    v = subset["index"][vi]
    display(Markdown(f"**Clip:** {v['video_id']} ({v['category_name']}) | representation: **{method_label(method)}**"))
    display(Image(filename=str(Path(cfg.thumbs_dir) / f"{v['video_id']}.jpg")))
    display(pd.DataFrame([{"rank": r + 1, "caption": subset["queries"][qi]["text"], "score": round(float(S[qi]), 4),
                           "relevant": "✔" if subset["queries"][qi]["video_idx"] == vi else ""}
                          for r, qi in enumerate(order)]))

video_to_text(0)
video_to_text(42, method="temporal_tfm")
""")

md("## 13. Efficiency")
code(r"""
tm = R["timing"]
display(pd.DataFrame([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in tm.items() if k != "search_ms_per_query"}]).T
        .rename(columns={0: "value"}))
pd.DataFrame([{"representation": method_label(m), "search ms/query (top-10)": round(v, 4)}
              for m, v in tm.get("search_ms_per_query", {}).items()])
""")

md("## 14. Technical report (.docx) generated from these results")
code(r"""
from mvr.report import build_report
report_path = build_report(cfg, Path(cfg.artifacts_dir) / "MIR_Assignment2_Technical_Report.docx")
print("report written to:", report_path.resolve())
Image(filename=str(cfg.figures_dir / "architecture.png"))
""")

md(r"""
## 15. Streamlit interactive dashboard
The dashboard is launched **from this notebook** (no manual script execution). Open the printed URL in the lab browser:
type any natural-language query, choose the video representation and Top-K, and play the retrieved clips with their
similarity scores. Other pages: Pipeline Runner, Video → Text, Evaluation, Research Investigation, Dataset Browser, Report.
""")
code(r"""
import subprocess, urllib.request
PORT = 8501
def dashboard_up():
    try:
        return urllib.request.urlopen(f"http://localhost:{PORT}/_stcore/health", timeout=2).read() == b"ok"
    except Exception:
        return False

if dashboard_up():
    print(f"Dashboard already running at http://localhost:{PORT}")
else:
    log = open("streamlit.log", "w")
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(PORT),
                             "--server.headless", "true"], stdout=log, stderr=subprocess.STDOUT)
    for _ in range(30):
        time.sleep(1)
        if dashboard_up():
            break
    print(("Dashboard running" if dashboard_up() else "Dashboard still starting - see streamlit.log")
          + f" at http://localhost:{PORT}  (pid {proc.pid})")
""")
code(r"""
from IPython.display import IFrame
IFrame(f"http://localhost:{PORT}", width="100%", height=650)
""")

md(r"""
## 16. Summary
* **Retrieval quality** (Text→Video, mean-pooled CLIP frames) and all other representations are reported in Section 8.
* **Research Investigation 1** – the data-driven answer is printed at the end of Section 9.
* **Deliverables:** this notebook (with outputs), the technical report (`artifacts/MIR_Assignment2_Technical_Report.docx`),
  and screenshots of the Streamlit dashboard running in the BITS Virtual Lab.
""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python (mmir-venv)", "language": "python", "name": "mmir-venv"},
                  "language_info": {"name": "python"}}
out = ROOT / "MIR_Assignment2.ipynb"
nbf.write(nb, out)
print("wrote", out, "with", len(cells), "cells")
