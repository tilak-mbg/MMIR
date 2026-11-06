"""Generates the Technical Report (.docx) directly from the evaluation results,
so every number, table and chart in the report comes from the actual run."""
from __future__ import annotations

import json
from pathlib import Path

from .config import LEARNED_METHODS, METHOD_INFO, Config, method_label
from .figures import make_all_figures


TEAM = [("TILAK", "2025AE05975", "System design, data pipeline, CLIP features, FAISS retrieval, lab execution"),
        ("OM PRAKASH UNIYAL", "2025AE05324", "Temporal models (pooling, Transformer, LSTM) and research ablation study"),
        ("KIRAN P", "2025AE05409", "Evaluation metrics, significance tests and results analysis"),
        ("GAURISHA MATHUR", "2025AE05642", "Streamlit dashboard and query edge-case handling"),
        ("KARAN RUPANI", "2025AE05813", "Experiments write-up and technical report")]


def build_report(cfg: Config, out: Path) -> Path:
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt

    R = json.loads(cfg.results_path.read_text())
    subset = json.loads(cfg.subset_path.read_text())
    figs = make_all_figures(R, cfg.figures_dir)
    C = R["conclusions"]

    doc = Document()
    sec = doc.sections[0]
    sec.left_margin = sec.right_margin = Cm(2.0)
    sec.top_margin = sec.bottom_margin = Cm(1.8)
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(10.5)

    def shade(cell, hexcol):
        tcPr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), hexcol)
        tcPr.append(shd)

    def table(header, rows, widths=None, font=8.5):
        t = doc.add_table(rows=1, cols=len(header))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, h in enumerate(header):
            c = t.rows[0].cells[i]
            c.text = ""
            r = c.paragraphs[0].add_run(str(h))
            r.bold = True
            r.font.size = Pt(font)
            shade(c, "DBE8F8")
        for row in rows:
            cells = t.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = ""
                r = cells[i].paragraphs[0].add_run(str(v))
                r.font.size = Pt(font)
        if widths:
            t.autofit = False
            for i, w in enumerate(widths):
                t.columns[i].width = Cm(w)
            for row in t.rows:
                for i, w in enumerate(widths):
                    row.cells[i].width = Cm(w)
        doc.add_paragraph()
        return t

    def fig(key, caption, width=16.5):
        p = figs.get(key)
        if p and Path(p).exists():
            doc.add_picture(str(p), width=Cm(width))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            cp = doc.add_paragraph(caption)
            cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cp.runs[0].italic = True
            cp.runs[0].font.size = Pt(9)

    def bullets(items):
        for it in items:
            doc.add_paragraph(it, style="List Bullet")

    def para(text, bold_prefix=None):
        p = doc.add_paragraph()
        if bold_prefix:
            p.add_run(bold_prefix).bold = True
        p.add_run(text)
        return p

    t2v, sd, v2t = R["t2v"], R["t2v_std"], R["v2t"]
    mp = t2v["mean_pool"]

    # ------------------------------------------------------------------ cover
    h = doc.add_heading("Assignment 2: Natural-Language Video Retrieval Using MSR-VTT", 0)
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph("Text-to-Video Retrieval Using Semantic and Temporal Video Representations")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.runs[0].font.size = Pt(13)
    p = doc.add_paragraph("Course: Multimodal Information Retrieval  ·  Environment: BITS Virtual Lab")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    table(["Group No.", "Member Name", "BITS ID", "Contribution"],
          [["10" if k == 0 else "", n, i, c] for k, (n, i, c) in enumerate(TEAM)], [2.2, 4.3, 3, 7.5])
    para(f"Results generated: {R['generated_at']}  ·  backbone: {R['config']['clip_model']}  ·  "
         f"{R['config']['num_frames']} frames/clip  ·  device: {R.get('timing', {}).get('device', 'n/a')}")

    doc.add_heading("Executive summary", 1)
    para(f"We built an end-to-end text-to-video retrieval system over a 100-clip MSR-VTT index (5 clips × 20 "
         f"categories) queried by 200 natural-language captions (a 200 × 100 search matrix), with a Streamlit dashboard "
         f"for arbitrary queries, playable Top-K results and live evaluation. Frames are encoded with a frozen CLIP "
         f"ViT and mapped, together with the text query, into CLIP's shared 512-d embedding space. With mean-pooled "
         f"frame embeddings the system reaches R@1 = {mp['R@1']:.1f}%, R@5 = {mp['R@5']:.1f}%, R@10 = {mp['R@10']:.1f}%, "
         f"MRR = {mp['MRR']:.3f} and a median rank of {mp['MdR']:.0f}.")
    para(C["answer"], "Research Investigation 1 – answer: ")

    # ------------------------------------------------------------------ 1 data
    doc.add_heading("1. Dataset and evaluation protocol", 1)
    pr = R["dataset"]
    table(["Item", "Value"], [
        ["Source", "MSR-VTT (Xu et al., CVPR 2016) – Hugging Face mirror friedrichor/MSR-VTT"],
        ["Index size N", f"{pr['n_index']} clips = {pr['videos_per_category']} clips × 20 categories, drawn from the 1k-A test split"],
        ["Query set Q", f"{pr['n_queries']} queries = {pr['captions_per_video']} human captions per clip "
                        "(the official 1k-A caption + one randomly chosen other caption)"],
        ["Search corpus matrix", f"{pr['n_queries']} × {pr['n_index']}"],
        ["Training pool (learned heads only)", f"{pr['n_train']} clips from the 9k train split, all 20 captions; "
                                                 "disjoint from the index (asserted in code)"],
        ["Sampling", f"stratified by category, fixed seed {pr['seed']}; undecodable clips are skipped and replaced"],
        ["Content filter", f"clips whose captions use sexual/explicit wording are excluded from index and training pool "
                           f"({pr.get('n_excluded_by_content_filter', 0)} candidate clips skipped)"],
        ["Relevance", "T2V: the clip a caption describes (1 relevant / query). V2T: the 2 captions of a clip"],
    ], [5, 12])
    cats = {}
    for v in subset["index"]:
        cats.setdefault(v["category_name"], 0)
        cats[v["category_name"]] += 1
    para("Categories (5 clips each): " + ", ".join(cats))

    # ------------------------------------------------------------------ 2 architecture
    doc.add_heading("2. System architecture", 1)
    fig("architecture", "Figure 1 – Offline indexing, training and online query pipelines.", 17)
    table(["Module (file)", "Responsibility"], [
        ["mvr/data.py", "Download annotations/videos, stratified 5×20 sampling, OpenCV frame decoding (TSN segment mid-points), thumbnails"],
        ["mvr/encoders.py", "CLIP image/text encoders (frozen) → shared 512-d space; CLIP pre-processing; optional X-CLIP"],
        ["mvr/temporal.py", "Mean/max/single/query-attentive pooling; learned temporal heads (MLP, Set-TFM, Temporal-TFM, LSTM) + InfoNCE training"],
        ["mvr/index.py", "FAISS IndexFlatIP per representation (cosine via inner product of unit vectors) + frame-level matrix"],
        ["mvr/query.py", "Query normalisation, unknown-word detection, spelling correction, prompt templates, long-query chunking"],
        ["mvr/metrics.py", "R@K, MRR, mAP, nDCG@10, MdR, MnR; paired bootstrap, Wilcoxon, Holm correction"],
        ["mvr/pipeline.py", "Orchestration with on-disk caching; evaluation & research experiments"],
        ["app.py", "Streamlit dashboard (pipeline runner, search, V2T, evaluation, research, dataset, report)"],
    ], [4, 13])

    # ------------------------------------------------------------------ 3 features
    doc.add_heading("3. Data ingestion and feature engineering", 1)
    tm = R.get("timing", {})
    bullets([
        f"Frame sampling: each clip is split into T = {R['config']['num_frames']} equal temporal segments and the "
        "middle frame of each segment is decoded (TSN-style uniform coverage of the whole clip). Frames are read "
        "sequentially with grab/retrieve for frame-accurate sampling; a full-decode fallback handles wrong frame-count metadata.",
        "Pre-processing: RGB conversion, bicubic resize of the short side to the model resolution, centre crop, "
        "normalisation with CLIP's channel mean/std.",
        f"Visual features: frozen CLIP ViT image tower + visual projection ({R['config']['clip_model']}), "
        "producing one L2-normalised 512-d embedding per frame (fp16 on GPU).",
        "Indexing: video-level vectors for every representation are stored in FAISS IndexFlatIP (exact inner-product "
        "search = cosine similarity on unit vectors); the (N × T × D) frame tensor is kept for query-conditioned pooling "
        "and per-frame similarity visualisation. Everything is cached to .npy/.faiss files.",
        f"Cost: decode {tm.get('decode_s_per_video', 0)*1000:.0f} ms/clip, CLIP encode "
        f"{tm.get('encode_s_per_video', 0)*1000:.0f} ms/clip, index build {tm.get('build_ms', 0):.1f} ms "
        f"({tm.get('backend', '')}).",
    ])

    # ------------------------------------------------------------------ 4 retrieval engine
    doc.add_heading("4. Cross-modal embedding and retrieval engine", 1)
    para("Queries are encoded by CLIP's text Transformer and text projection into the same 512-d space as the frames "
         "(CLIP was contrastively trained so that matching image/text pairs have high cosine similarity). "
         "Score(q, v) = cos(e_q, e_v) = e_q · e_v for unit vectors; the Top-K clips are returned with their scores. "
         "For query-attentive pooling, Score = Σ_t softmax_t(s_t/τ)·s_t with s_t = e_q · f_t "
         f"(τ = {R['tau']} chosen on the training pool, never on the test queries).")
    doc.add_heading("Representations compared", 2)
    table(["Representation", "Description", "Order-aware", "Trained"],
          [[method_label(m), METHOD_INFO[m][1], "yes" if m in ("temporal_tfm", "seq_lstm", "xclip") else "no",
            "yes" if m in LEARNED_METHODS else ("pre-trained" if m == "xclip" else "no")]
           for m in t2v], [4, 9.5, 1.8, 1.7])
    doc.add_heading("Edge-case handling", 2)
    table(["Edge case", "Handling in the engine"], [
        ["Empty / punctuation-only query", "Rejected with a helpful message (no meaningless search)"],
        ["Unicode noise, URLs, extra spaces", "NFKC normalisation, control-char removal, whitespace collapse"],
        ["Unknown / misspelled vocabulary", "Words not in CLIP's BPE vocabulary or the MSR-VTT training lexicon are "
                                            "flagged; frequency-ranked edit-distance (≤2) correction; BPE sub-words as fallback"],
        ["Very short queries (≤2 content words)", "Prompt-template ensemble: 'a video of {q}.', 'a video showing {q}.', '{q}'"],
        ["Long / detailed queries (>77 tokens)", "Sentence/clause-aware chunking, chunk embeddings averaged (no silent truncation)"],
        ["Non-English text", "Detected and flagged (CLIP text encoder is English-only)"],
        ["No good match in the collection", "Low-confidence warning when the top score is below the 5th percentile of correct-match scores"],
    ], [5, 12])
    rb = R.get("robustness", {})
    if "rows" in rb:
        para("Effect of the edge-case handling (mean-pool index, all 200 queries perturbed):")
        table(["Query condition", "Handling", "R@1", "R@5", "MRR", "MdR"],
              [[r["condition"], r["handling"], f"{r['R@1']:.1f}", f"{r['R@5']:.1f}", f"{r['MRR']:.3f}", f"{r['MdR']:.0f}"]
               for r in rb["rows"]], [4.5, 4.5, 1.6, 1.6, 1.8, 1.4])

    # ------------------------------------------------------------------ 5 dashboard
    doc.add_heading("5. Streamlit interactive dashboard", 1)
    bullets([
        "Pipeline Runner – every step (download, frame extraction, CLIP encoding, training, indexing, X-CLIP, "
        "evaluation) is launched from buttons with live progress; no manual script execution is required.",
        "Text → Video Search – free-text query, choice of representation, Top-K slider; results show playable "
        "clips, cosine scores, category, per-frame similarity profile, captions, query-processing diagnostics and latency.",
        "Video → Text – the reverse modality: pick a clip, retrieve its Top-K captions with relevance marks.",
        "Evaluation – metric tables (T2V and V2T), interactive charts, frame sweep, per-category heat-map, "
        "robustness and efficiency.",
        "Research Investigation – ablation ladder, significance tests, order ablation and conclusions.",
        "Dataset Browser and Report pages (this document is generated from the Report page).",
    ])

    # ------------------------------------------------------------------ 6 results
    doc.add_heading("6. Quantitative performance evaluation", 1)
    cols = ["R@1", "R@5", "R@10", "MRR", "mAP", "nDCG@10", "MdR", "MnR"]

    def fmt(m, c, tab, s):
        v, e = tab[m][c], s.get(m, {}).get(c, 0)
        if c.startswith("R@") or c in ("MdR", "MnR"):
            return f"{v:.1f}" + (f"±{e:.1f}" if e else "")
        return f"{v:.3f}" + (f"±{e:.3f}" if e else "")

    para("Table 1 – Text→Video retrieval (200 queries × 100 clips). Learned heads: mean ± sd over "
         f"{len(R['config']['train_seeds'])} seeds. With one relevant clip per query mAP equals MRR.", None)
    table(["Method"] + cols, [[method_label(m)] + [fmt(m, c, t2v, sd) for c in cols] for m in t2v],
          [4.2] + [1.6] * 8, 8)
    para("Table 2 – Video→Text retrieval (100 clips × 200 captions; 2 relevant captions per clip).")
    table(["Method"] + cols, [[method_label(m)] + [fmt(m, c, v2t, R["v2t_std"]) for c in cols] for m in v2t],
          [4.2] + [1.6] * 8, 8)
    fig("main_bars", "Figure 2 – T2V R@1 and MRR per representation (error bars: sd over seeds; dashed: mean-pool baseline).", 17)

    # ------------------------------------------------------------------ 7 research
    doc.add_heading("7. Research Investigation 1: is mean pooling sufficient?", 1)
    doc.add_heading("7.1 Experimental design", 2)
    para("To separate the effect of temporal modelling from confounds we use an ablation ladder in which each step "
         "adds exactly one capability on top of the same frozen CLIP frame features: (0) a single frame; (1) mean "
         "pooling; (1b) zero-shot max and query-attentive pooling; (2) an MLP adapter trained on in-domain data but "
         "with no frame interaction (control for domain adaptation); (3) a Set Transformer with frame–frame "
         "attention but no positional information (control for interaction without order); (4) a Temporal "
         "Transformer with learned temporal position embeddings and (4b) an LSTM – both explicitly model order. "
         "All learned heads are zero-initialised residuals of mean pooling, use the same data, loss (symmetric "
         "InfoNCE), optimiser (AdamW, one-cycle), temporal sub-sampling augmentation and early stopping on held-out "
         "training clips, and are run with multiple seeds. We additionally (i) shuffle/reverse frame order at test "
         "time, (ii) vary the number of frames, (iii) split queries by temporal language, and (iv) compare against "
         "X-CLIP, a natively spatio-temporal video-language model, when available.")
    doc.add_heading("7.2 Statistical tests", 2)
    para("Query-level paired bootstrap (10,000 resamples) 95% confidence intervals on ΔMRR and ΔR@K, Wilcoxon "
         "signed-rank tests on per-query reciprocal ranks, Holm–Bonferroni correction across comparisons.")
    table(["Comparison (A vs B)", "ΔMRR", "95% CI", "ΔR@1", "ΔR@5", "p (Wilcoxon)", "p (Holm)", "A better/worse/tie"],
          [[f"{method_label(r['method'])} vs {method_label(r['reference'])}", f"{r['MRR']['delta']:+.3f}",
            f"[{r['MRR']['ci_low']:+.3f}, {r['MRR']['ci_high']:+.3f}]", f"{r['R@1']['delta']:+.1f}",
            f"{r['R@5']['delta']:+.1f}", f"{r['wilcoxon_p']:.3g}", f"{r['wilcoxon_p_holm']:.3g}",
            f"{r['n_better']}/{r['n_worse']}/{r['n_tie']}"] for r in R["significance"]],
          [5, 1.4, 2.6, 1.3, 1.3, 1.6, 1.4, 2.3], 7.5)
    doc.add_heading("7.3 Temporal-order ablation and frame count", 2)
    oa = R["order_ablation"]
    table(["Method", "MRR original", "MRR shuffled", "MRR reversed", "R@1 original", "R@1 shuffled"],
          [[method_label(m), f"{oa[m]['original']['MRR']:.3f}", f"{oa[m]['shuffled']['MRR']:.3f}",
            f"{oa[m]['reversed']['MRR']:.3f}", f"{oa[m]['original']['R@1']:.1f}", f"{oa[m]['shuffled']['R@1']:.1f}"]
           for m in oa], [5, 2.4, 2.4, 2.4, 2.4, 2.4])
    fig("order", "Figure 3 – MRR when the frame order is shuffled or reversed at test time.", 16)
    fig("frame_sweep", "Figure 4 – MRR as a function of the number of sampled frames.", 12)
    tl = R["temporal_language"]
    para(f"Queries containing explicit temporal language ({tl['n_temporal']} of {tl['n_temporal'] + tl['n_other']}, "
         "e.g. 'then', 'after', 'starts'):")
    table(["Method", "MRR – temporal-language queries", "MRR – other queries"],
          [[method_label(m), f"{v['temporal_MRR']:.3f}" if v["temporal_MRR"] is not None else "–", f"{v['other_MRR']:.3f}"]
           for m, v in tl["by_method"].items()], [6, 5.5, 5.5])
    pcr = R.get("positive_control", {}).get("mrr")
    if pcr:
        para("Positive control – synthetic clips 'state A then state B' vs 'B then A' whose captions encode the "
             "ordered pair (only temporal order separates them). This verifies that the set-up can detect a "
             "temporal benefit when it exists:")
        table(["Method", "MRR (order-only task)"], [[method_label(m), f"{v:.3f}"] for m, v in pcr.items()], [8, 5])
    fig("per_category", "Figure 5 – Per-category R@5 (each cell = 10 queries).", 15)
    fig("training", "Figure 6 – Validation MRR during training; epoch 0 equals mean pooling.", 12)
    doc.add_heading("7.4 Findings and conclusion", 2)
    bullets(C["points"])
    para(C["answer"], "Conclusion: ")
    doc.add_heading("7.5 Insights", 2)
    bullets([
        "MSR-VTT captions mostly describe *what* is visible (people, objects, scenes, activities recognisable from "
        "single frames), so a frame-level image–text model already captures most of the retrievable signal; "
        "averaging frames adds robustness to unrepresentative frames.",
        "Gains from trained heads must be compared with an equally-trained, order-free control. Without the MLP "
        "adapter and Set-Transformer controls, domain-adaptation gains are easily misattributed to temporal modelling.",
        "Test-time frame shuffling is a cheap diagnostic of whether a 'temporal' model really uses order.",
        "Temporal modelling is most likely to matter for queries whose meaning depends on event order or motion "
        "direction (e.g. 'opens then closes', 'falls down'); such queries are rare in MSR-VTT, which limits the "
        "measurable benefit on this benchmark.",
    ])

    # ------------------------------------------------------------------ 8 efficiency / limits
    doc.add_heading("8. Efficiency", 1)
    if "search_ms_per_query" in tm:
        table(["Method", "Search latency (ms/query, top-10)"],
              [[method_label(m), f"{v:.3f}"] for m, v in tm["search_ms_per_query"].items()], [8, 6])
    para(f"Query text encoding: {tm.get('text_encode_ms_per_query', 0):.1f} ms/query (batched) on {tm.get('device')}.")
    doc.add_heading("9. Limitations and future work", 1)
    bullets([
        "200 queries give wide confidence intervals (one query = 0.5 R@1 points); results are reported with CIs and paired tests.",
        "Learned heads are trained on a few hundred clips with a frozen backbone; end-to-end fine-tuning on the full "
        "9k training split (as in CLIP4Clip) could change the picture.",
        "Audio and on-screen text (ASR/OCR) are not used although many MSR-VTT categories (music, news) rely on them.",
        "Future work: temporally-grounded benchmarks (e.g. order-sensitive caption sets), motion features (optical "
        "flow / VideoMAE), and hybrid scoring that adds temporal evidence only when the query contains temporal language.",
    ])
    doc.add_heading("10. Reproducibility", 1)
    bullets([
        "pip install -r requirements.txt  →  streamlit run app.py  →  Pipeline Runner → 'Run full pipeline'.",
        "All randomness is seeded (sampling seed, training seeds); outputs are cached under artifacts/.",
        f"Configuration used: {json.dumps({k: R['config'][k] for k in ('clip_model', 'num_frames', 'train_videos_per_category', 'epochs', 'train_seeds', 'seed')})}",
    ])
    doc.add_heading("Appendix A – Execution evidence (BITS Virtual Lab)", 1)
    for s in ["Pipeline Runner – full pipeline completed", "Text→Video search: 'a person is playing guitar on a stage' – Top-K playable clips with scores",
              "Text→Video search: edge case (typo / short / long query) with diagnostics", "Video→Text retrieval",
              "Evaluation page – metric tables and charts", "Research Investigation page – answer and significance tests"]:
        para(f"[Insert screenshot: {s}]")
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    return out
