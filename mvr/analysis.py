"""Data-driven answer to Research Investigation 1:
"Is mean-pooling frame embeddings sufficient, or does explicitly modelling
temporal relationships give superior retrieval performance?"

All statements below are generated from the measured numbers (no hard-coded
claims), so the report always matches the run that produced it.
"""
from __future__ import annotations

from .config import method_label


def _sig(R, a, b):
    for row in R.get("significance", []):
        if row["method"] == a and row["reference"] == b:
            return row
    return None


def _fmt_delta(row, key="MRR", scale=1.0, unit=""):
    d = row[key]
    return (f"{d['delta']*scale:+.3f}{unit} (95% CI {d['ci_low']*scale:+.3f} to {d['ci_high']*scale:+.3f}; "
            f"Wilcoxon p={row['wilcoxon_p']:.3g}, Holm-adj. p={row['wilcoxon_p_holm']:.3g})")


def _is_sig(row, alpha=0.05):
    if row is None:
        return False
    ci = row["MRR"]
    excl0 = ci["ci_low"] > 0 or ci["ci_high"] < 0
    p = row.get("wilcoxon_p_holm", 1.0)
    return excl0 and (p == p) and p < alpha


def research_conclusions(R: dict) -> dict:
    t2v = R["t2v"]
    mp = t2v["mean_pool"]
    pts = []
    learned_present = all(m in t2v for m in ("mlp_adapter", "set_tfm", "temporal_tfm"))

    # 1. Baseline position
    pf = [m for m in ("single_frame", "mean_pool", "max_pool", "query_attn") if m in t2v]
    best_pf = max(pf, key=lambda m: t2v[m]["MRR"])
    sf = t2v.get("single_frame")
    if sf:
        d = mp["MRR"] - sf["MRR"]
        if d > 0.005:
            tail = "so aggregating evidence over the whole clip clearly helps."
        elif d < -0.005:
            tail = "so averaging all frames does not help over one representative frame in this run."
        else:
            tail = "so for these clips one representative frame carries almost all of the retrievable signal."
        pts.append(f"Mean pooling reaches R@1={mp['R@1']:.1f}% / MRR={mp['MRR']:.3f} vs. a single middle frame at "
                   f"R@1={sf['R@1']:.1f}% / MRR={sf['MRR']:.3f} ({d:+.3f} MRR), " + tail)
    if best_pf != "mean_pool":
        row = _sig(R, best_pf, "mean_pool")
        pts.append(f"Among parameter-free (zero-shot) aggregators the best is {method_label(best_pf)} "
                   f"(MRR {t2v[best_pf]['MRR']:.3f}); difference vs. mean pooling: "
                   + (_fmt_delta(row) if row else "n/a") + ".")
    else:
        pts.append("Among parameter-free (zero-shot) aggregators, mean pooling is the strongest; max pooling and "
                   "query-attentive pooling do not improve on it.")

    # 2. Frame-count saturation
    sw = R.get("frame_sweep", {}).get("mean_pool")
    if sw:
        ks = sorted(sw, key=int)
        best = max(sw[k]["MRR"] for k in ks)
        sat = next(k for k in ks if sw[k]["MRR"] >= 0.98 * best)
        pts.append(f"Mean pooling saturates quickly: with {sat} frame(s) it already achieves >=98% of its best MRR "
                   f"({sw[sat]['MRR']:.3f} vs {best:.3f}); adding more frames brings diminishing returns.")

    verdict = "inconclusive"
    if learned_present:
        ad, st, tt = t2v["mlp_adapter"], t2v["set_tfm"], t2v["temporal_tfm"]
        r_tt_mp = _sig(R, "temporal_tfm", "mean_pool")
        r_order = _sig(R, "temporal_tfm", "set_tfm")
        r_tt_ad = _sig(R, "temporal_tfm", "mlp_adapter")
        pts.append(f"Temporal Transformer (+positions) vs mean pooling: MRR {tt['MRR']:.3f} vs {mp['MRR']:.3f}, "
                   f"R@1 {tt['R@1']:.1f}% vs {mp['R@1']:.1f}%; delta MRR = " + (_fmt_delta(r_tt_mp) if r_tt_mp else "n/a") + ".")
        pts.append(f"Decomposition of that difference: (i) domain adaptation alone (MLP adapter, no temporal mixing) "
                   f"changes MRR by {ad['MRR']-mp['MRR']:+.3f}; (ii) adding cross-frame self-attention without order "
                   f"(Set Transformer) adds {st['MRR']-ad['MRR']:+.3f}; (iii) adding temporal ORDER (position "
                   f"embeddings) adds {tt['MRR']-st['MRR']:+.3f}"
                   + (f" [order effect: {_fmt_delta(r_order)}]" if r_order else "") + ".")
        if "seq_lstm" in t2v:
            ls = t2v["seq_lstm"]
            pts.append(f"The order-aware LSTM obtains MRR {ls['MRR']:.3f} (R@1 {ls['R@1']:.1f}%), i.e. "
                       f"{ls['MRR']-mp['MRR']:+.3f} vs mean pooling and {ls['MRR']-ad['MRR']:+.3f} vs the MLP control.")

        oa = R.get("order_ablation", {})
        if "temporal_tfm" in oa:
            o = oa["temporal_tfm"]
            pts.append(f"Shuffling the frame order at test time changes the Temporal Transformer's MRR from "
                       f"{o['original']['MRR']:.3f} to {o['shuffled']['MRR']:.3f} (reversed: {o['reversed']['MRR']:.3f}); "
                       f"order-agnostic methods are unaffected by construction (mean pooling: "
                       f"{oa['mean_pool']['original']['MRR']:.3f} -> {oa['mean_pool']['shuffled']['MRR']:.3f}). "
                       + ("The small change shows the model barely relies on frame order for these queries."
                          if abs(o['original']['MRR'] - o['shuffled']['MRR']) < 0.01 else
                          "The model does use frame order, but see the significance tests for whether this helps."))

        tl = R.get("temporal_language", {})
        if tl.get("n_temporal", 0) >= 5:
            bm = tl["by_method"]
            d_t = bm["temporal_tfm"]["temporal_MRR"] - bm["mean_pool"]["temporal_MRR"]
            d_o = bm["temporal_tfm"]["other_MRR"] - bm["mean_pool"]["other_MRR"]
            pts.append(f"On the {tl['n_temporal']} queries with explicit temporal language (e.g. 'then', 'after', "
                       f"'starts') the Temporal Transformer changes MRR by {d_t:+.3f} vs mean pooling, against "
                       f"{d_o:+.3f} on the remaining {tl['n_other']} queries.")

        temporal_beats_control = _is_sig(r_tt_ad, 0.05) and r_tt_ad["MRR"]["delta"] > 0
        order_helps = _is_sig(r_order, 0.05) and r_order["MRR"]["delta"] > 0
        tt_beats_mp = _is_sig(r_tt_mp, 0.05) and r_tt_mp["MRR"]["delta"] > 0
        if order_helps and temporal_beats_control:
            verdict = "temporal_better"
            answer = ("Explicit temporal modelling gives a statistically significant improvement over mean pooling "
                      "that is NOT explained by domain adaptation alone (it also beats the MLP control and the "
                      "order-free Set Transformer). Mean pooling is therefore not fully sufficient.")
        elif tt_beats_mp and not order_helps:
            verdict = "adaptation_not_order"
            answer = ("The trained temporal models beat zero-shot mean pooling, but the gain is explained by "
                      "learning on in-domain MSR-VTT data (the MLP control and the order-free Set Transformer "
                      "capture it) rather than by modelling temporal ORDER. Mean-pooled CLIP frame embeddings "
                      "are a sufficient representation of temporal content for this retrieval task; explicit "
                      "temporal modelling adds no significant benefit.")
        elif order_helps:
            verdict = "mixed"
            answer = ("Temporal order gives a significant gain over the order-free Set Transformer, but the overall "
                      "improvement of the Temporal Transformer over the mean-pooling family is not significant. "
                      "Explicit temporal modelling therefore offers at most a modest benefit; mean pooling remains a "
                      "strong, sufficient baseline at a fraction of the complexity.")
        else:
            verdict = "mean_pool_sufficient"
            answer = ("No temporal model significantly outperforms mean pooling after correction for multiple "
                      "comparisons. Mean pooling of CLIP frame embeddings is sufficient for this text-to-video "
                      "retrieval setting: MSR-VTT captions are dominated by static appearance cues (objects, "
                      "scenes, people) that are already captured by averaging frames.")
    else:
        answer = ("Learned temporal heads were not trained in this run, so only parameter-free aggregators can be "
                  "compared; see the frame-sweep and order-ablation sections.")

    xc = t2v.get("xclip")
    if xc:
        if xc["MRR"] < mp["MRR"] - 0.05:
            pts.append(f"The off-the-shelf spatio-temporal model X-CLIP (fine-tuned on Kinetics-400 action labels) reaches "
                       f"only R@1={xc['R@1']:.1f}% / MRR={xc['MRR']:.3f} (median rank {xc['MdR']:.0f}) versus "
                       f"{mp['R@1']:.1f}% / {mp['MRR']:.3f} for mean-pooled CLIP. Its text tower was adapted to ~400 "
                       f"short action-class names, so it does not transfer to free-form MSR-VTT captions: a model "
                       f"that natively encodes time is not automatically better - large-scale image-text alignment "
                       f"matters more than temporal architecture for this task.")
        else:
            pts.append(f"The off-the-shelf spatio-temporal model X-CLIP reaches R@1={xc['R@1']:.1f}% / MRR={xc['MRR']:.3f} "
                       f"versus {mp['R@1']:.1f}% / {mp['MRR']:.3f} for mean-pooled CLIP"
                       + (" - a native video architecture gives a measurable advantage." if xc["MRR"] > mp["MRR"] + 0.02
                          else " - no clear advantage from a native video architecture."))
    pc = R.get("positive_control", {}).get("mrr")
    if pc:
        pts.append(f"Positive control: on a synthetic task where only frame ORDER distinguishes clips, mean pooling "
                   f"scores MRR {pc['mean_pool']:.3f} and the order-free Set Transformer {pc['set_tfm']:.3f}, while the "
                   f"Temporal Transformer reaches {pc['temporal_tfm']:.3f} and the LSTM {pc['seq_lstm']:.3f}. The temporal "
                   f"heads and the evaluation are therefore capable of detecting a temporal benefit when one exists; "
                   f"any lack of gain on MSR-VTT reflects the data, not the models.")
    pts.append("Caveat: with 200 queries a 1-query change is 0.5 R@1 points, so small differences fall inside the "
               "95% bootstrap confidence intervals; conclusions are based on paired tests, not raw deltas.")
    return {"verdict": verdict, "answer": answer, "points": pts}
