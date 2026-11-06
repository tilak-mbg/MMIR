"""Standard IR evaluation metrics and statistical tests.

Text->Video (T2V): each query has exactly ONE relevant video, so
    AP(q) = 1/rank(q)  =>  mAP == MRR   (reported for completeness)
Video->Text (V2T): each video has TWO relevant captions -> mAP differs from MRR.
"""
from __future__ import annotations

import numpy as np


def t2v_ranks(S: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """S (Q,N) similarity, gt (Q,) index of the relevant video -> 1-based ranks."""
    g = S[np.arange(len(gt)), gt]
    return 1 + (S > g[:, None]).sum(1)


def summarize_ranks(ranks: np.ndarray, ks=(1, 5, 10)) -> dict:
    ranks = np.asarray(ranks, dtype=float)
    out = {f"R@{k}": float((ranks <= k).mean() * 100) for k in ks}
    rr = 1.0 / ranks
    out["MRR"] = float(rr.mean())
    out["mAP"] = float(rr.mean())  # single relevant item
    out["nDCG@10"] = float(np.where(ranks <= 10, 1 / np.log2(ranks + 1), 0).mean())
    out["MdR"] = float(np.median(ranks))
    out["MnR"] = float(ranks.mean())
    return out


def v2t_metrics(S_vt: np.ndarray, rel: list[set], ks=(1, 5, 10)) -> tuple[dict, dict]:
    """S_vt (N,Q): video->caption similarities, rel[n] = set of relevant caption ids."""
    order = np.argsort(-S_vt, axis=1, kind="stable")
    first, aps, ndcgs = [], [], []
    for n in range(S_vt.shape[0]):
        hits = np.isin(order[n], list(rel[n]))
        pos = np.where(hits)[0] + 1
        first.append(pos[0])
        prec = np.arange(1, len(pos) + 1) / pos
        aps.append(prec.mean())
        dcg = sum(1 / np.log2(p + 1) for p in pos if p <= 10)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(rel[n]), 10)))
        ndcgs.append(dcg / idcg)
    first = np.array(first, dtype=float)
    out = {f"R@{k}": float((first <= k).mean() * 100) for k in ks}
    out["MRR"] = float((1 / first).mean())
    out["mAP"] = float(np.mean(aps))
    out["nDCG@10"] = float(np.mean(ndcgs))
    out["MdR"] = float(np.median(first))
    out["MnR"] = float(first.mean())
    return out, {"first_rank": first.tolist(), "ap": [float(a) for a in aps]}


# ---------------------------------------------------------------------------
# Significance testing (paired, query-level)
# ---------------------------------------------------------------------------
def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 10000, seed: int = 0) -> dict:
    """Bootstrap the mean difference (b - a) over queries. Returns delta, 95% CI, p."""
    rng = np.random.default_rng(seed)
    d = np.asarray(b, float) - np.asarray(a, float)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    boots = d[idx].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # two-sided p-value: how often the bootstrap distribution crosses zero
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"delta": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi), "p_boot": float(min(1.0, p))}


def wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from scipy.stats import wilcoxon
        d = np.asarray(b, float) - np.asarray(a, float)
        if np.allclose(d, 0):
            return 1.0
        return float(wilcoxon(a, b, zero_method="zsplit").pvalue)
    except Exception:
        return float("nan")


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni adjustment of a {name: p} dict."""
    items = sorted([(p, k) for k, p in pvals.items() if p == p], key=lambda x: x[0])
    m = len(items)
    adj, running = {}, 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    for k, p in pvals.items():
        adj.setdefault(k, float("nan"))
    return adj
