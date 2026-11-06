"""Positive-control experiment for the research question.

A synthetic retrieval task in which ONLY frame order distinguishes videos
("state A then B" vs "state B then A"). It verifies that the evaluation
set-up is able to detect a temporal benefit when one exists: order-aware heads
should succeed while mean pooling and order-free heads cannot. If, on real
MSR-VTT, the same heads show no gain, the absence is a property of the data,
not a failure of the temporal models.
"""
from __future__ import annotations

import numpy as np

from .temporal import head_embed, l2n, pool_video, train_head


def make_order_task(n_states=12, D=64, T=16, reps=6, seed=0):
    rng = np.random.default_rng(seed)
    S = l2n(rng.normal(size=(n_states, D)))
    W1, W2 = rng.normal(size=(D, D)) / 8, rng.normal(size=(D, D)) / 8
    frames, texts = [], []
    for a in range(n_states):
        for b in range(n_states):
            if a == b:
                continue
            for _ in range(reps):
                f = np.concatenate([np.repeat(S[a][None], T // 2, 0), np.repeat(S[b][None], T // 2, 0)])
                frames.append(l2n(f + 0.05 * rng.normal(size=f.shape)))
                texts.append(l2n(S[a] + S[b] + S[a] @ W1 + S[b] @ W2 + 0.05 * rng.normal(size=(5, D))))
    return np.array(frames, np.float32), np.array(texts, np.float32), reps


def _mrr(V, Q):
    S = Q @ V.T
    g = np.diag(S)
    return float((1 / (1 + (S > g[:, None]).sum(1))).mean())


def run_positive_control(epochs: int = 25, seed: int = 0) -> dict:
    F, Tx, reps = make_order_task(seed=seed)
    mask = np.ones(Tx.shape[:2], bool)
    te = np.arange(0, len(F), reps)
    tr = np.setdiff1d(np.arange(len(F)), te)
    Fte, Qte = F[te], Tx[te][:, 0]
    out = {"mean_pool": _mrr(pool_video(Fte, "mean_pool"), Qte),
           "max_pool": _mrr(pool_video(Fte, "max_pool"), Qte)}
    for arch in ("mlp_adapter", "set_tfm", "temporal_tfm", "seq_lstm"):
        h, _ = train_head(arch, F[tr], Tx[tr], mask[tr], seed=seed, epochs=epochs, lr=1e-3, heads=4, batch=128)
        out[arch] = _mrr(head_embed(h, Fte), Qte)
    return {"mrr": out, "n_test_queries": int(len(te)), "n_train_videos": int(len(tr))}
