"""Video-level representations built from per-frame CLIP embeddings.

Parameter-free (zero-shot) aggregators
    single_frame  : middle frame only
    mean_pool     : average of frame embeddings  (baseline in the research question)
    max_pool      : element-wise max
    query_attn    : query-conditioned softmax pooling of frame similarities

Learned heads (trained on a disjoint MSR-VTT training pool, CLIP frozen)
    mlp_adapter   : residual MLP on the mean-pooled vector  -> CONTROL (domain
                    adaptation without any inter-frame modelling)
    set_tfm       : Transformer over frames WITHOUT positional embeddings
                    (frame interactions, permutation-invariant)
    temporal_tfm  : Transformer over frames WITH temporal position embeddings
                    (explicit temporal order, CLIP4Clip "seqTransf")
    seq_lstm      : LSTM over ordered frames (CLIP4Clip "seqLSTM")

All learned heads are residual + zero-initialised, i.e. at initialisation they
are *exactly* mean pooling; any change in performance is therefore caused by
what the head learns on top of the baseline.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

ProgressFn = Optional[Callable[[float, str], None]]


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


# ---------------------------------------------------------------------------
# Parameter-free aggregation
# ---------------------------------------------------------------------------
def pool_video(frames: np.ndarray, method: str) -> np.ndarray:
    """frames: (N,T,D) L2-normalised -> (N,D) L2-normalised video vectors."""
    if method == "single_frame":
        return l2n(frames[:, frames.shape[1] // 2])
    if method == "mean_pool":
        return l2n(frames.mean(1))
    if method == "max_pool":
        return l2n(frames.max(1))
    raise ValueError(method)


def query_attn_scores(q: np.ndarray, frames: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Query-conditioned temporal pooling.

    s_{q,n,t} = <q, f_{n,t}>;  w = softmax_t(s / tau);  S_{q,n} = sum_t w * s.
    tau -> inf  == mean pooling of similarities; tau -> 0 == max over frames.
    Returns (S (Q,N), weights (Q,N,T)).
    """
    s = np.einsum("qd,ntd->qnt", q, frames)
    z = s / tau
    z = z - z.max(-1, keepdims=True)
    w = np.exp(z)
    w = w / w.sum(-1, keepdims=True)
    return (w * s).sum(-1), w


# ---------------------------------------------------------------------------
# Learned temporal heads
# ---------------------------------------------------------------------------
def _torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


def build_head(arch: str, dim: int, max_frames: int = 16, layers: int = 2, heads: int = 8,
               dropout: float = 0.1):
    torch, nn, F = _torch()

    class TemporalHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.arch = arch
            if arch == "mlp_adapter":
                self.f = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(),
                                       nn.Dropout(dropout), nn.Linear(2 * dim, dim))
                last = self.f[-1]
            elif arch in ("set_tfm", "temporal_tfm"):
                self.pos = nn.Embedding(max_frames, dim) if arch == "temporal_tfm" else None
                if self.pos is not None:
                    nn.init.normal_(self.pos.weight, std=0.02)
                layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout, batch_first=True,
                                                   norm_first=True, activation="gelu")
                self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
                self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
                last = self.out[-1]
            elif arch == "seq_lstm":
                self.lstm = nn.LSTM(dim, dim, batch_first=True)
                self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
                last = self.out[-1]
            else:
                raise ValueError(arch)
            nn.init.zeros_(last.weight)   # residual zero-init => starts as mean pooling
            nn.init.zeros_(last.bias)

        def forward(self, x, pos_idx=None):  # x: (B,T,D)
            if self.arch == "mlp_adapter":
                base = x.mean(1)
                v = base + self.f(base)
            elif self.arch in ("set_tfm", "temporal_tfm"):
                h = x
                if self.pos is not None:
                    if pos_idx is None:
                        pos_idx = torch.arange(x.shape[1], device=x.device)
                    h = h + self.pos(pos_idx).unsqueeze(0) if pos_idx.dim() == 1 else h + self.pos(pos_idx)
                h = self.enc(h)
                v = (x + self.out(h)).mean(1)
            else:  # seq_lstm
                h, _ = self.lstm(x)
                v = (x + self.out(h)).mean(1)
            return F.normalize(v, dim=-1)

    return TemporalHead()


def head_embed(head, frames: np.ndarray, pos_idx: np.ndarray | None = None, device: str = "cpu") -> np.ndarray:
    torch, _, _ = _torch()
    head.eval()
    with torch.no_grad():
        x = torch.from_numpy(frames.astype(np.float32)).to(device)
        p = None if pos_idx is None else torch.from_numpy(np.asarray(pos_idx)).long().to(device)
        return head(x, p).cpu().numpy()


def train_head(arch: str, train_frames: np.ndarray, train_text: np.ndarray, text_mask: np.ndarray,
               seed: int, epochs: int = 40, lr: float = 1e-4, weight_decay: float = 0.05,
               batch: int = 128, layers: int = 2, heads: int = 8, device: str = "cpu",
               progress: ProgressFn = None) -> tuple[object, dict]:
    """Contrastive (symmetric InfoNCE) training of a temporal head on frozen CLIP features.

    train_frames (M,T,D), train_text (M,C,D) caption embeddings, text_mask (M,C) bool.
    10% of the training videos are held out for early stopping (the evaluation
    index is never used for any training or model-selection decision).
    """
    torch, nn, F = _torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    M, T, D = train_frames.shape
    perm = rng.permutation(M)
    n_val = max(10, int(0.1 * M))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    head = build_head(arch, D, max_frames=T, layers=layers, heads=heads).to(device)
    logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.03), device=device))
    opt = torch.optim.AdamW(list(head.parameters()) + [logit_scale], lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, math.ceil(len(tr_idx) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per_epoch,
                                                pct_start=0.1)
    Xf = torch.from_numpy(train_frames.astype(np.float32)).to(device)
    Xt = torch.from_numpy(train_text.astype(np.float32)).to(device)

    def val_mrr() -> float:
        v = head_embed(head, train_frames[val_idx], device=device)
        caps, owner = [], []
        for j, vi in enumerate(val_idx):
            for c in np.where(text_mask[vi])[0][:5]:
                caps.append(train_text[vi, c])
                owner.append(j)
        S = np.stack(caps) @ v.T
        gt = S[np.arange(len(owner)), owner]
        ranks = 1 + (S > gt[:, None]).sum(1)
        return float((1.0 / ranks).mean())

    hist = {"epoch": [], "loss": [], "val_mrr": []}
    best, best_state, best_ep = val_mrr(), {k: v.detach().clone() for k, v in head.state_dict().items()}, 0
    hist["epoch"].append(0); hist["loss"].append(None); hist["val_mrr"].append(best)
    t0 = time.time()
    for ep in range(1, epochs + 1):
        head.train()
        order = rng.permutation(tr_idx)
        losses = []
        for b in range(0, len(order), batch):
            ids = order[b:b + batch]
            if len(ids) < 2:
                continue
            # one random caption per video
            cap = np.array([rng.choice(np.where(text_mask[i])[0]) for i in ids])
            t_emb = Xt[torch.as_tensor(ids, device=device), torch.as_tensor(cap, device=device)]
            x = Xf[torch.as_tensor(ids, device=device)]
            pos = None
            if rng.random() < 0.5:  # temporal sub-sampling augmentation (k frames, keep order)
                k = int(rng.choice([4, 8]))
                sel = np.sort(rng.choice(T, size=k, replace=False))
                x = x[:, torch.as_tensor(sel, device=device)]
                pos = torch.as_tensor(sel, device=device)
            v = head(x, pos)
            logits = logit_scale.exp().clamp(max=100) * t_emb @ v.t()
            lbl = torch.arange(len(ids), device=device)
            loss = 0.5 * (F.cross_entropy(logits, lbl) + F.cross_entropy(logits.t(), lbl))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        m = val_mrr()
        hist["epoch"].append(ep); hist["loss"].append(float(np.mean(losses)) if losses else None)
        hist["val_mrr"].append(m)
        if m > best:
            best, best_ep = m, ep
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        if progress:
            progress(ep / epochs, f"{arch} seed {seed}: epoch {ep}/{epochs} loss={hist['loss'][-1]:.3f} val MRR={m:.3f}")
    head.load_state_dict(best_state)
    hist.update({"best_val_mrr": best, "best_epoch": best_ep, "train_seconds": round(time.time() - t0, 1),
                 "n_train_videos": int(len(tr_idx)), "n_val_videos": int(len(val_idx)),
                 "n_params": int(sum(p.numel() for p in head.parameters()))})
    return head, hist


def save_head(head, hist: dict, path: Path) -> None:
    torch, _, _ = _torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), path)
    path.with_suffix(".json").write_text(json.dumps(hist, indent=1))


def load_head(arch: str, dim: int, path: Path, max_frames: int, layers: int, heads: int, device: str = "cpu"):
    torch, _, _ = _torch()
    head = build_head(arch, dim, max_frames=max_frames, layers=layers, heads=heads)
    head.load_state_dict(torch.load(path, map_location=device))
    return head.to(device).eval()


def select_tau(train_frames: np.ndarray, train_text: np.ndarray, text_mask: np.ndarray,
               grid=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2)) -> tuple[float, dict]:
    """Pick the query-attention temperature on the TRAINING pool (no test leakage)."""
    caps, owner = [], []
    for vi in range(len(train_frames)):
        c = np.where(text_mask[vi])[0]
        if len(c):
            caps.append(train_text[vi, c[0]])
            owner.append(vi)
    Q = np.stack(caps)
    res = {}
    for tau in grid:
        S, _ = query_attn_scores(Q, train_frames, tau)
        gt = S[np.arange(len(owner)), owner]
        ranks = 1 + (S > gt[:, None]).sum(1)
        res[tau] = float((1 / ranks).mean())
    best = max(res, key=res.get)
    return float(best), {str(k): v for k, v in res.items()}
