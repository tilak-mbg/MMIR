"""Vector index over the 100 video clips.

For every video representation (method) we store an (N,D) matrix of
L2-normalised vectors in a FAISS ``IndexFlatIP`` (exact inner-product search;
inner product of unit vectors == cosine similarity). Frame-level vectors
(N*T,D) are also kept for query-conditioned pooling and per-frame
visualisation. Falls back to NumPy matrix products if FAISS is unavailable.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None


class VideoIndex:
    def __init__(self, video_ids: list[str], frames: np.ndarray, pooled: dict[str, np.ndarray],
                 tau: float = 0.01):
        self.video_ids = list(video_ids)
        self.frames = frames.astype(np.float32)          # (N,T,D)
        self.pooled = {k: v.astype(np.float32) for k, v in pooled.items()}
        self.tau = tau
        self.backend = "faiss.IndexFlatIP" if faiss is not None else "numpy"
        self._faiss = {}
        if faiss is not None:
            for m, M in self.pooled.items():
                ix = faiss.IndexFlatIP(M.shape[1])
                ix.add(np.ascontiguousarray(M))
                self._faiss[m] = ix

    @property
    def methods(self) -> list[str]:
        return list(self.pooled.keys()) + ["query_attn"]

    def scores(self, q: np.ndarray, method: str) -> np.ndarray:
        """Full similarity matrix (Q,N) for a batch of query embeddings."""
        q = np.atleast_2d(q).astype(np.float32)
        if method == "query_attn":
            from .temporal import query_attn_scores
            return query_attn_scores(q, self.frames, self.tau)[0]
        return q @ self.pooled[method].T

    def search(self, q: np.ndarray, method: str, k: int = 5) -> dict:
        """Top-K search for one query embedding. Returns ids, scores, per-frame sims."""
        t0 = time.perf_counter()
        q = np.atleast_2d(q).astype(np.float32)
        k = int(min(max(k, 1), len(self.video_ids)))
        if method == "query_attn":
            S = self.scores(q, method)[0]
            top = np.argsort(-S, kind="stable")[:k]
            sc = S[top]
        elif method in self._faiss:
            sc, top = self._faiss[method].search(q, k)
            sc, top = sc[0], top[0]
        else:
            S = self.scores(q, method)[0]
            top = np.argsort(-S, kind="stable")[:k]
            sc = S[top]
        latency = (time.perf_counter() - t0) * 1000
        frame_sims = np.einsum("d,ktd->kt", q[0], self.frames[top])
        return {"idx": top.tolist(), "scores": [float(s) for s in sc], "latency_ms": latency,
                "frame_sims": frame_sims}

    # ------------------------------------------------------------------
    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "frames.npy", self.frames)
        for m, M in self.pooled.items():
            np.save(d / f"pooled_{m}.npy", M)
            if m in self._faiss:
                faiss.write_index(self._faiss[m], str(d / f"{m}.faiss"))
        (d / "meta.json").write_text(json.dumps({"video_ids": self.video_ids, "tau": self.tau,
                                                 "methods": list(self.pooled), "backend": self.backend}))

    @classmethod
    def load(cls, d: Path) -> "VideoIndex":
        meta = json.loads((d / "meta.json").read_text())
        frames = np.load(d / "frames.npy")
        pooled = {m: np.load(d / f"pooled_{m}.npy") for m in meta["methods"]}
        return cls(meta["video_ids"], frames, pooled, meta.get("tau", 0.01))
