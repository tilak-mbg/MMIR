"""Central configuration for the MSR-VTT text-to-video retrieval system.

Every value can be overridden from the Streamlit "Pipeline Runner" page or via
environment variables (MVR_<NAME>), so the same code runs on CPU or GPU and
online (Hugging Face) or offline (manually placed files).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Canonical MSR-VTT category names (Xu et al., CVPR 2016), id -> name.
MSRVTT_CATEGORIES = [
    "music", "people", "gaming", "sports/actions", "news/events/politics",
    "education", "tv shows", "movie/comedy", "animation", "vehicles/autos",
    "how-to", "travel", "science/technology", "animals/pets", "kids/family",
    "documentary", "food/drink", "cooking", "beauty/fashion", "advertisement",
]

# Methods evaluated. Order = display order.
PARAM_FREE_METHODS = ["single_frame", "mean_pool", "max_pool", "query_attn"]
LEARNED_METHODS = ["mlp_adapter", "set_tfm", "temporal_tfm", "seq_lstm"]

METHOD_INFO = {
    "single_frame": ("Single middle frame", "No temporal information (lower bound)."),
    "mean_pool": ("Mean pooling (baseline)", "Average of L2-normalised CLIP frame embeddings - order-agnostic."),
    "max_pool": ("Max pooling", "Element-wise max over frame embeddings - order-agnostic."),
    "query_attn": ("Query-attentive pooling", "Text-conditioned softmax over per-frame similarities (parameter-free, order-agnostic)."),
    "mlp_adapter": ("MLP adapter (control)", "Trained residual MLP on the mean-pooled vector; same data/loss as the temporal models but NO inter-frame modelling."),
    "set_tfm": ("Set Transformer (no positions)", "Trained self-attention across frames WITHOUT positional embeddings - models frame interactions but not order."),
    "temporal_tfm": ("Temporal Transformer (+positions)", "Trained self-attention across frames WITH learned temporal position embeddings (CLIP4Clip seqTransf-style) - explicit temporal order."),
    "seq_lstm": ("Temporal LSTM", "Trained uni-directional LSTM over the ordered frame sequence (CLIP4Clip seqLSTM-style)."),
    "xclip": ("X-CLIP (pretrained video model)", "Off-the-shelf spatio-temporal video-language model (cross-frame attention + multi-frame integration transformer)."),
}

ALL_SEARCH_METHODS = PARAM_FREE_METHODS + LEARNED_METHODS


@dataclass
class Config:
    # --- data ---
    hf_dataset_repo: str = "friedrichor/MSR-VTT"
    local_repo_dir: str = ""          # offline mirror of the HF repo layout (optional)
    data_dir: str = str(PROJECT_ROOT / "data" / "msrvtt")
    artifacts_dir: str = str(PROJECT_ROOT / "artifacts")
    videos_per_category: int = 5      # 20 categories x 5 = 100 index clips
    captions_per_video: int = 2       # 100 x 2 = 200 queries
    train_videos_per_category: int = 40  # disjoint training pool for learned temporal heads
    seed: int = 42

    # --- features ---
    clip_model: str = "openai/clip-vit-base-patch32"
    num_frames: int = 16              # uniformly sampled (segment mid-points)
    batch_size: int = 64
    device: str = "auto"              # auto | cuda | cpu

    # --- learned temporal heads ---
    train_seeds: tuple = (0, 1, 2)
    epochs: int = 40
    lr: float = 1e-4
    weight_decay: float = 0.05
    train_batch: int = 128
    tfm_layers: int = 2
    tfm_heads: int = 8

    cpu_fast: bool = True             # on CPU-only machines cap training size/epochs (see eff_* below)

    # --- optional external spatio-temporal model ---
    use_xclip: bool = True
    xclip_model: str = "microsoft/xclip-base-patch32"

    # --- evaluation ---
    bootstrap_samples: int = 10000
    ks: tuple = (1, 5, 10)

    # derived paths -------------------------------------------------------
    @property
    def videos_dir(self) -> Path:
        return Path(self.data_dir) / "videos"

    @property
    def subset_path(self) -> Path:
        return Path(self.data_dir) / "subset.json"

    @property
    def model_tag(self) -> str:
        return self.clip_model.rstrip("/").split("/")[-1].replace(".", "_")

    @property
    def features_path(self) -> Path:
        return Path(self.artifacts_dir) / f"features_{self.model_tag}_T{self.num_frames}.npz"

    @property
    def index_dir(self) -> Path:
        return Path(self.artifacts_dir) / f"index_{self.model_tag}_T{self.num_frames}"

    @property
    def models_dir(self) -> Path:
        return Path(self.artifacts_dir) / f"temporal_heads_{self.model_tag}"

    @property
    def results_path(self) -> Path:
        return Path(self.artifacts_dir) / f"results_{self.model_tag}_T{self.num_frames}.json"

    @property
    def figures_dir(self) -> Path:
        return Path(self.artifacts_dir) / "figures"

    @property
    def thumbs_dir(self) -> Path:
        return Path(self.artifacts_dir) / "thumbs"

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def eff_train_per_cat(self) -> int:
        """Training clips per category actually used (capped at 25 on CPU when cpu_fast)."""
        if self.cpu_fast and self.resolved_device() == "cpu":
            return min(self.train_videos_per_category, 25)
        return self.train_videos_per_category

    def eff_epochs(self) -> int:
        if self.cpu_fast and self.resolved_device() == "cpu":
            return min(self.epochs, 20)
        return self.epochs

    def to_dict(self) -> dict:
        d = asdict(self)
        d["train_seeds"] = list(self.train_seeds)
        d["ks"] = list(self.ks)
        return d

    # persistence so the dashboard remembers the last-used settings
    def save(self, path: Path | None = None) -> None:
        path = path or Path(self.artifacts_dir) / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        cfg = cls()
        path = path or Path(os.environ.get("MVR_ARTIFACTS_DIR", cfg.artifacts_dir)) / "config.json"
        if path.exists():
            try:
                d = json.loads(path.read_text())
                for k, v in d.items():
                    if hasattr(cfg, k):
                        if k in ("train_seeds", "ks"):
                            v = tuple(v)
                        setattr(cfg, k, v)
            except Exception:
                pass
        # environment overrides (MVR_CLIP_MODEL=..., etc.)
        for k in list(asdict(cfg).keys()):
            env = os.environ.get("MVR_" + k.upper())
            if env is None:
                continue
            cur = getattr(cfg, k)
            if isinstance(cur, bool):
                setattr(cfg, k, env.lower() in ("1", "true", "yes"))
            elif isinstance(cur, int):
                setattr(cfg, k, int(env))
            elif isinstance(cur, float):
                setattr(cfg, k, float(env))
            elif isinstance(cur, tuple):
                setattr(cfg, k, tuple(int(x) for x in env.split(",")))
            else:
                setattr(cfg, k, env)
        return cfg


def method_label(m: str) -> str:
    return METHOD_INFO.get(m, (m, ""))[0]
