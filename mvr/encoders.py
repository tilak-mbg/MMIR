"""Pretrained encoders that map video frames and text into ONE shared space.

* CLIP (ViT image tower + Transformer text tower, contrastively pre-trained on
  400M image-text pairs). Frame embeddings and query embeddings are both
  projected by CLIP's own projection heads into a 512-d joint space and
  L2-normalised, so cosine similarity == dot product.
* X-CLIP (optional): a video-native model with cross-frame attention, used as an
  external spatio-temporal reference.

Only stable, version-independent sub-module calls are used
(``vision_model -> visual_projection`` and ``text_model -> text_projection``)
so the code works with transformers 4.3x as well as 5.x.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def preprocess_frames(frames: Sequence[np.ndarray], size: int = 224):
    """CLIP pre-processing: bicubic resize of the short side -> centre crop ->
    scale to [0,1] -> per-channel normalisation. Returns float tensor (N,3,S,S)."""
    import torch
    from PIL import Image

    out = []
    for f in frames:
        im = Image.fromarray(f).convert("RGB")
        w, h = im.size
        s = size / min(w, h)
        im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
        w, h = im.size
        left, top = (w - size) // 2, (h - size) // 2
        im = im.crop((left, top, left + size, top + size))
        out.append(np.asarray(im, dtype=np.float32) / 255.0)
    x = torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2)
    mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
    return (x - mean) / std


def _l2n(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


class ClipEncoder:
    """Wraps a Hugging Face CLIP checkpoint for frame & text embedding."""

    def __init__(self, model_name: str, device: str = "cpu"):
        import torch
        from transformers import CLIPModel, CLIPTokenizer

        self.device = device
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = CLIPModel.from_pretrained(model_name).to(device=device, dtype=self.dtype).eval()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.image_size = int(self.model.config.vision_config.image_size)
        self.max_len = int(getattr(self.model.config.text_config, "max_position_embeddings", 77))
        self.dim = int(self.model.config.projection_dim)
        self.name = model_name

    # ----- images -----
    def encode_frames(self, frames: Sequence[np.ndarray], batch_size: int = 64) -> np.ndarray:
        import torch

        feats = []
        for i in range(0, len(frames), batch_size):
            px = preprocess_frames(frames[i:i + batch_size], self.image_size).to(self.device, self.dtype)
            with torch.no_grad():
                out = self.model.vision_model(pixel_values=px)
                emb = self.model.visual_projection(out.pooler_output)
            feats.append(emb.float().cpu().numpy())
        return _l2n(np.concatenate(feats, 0))

    # ----- text -----
    def tokenize(self, texts: Sequence[str]):
        return self.tokenizer(list(texts), padding=True, truncation=True,
                              max_length=self.max_len, return_tensors="pt")

    def encode_text(self, texts: Sequence[str], batch_size: int = 256) -> np.ndarray:
        import torch

        feats = []
        for i in range(0, len(texts), batch_size):
            tok = self.tokenize(texts[i:i + batch_size])
            with torch.no_grad():
                out = self.model.text_model(input_ids=tok["input_ids"].to(self.device),
                                            attention_mask=tok["attention_mask"].to(self.device))
                emb = self.model.text_projection(out.pooler_output)
            feats.append(emb.float().cpu().numpy())
        return _l2n(np.concatenate(feats, 0))

    def token_count(self, text: str) -> int:
        """Number of BPE tokens incl. <sot>/<eot> (no truncation)."""
        return len(self.tokenizer(text, truncation=False)["input_ids"])

    def word_pieces(self, word: str) -> list[str]:
        return self.tokenizer.tokenize(word)

    def in_vocab(self, word: str) -> bool:
        """True if the word is a single whole-word BPE token in CLIP's vocabulary."""
        vocab = self.tokenizer.get_vocab()
        return (word.lower() + "</w>") in vocab


class XClipScorer:
    """Optional X-CLIP video-text scorer (video-native spatio-temporal model).

    X-CLIP conditions its text embedding on the video (prompt generator), so a
    (query x video) score matrix is produced by a joint forward pass.
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        import torch
        from transformers import CLIPTokenizer, XCLIPModel

        self.device = device
        self.model = XCLIPModel.from_pretrained(model_name).to(device).eval()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.num_frames = int(getattr(self.model.config.vision_config, "num_frames", 8))
        self.image_size = int(self.model.config.vision_config.image_size)
        self.torch = torch

    def video_pixels(self, frames: Sequence[np.ndarray]):
        idx = np.linspace(0, len(frames) - 1, self.num_frames).round().astype(int)
        return preprocess_frames([frames[i] for i in idx], self.image_size)  # (T,3,H,W)

    def score_matrix(self, texts: Sequence[str], videos_pixels: list, batch_videos: int = 8) -> np.ndarray:
        torch = self.torch
        tok = self.tokenizer(list(texts), padding=True, truncation=True, max_length=77, return_tensors="pt")
        tok = {k: v.to(self.device) for k, v in tok.items()}
        cols = []
        for i in range(0, len(videos_pixels), batch_videos):
            px = torch.stack(videos_pixels[i:i + batch_videos]).to(self.device)  # (B,T,3,H,W)
            with torch.no_grad():
                out = self.model(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"],
                                 pixel_values=px)
            cols.append(out.logits_per_text.float().cpu().numpy())  # (n_text, B)
        return np.concatenate(cols, axis=1)  # (n_text, n_videos)
