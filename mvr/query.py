"""Natural-language query processing with explicit edge-case handling.

Edge cases handled
  * empty / whitespace / punctuation-only input  -> rejected with a clear message
  * unicode noise, control chars, repeated spaces -> NFKC normalisation + cleanup
  * unknown / misspelled vocabulary               -> detected against CLIP's BPE
        vocabulary and an MSR-VTT caption lexicon; optional edit-distance
        spelling correction (Norvig-style, edits <= 2, frequency-ranked)
  * very short queries (<= 2 content words)       -> prompt-template ensembling
        ("a video of {q}.", ...) which moves the query into CLIP's caption
        distribution
  * very long / detailed queries (> 77 BPE tokens)-> sentence-aware chunking;
        chunk embeddings are averaged instead of silently truncating
  * non-English input                             -> warning (CLIP text tower is English)
"""
from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

STOPWORDS = set("""a an the of in on at to for with and or is are was were be being been by from
this that these those it its as into onto over under up down out off some any very
his her their our your my he she they we you i me him them who which while""".split())

TEMPLATES = ["a video of {}.", "a video showing {}.", "{}"]
MAX_CHARS = 1000


class QueryError(ValueError):
    pass


@dataclass
class ProcessedQuery:
    original: str
    normalized: str
    final_text: str
    texts_to_encode: list
    mode: str = "normal"                 # short | normal | long
    n_tokens: int = 0
    truncated_without_chunking: bool = False
    unknown_words: list = field(default_factory=list)
    corrections: dict = field(default_factory=dict)
    steps: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class SpellCorrector:
    """Frequency-ranked edit-distance corrector over a caption lexicon."""

    letters = string.ascii_lowercase

    def __init__(self, counts: Counter):
        self.counts = counts

    def _edits1(self, w):
        splits = [(w[:i], w[i:]) for i in range(len(w) + 1)]
        deletes = [a + b[1:] for a, b in splits if b]
        transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
        replaces = [a + c + b[1:] for a, b in splits if b for c in self.letters]
        inserts = [a + c + b for a, b in splits for c in self.letters]
        return set(deletes + transposes + replaces + inserts)

    def correct(self, w: str) -> str | None:
        if w in self.counts:
            return None
        c1 = [e for e in self._edits1(w) if e in self.counts]
        if c1:
            return max(c1, key=self.counts.get)
        if len(w) >= 5:
            c2 = {e2 for e1 in self._edits1(w) for e2 in self._edits1(e1) if e2 in self.counts}
            if c2:
                return max(c2, key=self.counts.get)
        return None


def build_lexicon(captions) -> Counter:
    c = Counter()
    for cap in captions:
        c.update(re.findall(r"[a-z]+", cap.lower()))
    return c


class QueryProcessor:
    def __init__(self, encoder, lexicon: Counter | None = None):
        self.enc = encoder
        self.lexicon = lexicon or Counter()
        self.speller = SpellCorrector(self.lexicon) if self.lexicon else None
        self.max_tokens = getattr(encoder, "max_len", 77)

    # ------------------------------------------------------------------
    @staticmethod
    def normalize(raw: str) -> str:
        if raw is None:
            raise QueryError("Please enter a query.")
        s = unicodedata.normalize("NFKC", str(raw))
        s = "".join(ch if unicodedata.category(ch)[0] != "C" else " " for ch in s)
        s = re.sub(r"https?://\S+", " ", s)
        s = re.sub(r"\s+", " ", s).strip().strip("\"'`")
        return s[:MAX_CHARS]

    def _known(self, w: str) -> bool:
        if w in self.lexicon:
            return True
        try:
            return self.enc.in_vocab(w)
        except Exception:
            return True

    def _chunks(self, text: str) -> list[str]:
        """Greedy sentence/clause packing so each chunk fits CLIP's 77-token window."""
        parts = [p.strip() for p in re.split(r"(?<=[.!?;])\s+|,\s+", text) if p.strip()]
        chunks, cur = [], ""
        for p in parts:
            cand = (cur + ", " + p) if cur else p
            if self.enc.token_count(cand) <= self.max_tokens:
                cur = cand
                continue
            if cur:
                chunks.append(cur)
            # a single clause longer than the window: split by words
            if self.enc.token_count(p) > self.max_tokens:
                words, buf = p.split(), ""
                for w in words:
                    c2 = (buf + " " + w).strip()
                    if self.enc.token_count(c2) > self.max_tokens:
                        chunks.append(buf)
                        buf = w
                    else:
                        buf = c2
                cur = buf
            else:
                cur = p
        if cur:
            chunks.append(cur)
        return chunks

    # ------------------------------------------------------------------
    def process(self, raw: str, auto_correct: bool = True, use_templates: bool = True,
                use_chunking: bool = True) -> ProcessedQuery:
        norm = self.normalize(raw)
        if not norm or not re.search(r"[A-Za-z0-9]", norm):
            raise QueryError("The query is empty or contains no letters/digits. "
                             "Please describe the video, e.g. 'a man is playing guitar on stage'.")
        pq = ProcessedQuery(original=str(raw), normalized=norm, final_text=norm, texts_to_encode=[norm])
        if norm != str(raw):
            pq.steps.append("normalised unicode / whitespace")

        non_ascii = sum(1 for ch in norm if ch.isalpha() and ord(ch) > 127)
        alpha = sum(1 for ch in norm if ch.isalpha()) or 1
        if non_ascii / alpha > 0.3:
            pq.warnings.append("Query looks non-English; CLIP's text encoder is trained on English captions, "
                               "so results may be unreliable.")

        # ---- unknown vocabulary / spelling ----
        words = re.findall(r"[A-Za-z]+", norm)
        text = norm
        for w in words:
            lw = w.lower()
            if len(lw) < 3 or self._known(lw):
                continue
            pq.unknown_words.append(w)
            fix = self.speller.correct(lw) if self.speller else None
            if fix and auto_correct:
                pq.corrections[w] = fix
                text = re.sub(rf"\b{re.escape(w)}\b", fix, text)
        if pq.unknown_words:
            if pq.corrections:
                pq.steps.append("spelling corrected: " + ", ".join(f"{k}->{v}" for k, v in pq.corrections.items()))
            rest = [w for w in pq.unknown_words if w not in pq.corrections]
            if rest:
                pq.warnings.append("Out-of-vocabulary words (handled as BPE sub-words, may carry little meaning): "
                                   + ", ".join(rest))
        pq.final_text = text

        # ---- length handling ----
        content = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]
        pq.n_tokens = self.enc.token_count(text)
        if pq.n_tokens > self.max_tokens:
            pq.mode = "long"
            if use_chunking:
                pq.texts_to_encode = self._chunks(text)
                pq.steps.append(f"long query ({pq.n_tokens} tokens > {self.max_tokens}): split into "
                                f"{len(pq.texts_to_encode)} chunks, embeddings averaged")
            else:
                pq.truncated_without_chunking = True
                pq.texts_to_encode = [text]
                pq.warnings.append(f"Query truncated to {self.max_tokens} tokens.")
        elif len(content) <= 2:
            pq.mode = "short"
            if use_templates:
                pq.texts_to_encode = [t.format(text.rstrip(".")) for t in TEMPLATES]
                pq.steps.append("short query: prompt-template ensemble " + str(TEMPLATES))
            else:
                pq.texts_to_encode = [text]
            if len(content) == 0:
                pq.warnings.append("Query has only stop-words; results will be weak.")
        else:
            pq.texts_to_encode = [text]
        return pq

    def encode(self, pq: ProcessedQuery) -> np.ndarray:
        e = self.enc.encode_text(pq.texts_to_encode).mean(0)
        return e / max(np.linalg.norm(e), 1e-12)

    def encode_many(self, texts, **kw) -> np.ndarray:
        """Batch helper used by the robustness experiments."""
        pqs = [self.process(t, **kw) for t in texts]
        flat, owner = [], []
        for i, pq in enumerate(pqs):
            flat.extend(pq.texts_to_encode)
            owner.extend([i] * len(pq.texts_to_encode))
        E = self.enc.encode_text(flat)
        out = np.zeros((len(pqs), E.shape[1]), dtype=np.float32)
        np.add.at(out, np.array(owner), E)
        return out / np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-12, None)


# ---------------------------------------------------------------------------
# Query perturbations for the robustness study
# ---------------------------------------------------------------------------
def inject_typos(text: str, rng: np.random.Generator, rate: float = 0.35) -> str:
    words = text.split()
    out = []
    for w in words:
        if len(w) >= 4 and w.isalpha() and rng.random() < rate:
            i = int(rng.integers(1, len(w) - 1))
            op = rng.integers(0, 3)
            if op == 0:
                w = w[:i] + w[i + 1] + w[i] + w[i + 2:]          # transpose
            elif op == 1:
                w = w[:i] + w[i + 1:]                           # delete
            else:
                w = w[:i] + "abcdefghijklmnopqrstuvwxyz"[rng.integers(0, 26)] + w[i + 1:]  # substitute
        out.append(w)
    return " ".join(out)


def shorten(text: str, n_content: int = 2) -> str:
    """Keep only the first n content words (e.g. 'man guitar')."""
    content = [w for w in re.findall(r"[A-Za-z]+", text) if w.lower() not in STOPWORDS]
    return " ".join(content[:n_content]) or text
