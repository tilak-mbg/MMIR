"""Data ingestion: MSR-VTT annotations, stratified sampling, video extraction
and frame decoding/pre-processing.

Data source: Hugging Face dataset repo ``friedrichor/MSR-VTT``
  * msrvtt_test_1k.json   - 1k-A test videos (category, 1 caption each)
  * msrvtt_train_9k.json  - 9k training videos (category, 20 captions each)
  * raw_data/MSRVTT_data.json - original annotation (all 20 captions / video)
  * raw_data/category.txt - category names
  * MSRVTT_Videos.zip     - all 10k .mp4 clips

Protocol implemented (as per the assignment):
  * Index  : 100 clips = 5 clips x 20 categories, drawn from the 1k-A TEST videos
  * Queries: 200 = 2 human captions per index clip
  * Train  : a disjoint pool from the 9k TRAIN videos, used only to fit the
             learned temporal heads (never overlaps with the index).
"""
from __future__ import annotations

import json
import random
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .config import MSRVTT_CATEGORIES, Config

ProgressFn = Optional[Callable[[float, str], None]]


def _p(progress: ProgressFn, frac: float, msg: str) -> None:
    if progress:
        progress(min(max(frac, 0.0), 1.0), msg)
    else:
        print(f"[{frac*100:5.1f}%] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Fetching files (Hugging Face hub or an offline local mirror)
# ---------------------------------------------------------------------------
def fetch_repo_file(cfg: Config, filename: str) -> Path:
    """Return a local path to ``filename`` from the dataset repo.

    Order: (1) already in data_dir, (2) local mirror dir, (3) HF hub download.
    """
    target = Path(cfg.data_dir) / filename
    if target.exists() and target.stat().st_size > 0:
        return target
    if cfg.local_repo_dir:
        src = Path(cfg.local_repo_dir) / filename
        if src.exists():
            return src
        raise FileNotFoundError(f"{filename} not found in local mirror {cfg.local_repo_dir}")
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=cfg.hf_dataset_repo, filename=filename,
                           repo_type="dataset", local_dir=cfg.data_dir)
    return Path(path)


def _load_json_records(path: Path) -> list:
    """Load either a JSON array or JSON-lines file."""
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    if txt[0] == "[":
        return json.loads(txt)
    if txt[0] == "{":
        try:
            obj = json.loads(txt)
            return obj if isinstance(obj, list) else [obj]
        except json.JSONDecodeError:
            pass
    return [json.loads(line) for line in txt.splitlines() if line.strip()]


def _vid_key(v) -> str:
    """Normalise video identifiers: 7020 / 'video7020' / 'video7020.mp4' -> 'video7020'."""
    s = str(v).strip()
    s = Path(s).stem if s.endswith(".mp4") else s
    if s.isdigit():
        s = f"video{s}"
    return s


def load_category_names(cfg: Config) -> list[str]:
    names = list(MSRVTT_CATEGORIES)
    try:
        p = fetch_repo_file(cfg, "raw_data/category.txt")
        for line in p.read_text(encoding="utf-8").splitlines():
            parts = re.split(r"[\t,]+|\s{2,}", line.strip())
            nums = [x for x in parts if x.strip().isdigit()]
            words = [x for x in parts if x.strip() and not x.strip().isdigit()]
            if nums and words and int(nums[0]) < 20:
                names[int(nums[0])] = words[0].strip()
    except Exception:
        pass  # fall back to canonical names
    return names


def load_annotations(cfg: Config, progress: ProgressFn = None) -> dict:
    """Merge all available annotation files into one table.

    Returns {video_id: {"category": int, "captions": [str,...], "split": str,
                        "url": str, "start": float, "end": float}}
    """
    table: dict[str, dict] = {}

    def upsert(vid, split=None, category=None, captions=None, url=None, start=None, end=None):
        rec = table.setdefault(vid, {"category": None, "captions": [], "split": None,
                                     "url": "", "start": None, "end": None})
        if category is not None and rec["category"] is None:
            rec["category"] = int(category)
        if split and rec["split"] in (None, "raw"):
            rec["split"] = split
        if url and not rec["url"]:
            rec["url"] = url
        if start is not None and rec["start"] is None:
            rec["start"], rec["end"] = start, end
        for c in captions or []:
            c = str(c).strip()
            if c and c not in rec["captions"]:
                rec["captions"].append(c)

    files = [("msrvtt_test_1k.json", "test1k"), ("msrvtt_train_9k.json", "train9k")]
    for i, (fname, split) in enumerate(files):
        _p(progress, 0.05 + 0.1 * i, f"Downloading annotation {fname}")
        path = fetch_repo_file(cfg, fname)
        for r in _load_json_records(path):
            vid = _vid_key(r.get("video_id") or r.get("video") or r.get("id"))
            caps = r.get("caption", r.get("captions", []))
            caps = [caps] if isinstance(caps, str) else list(caps or [])
            upsert(vid, split, r.get("category"), caps, r.get("url"),
                   r.get("start time"), r.get("end time"))

    # Original annotation: all 20 captions for every video (incl. test 1k-A).
    try:
        _p(progress, 0.30, "Downloading raw_data/MSRVTT_data.json (all 20 captions/video)")
        raw = json.loads(fetch_repo_file(cfg, "raw_data/MSRVTT_data.json").read_text(encoding="utf-8"))
        for v in raw.get("videos", []):
            upsert(_vid_key(v.get("video_id")), None, v.get("category"), None,
                   v.get("url"), v.get("start time"), v.get("end time"))
        for s in raw.get("sentences", []):
            vid = _vid_key(s.get("video_id"))
            if vid in table:
                upsert(vid, captions=[s.get("caption", "")])
    except Exception as e:  # not fatal - the sampler falls back to train9k videos
        _p(progress, 0.30, f"MSRVTT_data.json unavailable ({e}); continuing with split files only")
    return table


# ---------------------------------------------------------------------------
# Video archive handling
# ---------------------------------------------------------------------------
class VideoArchive:
    """Extracts individual clips from MSRVTT_Videos.zip (or a local folder)."""

    def __init__(self, cfg: Config, progress: ProgressFn = None):
        self.cfg = cfg
        self.out_dir = cfg.videos_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.members: dict[str, str] = {}
        self.zf: zipfile.ZipFile | None = None
        self.local_folder: Path | None = None
        manual = Path(cfg.data_dir) / "videos_manual"   # user may drop .mp4s here
        if manual.exists() and any(manual.glob("*.mp4")):
            self.local_folder = manual
            return
        _p(progress, 0.35, "Fetching MSRVTT_Videos.zip (~2.2 GB, cached after first run)")
        zpath = fetch_repo_file(cfg, "MSRVTT_Videos.zip")
        self.zf = zipfile.ZipFile(zpath)
        for name in self.zf.namelist():
            if name.lower().endswith(".mp4") and not Path(name).name.startswith("._"):
                self.members[Path(name).stem] = name

    def available(self, vid: str) -> bool:
        if (self.out_dir / f"{vid}.mp4").exists():
            return True
        if self.local_folder is not None:
            return (self.local_folder / f"{vid}.mp4").exists()
        return vid in self.members

    def extract(self, vid: str) -> Path | None:
        dst = self.out_dir / f"{vid}.mp4"
        if dst.exists() and dst.stat().st_size > 0:
            return dst
        if self.local_folder is not None:
            src = self.local_folder / f"{vid}.mp4"
            if not src.exists():
                return None
            shutil.copy(src, dst)
            return dst
        name = self.members.get(vid)
        if name is None or self.zf is None:
            return None
        with self.zf.open(name) as fsrc, open(dst, "wb") as fdst:
            shutil.copyfileobj(fsrc, fdst)
        return dst


# ---------------------------------------------------------------------------
# Frame decoding & pre-processing
# ---------------------------------------------------------------------------
def segment_indices(n_frames: int, num: int) -> np.ndarray:
    """TSN-style uniform sampling: mid-point of ``num`` equal temporal segments."""
    if n_frames <= 0:
        return np.zeros(num, dtype=int)
    idx = ((np.arange(num) + 0.5) * n_frames / num).astype(int)
    return np.clip(idx, 0, n_frames - 1)


def read_video_frames(path: str | Path, num: int) -> tuple[list[np.ndarray], dict]:
    """Decode ``num`` uniformly-spaced RGB frames with OpenCV.

    Frames are read sequentially (grab/retrieve) which is frame-accurate and
    robust to the unreliable seeking of some MP4 encodes. Returns (frames, meta).
    Raises ValueError for unreadable/corrupt videos.
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frames: list[np.ndarray] = []
    if n > 0:
        wanted = segment_indices(n, num)
        wanted_set = defaultdict(int)
        for w in wanted:
            wanted_set[int(w)] += 1
        i = 0
        last_ok = None
        while i <= wanted.max():
            ok = cap.grab()
            if not ok:
                break
            if i in wanted_set:
                ok, f = cap.retrieve()
                if ok:
                    last_ok = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    frames.extend([last_ok] * wanted_set[i])
            i += 1
    cap.release()
    if len(frames) < num:
        # frame count metadata was wrong -> decode everything, then sample
        cap = cv2.VideoCapture(str(path))
        allf = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            allf.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        if not allf:
            raise ValueError(f"no decodable frames in {path}")
        n = len(allf)
        frames = [allf[i] for i in segment_indices(n, num)]
    duration = n / fps if fps > 0 else 0.0
    return frames, {"fps": fps, "n_frames": n, "width": width, "height": height,
                    "duration": round(duration, 2)}


def is_browser_playable(path: Path) -> bool:
    import cv2
    cap = cv2.VideoCapture(str(path))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()
    code = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).lower()
    return code in ("avc1", "h264", "x264", "vp80", "vp90", "av01")


def make_browser_playable(path: Path) -> Path:
    """Best-effort transcode to H.264 if the clip codec is not playable in a browser."""
    try:
        if is_browser_playable(path):
            return path
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            try:
                import imageio_ffmpeg  # type: ignore
                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                return path
        import subprocess
        tmp = path.with_suffix(".h264.mp4")
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(path), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(tmp)],
                       check=True, timeout=120)
        tmp.replace(path)
    except Exception:
        pass
    return path


def save_thumb_strip(frames: list[np.ndarray], out: Path, height: int = 96) -> None:
    """Save a horizontal strip of sampled frames (used by the dashboard)."""
    from PIL import Image
    ims = []
    for f in frames:
        im = Image.fromarray(f)
        w = int(im.width * height / max(im.height, 1))
        ims.append(im.resize((max(w, 1), height)))
    strip = Image.new("RGB", (sum(i.width for i in ims), height))
    x = 0
    for im in ims:
        strip.paste(im, (x, 0))
        x += im.width
    out.parent.mkdir(parents=True, exist_ok=True)
    strip.save(out, quality=85)


# ---------------------------------------------------------------------------
# Subset construction (100 index clips / 200 queries / disjoint train pool)
# ---------------------------------------------------------------------------
def prepare_subset(cfg: Config, progress: ProgressFn = None, force: bool = False) -> dict:
    if cfg.subset_path.exists() and not force:
        subset = json.loads(cfg.subset_path.read_text())
        if all((cfg.videos_dir / f"{v['video_id']}.mp4").exists() for v in subset["index"]):
            _p(progress, 1.0, "Subset already prepared (cached)")
            return subset

    rng = random.Random(cfg.seed)
    cat_names = load_category_names(cfg)
    table = load_annotations(cfg, progress)
    archive = VideoArchive(cfg, progress)

    def usable(vid, rec, min_caps):
        return rec["category"] is not None and len(rec["captions"]) >= min_caps and archive.available(vid)

    # ---- index: prefer the official 1k-A test videos, then held-out train9k ----
    by_cat_test, by_cat_other = defaultdict(list), defaultdict(list)
    for vid, rec in sorted(table.items()):
        if not usable(vid, rec, cfg.captions_per_video):
            continue
        (by_cat_test if rec["split"] == "test1k" else by_cat_other)[rec["category"]].append(vid)

    index, used = [], set()
    n_cat = 20
    for ci, c in enumerate(range(n_cat)):
        _p(progress, 0.40 + 0.25 * ci / n_cat, f"Sampling + validating index clips for '{cat_names[c]}'")
        pool_a = by_cat_test[c][:]
        pool_b = by_cat_other[c][:]
        rng.shuffle(pool_a)
        rng.shuffle(pool_b)
        chosen = []
        for vid in pool_a + pool_b:
            if len(chosen) == cfg.videos_per_category:
                break
            path = archive.extract(vid)
            if path is None:
                continue
            try:
                frames, meta = read_video_frames(path, cfg.num_frames)   # validates decodability
            except Exception:
                continue
            make_browser_playable(path)
            save_thumb_strip(frames[:: max(1, len(frames) // 8)], cfg.thumbs_dir / f"{vid}.jpg")
            rec = table[vid]
            caps = rec["captions"]
            # caption 1 = the official 1k-A caption (first stored); caption 2 = random other one
            q = [caps[0]] + rng.sample(caps[1:], k=min(cfg.captions_per_video - 1, len(caps) - 1))
            chosen.append({"video_id": vid, "category": c, "category_name": cat_names[c],
                           "queries": q, "all_captions": caps, "url": rec["url"],
                           "source_split": rec["split"], **meta})
            used.add(vid)
        if len(chosen) < cfg.videos_per_category:
            raise RuntimeError(f"Only {len(chosen)} usable clips found for category {c} ({cat_names[c]})")
        index.extend(chosen)

    queries = []
    for vi, v in enumerate(index):
        for qi, q in enumerate(v["queries"]):
            queries.append({"qid": len(queries), "text": q, "video_idx": vi, "video_id": v["video_id"],
                            "category": v["category"]})

    # ---- disjoint training pool for learned temporal heads (train9k only) ----
    train = []
    n_train_per_cat = cfg.eff_train_per_cat()
    if n_train_per_cat > 0:
        tr_by_cat = defaultdict(list)
        for vid, rec in sorted(table.items()):
            if vid in used or rec["split"] != "train9k":
                continue
            if usable(vid, rec, 5):
                tr_by_cat[rec["category"]].append(vid)
        for ci, c in enumerate(range(n_cat)):
            _p(progress, 0.65 + 0.33 * ci / n_cat, f"Extracting training clips for '{cat_names[c]}'")
            pool = tr_by_cat[c][:]
            rng.shuffle(pool)
            got = 0
            for vid in pool:
                if got == n_train_per_cat:
                    break
                if archive.extract(vid) is None:
                    continue
                train.append({"video_id": vid, "category": c, "captions": table[vid]["captions"][:20]})
                got += 1

    assert not (set(v["video_id"] for v in index) & set(t["video_id"] for t in train)), "index/train overlap!"

    # caption lexicon for spelling correction - built ONLY from non-index videos (no leakage)
    from collections import Counter
    lex = Counter()
    for vid, rec in table.items():
        if vid in used:
            continue
        for c in rec["captions"]:
            lex.update(re.findall(r"[a-z]+", c.lower()))
    (Path(cfg.data_dir) / "lexicon.json").write_text(json.dumps(dict(lex.most_common())))
    subset = {"category_names": cat_names, "index": index, "queries": queries, "train": train,
              "protocol": {"videos_per_category": cfg.videos_per_category,
                           "captions_per_video": cfg.captions_per_video,
                           "n_index": len(index), "n_queries": len(queries), "n_train": len(train),
                           "seed": cfg.seed}}
    cfg.subset_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.subset_path.write_text(json.dumps(subset, indent=1))
    _p(progress, 1.0, f"Prepared {len(index)} index clips, {len(queries)} queries, {len(train)} training clips")
    return subset


def load_subset(cfg: Config) -> dict | None:
    if cfg.subset_path.exists():
        return json.loads(cfg.subset_path.read_text())
    return None
