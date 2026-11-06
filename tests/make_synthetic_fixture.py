"""Build an OFFLINE test fixture that mimics the Hugging Face repo layout of
friedrichor/MSR-VTT plus a tiny randomly-initialised CLIP / X-CLIP checkpoint.

Used only for smoke-testing the full pipeline without internet. Real runs use
the actual dataset and pretrained weights from Hugging Face.

usage: python tests/make_synthetic_fixture.py <out_dir> <path/to/bpe_simple_vocab_16e6.txt.gz>
"""
import gzip
import json
import random
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

OBJ = ["man", "woman", "dog", "cat", "car", "guitar", "ball", "chef", "child", "robot"]
ACT = ["playing", "running", "cooking", "talking", "dancing", "driving", "singing", "jumping"]
PLC = ["on a stage", "in a kitchen", "on a road", "in a park", "in a studio", "at the beach"]


def caption(rng):
    s = f"a {rng.choice(OBJ)} is {rng.choice(ACT)} {rng.choice(PLC)}"
    if rng.random() < 0.2:
        s += f" and then {rng.choice(ACT)}"
    return s


def make_video(path, rng, n=40):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (96, 72))
    base = np.array([rng.randint(0, 255) for _ in range(3)], np.uint8)
    for t in range(n):
        f = np.zeros((72, 96, 3), np.uint8) + base
        x = int(10 + 70 * t / n)
        cv2.circle(f, (x, 36), 10, (255 - int(base[0]), 128, 255), -1)
        w.write(f)
    w.release()


def build_tokenizer_files(bpe_gz, out):
    def bytes_to_unicode():
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return dict(zip(bs, [chr(c) for c in cs]))
    merges = gzip.open(bpe_gz).read().decode("utf-8").split("\n")[1:49152 - 256 - 2 + 1]
    merges = [tuple(m.split()) for m in merges]
    vocab = list(bytes_to_unicode().values())
    vocab = vocab + [v + "</w>" for v in vocab]
    vocab += ["".join(m) for m in merges]
    vocab += ["<|startoftext|>", "<|endoftext|>"]
    (out / "vocab.json").write_text(json.dumps({v: i for i, v in enumerate(vocab)}))
    (out / "merges.txt").write_text("#version: 0.2\n" + "\n".join(" ".join(m) for m in merges) + "\n")


def build_models(out, bpe_gz):
    from transformers import (CLIPConfig, CLIPModel, CLIPTokenizer, XCLIPConfig, XCLIPModel)
    tiny_text = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4)
    tiny_vis = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                    image_size=64, patch_size=16)
    for name, Cfg, Model, extra in (("tiny-clip", CLIPConfig, CLIPModel, {}),
                                    ("tiny-xclip", XCLIPConfig, XCLIPModel, {"num_frames": 8})):
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        cfg = Cfg(text_config=tiny_text, vision_config={**tiny_vis, **extra}, projection_dim=32)
        if name == "tiny-xclip":
            vis = {**tiny_vis, **extra, "mit_hidden_size": 32, "mit_intermediate_size": 64,
                   "mit_num_hidden_layers": 1, "mit_num_attention_heads": 4}
            cfg = Cfg(text_config=tiny_text, vision_config=vis, projection_dim=32,
                      prompt_layers=1, prompt_num_attention_heads=4)
        Model(cfg).save_pretrained(d)
        build_tokenizer_files(bpe_gz, d)
        CLIPTokenizer(str(d / "vocab.json"), str(d / "merges.txt")).save_pretrained(d)


def main(out, bpe_gz):
    out = Path(out)
    rng = random.Random(0)
    repo = out / "repo"
    (repo / "raw_data").mkdir(parents=True, exist_ok=True)
    vids_dir = out / "vids"
    vids_dir.mkdir(exist_ok=True)
    test, train, videos, sentences = [], [], [], []
    vid = 0
    for split, per_cat in (("test", 7), ("train", 12)):
        for c in range(20):
            for _ in range(per_cat):
                caps = [caption(rng) for _ in range(20)]
                rec = {"video_id": f"video{vid}", "video": f"video{vid}.mp4", "source": "MSR-VTT",
                       "category": c, "url": "https://www.youtube.com/watch?v=x", "start time": 1.0,
                       "end time": 5.0, "id": vid}
                if split == "test":
                    test.append({**rec, "caption": caps[0]})
                else:
                    train.append({**rec, "caption": caps})
                videos.append({"category": c, "video_id": f"video{vid}", "split": split, "id": vid})
                sentences += [{"caption": s, "video_id": f"video{vid}", "sen_id": len(sentences)} for s in caps]
                make_video(vids_dir / f"video{vid}.mp4", rng)
                vid += 1
    # one corrupt file to exercise the robustness path
    (vids_dir / f"video{vid-1}.mp4").write_bytes(b"not a video")
    (repo / "msrvtt_test_1k.json").write_text(json.dumps(test))
    (repo / "msrvtt_train_9k.json").write_text(json.dumps(train))
    (repo / "raw_data" / "MSRVTT_data.json").write_text(json.dumps({"videos": videos, "sentences": sentences}))
    (repo / "raw_data" / "category.txt").write_text("\n".join(f"cat{i}\t{i}" for i in range(20)))
    with zipfile.ZipFile(repo / "MSRVTT_Videos.zip", "w") as z:
        for p in sorted(vids_dir.glob("*.mp4")):
            z.write(p, f"MSRVTT_Videos/{p.name}")
    build_models(out, bpe_gz)
    print("fixture ready at", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
