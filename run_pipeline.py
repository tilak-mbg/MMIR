"""Optional command-line runner (the Streamlit dashboard can do all of this).

    python run_pipeline.py                 # full pipeline with saved/default settings
    python run_pipeline.py --force         # recompute everything
    python run_pipeline.py --report        # also write the .docx report
"""
import argparse
from pathlib import Path

from mvr.config import Config
from mvr.pipeline import run_all

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--clip-model", default=None)
    ap.add_argument("--frames", type=int, default=None)
    a = ap.parse_args()
    cfg = Config.load()
    if a.clip_model:
        cfg.clip_model = a.clip_model
    if a.frames:
        cfg.num_frames = a.frames
    print("Config:", cfg.to_dict())
    run_all(cfg, force=a.force)
    if a.report:
        from mvr.report import build_report
        out = build_report(cfg, Path(cfg.artifacts_dir) / "MIR_Assignment2_Technical_Report.docx")
        print("Report written to", out)
