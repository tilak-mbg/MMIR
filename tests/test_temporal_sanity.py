"""Sanity check: on a synthetic task where ONLY frame order distinguishes videos,
order-aware heads must beat order-free ones.  Run: python tests/test_temporal_sanity.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mvr.control import run_positive_control  # noqa: E402

if __name__ == "__main__":
    r = run_positive_control()["mrr"]
    for k, v in r.items():
        print(f"{k:13s} MRR {v:.3f}")
    assert r["temporal_tfm"] > r["set_tfm"] + 0.05, "temporal transformer should exploit order"
    assert r["seq_lstm"] > r["mlp_adapter"] + 0.05, "LSTM should exploit order"
    print("OK - order-aware heads exploit temporal order; order-free heads cannot.")
