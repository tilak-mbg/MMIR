"""Static figures (PNG) for the technical report. The dashboard draws its own
interactive versions of the same charts with Altair."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import LEARNED_METHODS, PARAM_FREE_METHODS, method_label  # noqa: E402

# reference categorical palette (fixed order), text inks, surface
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FAMILY_COLOR = {"param_free": BLUE, "learned": ORANGE, "external": AQUA}


def family(m: str) -> str:
    return "param_free" if m in PARAM_FREE_METHODS else ("learned" if m in LEARNED_METHODS else "external")


def _style(ax, title, ylabel=None):
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _legend_family(ax, present):
    from matplotlib.patches import Patch
    names = {"param_free": "Parameter-free (zero-shot)", "learned": "Learned head (3 seeds, mean ± sd)",
             "external": "External video model"}
    ax.legend(handles=[Patch(color=FAMILY_COLOR[f], label=names[f]) for f in present],
              frameon=False, fontsize=8, loc="upper left")


def fig_main_bars(R, out: Path):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    t2v, sd = R["t2v"], R["t2v_std"]
    ms = list(t2v)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, key, unit in ((axes[0], "R@1", "%"), (axes[1], "MRR", "")):
        vals = [t2v[m][key] for m in ms]
        err = [sd.get(m, {}).get(key, 0) for m in ms]
        cols = [FAMILY_COLOR[family(m)] for m in ms]
        bars = ax.bar(range(len(ms)), vals, yerr=err, color=cols, width=0.62, capsize=3,
                      error_kw={"elinewidth": 1, "ecolor": INK2})
        ax.axhline(t2v["mean_pool"][key], color=INK2, linestyle="--", linewidth=1)
        top = max(v + e for v, e in zip(vals, err))
        ax.set_ylim(0, top * 1.15 if top > 0 else 1)
        for b, v, e in zip(bars, vals, err):
            ax.text(b.get_x() + b.get_width() / 2, v + e + top * 0.01, f"{v:.1f}" if unit else f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_xticks(range(len(ms)))
        ax.set_xticklabels([method_label(m).replace(" (", "\n(") for m in ms], rotation=35, ha="right", fontsize=7)
        _style(ax, f"Text-to-video {key}", key + (" (%)" if unit else ""))
    names = {"param_free": "Parameter-free (zero-shot)", "learned": "Learned head (mean ± sd over seeds)",
             "external": "External video model"}
    present = [f for f in ("param_free", "learned", "external") if any(family(m) == f for m in ms)]
    handles = [Patch(color=FAMILY_COLOR[f], label=names[f]) for f in present]
    handles.append(Line2D([0], [0], color=INK2, linestyle="--", label="Mean-pool baseline"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_frame_sweep(R, out: Path):
    sw = R["frame_sweep"]
    show = [m for m in ("mean_pool", "max_pool", "query_attn", "temporal_tfm") if m in sw]
    cols = [BLUE, ORANGE, AQUA, YELLOW]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for m, c in zip(show, cols):
        ks = sorted(sw[m], key=int)
        ax.plot([int(k) for k in ks], [sw[m][k]["MRR"] for k in ks], marker="o", markersize=5, linewidth=2,
                color=c, label=method_label(m))
    ax.set_xscale("log", base=2)
    ax.set_xticks([int(k) for k in sorted(sw[show[0]], key=int)])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Frames sampled per clip", color=INK2)
    _style(ax, "MRR vs number of sampled frames", "MRR")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_order(R, out: Path):
    oa = R["order_ablation"]
    ms = list(oa)
    conds, cols = ["original", "shuffled", "reversed"], [BLUE, ORANGE, AQUA]
    x = np.arange(len(ms))
    w = 0.26
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (c, col) in enumerate(zip(conds, cols)):
        ax.bar(x + (i - 1) * w, [oa[m][c]["MRR"] for m in ms], width=w - 0.02, color=col, label=f"{c} frame order")
    ax.set_xticks(x)
    ax.set_xticklabels([method_label(m).replace(" (", "\n(") for m in ms], rotation=30, ha="right", fontsize=7)
    _style(ax, "Temporal-order ablation: MRR with original / shuffled / reversed frames", "MRR")
    ax.title.set_position((0, 1.14))
    lo = min(oa[m][c]["MRR"] for m in ms for c in conds)
    hi = max(oa[m][c]["MRR"] for m in ms for c in conds)
    ax.set_ylim(max(0, lo - 0.08), hi * 1.12)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_per_category(R, out: Path):
    pc = R["per_category"]
    ms = [m for m in pc]
    cats = list(next(iter(pc.values())).keys())
    M = np.array([[pc[m][c]["R@5"] for m in ms] for c in cats])
    fig, ax = plt.subplots(figsize=(1.0 * len(ms) + 3, 7))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(ms)))
    ax.set_xticklabels([method_label(m) for m in ms], rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats, fontsize=8)
    for i in range(len(cats)):
        for j in range(len(ms)):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                    color="white" if M[i, j] > 60 else INK)
    ax.set_title("Per-category R@5 (%) - 10 queries per category", loc="left", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_training(R, out: Path):
    tr = R.get("training", {})
    show = [(a, c) for a, c in zip(LEARNED_METHODS, [BLUE, ORANGE, AQUA, YELLOW]) if f"{a}_seed0" in tr]
    if not show:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for a, c in show:
        h = tr[f"{a}_seed0"]
        ax.plot(h["epoch"], h["val_mrr"], color=c, linewidth=2, label=method_label(a))
    ax.set_xlabel("Epoch (0 = initialisation = mean pooling)", color=INK2)
    _style(ax, "Validation MRR during training (held-out TRAIN clips, seed 0)", "Val MRR")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_architecture(out: Path):
    """Block diagram of the retrieval system."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                    fc=fc, ec=INK2, lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, color=INK, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, color=INK2, lw=1.2))

    off = "#dbe8f8"; tx = "#fde3d8"; idx = "#d3f1e6"; ui = "#fbefcf"
    ax.text(0.1, 5.75, "OFFLINE (index build)", fontsize=9, fontweight="bold", color=INK2)
    box(0.1, 4.3, 1.8, 1.1, "MSR-VTT\n100 clips\n(5 x 20 categories)", off)
    box(2.3, 4.3, 1.8, 1.1, "Frame sampling\nT=16 segment\nmid-points (OpenCV)", off)
    box(4.5, 4.3, 1.8, 1.1, "CLIP ViT image\nencoder (frozen)\n+ projection", off)
    box(6.7, 4.3, 2.2, 1.1, "Temporal aggregation\nmean / max / single /\nTransformer / LSTM", off)
    box(9.3, 4.3, 2.5, 1.1, "Video index\nL2-norm vectors\nFAISS IndexFlatIP", idx)
    arrow(1.9, 4.85, 2.3, 4.85); arrow(4.1, 4.85, 4.5, 4.85); arrow(6.3, 4.85, 6.7, 4.85); arrow(8.9, 4.85, 9.3, 4.85)

    ax.text(0.1, 3.55, "TRAINING (disjoint MSR-VTT train clips)", fontsize=9, fontweight="bold", color=INK2)
    box(4.5, 2.6, 1.8, 0.8, "Train clips x 20\ncaptions (frozen CLIP)", tx)
    box(6.7, 2.6, 2.2, 0.8, "InfoNCE on temporal\nheads (3 seeds)", tx)
    arrow(6.3, 3.0, 6.7, 3.0); arrow(7.8, 3.4, 7.8, 4.3)

    ax.text(0.1, 1.85, "ONLINE (query time)", fontsize=9, fontweight="bold", color=INK2)
    box(0.1, 0.4, 1.8, 1.1, "Streamlit UI\nnatural-language\nquery", ui)
    box(2.3, 0.4, 1.8, 1.1, "Query processing\nnormalise, spell-fix,\ntemplates, chunking", ui)
    box(4.5, 0.4, 1.8, 1.1, "CLIP text encoder\n(frozen) +\nprojection", ui)
    box(6.7, 0.4, 2.2, 1.1, "Shared 512-d space\ncosine = dot product\n(query-attn optional)", idx)
    box(9.3, 0.4, 2.5, 1.1, "Top-K clips + scores\nplayable videos,\nper-frame similarity", ui)
    arrow(1.9, 0.95, 2.3, 0.95); arrow(4.1, 0.95, 4.5, 0.95); arrow(6.3, 0.95, 6.7, 0.95); arrow(8.9, 0.95, 9.3, 0.95)
    arrow(10.55, 4.3, 10.55, 1.5)
    ax.text(10.65, 2.9, "search", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def make_all_figures(R: dict, d: Path) -> dict:
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, fn in (("main_bars", fig_main_bars), ("frame_sweep", fig_frame_sweep), ("order", fig_order),
                     ("per_category", fig_per_category), ("training", fig_training)):
        p = d / f"{name}.png"
        try:
            fn(R, p)
            if p.exists():
                out[name] = p
        except Exception as e:
            print(f"figure {name} failed: {e}")
    p = d / "architecture.png"
    fig_architecture(p)
    out["architecture"] = p
    return out
