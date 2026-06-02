"""Regenerate the data figures from the SJD-PAC paper as standalone PNGs.

All numbers are taken directly from the paper (Fig. 1, Fig. 4, Tab. 2).
Run: python figs/make_figs.py
"""
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 1.0,
    "figure.dpi": 150,
})

HERE = os.path.dirname(os.path.abspath(__file__))

TEAL = "#127a8a"
BLUE = "#2563c9"
GREEN = "#3f8f4f"
ORANGE = "#d9822b"
GREY = "#9aa3ad"


def teaser():
    """Figure 1: acceptance-length distribution + contribution to speedup."""
    lengths = np.arange(1, 16)
    prop = np.array([0.49, 0.22, 0.11, 0.075, 0.045, 0.028, 0.018, 0.012,
                     0.009, 0.006, 0.004, 0.003, 0.0022, 0.0015, 0.001])
    prop = prop / prop.sum()
    contrib = np.array([0.0, 0.175, 0.197, 0.17, 0.135, 0.075, 0.05, 0.035,
                        0.025, 0.018, 0.012, 0.008, 0.006, 0.004, 0.003])

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))

    # (a) frequency distribution — teal->green gradient bars
    cmap = plt.cm.get_cmap("viridis")
    colors = [cmap(0.15 + 0.7 * i / len(lengths)) for i in range(len(lengths))]
    axes[0].bar(lengths, prop, color=colors, width=0.78, zorder=3)
    axes[0].annotate("~50% of steps\naccept just 1 token",
                     xy=(1, prop[0]), xytext=(3.4, prop[0] * 0.86),
                     fontsize=11, color="#0d4f5a", fontweight="bold",
                     ha="left", va="center")
    axes[0].set_xlabel("Acceptance length (tokens per step)")
    axes[0].set_ylabel("Proportion")
    axes[0].set_title("(a) Where SJD wastes its budget", fontsize=12.5, fontweight="bold")
    axes[0].set_xticks(range(1, 16, 2))
    axes[0].grid(axis="y", color="#e6e6e6", zorder=0)

    # (b) contribution to speedup
    axes[1].bar(lengths, contrib, color=TEAL, width=0.78, zorder=3)
    axes[1].set_xlabel("Acceptance length (tokens per step)")
    axes[1].set_ylabel("Contribution to speedup")
    axes[1].set_title("(b) Single-token steps add nothing", fontsize=12.5, fontweight="bold")
    axes[1].set_xticks(range(1, 16, 2))
    axes[1].grid(axis="y", color="#e6e6e6", zorder=0)

    fig.tight_layout()
    out = os.path.join(HERE, "acceptance_distribution.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


def tv_distance():
    """Figure 4: TV distance vs. perturbation offset for text vs. image."""
    j = np.array([1, 3, 5, 7, 9, 11, 13, 15])
    text = np.array([0.32, 0.15, 0.11, 0.095, 0.088, 0.083, 0.080, 0.078])
    image = np.array([0.32, 0.06, 0.042, 0.034, 0.030, 0.027, 0.025, 0.024])

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(j, text, "-o", color=BLUE, lw=2.2, ms=6, label="Text — stays sensitive", zorder=3)
    ax.plot(j, image, "-o", color=GREEN, lw=2.2, ms=6, label="Image — forgets fast", zorder=3)
    ax.fill_between(j, image, text, color="#dfeaf6", alpha=0.6, zorder=1)
    ax.set_yscale("log")
    ax.set_yticks([0.02, 0.04, 0.08, 0.16, 0.32])
    ax.set_yticklabels(["0.02", "0.04", "0.08", "0.16", "0.32"])
    ax.set_xlabel("Perturbation offset $j$")
    ax.set_ylabel(r"$d_{\mathrm{TV}}$ (log scale)")
    ax.set_title("Why stale drafts still work for images", fontsize=12.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=11)
    ax.grid(color="#ececec", zorder=0)
    fig.tight_layout()
    out = os.path.join(HERE, "tv_distance.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


def ablation():
    """Table 2: each component compounds (step compression, MS-COCO)."""
    labels = ["SJD baseline\n(L=32)", "+ PD\n(L=32)", "+ PD + AC\n(L=32)", "+ PD + AC\n(L=64)"]
    vals = [2.31, 2.71, 3.52, 4.51]
    colors = [GREY, TEAL, BLUE, GREEN]

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bars = ax.barh(range(len(vals)), vals, color=colors, height=0.62, zorder=3)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Step compression ratio  (MS-COCO, Lumina-mGPT)")
    ax.set_xlim(0, 5.0)
    ax.set_title("Each piece compounds", fontsize=12.5, fontweight="bold")
    for b, v in zip(bars, vals):
        ax.text(v + 0.08, b.get_y() + b.get_height() / 2,
                f"{v:.2f}×", va="center", ha="left", fontweight="bold", fontsize=12)
    ax.grid(axis="x", color="#ececec", zorder=0)
    fig.tight_layout()
    out = os.path.join(HERE, "ablation.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


def speedup():
    """Teaser: main-results speedup on PartiPrompts / Lumina-mGPT (Tab. 1)."""
    methods = ["Baseline", "EAGLE", "SJD", "GSD\n(lossy)", "SJD-PAC\n(ours)"]
    step = [1.00, 2.86, 2.28, 3.76, 4.62]
    latency = [1.00, 2.01, 2.13, 4.65, 3.97]
    x = np.arange(len(methods))
    w = 0.38

    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    step_colors = [GREY, GREY, GREY, ORANGE, GREEN]
    lat_colors = ["#c7ccd1", "#c7ccd1", "#c7ccd1", "#eab06a", "#7bbf8a"]
    b1 = ax.bar(x - w / 2, step, w, color=step_colors, zorder=3, label="Step compression")
    b2 = ax.bar(x + w / 2, latency, w, color=lat_colors, zorder=3, label="Wall-clock speedup")

    for bars, vals in ((b1, step), (b2, latency)):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.06, f"{v:.2f}×",
                    ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel("Acceleration over autoregressive baseline")
    ax.set_ylim(0, 5.4)
    ax.set_title("Lossless acceleration on PartiPrompts · Lumina-mGPT",
                 fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=10.5, loc="upper left")
    ax.grid(axis="y", color="#ececec", zorder=0)
    ax.annotate("matches lossy GSD —\nwithout the artifacts",
                xy=(4 + w / 2, 3.97), xytext=(3.05, 5.0),
                fontsize=10, color=GREEN, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5))
    fig.tight_layout()
    out = os.path.join(HERE, "speedup.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    speedup()
    teaser()
    tv_distance()
    ablation()
    print("done")
