"""Figure for DermaMNIST Study D1: per-run test ACC and AUC at 50 features, with the all-features reference.

Output: Derma/feature_selection/analysis_d1/test_scores.png

Usage (from project root):
    .venv\\Scripts\\python.exe Derma\\feature_selection\\plot_d1.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CONTROL, BIO = "#8a8984", "#2a78d6"      # recessive gray for controls, categorical slot 1 for the searches
ORDER = [("FILTER", "ANOVA filter", CONTROL), ("RANDOM", "Random search", CONTROL),
         ("GA", "GA", BIO), ("CSO", "CSO", BIO)]


def main():
    with open(os.path.join(HERE, "results_d1", "runs.jsonl")) as f:
        runs = [json.loads(line) for line in f]
    ref = next(r for r in runs if r["method"] == "ALL1820")
    rng = np.random.default_rng(0)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), facecolor=SURFACE)
    for ax, (met, label) in zip(axes, (("test_acc", "Test accuracy"), ("test_auc", "Test AUC (macro one-vs-rest)"))):
        ax.set_facecolor(SURFACE)
        for i, (m, name, col) in enumerate(ORDER):
            v = np.array([r[met] for r in runs if r["method"] == m])
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=22, color=col, alpha=0.75,
                       edgecolor=SURFACE, linewidth=0.8, zorder=3)
            ax.plot([i - 0.28, i + 0.28], [v.mean()] * 2, color=INK, linewidth=2, solid_capstyle="round", zorder=4)
        ax.axhline(ref[met], color=INK2, linestyle=(0, (4, 3)), linewidth=1.2, zorder=2)
        ax.text(3.45, ref[met], f"All 1,820 features  {ref[met]:.3f}", color=INK2, ha="right", va="bottom", fontsize=8)
        ax.set_xticks(range(len(ORDER)), [n for _, n, _ in ORDER])
        ax.set_xlim(-0.55, 3.55)
        ax.set_title(label, color=INK, fontsize=10, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.tick_params(length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    fig.text(0.01, 0.01, "Dots: 30 runs (paired seeds), 50 features each. Black bar: mean. Gray: controls; blue: "
             "bio-inspired search.", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = os.path.join(HERE, "analysis_d1", "test_scores.png")
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print("saved", out)


if __name__ == "__main__":
    main()
