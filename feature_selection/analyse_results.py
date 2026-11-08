"""Analyse the feature-selection benchmark (step 9): results tables, Wilcoxon tests, plots.

Inputs:
    feature_selection/results/runs.jsonl   one record per (method, feature set, run) from run_benchmark.py
    svm_baseline/results.csv               all-512 SVM and original ResNet-18 head, per feature set

Outputs in feature_selection/analysis/:
    summary_table.csv / .md     mean ± std per method on test (+ val score, features kept, runtime)
    summary_by_set.csv          the same, split by feature set (released weights 1 / 2 / 3)
    wilcoxon.csv / .md          paired Wilcoxon signed-rank tests on test metrics:
                                every method vs all-512 SVM, and every pair of methods.
                                Holm correction within each metric (each method vs all-512 + every method pair).
    convergence.png             best-so-far fitness vs evaluations, mean ± std, one panel per feature set
    features_vs_accuracy.png    features kept vs test ACC and vs val ACC (shows validation overfitting)
    alpha_ablation.csv / .md    α ∈ {0.90, 0.95, 0.99} per method: features kept, val/test metrics, and how
                                many runs return exactly the same mask as at α = 0.99 (paired by seed)

Main tables, tests and plots use only α = 0.99 runs; other α values appear only in the ablation table.

Pairing: runs share seeds across methods (seed = 100 * feature_set + run), so method-vs-method
tests pair on (feature_set, run). The all-512 baseline is deterministic per feature set, so each run
is paired with its own feature set's baseline value.

Usage (from project root):
    .venv\\Scripts\\python.exe feature_selection\\analyse_results.py
"""

import csv
import itertools
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "results", "runs.jsonl")
BASELINE = os.path.join(ROOT, "svm_baseline", "results.csv")
OUT = os.path.join(HERE, "analysis")

METHODS = ["RANDOM", "PSO", "GA", "GWO", "WOA", "BPSO", "VBPSO"]
METRICS = ["acc", "auc", "macro_f1", "malignant_recall", "specificity"]
LABELS = {"acc": "ACC", "auc": "AUC", "macro_f1": "Macro-F1", "malignant_recall": "Malignant recall",
          "specificity": "Specificity"}
SETS = (1, 2, 3)
MAIN_ALPHA = 0.99

# Validated light-mode categorical slots 1-6 for the algorithms; the random-search control and the
# all-512 reference are neutral so colour only carries algorithm identity. Markers + direct labels
# are the secondary encoding (three slots sit below 3:1 contrast on the light surface).
STYLE = {
    "RANDOM": {"color": "#898781", "marker": "o", "label": "Random search"},
    "PSO": {"color": "#2a78d6", "marker": "s", "label": "PSO"},
    "GA": {"color": "#eb6834", "marker": "^", "label": "GA"},
    "GWO": {"color": "#1baf7a", "marker": "D", "label": "GWO"},
    "WOA": {"color": "#eda100", "marker": "v", "label": "WOA"},
    "BPSO": {"color": "#e87ba4", "marker": "P", "label": "BPSO"},
    "VBPSO": {"color": "#008300", "marker": "X", "label": "BPSO (V-shaped)"},
}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def load():
    """Main analysis uses only the main protocol (α = 0.99, val ACC, 1000 evals); ablation runs are kept apart."""
    with open(RUNS) as f:
        all_runs = [json.loads(line) for line in f]
    runs = [r for r in all_runs if (r["alpha"], r["metric"], r["budget"]) == (MAIN_ALPHA, "acc", 1000)]
    by = {m: sorted([r for r in runs if r["method"] == m], key=lambda r: (r["feature_set"], r["run"]))
          for m in METHODS}
    base, head = {}, {}
    with open(BASELINE) as f:
        for r in csv.DictReader(f):
            s = int(r["weights"][-1])
            vals = {m: float(r[m]) for m in METRICS}
            target = base if r["model"] == "svm_all_512" else head
            target.setdefault(s, {})[r["split"]] = vals
    return runs, by, base, head, all_runs


def mean_std(xs):
    xs = np.asarray(xs, dtype=float)
    return xs.mean(), (xs.std(ddof=1) if len(xs) > 1 else 0.0)


def fmt(xs, nd=3):
    m, s = mean_std(xs)
    return f"{m:.{nd}f} ± {s:.{nd}f}"


def summary_rows(by, base, head, sets=SETS):
    rows = []
    for name, ref in (("ResNet-18 head (released weights)", head), ("All 512 features + SVM", base)):
        rows.append({"method": name, "n_runs": len(sets),
                     "val_acc": fmt([ref[s]["val"]["acc"] for s in sets]),
                     "n_features": "512", "pct_features": "100.0",
                     **{m: fmt([ref[s]["test"][m] for s in sets]) for m in METRICS}, "runtime_s": "–"})
    for meth in METHODS:
        R = [r for r in by[meth] if r["feature_set"] in sets]
        nf = [r["n_features"] for r in R]
        rows.append({"method": STYLE[meth]["label"] + " + SVM", "n_runs": len(R),
                     "val_acc": fmt([r["val_score"] for r in R]),
                     "n_features": f"{np.mean(nf):.1f} ± {np.std(nf, ddof=1):.1f}",
                     "pct_features": f"{100 * np.mean(nf) / 512:.1f}",
                     **{m: fmt([r[f"test_{m}"] for r in R]) for m in METRICS},
                     "runtime_s": f"{np.mean([r['runtime_s'] for r in R]):.1f}"})
    return rows


def ablation_rows(all_runs):
    key = lambda r: (r["feature_set"], r["run"])  # noqa: E731
    main = {(r["method"], key(r)): r["mask_hex"] for r in all_runs if r["alpha"] == MAIN_ALPHA}
    rows = []
    for meth in METHODS:
        for alpha in sorted({r["alpha"] for r in all_runs if r["method"] == meth}, reverse=True):
            R = [r for r in all_runs if r["method"] == meth and r["alpha"] == alpha
                 and (r["metric"], r["budget"]) == ("acc", 1000)]
            if len(R) == 0:
                continue
            same = sum(main.get((meth, key(r))) == r["mask_hex"] for r in R)
            rows.append({"method": STYLE[meth]["label"], "alpha": alpha, "n_runs": len(R),
                         "pct_features": f"{np.mean([r['pct_features'] for r in R]):.1f}",
                         "val_acc": fmt([r["val_score"] for r in R]),
                         **{m: fmt([r[f"test_{m}"] for r in R]) for m in ("acc", "auc", "malignant_recall")},
                         "same_mask_as_0.99": f"{same}/{len(R)}"})
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def md_table(rows, cols, headers):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def paired_test(a, b):
    d = np.asarray(a) - np.asarray(b)
    if np.allclose(d, 0):
        return 1.0
    return float(wilcoxon(a, b, zero_method="zsplit").pvalue)


def holm(pvals):
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def wilcoxon_rows(by, base):
    rows = []
    for m in METRICS:
        comps = []
        for meth in METHODS:
            a = [r[f"test_{m}"] for r in by[meth]]
            b = [base[r["feature_set"]]["test"][m] for r in by[meth]]
            comps.append((meth, "ALL512", a, b))
        for x, y in itertools.combinations(METHODS, 2):
            comps.append((x, y, [r[f"test_{m}"] for r in by[x]], [r[f"test_{m}"] for r in by[y]]))
        p = np.array([paired_test(a, b) for _, _, a, b in comps])
        p_holm = holm(p)
        for (x, y, a, b), pi, ph in zip(comps, p, p_holm):
            diff = float(np.mean(a) - np.mean(b))
            rows.append({"metric": m, "method": x, "vs": y, "mean_diff": round(diff, 5),
                         "wins": int(np.sum(np.asarray(a) > np.asarray(b))),
                         "ties": int(np.sum(np.asarray(a) == np.asarray(b))),
                         "losses": int(np.sum(np.asarray(a) < np.asarray(b))),
                         "p": round(pi, 5), "p_holm": round(float(ph), 5),
                         "significant_holm_0.05": bool(ph < 0.05)})
    return rows


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9)


def plot_convergence(by, base, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=False, facecolor=SURFACE)
    for ax, s in zip(axes, SETS):
        style_axes(ax)
        base_fit = 0.99 * (1 - base[s]["val"]["acc"]) + 0.01
        ax.axhline(base_fit, color=INK2, linestyle=(0, (4, 3)), linewidth=1.2)
        ax.annotate("All 512 features", xy=(1000, base_fit), xytext=(-4, 4), textcoords="offset points",
                    ha="right", va="bottom", fontsize=8.5, color=INK2)
        for meth in METHODS:
            H = np.array([r["history"] for r in by[meth] if r["feature_set"] == s])
            x = np.arange(1, H.shape[1] + 1)
            mu, sd = H.mean(0), H.std(0, ddof=1)
            st = STYLE[meth]
            ax.fill_between(x, mu - sd, mu + sd, color=st["color"], alpha=0.12, linewidth=0)
            ax.plot(x, mu, color=st["color"], linewidth=2, label=st["label"],
                    marker=st["marker"], markevery=[len(x) - 1], markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
            # Direct-label only the lines that separate; the overlapping cluster is named by the legend.
            if meth in ("GWO", "WOA"):
                ax.annotate(st["label"], xy=(1000, mu[-1]), xytext=(12, 0), textcoords="offset points",
                            fontsize=8.5, color=INK, va="center", annotation_clip=False)
        ax.set_xlim(0, 1000)
        ax.set_title(f"Feature set {s} (released weights {s})", fontsize=10.5, color=INK, loc="left")
        ax.set_xlabel("Fitness evaluations", fontsize=9.5, color=INK2)
    axes[0].set_ylabel("Best fitness so far (lower is better)", fontsize=9.5, color=INK2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(METHODS), frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, 1.02), labelcolor=INK)
    fig.suptitle("Convergence: mean ± 1 std over 10 runs, validation fitness (α = 0.99)",
                 x=0.01, y=1.09, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 0.97, 1))
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_features_vs_accuracy(by, base, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, facecolor=SURFACE)
    panels = (("test_acc", "acc", "test", "Test ACC (held out, used once per run)"),
              ("val_score", "acc", "val", "Validation ACC (what the search optimised)"))
    for ax, (key, bkey, split, title) in zip(axes, panels):
        style_axes(ax)
        ax.set_xscale("log")
        for meth in METHODS:
            R, st = by[meth], STYLE[meth]
            pct = np.array([r["pct_features"] for r in R])
            y = np.array([r[key] for r in R])
            # Horizontal jitter only, so accuracy values stay exact.
            jitter = np.exp(np.random.default_rng(0).normal(0, 0.04, len(y)))
            ax.scatter(pct * jitter, y, s=22, color=st["color"], alpha=0.35, marker=st["marker"], linewidths=0)
            mx, my, sy = np.exp(np.log(pct).mean()), y.mean(), y.std(ddof=1)
            ax.errorbar(mx, my, yerr=sy, color=st["color"], marker=st["marker"], markersize=10,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, linewidth=2, capsize=0, zorder=5,
                        label=st["label"])
            # Direct-label only the methods that sit apart; the 25-45% cluster is named by the legend.
            if meth in ("GWO", "WOA"):
                ax.annotate(f"{st['label']} {my:.3f}", xy=(mx, my), xytext=(10, 0), textcoords="offset points",
                            fontsize=8.5, color=INK, va="center", zorder=6)
        b = np.mean([base[s][split][bkey] for s in SETS])
        ax.scatter([100], [b], s=90, color=INK, marker="*", zorder=5, label="All 512 features")
        ax.axhline(b, color=INK2, linestyle=(0, (4, 3)), linewidth=1)
        ax.annotate(f"All 512 features {b:.3f}", xy=(0.13, b), xytext=(0, 4), textcoords="offset points",
                    ha="left", va="bottom", fontsize=8.5, color=INK2)
        ax.set_xlim(0.12, 160)
        ax.set_xticks([0.2, 0.5, 1, 2, 5, 10, 20, 50, 100])
        ax.set_xticklabels(["0.2", "0.5", "1", "2", "5", "10", "20", "50", "100"])
        ax.set_xlabel("Features kept (% of 512, log scale)", fontsize=9.5, color=INK2)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
    axes[0].set_ylabel("Accuracy", fontsize=9.5, color=INK2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(METHODS) + 1, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, 1.0), labelcolor=INK)
    fig.suptitle("Fewer features raise validation accuracy but not test accuracy "
                 "(dots = runs, marker = mean ± 1 std, 30 runs per method)",
                 x=0.01, y=1.06, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    runs, by, base, head, all_runs = load()
    evals = sorted({r["n_evals"] for r in runs})
    counts = {m: len(by[m]) for m in METHODS}
    print(f"loaded {len(runs)} runs {counts}, evaluations per run: {evals}")

    rows = summary_rows(by, base, head)
    write_csv(os.path.join(OUT, "summary_table.csv"), rows)
    by_set = []
    for s in SETS:
        by_set += [{"feature_set": s, **r} for r in summary_rows(by, base, head, sets=(s,))]
    write_csv(os.path.join(OUT, "summary_by_set.csv"), by_set)

    wrows = wilcoxon_rows(by, base)
    write_csv(os.path.join(OUT, "wilcoxon.csv"), wrows)

    cols = ["method", "n_runs", "val_acc", "n_features", "pct_features"] + METRICS + ["runtime_s"]
    heads = ["Method", "Runs", "Val ACC", "# Features", "% Features"] + [LABELS[m] for m in METRICS] + ["Runtime (s)"]
    sig = [r for r in wrows if r["significant_holm_0.05"]]
    sig_lines = [f"- {r['method']} vs {r['vs']}, {LABELS[r['metric']]}: diff {r['mean_diff']:+.4f}, "
                 f"p = {r['p']:.4f}, Holm p = {r['p_holm']:.4f}" for r in sig] or ["- none"]
    vs_base = [r for r in wrows if r["vs"] == "ALL512"]
    wb_cols = ["metric", "method", "mean_diff", "wins", "ties", "losses", "p", "p_holm"]
    with open(os.path.join(OUT, "summary_table.md"), "w", encoding="utf-8") as f:
        f.write("# Feature-selection benchmark — test results (mean ± std)\n\n")
        f.write(f"{len(runs)} runs; {evals[0]} fitness evaluations each; α = 0.99; error = 1 − val ACC.\n\n")
        f.write(md_table(rows, cols, heads) + "\n")
    with open(os.path.join(OUT, "wilcoxon.md"), "w", encoding="utf-8") as f:
        f.write("# Paired Wilcoxon signed-rank tests on test metrics\n\n")
        n_tests = len(METHODS) + len(METHODS) * (len(METHODS) - 1) // 2
        f.write(f"Holm correction within each metric ({len(METHODS)} vs all-512 + "
                f"{n_tests - len(METHODS)} method pairs = {n_tests} tests).\n\n")
        f.write("## Significant after Holm (p < 0.05)\n\n" + "\n".join(sig_lines) + "\n\n")
        f.write("## Each method vs all 512 features + SVM\n\n")
        f.write(md_table(vs_base, wb_cols, ["Metric", "Method", "Mean diff", "Wins", "Ties", "Losses", "p", "Holm p"]))
        f.write("\n\nAll pairwise comparisons: `wilcoxon.csv`.\n")

    arows = ablation_rows(all_runs)
    if any(r["alpha"] != MAIN_ALPHA for r in arows):
        write_csv(os.path.join(OUT, "alpha_ablation.csv"), arows)
        acols = ["method", "alpha", "n_runs", "pct_features", "val_acc", "acc", "auc", "malignant_recall",
                 "same_mask_as_0.99"]
        with open(os.path.join(OUT, "alpha_ablation.md"), "w", encoding="utf-8") as f:
            f.write("# α ablation — fitness = α·error + (1 − α)·features/512\n\n")
            f.write("Same seeds at every α, so masks are compared run for run with α = 0.99.\n\n")
            f.write(md_table(arows, acols, ["Method", "α", "Runs", "% Features", "Val ACC", "Test ACC", "Test AUC",
                                            "Malignant recall", "Same mask as α = 0.99"]) + "\n")
        print("\nα ablation:\n" + md_table(arows, ["method", "alpha", "pct_features", "val_acc", "acc", "auc",
                                                   "same_mask_as_0.99"],
                                           ["Method", "α", "% feat", "Val ACC", "ACC", "AUC", "Same mask"]))

    plot_convergence(by, base, os.path.join(OUT, "convergence.png"))
    plot_features_vs_accuracy(by, base, os.path.join(OUT, "features_vs_accuracy.png"))

    print(md_table(rows, ["method", "val_acc", "pct_features", "acc", "auc", "macro_f1", "malignant_recall"],
                   ["Method", "Val ACC", "% feat", "ACC", "AUC", "Macro-F1", "Mal. recall"]))
    print("\nSignificant after Holm correction:\n" + "\n".join(sig_lines))
    print(f"\nsaved to {os.path.relpath(OUT, ROOT)}: " + ", ".join(sorted(os.listdir(OUT))))


if __name__ == "__main__":
    main()
