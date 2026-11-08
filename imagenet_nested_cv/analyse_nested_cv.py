"""Analyse the option-2 nested-CV benchmark (ImageNet ResNet-18 features), as pre-registered in
docs/project_context.md, step 13.

Comparisons per metric (9): each algorithm (PSO, GA, GWO, WOA) vs RANDOM, each algorithm vs ALL512,
and RANDOM vs ALL512. Pairs are matched on (repeat, outer fold).

Tests:
    primary    corrected resampled t-test (Nadeau & Bengio 2003): the paired differences d_j over the
               J outer folds of repeated k-fold CV overlap, so the variance is inflated by
               (1/J + n_test/n_train) instead of 1/J; two-sided, df = J - 1
    secondary  Wilcoxon signed-rank (ignores the overlap, so it is optimistic)
Holm correction within each metric, separately for each test. Primary metrics: test ACC and AUC.

Outputs in imagenet_nested_cv/analysis_<backend>/: summary.csv / .md, tests.csv / .md

Fixed-size mode (--k K, step 14): reads results_<backend>_k<K>/, adds SINGLE (the first random K-subset
drawn by RANDOM, no search) and reuses ALL512 from the free-size run (same folds, same deterministic
classifier). Comparisons per metric (12): each algorithm vs RANDOM, vs SINGLE and vs ALL512.
Outputs in analysis_<backend>_k<K>/.

Usage (from project root):
    .venv\\Scripts\\python.exe imagenet_nested_cv\\analyse_nested_cv.py               # GPU LS-SVM runs
    .venv\\Scripts\\python.exe imagenet_nested_cv\\analyse_nested_cv.py --backend cpu # CPU SVC runs
    .venv\\Scripts\\python.exe imagenet_nested_cv\\analyse_nested_cv.py --k 20        # fixed-size runs
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "feature_selection"))
from analyse_results import fmt, holm, md_table, paired_test, write_csv  # noqa: E402

ALGOS = ["PSO", "GA", "GWO", "WOA"]
METHODS = ["ALL512", "RANDOM"] + ALGOS
METRICS = ["acc", "auc", "macro_f1", "malignant_recall", "specificity"]
LABELS = {"acc": "ACC", "auc": "AUC", "macro_f1": "Macro-F1", "malignant_recall": "Malignant recall",
          "specificity": "Specificity"}
NAMES = {"ALL512": "All 512 features", "RANDOM": "Random search", "SINGLE": "Single random k-subset (no search)"}
RESULTS = {"gpu": "results_gpu_lssvm", "cpu": "results_cpu_svc"}


def corrected_ttest(a, b, test_train_ratio):
    """Nadeau & Bengio corrected resampled t-test on paired per-fold scores; returns (t, two-sided p)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    J = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1.0 / J + test_train_ratio) * var)
    return float(t), float(2 * stats.t.sf(abs(t), df=J - 1))


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main(backend, k=None):
    order = lambda r: (r["repeat"], r["fold"])  # noqa: E731
    methods = METHODS if k is None else ["ALL512", "SINGLE", "RANDOM"] + ALGOS
    runs = load(os.path.join(HERE, RESULTS[backend] + (f"_k{k}" if k else ""), "runs.jsonl"))
    if k is not None:
        runs = [r for r in runs if r["method"] != "ALL512"]
        runs += [r for r in load(os.path.join(HERE, RESULTS[backend], "runs.jsonl")) if r["method"] == "ALL512"]
        runs += [{**r, "method": "SINGLE", "inner_cv_acc": float("nan"), "n_features": k, "runtime_s": 0.0,
                  **{f"test_{m}": r[f"single_test_{m}"] for m in METRICS}}
                 for r in runs if r["method"] == "RANDOM"]
    by = {m: sorted([r for r in runs if r["method"] == m], key=order) for m in methods}
    keys = [order(r) for r in by["ALL512"]]
    present = [m for m in methods if len(by[m]) == len(keys) and [order(r) for r in by[m]] == keys]
    missing = [m for m in methods if m not in present]
    ratio = float(np.mean([r["n_test"] / r["n_train"] for r in by["ALL512"]]))
    print(f"backend {backend}: {len(runs)} jobs, {len(keys)} outer folds; complete methods {present}"
          + (f"; incomplete/missing {missing} (left out)" if missing else "") + f"; n_test/n_train = {ratio:.3f}")

    srows = []
    for m in present:
        R = by[m]
        srows.append({"method": NAMES.get(m, m), "n_folds": len(R),
                      "inner_cv_acc": "–" if m == "SINGLE" else fmt([r["inner_cv_acc"] for r in R]),
                      "n_features": f"{np.mean([r['n_features'] for r in R]):.0f}",
                      **{k: fmt([r[f"test_{k}"] for r in R]) for k in METRICS},
                      "runtime_s": f"{np.mean([r['runtime_s'] for r in R]):.1f}"})

    if k is None:
        comps = [(a, "RANDOM") for a in ALGOS] + [(a, "ALL512") for a in ALGOS] + [("RANDOM", "ALL512")]
    else:
        comps = [(a, ref) for ref in ("RANDOM", "SINGLE", "ALL512") for a in ALGOS]
    comps = [(a, b) for a, b in comps if a in present and b in present]
    trows = []
    for met in METRICS:
        block = []
        for a, b in comps:
            x = [r[f"test_{met}"] for r in by[a]]
            y = [r[f"test_{met}"] for r in by[b]]
            t, p_t = corrected_ttest(x, y, ratio)
            x, y = np.asarray(x), np.asarray(y)
            block.append({"metric": met, "method": a, "vs": b, "mean_diff": round(float(x.mean() - y.mean()), 5),
                          "wins": int((x > y).sum()), "ties": int((x == y).sum()), "losses": int((x < y).sum()),
                          "t_corrected": round(t, 3), "p_corrected": p_t, "p_wilcoxon": paired_test(x, y)})
        for col in ("p_corrected", "p_wilcoxon"):
            adj = holm(np.array([r[col] for r in block]))
            for r, h in zip(block, adj):
                r[col + "_holm"] = round(float(h), 5)
                r[col] = round(float(r[col]), 5)
        trows += block

    out = os.path.join(HERE, f"analysis_{backend}" + (f"_k{k}" if k else ""))
    os.makedirs(out, exist_ok=True)
    write_csv(os.path.join(out, "summary.csv"), srows)
    write_csv(os.path.join(out, "tests.csv"), trows)
    scols = ["method", "n_folds", "inner_cv_acc", "n_features"] + METRICS + ["runtime_s"]
    sheads = ["Method", "Outer folds", "Inner-CV ACC", "# Features"] + [LABELS[k] for k in METRICS] + ["Runtime (s)"]
    tcols = ["metric", "method", "vs", "mean_diff", "wins", "ties", "losses", "t_corrected", "p_corrected",
             "p_corrected_holm", "p_wilcoxon", "p_wilcoxon_holm"]
    theads = ["Metric", "Method", "vs", "Mean diff", "Better", "Tied", "Worse", "t (corr.)", "p (corr.)",
              "Holm p (corr.)", "p (Wilcoxon)", "Holm p (Wilcoxon)"]
    sig = [r for r in trows if r["p_corrected_holm"] < 0.05]
    sig_lines = [f"- {LABELS[r['metric']]}: {r['method']} vs {r['vs']}: diff {r['mean_diff']:+.4f} "
                 f"({r['wins']}/{r['ties']}/{r['losses']}), corrected p = {r['p_corrected']:.4f}, "
                 f"Holm p = {r['p_corrected_holm']:.4f}" for r in sig] or ["- none"]
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(f"# Option 2{'b' if k else ''} — ImageNet ResNet-18 features, nested CV ({backend} backend"
                + (f", exactly {k} features" if k else "") + ")\n\n")
        f.write(f"Outer-test results over {len(keys)} outer folds (stratified 5-fold × 6 repeats), mean ± std.\n\n")
        f.write(md_table(srows, scols, sheads) + "\n")
    with open(os.path.join(out, "tests.md"), "w", encoding="utf-8") as f:
        f.write(f"# Option 2 — paired tests on outer-test metrics ({backend} backend)\n\n")
        f.write("Primary: corrected resampled t-test (Nadeau & Bengio 2003). Secondary: Wilcoxon signed-rank. "
                f"Holm within each metric over {len(comps)} comparisons.\n\n")
        f.write("## Significant after Holm (primary test, p < 0.05)\n\n" + "\n".join(sig_lines) + "\n\n")
        f.write(md_table(trows, tcols, theads) + "\n")

    print(md_table(srows, ["method", "inner_cv_acc", "n_features", "acc", "auc", "malignant_recall"],
                   ["Method", "Inner-CV ACC", "# Feat", "ACC", "AUC", "Mal. recall"]))
    print("\nPrimary metrics (ACC, AUC):")
    print(md_table([r for r in trows if r["metric"] in ("acc", "auc")],
                   ["metric", "method", "vs", "mean_diff", "wins", "losses", "p_corrected", "p_corrected_holm",
                    "p_wilcoxon_holm"],
                   ["Metric", "Method", "vs", "Diff", "Better", "Worse", "p corr", "Holm corr", "Holm Wilc"]))
    print("\nSignificant after Holm (primary test):\n" + "\n".join(sig_lines))
    print(f"\nsaved to {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Analyse the option-2 nested-CV benchmark")
    p.add_argument("--backend", default="gpu", choices=["gpu", "cpu"])
    p.add_argument("--k", type=int, default=None, help="analyse the fixed-size run with exactly k features")
    a = p.parse_args()
    main(a.backend, a.k)
