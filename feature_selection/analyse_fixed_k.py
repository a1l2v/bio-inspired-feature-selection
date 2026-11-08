"""Analyse the equal-size (fixed-k) benchmark: does a selected k-subset beat random k-subsets?

Input:  feature_selection/results_fixed_k/runs.jsonl   (run_benchmark.py --k K --out results_fixed_k)
        svm_baseline/results.csv                       (all-512 SVM per feature set)
Output: feature_selection/analysis_fixed_k/
    summary.csv / .md   per k: test metrics (mean ± std) for every method, the single random k-subset
                        (no search) and all 512 features
    wilcoxon.csv / .md  per k and metric, each algorithm (PSO, GA, GWO, WOA) vs
                          RANDOM  - random search, same k and budget   (does the algorithm beat searching?)
                          SINGLE  - one random k-subset, no search    (does selection beat chance?)
                          ALL512  - SVM on all 512 features
                        Paired Wilcoxon signed-rank on (feature set, run); Holm within each (k, metric)
                        over the 12 comparisons (4 algorithms x 3 references).

Usage (from project root):
    .venv\\Scripts\\python.exe feature_selection\\analyse_fixed_k.py
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyse_results import BASELINE, ROOT, fmt, holm, md_table, paired_test, write_csv  # noqa: E402

RUNS = os.path.join(HERE, "results_fixed_k", "runs.jsonl")
OUT = os.path.join(HERE, "analysis_fixed_k")
ALGOS = ["PSO", "GA", "GWO", "WOA"]
METRICS = ["acc", "auc", "macro_f1", "malignant_recall", "specificity"]
LABELS = {"acc": "ACC", "auc": "AUC", "macro_f1": "Macro-F1", "malignant_recall": "Malignant recall",
          "specificity": "Specificity"}


def load():
    with open(RUNS) as f:
        runs = [json.loads(line) for line in f]
    by = {}
    for r in runs:
        by.setdefault((r["k"], r["method"]), []).append(r)
    for v in by.values():
        v.sort(key=lambda r: (r["feature_set"], r["run"]))
    base = {}
    import csv
    with open(BASELINE) as f:
        for r in csv.DictReader(f):
            if r["model"] == "svm_all_512" and r["split"] == "test":
                base[int(r["weights"][-1])] = {m: float(r[m]) for m in METRICS}
    return runs, by, base


def main():
    os.makedirs(OUT, exist_ok=True)
    runs, by, base = load()
    ks = sorted({r["k"] for r in runs})
    print(f"loaded {len(runs)} runs; k = {ks}; evaluations per run: {sorted({r['n_evals'] for r in runs})}")

    srows, wrows = [], []
    for k in ks:
        rand = by[(k, "RANDOM")]
        refs = {
            "RANDOM": lambda m, R=rand: [r[f"test_{m}"] for r in R],
            "SINGLE": lambda m, R=rand: [r[f"single_test_{m}"] for r in R],
            "ALL512": lambda m, R=rand: [base[r["feature_set"]][m] for r in R],
        }
        srows.append({"k": k, "method": "All 512 features", "n_runs": 3, "val_acc": "–",
                      **{m: fmt([base[s][m] for s in (1, 2, 3)]) for m in METRICS}})
        srows.append({"k": k, "method": "Single random k-subset (no search)", "n_runs": len(rand), "val_acc": "–",
                      **{m: fmt(refs["SINGLE"](m)) for m in METRICS}})
        for meth in ["RANDOM"] + ALGOS:
            R = by[(k, meth)]
            srows.append({"k": k, "method": "Random search" if meth == "RANDOM" else meth, "n_runs": len(R),
                          "val_acc": fmt([r["val_score"] for r in R]),
                          **{m: fmt([r[f"test_{m}"] for r in R]) for m in METRICS}})

        for m in METRICS:
            comps = []
            for meth in ALGOS:
                R = by[(k, meth)]
                assert [(r["feature_set"], r["run"]) for r in R] == [(r["feature_set"], r["run"]) for r in rand]
                a = [r[f"test_{m}"] for r in R]
                for ref, get in refs.items():
                    comps.append((meth, ref, a, get(m)))
            p = np.array([paired_test(a, b) for _, _, a, b in comps])
            for (meth, ref, a, b), pi, ph in zip(comps, p, holm(p)):
                a, b = np.asarray(a), np.asarray(b)
                wrows.append({"k": k, "metric": m, "method": meth, "vs": ref,
                              "mean_diff": round(float(a.mean() - b.mean()), 5),
                              "wins": int((a > b).sum()), "ties": int((a == b).sum()), "losses": int((a < b).sum()),
                              "p": round(float(pi), 5), "p_holm": round(float(ph), 5),
                              "significant_holm_0.05": bool(ph < 0.05)})

    write_csv(os.path.join(OUT, "summary.csv"), srows)
    write_csv(os.path.join(OUT, "wilcoxon.csv"), wrows)

    scols = ["k", "method", "n_runs", "val_acc"] + METRICS
    sheads = ["k", "Method", "Runs", "Val ACC"] + [LABELS[m] for m in METRICS]
    sig = [r for r in wrows if r["significant_holm_0.05"]]
    sig_lines = [f"- k = {r['k']}: {r['method']} vs {r['vs']}, {LABELS[r['metric']]}: diff {r['mean_diff']:+.4f} "
                 f"({r['wins']} better / {r['ties']} tied / {r['losses']} worse), p = {r['p']:.4f}, "
                 f"Holm p = {r['p_holm']:.4f}" for r in sig] or ["- none"]
    key = [r for r in wrows if r["metric"] in ("acc", "auc")]
    wcols = ["k", "metric", "method", "vs", "mean_diff", "wins", "ties", "losses", "p", "p_holm"]
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
        f.write("# Equal-size benchmark — test results (mean ± std)\n\n")
        f.write("Every method selects exactly k features; fitness = 1 − val ACC; 1000 evaluations; "
                "30 paired runs (3 feature sets × 10 seeds).\n\n")
        f.write(md_table(srows, scols, sheads) + "\n")
    with open(os.path.join(OUT, "wilcoxon.md"), "w", encoding="utf-8") as f:
        f.write("# Equal-size benchmark — paired Wilcoxon tests on test metrics\n\n")
        f.write("Holm correction within each (k, metric): 4 algorithms × 3 references = 12 tests.\n\n")
        f.write("## Significant after Holm (p < 0.05)\n\n" + "\n".join(sig_lines) + "\n\n")
        f.write("## ACC and AUC, all comparisons\n\n")
        f.write(md_table(key, wcols, ["k", "Metric", "Method", "vs", "Mean diff", "Better", "Tied", "Worse",
                                      "p", "Holm p"]) + "\n\nAll metrics: `wilcoxon.csv`.\n")

    print(md_table(srows, ["k", "method", "val_acc", "acc", "auc", "malignant_recall"],
                   ["k", "Method", "Val ACC", "ACC", "AUC", "Mal. recall"]))
    smallest = sorted([r for r in wrows if r["vs"] == "RANDOM" and r["mean_diff"] > 0], key=lambda r: r["p"])[:3]
    print("\nSmallest raw p, algorithm better than random search:")
    for r in smallest:
        print(f"  k = {r['k']} {r['method']} {LABELS[r['metric']]}: diff {r['mean_diff']:+.4f}, p = {r['p']:.4f}, "
              f"Holm p = {r['p_holm']:.4f}")
    print("\nSignificant after Holm:\n" + "\n".join(sig_lines))
    print(f"\nsaved to {os.path.relpath(OUT, ROOT)}: " + ", ".join(sorted(os.listdir(OUT))))


if __name__ == "__main__":
    main()
