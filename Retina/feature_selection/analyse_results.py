"""Analyse the RetinaMNIST benchmark as fixed in Retina/README.md.

Hypotheses per metric (4): GA vs RANDOM, GWO vs RANDOM, GA vs ALL512, GWO vs ALL512; two-sided paired
Wilcoxon signed-rank on (feature set, run), Holm within each metric. Primary metrics: test ACC and AUC;
macro-F1 and QWK secondary. ALL512 is deterministic per feature set, so each run is paired with its own
feature set's ALL512 value.

Output: Retina/feature_selection/analysis_<backend>/summary.{csv,md}, tests.{csv,md}

Study R2 (--study r2, design in Retina/README.md): methods ALL512, RANDOM, GA, HHO, CSO from
results_r2_gpu/. Primary Holm family per metric: HHO vs RANDOM, CSO vs RANDOM, HHO vs ALL512, CSO vs ALL512.
Secondary (reported, not in the Holm family): GA vs RANDOM, GA vs ALL512, HHO vs GA, CSO vs GA.
Output: analysis_r2/.

Usage (from project root):
    .venv\\Scripts\\python.exe Retina\\feature_selection\\analyse_results.py
    .venv\\Scripts\\python.exe Retina\\feature_selection\\analyse_results.py --study r2
"""

import argparse
import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
# Shared helpers from the BreastMNIST analysis (same file name as this one, so load it by path).
_spec = importlib.util.spec_from_file_location("breast_analyse", os.path.join(ROOT, "feature_selection", "analyse_results.py"))
_b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_b)
fmt, holm, md_table, paired_test, write_csv = _b.fmt, _b.holm, _b.md_table, _b.paired_test, _b.write_csv

METHODS = ["ALL512", "RANDOM", "GA", "GWO"]
METRICS = ["acc", "auc", "macro_f1", "qwk"]
LABELS = {"acc": "ACC", "auc": "AUC", "macro_f1": "Macro-F1", "qwk": "QWK"}
NAMES = {"ALL512": "All 512 features", "RANDOM": "Random search"}
COMPARISONS = [("GA", "RANDOM"), ("GWO", "RANDOM"), ("GA", "ALL512"), ("GWO", "ALL512")]


def main(backend, study="r1"):
    if study == "r2":
        methods, primary = ["ALL512", "RANDOM", "GA", "HHO", "CSO"], [("HHO", "RANDOM"), ("CSO", "RANDOM"),
                                                                        ("HHO", "ALL512"), ("CSO", "ALL512")]
        secondary = [("GA", "RANDOM"), ("GA", "ALL512"), ("HHO", "GA"), ("CSO", "GA")]
        path, fit_key, out = os.path.join(HERE, "results_r2_gpu", "runs.jsonl"), "cv_acc", os.path.join(HERE, "analysis_r2")
    else:
        methods, primary, secondary = METHODS, COMPARISONS, []
        path = os.path.join(HERE, f"results_{'gpu_lssvm' if backend == 'gpu' else 'cpu_svc'}", "runs.jsonl")
        fit_key, out = "val_acc", os.path.join(HERE, f"analysis_{backend}")
    with open(path) as f:
        runs = [json.loads(line) for line in f]
    by = {m: sorted([r for r in runs if r["method"] == m], key=lambda r: (r["feature_set"], r["run"])) for m in methods}
    base = {r["feature_set"]: r for r in by["ALL512"]}
    print(f"{len(runs)} runs: " + ", ".join(f"{m} {len(by[m])}" for m in methods)
          + f"; search evaluations per run {sorted({r['n_evals'] for r in runs if r['method'] != 'ALL512'})}")

    srows = [{"method": NAMES.get(m, m), "n_runs": len(by[m]), "val_acc": fmt([r[fit_key] for r in by[m]]),
              "n_features": f"{np.mean([r['n_features'] for r in by[m]]):.1f}",
              **{k: fmt([r[f"test_{k}"] for r in by[m]]) for k in METRICS}} for m in methods]

    def compare(a, b, met):
        x = np.array([r[f"test_{met}"] for r in by[a]])
        y = np.array([base[r["feature_set"]][f"test_{met}"] if b == "ALL512" else r[f"test_{met}"]
                      for r in (by[a] if b == "ALL512" else by[b])])
        if b != "ALL512":
            assert [(r["feature_set"], r["run"]) for r in by[a]] == [(r["feature_set"], r["run"]) for r in by[b]]
        return {"metric": met, "method": a, "vs": b, "mean_diff": round(float(x.mean() - y.mean()), 5),
                "wins": int((x > y).sum()), "ties": int((x == y).sum()), "losses": int((x < y).sum()),
                "p": paired_test(x, y)}

    trows = []
    for met in METRICS:
        block = [compare(a, b, met) for a, b in primary]
        for r, h in zip(block, holm(np.array([r["p"] for r in block]))):
            r["p"], r["p_holm"], r["family"] = round(float(r["p"]), 5), round(float(h), 5), "primary"
        extra = [compare(a, b, met) for a, b in secondary]
        for r in extra:
            r["p"], r["p_holm"], r["family"] = round(float(r["p"]), 5), "–", "secondary (not corrected)"
        trows += block + extra
    os.makedirs(out, exist_ok=True)
    write_csv(os.path.join(out, "summary.csv"), srows)
    write_csv(os.path.join(out, "tests.csv"), trows)
    scols = ["method", "n_runs", "val_acc", "n_features"] + METRICS
    sheads = ["Method", "Runs", "CV ACC" if study == "r2" else "Val ACC", "# Features"] + [LABELS[k] for k in METRICS]
    tcols = ["metric", "family", "method", "vs", "mean_diff", "wins", "ties", "losses", "p", "p_holm"]
    theads = ["Metric", "Family", "Method", "vs", "Mean diff", "Better", "Tied", "Worse", "p", "Holm p"]
    sig = [r for r in trows if r["family"] == "primary" and r["p_holm"] < 0.05]
    sig_lines = [f"- {LABELS[r['metric']]}: {r['method']} vs {r['vs']}: diff {r['mean_diff']:+.4f} "
                 f"({r['wins']}/{r['ties']}/{r['losses']}), p = {r['p']:.4f}, Holm p = {r['p_holm']:.4f}"
                 for r in sig] or ["- none"]
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(f"# RetinaMNIST — test results ({backend} backend), mean ± std\n\n" + md_table(srows, scols, sheads) + "\n")
    with open(os.path.join(out, "tests.md"), "w", encoding="utf-8") as f:
        f.write(f"# RetinaMNIST — paired Wilcoxon tests ({backend} backend), Holm within metric (4 tests)\n\n")
        f.write("## Significant after Holm\n\n" + "\n".join(sig_lines) + "\n\n" + md_table(trows, tcols, theads) + "\n")

    print(md_table(srows, scols, sheads))
    print("\n" + md_table(trows, tcols, theads))
    print("\nSignificant after Holm:\n" + "\n".join(sig_lines))
    print(f"\nsaved to {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Analyse the RetinaMNIST feature-selection benchmark")
    p.add_argument("--backend", default="gpu", choices=["gpu", "cpu"])
    p.add_argument("--study", default="r1", choices=["r1", "r2"])
    a = p.parse_args()
    main(a.backend, a.study)
