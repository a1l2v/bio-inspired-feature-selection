"""Analyse DermaMNIST Study D1 as fixed in Derma/README.md.

Primary (per metric ACC and AUC, Holm over 4 tests): GA vs RANDOM, CSO vs RANDOM, GA vs FILTER, CSO vs FILTER.
Secondary (not corrected): GA / CSO vs ALL1820, GA vs CSO, FILTER vs RANDOM; macro-F1 and balanced accuracy
get the same comparisons, all uncorrected. Two-sided paired Wilcoxon over the 30 runs (paired by seed);
ALL1820 is one deterministic model, so each run is compared with that single value.

Output: Derma/feature_selection/analysis_d1/summary.{csv,md}, tests.{csv,md}
Study D2 (KNN classifier, same hypotheses): --study d2 reads results_d2/ and writes analysis_d2/.

Usage (from project root):
    .venv\\Scripts\\python.exe Derma\\feature_selection\\analyse_d1.py
"""

import argparse
import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
_spec = importlib.util.spec_from_file_location("breast_analyse", os.path.join(ROOT, "feature_selection", "analyse_results.py"))
_b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_b)
fmt, holm, md_table, paired_test, write_csv = _b.fmt, _b.holm, _b.md_table, _b.paired_test, _b.write_csv

METHODS = ["ALL1820", "FILTER", "RANDOM", "GA", "CSO"]
NAMES = {"ALL1820": "All 1,820 features", "FILTER": "ANOVA filter top 50", "RANDOM": "Random search (50)",
         "GA": "GA (50)", "CSO": "CSO (50)"}
METRICS = ["acc", "auc", "macro_f1", "bal_acc"]
LABELS = {"acc": "ACC", "auc": "AUC", "macro_f1": "Macro-F1", "bal_acc": "Bal. ACC"}
PRIMARY = [("GA", "RANDOM"), ("CSO", "RANDOM"), ("GA", "FILTER"), ("CSO", "FILTER")]
SECONDARY = [("GA", "ALL1820"), ("CSO", "ALL1820"), ("GA", "CSO"), ("FILTER", "RANDOM")]


def main(study):
    title = {"d1": "Study D1 (LS-SVM)", "d2": "Study D2 (5-NN)"}[study]
    with open(os.path.join(HERE, f"results_{study}", "runs.jsonl")) as f:
        runs = [json.loads(line) for line in f]
    by = {m: sorted([r for r in runs if r["method"] == m], key=lambda r: r["run"]) for m in METHODS}
    print(f"{len(runs)} runs: " + ", ".join(f"{m} {len(by[m])}" for m in METHODS) + "; search evaluations "
          + str(sorted({r["n_evals"] for m in ("RANDOM", "GA", "CSO") for r in by[m]})))

    srows = [{"method": NAMES[m], "n_runs": len(by[m]), "cv_acc": fmt([r["cv_acc"] for r in by[m]]),
              "n_features": by[m][0]["n_features"], **{k: fmt([r[f"test_{k}"] for r in by[m]]) for k in METRICS}}
             for m in METHODS]

    def compare(a, b, met):
        x = np.array([r[f"test_{met}"] for r in by[a]])
        if b == "ALL1820":
            y = np.full_like(x, by["ALL1820"][0][f"test_{met}"])
        else:
            assert [r["run"] for r in by[a]] == [r["run"] for r in by[b]]
            y = np.array([r[f"test_{met}"] for r in by[b]])
        return {"metric": met, "method": a, "vs": b, "mean_diff": round(float(x.mean() - y.mean()), 5),
                "wins": int((x > y).sum()), "ties": int((x == y).sum()), "losses": int((x < y).sum()),
                "p": paired_test(x, y)}

    trows = []
    for met in METRICS:
        block = [compare(a, b, met) for a, b in PRIMARY]
        if met in ("acc", "auc"):
            for r, h in zip(block, holm(np.array([r["p"] for r in block]))):
                r["p"], r["p_holm"], r["family"] = round(float(r["p"]), 5), round(float(h), 5), "primary"
        else:
            for r in block:
                r["p"], r["p_holm"], r["family"] = round(float(r["p"]), 5), "–", "secondary (not corrected)"
        extra = [compare(a, b, met) for a, b in SECONDARY]
        for r in extra:
            r["p"], r["p_holm"], r["family"] = round(float(r["p"]), 5), "–", "secondary (not corrected)"
        trows += block + extra

    out = os.path.join(HERE, f"analysis_{study}")
    os.makedirs(out, exist_ok=True)
    write_csv(os.path.join(out, "summary.csv"), srows)
    write_csv(os.path.join(out, "tests.csv"), trows)
    scols = ["method", "n_runs", "cv_acc", "n_features"] + METRICS
    sheads = ["Method", "Runs", "Pool CV ACC", "# Features"] + [LABELS[k] for k in METRICS]
    tcols = ["metric", "family", "method", "vs", "mean_diff", "wins", "ties", "losses", "p", "p_holm"]
    theads = ["Metric", "Family", "Method", "vs", "Mean diff", "Better", "Tied", "Worse", "p", "Holm p"]
    sig = [r for r in trows if r["family"] == "primary" and r["p_holm"] < 0.05]
    sig_lines = [f"- {LABELS[r['metric']]}: {r['method']} vs {r['vs']}: diff {r['mean_diff']:+.4f} "
                 f"({r['wins']}/{r['ties']}/{r['losses']}), p = {r['p']:.2e}, Holm p = {r['p_holm']:.2e}"
                 for r in sig] or ["- none"]
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(f"# DermaMNIST {title} — test results, mean ± std over 30 runs\n\n" + md_table(srows, scols, sheads) + "\n")
    with open(os.path.join(out, "tests.md"), "w", encoding="utf-8") as f:
        f.write(f"# DermaMNIST {title} — paired Wilcoxon tests, Holm over 4 primary tests per primary metric\n\n")
        f.write("## Primary hypotheses significant after Holm\n\n" + "\n".join(sig_lines) + "\n\n"
                + md_table(trows, tcols, theads) + "\n")
    print(md_table(srows, scols, sheads))
    print("\n" + md_table(trows, tcols, theads))
    print("\nPrimary hypotheses significant after Holm:\n" + "\n".join(sig_lines))
    print(f"\nsaved to {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Analyse DermaMNIST Study D1 or D2")
    p.add_argument("--study", default="d1", choices=["d1", "d2"])
    main(p.parse_args().study)
