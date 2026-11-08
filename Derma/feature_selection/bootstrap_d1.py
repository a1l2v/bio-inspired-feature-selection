"""Post-hoc robustness check for Study D1 (not part of the pre-specified design): test-set sampling uncertainty.

The pre-specified Wilcoxon tests pair 30 runs that differ in their selection (pool, folds, search seed) but are
all scored on the same 2,005 test images, so they do not cover sampling noise in the test set itself. Here:

1. Every run's selected 50 features (saved in runs.jsonl) are re-fitted exactly as in run_fixed_k.py; the
   re-computed test ACC / AUC must equal the stored values (integrity check).
2. Paired bootstrap over test images (2,000 resamples, seed 0): in each resample, every method's mean ACC and
   mean macro AUC over its 30 runs is recomputed, and the GA / CSO minus RANDOM / FILTER differences recorded.
   Reported: the observed difference, the 95 % percentile interval, and the share of resamples <= 0.

Output: Derma/feature_selection/analysis_d1/bootstrap.json, bootstrap.md

Usage (from project root):
    .venv\\Scripts\\python.exe Derma\\feature_selection\\bootstrap_d1.py
"""

import json
import os
import sys

import numpy as np
import torch
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_fixed_k import FEATS, N_CLASSES, evaluate  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "Retina", "feature_selection"))
from gpu_lssvm import LSSVMOvR  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

METHODS = ["FILTER", "RANDOM", "GA", "CSO"]
PAIRS = [("GA", "RANDOM"), ("CSO", "RANDOM"), ("GA", "FILTER"), ("CSO", "FILTER")]
B = 2000


def auc_matrix(scores, y, idx):
    """Macro one-vs-rest AUC for every row of scores (R x n x C) on the resampled test indices idx."""
    s, yy = scores[:, idx, :], y[idx]
    n = len(idx)
    out = np.zeros(len(scores))
    for c in range(N_CLASSES):
        pos = yy == c
        n1 = pos.sum()
        ranks = rankdata(s[:, :, c], axis=1)                       # ties get average ranks, as in roc_auc_score
        out += (ranks[:, pos].sum(1) - n1 * (n1 + 1) / 2) / (n1 * (n - n1))
    return out / N_CLASSES


def main():
    torch.backends.cuda.preferred_linalg_library("cusolver")
    with open(os.path.join(HERE, "results_d1", "runs.jsonl")) as f:
        runs = [json.loads(line) for line in f]
    d = np.load(FEATS)
    Xtv = np.vstack([d["X_train"], d["X_val"]])
    ytv = np.concatenate([d["y_train"], d["y_val"]])
    sc = StandardScaler().fit(Xtv)
    dev = torch.device("cuda")
    Xtv_t = torch.tensor(sc.transform(Xtv).astype(np.float32), device=dev)
    Xte_t = torch.tensor(sc.transform(d["X_test"]).astype(np.float32), device=dev)
    ytv_t, yte = torch.tensor(ytv, device=dev), d["y_test"]

    rows, scores, max_dev = [], [], 0.0
    for m in METHODS:
        for r in sorted([r for r in runs if r["method"] == m], key=lambda r: r["run"]):
            sel = torch.as_tensor(r["selected"], device=dev)
            s = LSSVMOvR(N_CLASSES).fit(Xtv_t[:, sel], ytv_t).decision_function(Xte_t[:, sel]).cpu().numpy()
            ev = evaluate(yte, s)
            max_dev = max(max_dev, abs(ev["acc"] - r["test_acc"]), abs(ev["auc"] - r["test_auc"]))
            rows.append(m)
            scores.append(s)
    rows, scores = np.array(rows), np.stack(scores)
    print(f"integrity: max |recomputed - stored| over {len(rows)} runs = {max_dev:.2e}", flush=True)
    assert max_dev < 1e-4, "re-fitted scores do not match the stored results"
    correct = (scores.argmax(2) == yte).astype(float)              # runs x test images

    def method_means(idx):
        acc = correct[:, idx].mean(1)
        auc = auc_matrix(scores, yte, idx)
        return {m: (acc[rows == m].mean(), auc[rows == m].mean()) for m in METHODS}

    obs = method_means(np.arange(len(yte)))
    rng = np.random.default_rng(0)
    boot = {f"{a}-{b}": {"acc": [], "auc": []} for a, b in PAIRS}
    for i in range(B):
        mm = method_means(rng.integers(0, len(yte), len(yte)))
        for a, b in PAIRS:
            boot[f"{a}-{b}"]["acc"].append(mm[a][0] - mm[b][0])
            boot[f"{a}-{b}"]["auc"].append(mm[a][1] - mm[b][1])
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{B} resamples", flush=True)

    res = {"B": B, "integrity_max_abs_dev": max_dev, "pairs": {}}
    lines = ["# Study D1 — post-hoc test-set bootstrap (2,000 paired resamples of the 2,005 test images)", "",
             "| Comparison | Metric | Observed diff | 95 % interval | Resamples <= 0 |", "|---|---|---|---|---|"]
    for a, b in PAIRS:
        key = f"{a}-{b}"
        for k, j in (("acc", 0), ("auc", 1)):
            v = np.array(boot[key][k])
            o = obs[a][j] - obs[b][j]
            lo, hi = np.percentile(v, [2.5, 97.5])
            frac = float((v <= 0).mean())
            res["pairs"].setdefault(key, {})[k] = {"observed": o, "ci95": [lo, hi], "frac_le_0": frac}
            lines.append(f"| {a} vs {b} | {k.upper()} | {o:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {frac:.4f} |")
    out = os.path.join(HERE, "analysis_d1")
    with open(os.path.join(out, "bootstrap.json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(out, "bootstrap.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
