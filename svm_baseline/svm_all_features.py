"""Baseline: SVM on all 512 ResNet-18 (28) features for BreastMNIST.

This is the "All 512 features + SVM" row of the results table, the reference point that the
bio-inspired feature selection has to beat with fewer features.

Protocol (kept identical for every later feature-selection method):
    - features from features/resnet18_28_{1,2,3}.npz (released MedMNIST weights)
    - StandardScaler fit on train only
    - SVC(kernel="rbf", C=1.0, gamma="scale"), fixed, no tuning (val is reserved for the
      feature-selection fitness; tuning on it here would leak into that comparison)
    - fit on train; report val (what the fitness function sees) and test (used once)
    - AUC from decision_function (deterministic; no Platt-scaling randomness)

Metrics: ACC, AUC, macro-F1, malignant recall (label 0 = malignant -> pos_label=0), specificity.
For comparison, the original ResNet-18 head is scored the same way from its saved probabilities.

Outputs in svm_baseline/:
    results.csv   one row per (weights, model, split)
    summary.json  mean +- std over the 3 weights files, test split

Usage (from project root):
    .venv\\Scripts\\python.exe svm_baseline\\svm_all_features.py
"""

import argparse
import csv
import json
import os
import time

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_FILES = [os.path.join(ROOT, "features", f"resnet18_28_{i}.npz") for i in (1, 2, 3)]
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
MALIGNANT = 0
SVM_PARAMS = {"kernel": "rbf", "C": 1.0, "gamma": "scale"}
METRICS = ("acc", "auc", "macro_f1", "malignant_recall", "specificity")


def evaluate(y_true, y_pred, score_benign):
    """Binary metrics. `score_benign` is any score that increases with P(label 1 = normal/benign)."""
    tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "acc": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, score_benign),  # same value whichever class is called positive
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "malignant_recall": recall_score(y_true, y_pred, pos_label=MALIGNANT),
        "specificity": recall_score(y_true, y_pred, pos_label=1),  # non-malignant correctly cleared
        "confusion": {"malignant_as_malignant": int(tn_b), "malignant_as_benign": int(fp_b),
                      "benign_as_malignant": int(fn_b), "benign_as_benign": int(tp_b)},
    }


def run_svm(d):
    scaler = StandardScaler().fit(d["X_train"])
    clf = SVC(**SVM_PARAMS).fit(scaler.transform(d["X_train"]), d["y_train"])
    out = {}
    for split in ("train", "val", "test"):
        X = scaler.transform(d[f"X_{split}"])
        out[split] = evaluate(d[f"y_{split}"], clf.predict(X), clf.decision_function(X))
    out["n_support"] = clf.n_support_.tolist()
    return out


def run_original_head(d):
    out = {}
    for split in ("train", "val", "test"):
        prob = d[f"prob_{split}"]
        out[split] = evaluate(d[f"y_{split}"], prob.argmax(1), prob[:, 1])
    return out


def main(feature_files):
    rows, per_model = [], {"resnet18_head": [], "svm_all_512": []}
    for path in feature_files:
        name = os.path.splitext(os.path.basename(path))[0]
        d = np.load(path)
        t = time.perf_counter()
        svm = run_svm(d)
        svm_seconds = time.perf_counter() - t
        head = run_original_head(d)

        print(f"==> {name}  (SVM fit+eval {svm_seconds:.2f}s, support vectors per class {svm['n_support']})")
        for model, res in (("resnet18_head", head), ("svm_all_512", svm)):
            per_model[model].append(res["test"])
            for split in ("train", "val", "test"):
                r = res[split]
                rows.append({"weights": name, "model": model, "split": split, "n_features": 512,
                             **{m: round(r[m], 5) for m in METRICS}, **r["confusion"]})
                if split != "train":
                    print(f"    {model:14} {split:5} " + "  ".join(f"{m} {r[m]:.4f}" for m in METRICS))
            c = res["test"]["confusion"]
            print(f"    {model:14} test confusion: malignant caught {c['malignant_as_malignant']}"
                  f"/{c['malignant_as_malignant'] + c['malignant_as_benign']}, "
                  f"benign flagged {c['benign_as_malignant']}/{c['benign_as_malignant'] + c['benign_as_benign']}")

    with open(os.path.join(OUT_DIR, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = {"protocol": {"svm": SVM_PARAMS, "scaler": "StandardScaler fit on train",
                            "features": [os.path.relpath(p, ROOT).replace("\\", "/") for p in feature_files]},
               "test_mean_std": {}}
    print("\n==> Test, mean ± std over weights files")
    for model, results in per_model.items():
        stats = {m: {"mean": round(float(np.mean([r[m] for r in results])), 5),
                     "std": round(float(np.std([r[m] for r in results], ddof=1)), 5)} for m in METRICS}
        summary["test_mean_std"][model] = stats
        print(f"    {model:14} " + "  ".join(f"{m} {s['mean']:.4f}±{s['std']:.4f}" for m, s in stats.items()))
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved {os.path.relpath(os.path.join(OUT_DIR, 'results.csv'), ROOT)} and summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVM baseline on all 512 ResNet-18 features")
    parser.add_argument("--features", nargs="+", default=FEATURE_FILES)
    main(parser.parse_args().features)
