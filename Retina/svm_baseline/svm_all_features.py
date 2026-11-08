"""RetinaMNIST baseline: SVM on all 512 ResNet-18 (28) features (5-class diabetic-retinopathy grading).

Same protocol as svm_baseline/svm_all_features.py for BreastMNIST, adapted to 5 classes:
    - features from Retina/features/resnet18_28_{1,2,3}.npz (released MedMNIST weights)
    - StandardScaler fit on train; SVC(kernel="rbf", C=1, gamma="scale"), fixed, fit on train
    - report val (what the feature-selection fitness sees) and test (used once)

Metrics (5 classes):
    acc       accuracy
    auc       macro one-vs-rest AUC from per-class scores, computed exactly as medmnist.Evaluator does
              (mean over classes of AUC(y == c, score[:, c])); SVM scores = decision_function (ovr)
    macro_f1  macro-averaged F1
    qwk       quadratic weighted Cohen's kappa (the labels are ordinal grades)
The original ResNet-18 head is scored the same way from its saved softmax probabilities.

Usage (from project root):
    .venv\\Scripts\\python.exe Retina\\svm_baseline\\svm_all_features.py
"""

import csv
import json
import os

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RETINA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_FILES = [os.path.join(RETINA, "features", f"resnet18_28_{i}.npz") for i in (1, 2, 3)]
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
N_CLASSES = 5
SVM_PARAMS = {"kernel": "rbf", "C": 1.0, "gamma": "scale", "decision_function_shape": "ovr"}
METRICS = ("acc", "auc", "macro_f1", "qwk")


def macro_ovr_auc(y, score):
    return float(np.mean([roc_auc_score(y == c, score[:, c]) for c in range(N_CLASSES)]))


def evaluate(y_true, y_pred, score):
    return {"acc": accuracy_score(y_true, y_pred),
            "auc": macro_ovr_auc(y_true, score),
            "macro_f1": f1_score(y_true, y_pred, average="macro", labels=list(range(N_CLASSES)), zero_division=0),
            "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic")}


def main():
    rows, test = {"resnet18_head": [], "svm_all_512": []}, []
    for path in FEATURE_FILES:
        name = os.path.splitext(os.path.basename(path))[0]
        d = np.load(path)
        sc = StandardScaler().fit(d["X_train"])
        clf = SVC(**SVM_PARAMS).fit(sc.transform(d["X_train"]), d["y_train"])
        print(f"==> {name}")
        for split in ("train", "val", "test"):
            X = sc.transform(d[f"X_{split}"])
            for model, res in (("resnet18_head", evaluate(d[f"y_{split}"], d[f"prob_{split}"].argmax(1), d[f"prob_{split}"])),
                               ("svm_all_512", evaluate(d[f"y_{split}"], clf.predict(X), clf.decision_function(X)))):
                test.append({"weights": name, "model": model, "split": split, **{m: round(res[m], 5) for m in METRICS}})
                if split == "test":
                    rows[model].append(res)
                if split != "train":
                    print(f"    {model:14} {split:5} " + "  ".join(f"{m} {res[m]:.4f}" for m in METRICS))

    with open(os.path.join(OUT_DIR, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(test[0]))
        w.writeheader()
        w.writerows(test)
    summary = {"protocol": {k: v for k, v in SVM_PARAMS.items()}, "test_mean_std": {}}
    print("\n==> Test, mean ± std over the 3 released checkpoints")
    for model, res in rows.items():
        s = {m: {"mean": round(float(np.mean([r[m] for r in res])), 5),
                 "std": round(float(np.std([r[m] for r in res], ddof=1)), 5)} for m in METRICS}
        summary["test_mean_std"][model] = s
        print(f"    {model:14} " + "  ".join(f"{m} {v['mean']:.4f}±{v['std']:.4f}" for m, v in s.items()))
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
