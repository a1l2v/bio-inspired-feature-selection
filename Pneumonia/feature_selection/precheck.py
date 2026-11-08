"""Pre-check: is there headroom for feature selection on the PneumoniaMNIST radiomics features?

Uses the 4,708 TRAINING images only (val and test untouched), 5-fold stratified CV (random_state 0), features
standardised on the training set. Classifiers on the GPU: LS-SVM (RBF, gamma = "scale", C = 1; primary, as in
the Retina studies) and 5-NN (Euclidean, majority vote; secondary).

Measured (CV accuracy, pooled over folds):
    A  all 1,359 features
    B  random subsets, k in 25/50/100/200/400/680, 20 draws each (draw seed = 1000 * k + i)
    C  ANOVA-F filter top-k, ranking recomputed inside each fold's training part (no leakage), same k
    D  each feature group alone (fo, hist, ..., fft)

Decision rule, fixed before running:
    H1 (features not interchangeable): mean random subset at k <= 100 >= 1.0 point BELOW all features (SVM).
    H2 (headroom): best filter top-k or best single group >= 1.0 point ABOVE all features (SVM).
    GO for the full bio-inspired study if H2 holds; H1 is supporting evidence. KNN is reported, not decisive.
    (The SE of a 4,708-image CV accuracy near 0.9 is about 0.4 points.)

Output: Pneumonia/feature_selection/precheck/precheck.json, precheck.md

Generalised for other datasets (DermaMNIST): --features, --out, --n_classes. KNN predicts the majority class
of the 5 neighbours (identical to the binary rule used for the PneumoniaMNIST run). The last k is half the
number of features, rounded up (680 for 1,359). Groups are the name prefix ("grey_<group>" for grey features of colour data).
    .venv\\Scripts\\python.exe Pneumonia\\feature_selection\\precheck.py --features Derma\\features\\radiomics_128.npz --out Derma\\feature_selection\\precheck --n_classes 7

Usage (from project root):
    .venv\\Scripts\\python.exe Pneumonia\\feature_selection\\precheck.py
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
PNEU = os.path.dirname(HERE)
ROOT = os.path.dirname(PNEU)
sys.path.insert(0, os.path.join(ROOT, "Retina", "feature_selection"))
from gpu_lssvm import BatchedFoldLSSVMOvR, LSSVMOvR  # noqa: E402

FEATS = os.path.join(PNEU, "features", "radiomics_128.npz")
OUT = os.path.join(HERE, "precheck")
torch.backends.cuda.preferred_linalg_library("cusolver")    # MAGMA's batched solve prints a warning per call
DRAWS, KNN_K, FOLDS = 20, 5, 5


class BatchedFoldKNN:
    def __init__(self, X, y, splits, n_classes):
        self.X, self.y, self.C = X, y, n_classes
        self.splits = [(torch.as_tensor(a, device=X.device), torch.as_tensor(b, device=X.device)) for a, b in splits]

    @torch.no_grad()
    def accuracy(self, mask, per_fold=None):
        correct = 0
        for f, (tr, va) in enumerate(self.splits):
            m = torch.as_tensor(per_fold[f] if per_fold is not None else mask, device=self.X.device)
            Xm = self.X[:, m]
            d = torch.cdist(Xm[va], Xm[tr])
            nn = self.y[tr][d.topk(KNN_K, largest=False).indices]
            votes = torch.nn.functional.one_hot(nn, self.C).sum(1)
            correct += int((votes.argmax(1) == self.y[va]).sum())
        return correct / len(self.y)


def svm_per_fold(X, y, splits, masks, n_classes):
    correct = 0
    for (tr, va), mk in zip(splits, masks):
        m = torch.as_tensor(mk, device=X.device)
        tr, va = torch.as_tensor(tr, device=X.device), torch.as_tensor(va, device=X.device)
        pred = LSSVMOvR(n_classes).fit(X[tr][:, m], y[tr]).predict(X[va][:, m])
        correct += int((pred == y[va]).sum())
    return correct / len(y)


def group_of(name):
    parts = name.split("_")
    return f"grey_{parts[1]}" if parts[0] == "grey" else parts[0]


def main(feats, out, C):
    t0 = time.perf_counter()
    d = np.load(feats)
    names = json.load(open(feats.replace(".npz", ".json")))["feature_names"]
    X, y = StandardScaler().fit_transform(d["X_train"]).astype(np.float32), d["y_train"]
    n_feat = X.shape[1]
    KS = [25, 50, 100, 200, 400, (n_feat + 1) // 2]
    splits = list(StratifiedKFold(FOLDS, shuffle=True, random_state=0).split(X, y))
    dev = torch.device("cuda")
    Xt, yt = torch.tensor(X, device=dev), torch.tensor(y, device=dev)
    svm, knn = BatchedFoldLSSVMOvR(Xt, yt, splits, C), BatchedFoldKNN(Xt, yt, splits, C)

    # float32 sanity check: fold 0, all features, float32 vs float64 LS-SVM predictions
    tr, va = (torch.as_tensor(i, device=dev) for i in splits[0])
    p32 = LSSVMOvR(C).fit(Xt[tr], yt[tr]).predict(Xt[va])
    p64 = LSSVMOvR(C).fit(Xt[tr].double(), yt[tr]).predict(Xt[va].double())
    agree = float((p32 == p64).float().mean())
    print(f"float32 vs float64 LS-SVM agreement on fold 0: {agree:.4f}", flush=True)

    res = {"n_train": len(y), "n_features": n_feat, "class_counts": np.bincount(y).tolist(),
           "majority_acc": round(float(np.bincount(y).max() / len(y)), 4), "fp32_fp64_agreement": agree}
    allm = np.ones(n_feat, bool)
    res["all"] = {"svm": svm.accuracy(allm), "knn": knn.accuracy(allm)}
    print(f"A  all {n_feat}: SVM {res['all']['svm']:.4f}  KNN {res['all']['knn']:.4f}", flush=True)

    res["random"] = {}
    for k in KS:
        s, n = [], []
        for i in range(DRAWS):
            m = np.zeros(n_feat, bool)
            m[np.random.default_rng(1000 * k + i).choice(n_feat, k, replace=False)] = True
            s.append(svm.accuracy(m))
            n.append(knn.accuracy(m))
        res["random"][k] = {"svm_mean": float(np.mean(s)), "svm_std": float(np.std(s)), "svm_max": float(np.max(s)),
                            "knn_mean": float(np.mean(n)), "knn_std": float(np.std(n)), "knn_max": float(np.max(n))}
        print(f"B  random k={k:3d}: SVM {np.mean(s):.4f} ± {np.std(s):.4f} (max {np.max(s):.4f})  "
              f"KNN {np.mean(n):.4f} ± {np.std(n):.4f}", flush=True)

    ranks = []
    for tr_idx, _ in splits:
        F, _ = f_classif(X[tr_idx], y[tr_idx])
        ranks.append(np.argsort(-np.nan_to_num(F)))
    res["filter"] = {}
    for k in KS:
        masks = []
        for r in ranks:
            m = np.zeros(n_feat, bool)
            m[r[:k]] = True
            masks.append(m)
        res["filter"][k] = {"svm": svm_per_fold(Xt, yt, splits, masks, C), "knn": knn.accuracy(None, per_fold=masks)}
        print(f"C  filter top-{k:3d}: SVM {res['filter'][k]['svm']:.4f}  KNN {res['filter'][k]['knn']:.4f}", flush=True)

    res["groups"] = {}
    gname = np.array([group_of(n) for n in names])
    for g in dict.fromkeys(gname):
        m = gname == g
        res["groups"][g] = {"n": int(m.sum()), "svm": svm.accuracy(m), "knn": knn.accuracy(m)}
        print(f"D  group {g:10s} ({m.sum():3d}): SVM {res['groups'][g]['svm']:.4f}  KNN {res['groups'][g]['knn']:.4f}",
              flush=True)

    base = res["all"]["svm"]
    small = np.mean([res["random"][k]["svm_mean"] for k in KS if k <= 100])
    best_filter = max(res["filter"].items(), key=lambda kv: kv[1]["svm"])
    best_group = max(res["groups"].items(), key=lambda kv: kv[1]["svm"])
    best = max(best_filter[1]["svm"], best_group[1]["svm"])
    res["decision"] = {"all_svm": base, "random_le100_svm": small, "H1_gap_points": 100 * (base - small),
                       "H1": bool(base - small >= 0.01),
                       "best_filter": [best_filter[0], best_filter[1]["svm"]],
                       "best_group": [best_group[0], best_group[1]["svm"]],
                       "H2_gain_points": 100 * (best - base), "H2": bool(best - base >= 0.01)}
    res["runtime_s"] = round(time.perf_counter() - t0, 1)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "precheck.json"), "w") as f:
        json.dump(res, f, indent=1)
    dd = res["decision"]
    lines = [f"# {os.path.basename(os.path.dirname(os.path.dirname(feats)))} radiomics pre-check (train only, 5-fold CV ACC)", "",
             f"All {n_feat} features: SVM {base:.4f}, KNN {res['all']['knn']:.4f} (majority {res['majority_acc']})", "",
             "| k | Random SVM (mean ± std, max) | Filter SVM | Random KNN | Filter KNN |", "|---|---|---|---|---|"]
    for k in KS:
        r, fl = res["random"][k], res["filter"][k]
        lines.append(f"| {k} | {r['svm_mean']:.4f} ± {r['svm_std']:.4f}, {r['svm_max']:.4f} | {fl['svm']:.4f} "
                     f"| {r['knn_mean']:.4f} ± {r['knn_std']:.4f} | {fl['knn']:.4f} |")
    lines += ["", "| Group | n | SVM | KNN |", "|---|---|---|---|"]
    lines += [f"| {g} | {v['n']} | {v['svm']:.4f} | {v['knn']:.4f} |" for g, v in res["groups"].items()]
    lines += ["", f"- H1: random k<=100 is {dd['H1_gap_points']:+.2f} points below all -> {'holds' if dd['H1'] else 'fails'}",
              f"- H2: best subset ({'filter top-' + str(best_filter[0]) if best_filter[1]['svm'] >= best_group[1]['svm'] else 'group ' + best_group[0]})"
              f" is {dd['H2_gain_points']:+.2f} points above all -> {'holds (GO)' if dd['H2'] else 'fails (NO GO)'}"]
    with open(os.path.join(out, "precheck.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[-2:]))
    print(f"runtime {res['runtime_s']}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Feature-selection headroom pre-check on training data only")
    p.add_argument("--features", default=FEATS)
    p.add_argument("--out", default=OUT)
    p.add_argument("--n_classes", type=int, default=2)
    a = p.parse_args()
    main(os.path.abspath(a.features), os.path.abspath(a.out), a.n_classes)
