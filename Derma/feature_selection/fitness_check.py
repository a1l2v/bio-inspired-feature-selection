"""Design check for Study D1 (training images only; val and test untouched).

1. Speed: seconds per fitness evaluation (5-fold CV LS-SVM, k = 50, on a 3,000-image stratified pool).
2. Validity: does the pool CV accuracy rank 50-feature subsets like an independent estimate? For 60 subsets
   (20 uniform random, 20 random from the per-pool ANOVA top-200, 20 random from the colour groups), compare
   pool CV ACC with the accuracy of an LS-SVM trained on the pool and scored on the 4,007 training images
   outside it. Spearman rho is reported overall and within the uniform-random group.

Output: Derma/feature_selection/fitness_check.json

Usage (from project root):
    .venv\\Scripts\\python.exe Derma\\feature_selection\\fitness_check.py
"""

import json
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DERMA = os.path.dirname(HERE)
ROOT = os.path.dirname(DERMA)
sys.path.insert(0, os.path.join(ROOT, "Retina", "feature_selection"))
from gpu_lssvm import BatchedFoldLSSVMOvR, LSSVMOvR  # noqa: E402

torch.backends.cuda.preferred_linalg_library("cusolver")
FEATS = os.path.join(DERMA, "features", "radiomics_128.npz")
POOL, K, C = 3000, 50, 7


def main():
    d = np.load(FEATS)
    names = json.load(open(FEATS.replace(".npz", ".json")))["feature_names"]
    X, y = StandardScaler().fit_transform(d["X_train"]).astype(np.float32), d["y_train"]
    pool, rest = train_test_split(np.arange(len(y)), train_size=POOL, stratify=y, random_state=0)
    splits = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X[pool], y[pool]))
    dev = torch.device("cuda")
    Xp, yp = torch.tensor(X[pool], device=dev), torch.tensor(y[pool], device=dev)
    Xr, yr = torch.tensor(X[rest], device=dev), torch.tensor(y[rest], device=dev)
    scorer = BatchedFoldLSSVMOvR(Xp, yp, splits, C)

    rng = np.random.default_rng(0)
    n = X.shape[1]
    F, _ = f_classif(X[pool], y[pool])
    top200 = np.argsort(-np.nan_to_num(F))[:200]
    colour = np.array([i for i, nm in enumerate(names) if nm.split("_")[0] in ("col", "chist", "cgrid", "cr")])
    sources = {"random": np.arange(n), "anova_top200": top200, "colour": colour}

    torch.cuda.synchronize()
    m = np.zeros(n, bool)
    m[rng.choice(n, K, replace=False)] = True
    scorer.accuracy(m)                                          # warm-up
    t = time.perf_counter()
    for _ in range(20):
        m = np.zeros(n, bool)
        m[rng.choice(n, K, replace=False)] = True
        scorer.accuracy(m)
    torch.cuda.synchronize()
    sec = (time.perf_counter() - t) / 20
    print(f"fitness evaluation: {sec * 1000:.0f} ms (pool {POOL}, k {K})", flush=True)

    rows = []
    for src, idx in sources.items():
        for _ in range(20):
            m = np.zeros(n, bool)
            m[rng.choice(idx, K, replace=False)] = True
            mt = torch.as_tensor(m, device=dev)
            held = float((LSSVMOvR(C).fit(Xp[:, mt], yp).predict(Xr[:, mt]) == yr).float().mean())
            rows.append({"source": src, "cv": scorer.accuracy(m), "held_out": held})
    cv, ho = np.array([r["cv"] for r in rows]), np.array([r["held_out"] for r in rows])
    rnd = np.array([r["source"] == "random" for r in rows])
    res = {"pool": POOL, "k": K, "sec_per_eval": sec,
           "rho_all": float(spearmanr(cv, ho)[0]), "rho_random": float(spearmanr(cv[rnd], ho[rnd])[0]),
           "by_source": {s: {"cv_mean": float(cv[[r["source"] == s for r in rows]].mean()),
                             "held_out_mean": float(ho[[r["source"] == s for r in rows]].mean())} for s in sources},
           "rows": rows}
    for s, v in res["by_source"].items():
        print(f"{s:13s}: pool CV {v['cv_mean']:.4f}  held-out {v['held_out_mean']:.4f}")
    print(f"Spearman rho: all 60 = {res['rho_all']:.3f}, random 20 = {res['rho_random']:.3f}")
    with open(os.path.join(HERE, "fitness_check.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
