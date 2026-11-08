"""RetinaMNIST Study R2: HHO and CSO feature selection with a 1,200-image cross-validated fitness (GPU).

Design fixed in Retina/README.md ("Study R2") before any R2 test result:
    fitness  0.99 * (1 - 5-fold CV ACC on train + val pooled) + 0.01 * k / 512, threshold encoding > 0.5,
             CV partition random_state = seed (shared by all methods), 5-class GPU LS-SVM, batched folds
    test     best mask -> LS-SVM trained on all 1,200 -> scored once on the 400 test images
    methods  ALL512, RANDOM, GA (secondary), HHO (mealpy OriginalHHO, epoch 27), CSO (own implementation)
    budget   exactly 1000 evaluations; 3 feature sets x 10 seeds; seed = 100 * set + run

Results -> Retina/feature_selection/results_r2_gpu/runs.jsonl (resumable); worker processes share the GPU.

Usage (from project root):
    .venv\\Scripts\\python.exe Retina\\feature_selection\\run_r2_benchmark.py --workers 8
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RETINA = os.path.dirname(HERE)
ROOT = os.path.dirname(RETINA)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "feature_selection"))
sys.path.insert(0, os.path.join(RETINA, "svm_baseline"))
from svm_all_features import METRICS, evaluate  # noqa: E402

_spec = importlib.util.spec_from_file_location("breast_run_benchmark", os.path.join(ROOT, "feature_selection", "run_benchmark.py"))
_breast = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_breast)
BudgetExhausted, FSProblem = _breast.BudgetExhausted, _breast.FSProblem

FEATURE_SETS = {i: os.path.join(RETINA, "features", f"resnet18_28_{i}.npz") for i in (1, 2, 3)}
OUT = os.path.join(HERE, "results_r2_gpu", "runs.jsonl")
N_FEATURES, N_CLASSES, FOLDS = 512, 5, 5
METHODS = ["ALL512", "RANDOM", "GA", "HHO", "CSO"]
HHO_EPOCHS = 27         # ~37 evaluations per iteration -> ~1,019 natural evaluations, capped at 1,000
CSO_PHI = 0.1


class CVFitness:
    def __init__(self, X, y, splits, alpha, scorer):
        self.X, self.y, self.alpha, self.scorer = X, y, alpha, scorer
        self._cache, self.n_evals, self.n_fits = {}, 0, 0
        self.best_fitness, self.best_mask, self.history = np.inf, None, []

    def cv_acc(self, mask):
        key = np.packbits(mask).tobytes()
        if key not in self._cache:
            self._cache[key] = self.scorer.accuracy(mask)
            self.n_fits += 1
        return self._cache[key]

    def fitness_of_mask(self, mask):
        k = int(mask.sum())
        if k == 0:
            return 1.0
        return self.alpha * (1.0 - self.cv_acc(mask)) + (1.0 - self.alpha) * k / N_FEATURES

    def __call__(self, x):
        mask = np.asarray(x, dtype=float) > 0.5
        f = self.fitness_of_mask(mask)
        self.n_evals += 1
        if f < self.best_fitness:
            self.best_fitness, self.best_mask = f, mask.copy()
        self.history.append(self.best_fitness)
        return f


def run_cso(fn, budget, pop_size, rng, phi=CSO_PHI):
    """Competitive Swarm Optimiser (Cheng & Jin 2015) on [0, 1]^D; stops by raising BudgetExhausted."""
    assert pop_size % 2 == 0
    X = rng.random((pop_size, N_FEATURES))
    V = np.zeros_like(X)
    fit = np.array([fn(x) for x in X])
    while True:
        mean = X.mean(0)
        perm = rng.permutation(pop_size)
        for i, j in zip(perm[0::2], perm[1::2]):
            w, l = (i, j) if fit[i] <= fit[j] else (j, i)
            r1, r2, r3 = rng.random(N_FEATURES), rng.random(N_FEATURES), rng.random(N_FEATURES)
            V[l] = r1 * V[l] + r2 * (X[w] - X[l]) + phi * r3 * (mean - X[l])
            X[l] = np.clip(X[l] + V[l], 0.0, 1.0)
            fit[l] = fn(X[l])


def run_job(job):
    import torch
    from gpu_lssvm import BatchedFoldLSSVMOvR, LSSVMOvR
    from mealpy import GA, HHO

    method, fs, run, budget, pop_size, alpha = job
    seed = 100 * fs + run
    d = np.load(FEATURE_SETS[fs])
    X = np.vstack([d["X_train"], d["X_val"]])
    y = np.concatenate([d["y_train"], d["y_val"]])
    sc = StandardScaler().fit(X)
    X, Xte, yte = sc.transform(X).astype(np.float32), sc.transform(d["X_test"]).astype(np.float32), d["y_test"]
    splits = list(StratifiedKFold(FOLDS, shuffle=True, random_state=seed).split(X, y))
    dev = torch.device("cuda")
    Xt, yt, Xte_t = torch.tensor(X, device=dev), torch.tensor(y, device=dev), torch.tensor(Xte, device=dev)
    fit = CVFitness(X, y, splits, alpha, BatchedFoldLSSVMOvR(Xt, yt, splits, N_CLASSES))

    def capped(x):
        if fit.n_evals >= budget:
            raise BudgetExhausted
        return fit(x)

    t = time.perf_counter()
    try:
        if method == "ALL512":
            fit(np.ones(N_FEATURES))
        elif method == "RANDOM":
            rng = np.random.default_rng(seed)
            for _ in range(budget):
                capped(rng.random(N_FEATURES))
        elif method == "CSO":
            run_cso(capped, budget, pop_size, np.random.default_rng(seed))
        else:
            model = (GA.BaseGA((budget - pop_size) // pop_size, pop_size) if method == "GA"
                     else HHO.OriginalHHO(HHO_EPOCHS, pop_size))
            model.solve(FSProblem(capped), seed=seed)
    except BudgetExhausted:
        pass
    runtime = time.perf_counter() - t

    mask = fit.best_mask
    m = torch.as_tensor(mask, device=dev)
    score = LSSVMOvR(N_CLASSES).fit(Xt[:, m], yt).decision_function(Xte_t[:, m]).cpu().numpy()
    test = evaluate(yte, score.argmax(1), score)
    return {"method": method, "feature_set": fs, "run": run, "seed": seed, "budget": budget, "alpha": alpha,
            "n_evals": fit.n_evals, "n_fits": fit.n_fits, "runtime_s": round(runtime, 2),
            "best_fitness": round(float(fit.best_fitness), 6), "cv_acc": round(fit.cv_acc(mask), 5),
            "n_features": int(mask.sum()), "pct_features": round(100 * mask.sum() / N_FEATURES, 2),
            **{f"test_{k}": round(float(test[k]), 5) for k in METRICS},
            "mask_hex": np.packbits(mask).tobytes().hex(), "history": [round(float(h), 6) for h in fit.history]}


def main(args):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            done = {(r["method"], r["feature_set"], r["run"]) for r in map(json.loads, f)}
    jobs = [(m, fs, run, args.budget, args.pop_size, args.alpha)
            for m in args.methods for fs in (1, 2, 3) for run in range(1 if m == "ALL512" else args.runs)
            if (m, fs, run) not in done]
    print(f"{len(jobs)} jobs to run ({len(done)} already done), {args.workers} workers -> {OUT}", flush=True)
    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_job, jobs), 1):
            with open(OUT, "a") as f:
                f.write(json.dumps(r) + "\n")
            flag = "" if r["method"] == "ALL512" or r["n_evals"] == args.budget else f"  !! {r['n_evals']} evals"
            print(f"[{i}/{len(jobs)}] {r['method']:6} set {r['feature_set']} run {r['run']}: CV {r['cv_acc']:.4f} "
                  f"| {r['n_features']:3d} feats | test acc {r['test_acc']:.4f} auc {r['test_auc']:.4f} "
                  f"| {r['runtime_s']:.0f}s | elapsed {(time.perf_counter() - t0) / 60:.1f} min{flag}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="RetinaMNIST Study R2: HHO and CSO with a CV fitness on the GPU")
    p.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--budget", type=int, default=1000)
    p.add_argument("--pop_size", type=int, default=20)
    p.add_argument("--alpha", type=float, default=0.99)
    p.add_argument("--workers", type=int, default=8)
    main(p.parse_args())
