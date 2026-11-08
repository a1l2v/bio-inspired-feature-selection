"""Option 2 benchmark: feature selection on frozen ImageNet ResNet-18 features with nested CV.

Design (fixed before any outer-test fold was scored; see docs/project_context.md, step 13):
    outer   stratified 5-fold x 6 repeats over all 780 images = 30 outer folds (random_state = repeat)
    inner   fitness = 0.99 * (1 - inner 5-fold CV ACC on the outer-train part) + 0.01 * k / 512,
            inner partition fixed per outer fold (random_state = seed) and shared by every method
    search  RANDOM, PSO, GA, GWO, WOA, exactly 1000 evaluations each (20 x 50), threshold encoding > 0.5
    ref     ALL512 = SVM on all 512 features (no selection)
    SVM     StandardScaler fit on outer-train, SVC(rbf, C=1, gamma="scale") - same as the main benchmark
    seeds   seed = 1000 * repeat + fold, identical across methods -> results are paired by outer fold
The outer-test fold is scored once per (method, outer fold), on the best mask found.

Backends (--backend):
    gpu  (default) GPU least-squares SVM, all inner folds in one batched solve (gpu_lssvm.py);
         results in imagenet_nested_cv/results_gpu_lssvm/runs.jsonl
    cpu  original scikit-learn SVC; results in imagenet_nested_cv/results_cpu_svc/runs.jsonl
         (kept from before the 1 Nov 2026 design amendment; see docs/project_context.md, step 13)

Fixed-size mode (--k K, step 14): every method keeps exactly K features (agent -> its K largest
coordinates), fitness = 1 - inner-CV ACC, results in results_<backend>_k<K>/. Random search also scores
its first random K-subset on the outer test fold ("single_test_*"), the no-search control.

Jobs run in parallel worker processes (one BLAS thread each; with --backend gpu every worker drives the
same GPU). Finished jobs are skipped on re-launch.

Usage (from project root):
    .venv\\Scripts\\python.exe imagenet_nested_cv\\run_nested_cv.py --workers 6
    .venv\\Scripts\\python.exe imagenet_nested_cv\\run_nested_cv.py --methods PSO --backend cpu --workers 10
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")    # one BLAS thread per worker process
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "feature_selection"))
sys.path.insert(0, os.path.join(ROOT, "svm_baseline"))
from run_benchmark import ALGORITHMS, BudgetExhausted, FSProblem  # noqa: E402
from svm_all_features import SVM_PARAMS, evaluate  # noqa: E402

FEATURES = os.path.join(ROOT, "features", "imagenet_resnet18.npz")
OUT = {"gpu": os.path.join(HERE, "results_gpu_lssvm", "runs.jsonl"),
       "cpu": os.path.join(HERE, "results_cpu_svc", "runs.jsonl")}
N_FEATURES = 512
OUTER_FOLDS, INNER_FOLDS = 5, 5
METHODS = ["ALL512", "RANDOM", "PSO", "GA", "GWO", "WOA"]
METRICS = ("acc", "auc", "macro_f1", "malignant_recall", "specificity")


class NestedCVFitness:
    """Same interface as feature_selection/fitness.py, but scored by inner CV on the outer-train part."""

    def __init__(self, X, y, inner_splits, alpha=0.99, gpu_scorer=None, k=None):
        self.X, self.y, self.splits, self.alpha, self.k = X, y, inner_splits, alpha, k
        self.gpu_scorer = gpu_scorer       # BatchedFoldLSSVM for --backend gpu, else None (sklearn SVC)
        self._cache = {}
        self.n_evals, self.n_fits = 0, 0
        self.best_fitness, self.best_mask, self.history = np.inf, None, []

    def cv_acc(self, mask):
        key = np.packbits(mask).tobytes()
        if key not in self._cache:
            if self.gpu_scorer is not None:
                self._cache[key] = self.gpu_scorer.accuracy(mask)
                self.n_fits += len(self.splits)
                return self._cache[key]
            correct = 0
            for a, b in self.splits:
                clf = SVC(**SVM_PARAMS).fit(self.X[a][:, mask], self.y[a])
                correct += int((clf.predict(self.X[b][:, mask]) == self.y[b]).sum())
                self.n_fits += 1
            self._cache[key] = correct / len(self.y)
        return self._cache[key]

    def fitness_of_mask(self, mask):
        k = int(mask.sum())
        if k == 0:
            return 1.0
        return self.alpha * (1.0 - self.cv_acc(mask)) + (1.0 - self.alpha) * k / N_FEATURES

    def to_mask(self, x):
        x = np.asarray(x, dtype=float)
        if self.k is None:
            return x > 0.5
        mask = np.zeros(N_FEATURES, dtype=bool)          # fixed size: keep the k largest coordinates
        mask[np.argsort(x, kind="stable")[-self.k:]] = True
        return mask

    def __call__(self, x):
        mask = self.to_mask(x)
        f = self.fitness_of_mask(mask)
        self.n_evals += 1
        if f < self.best_fitness:
            self.best_fitness, self.best_mask = f, mask.copy()
        self.history.append(self.best_fitness)
        return f


def outer_split(X, y, repeat, fold):
    splits = list(StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=repeat).split(X, y))
    return splits[fold]


def run_job(job):
    method, repeat, fold, budget, pop_size, alpha, backend, k = job
    seed = 1000 * repeat + fold
    d = np.load(FEATURES)
    X, y = d["X"], d["y"]
    tr, te = outer_split(X, y, repeat, fold)
    scaler = StandardScaler().fit(X[tr])
    Xtr, Xte, ytr, yte = scaler.transform(X[tr]), scaler.transform(X[te]), y[tr], y[te]
    inner = list(StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=seed).split(Xtr, ytr))
    if backend == "gpu":
        import torch
        from gpu_lssvm import BatchedFoldLSSVM, GPULSSVM
        dev = torch.device("cuda")
        Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=dev)
        ytr_t = torch.tensor(ytr, device=dev)
        fit = NestedCVFitness(Xtr, ytr, inner, alpha, gpu_scorer=BatchedFoldLSSVM(Xtr_t, ytr_t, inner), k=k)
    else:
        fit = NestedCVFitness(Xtr, ytr, inner, alpha, k=k)

    def score_test(mask):
        if backend == "gpu":
            m_t = torch.as_tensor(mask, device=dev)
            clf = GPULSSVM().fit(Xtr_t[:, m_t], ytr_t)
            score = clf.decision_function(torch.tensor(Xte, dtype=torch.float32, device=dev)[:, m_t]).cpu().numpy()
            return evaluate(yte, (score > 0).astype(int), score)
        clf = SVC(**SVM_PARAMS).fit(Xtr[:, mask], ytr)
        return evaluate(yte, clf.predict(Xte[:, mask]), clf.decision_function(Xte[:, mask]))

    def capped(x):
        if fit.n_evals >= budget:
            raise BudgetExhausted
        return fit(x)

    extra = {}
    t = time.perf_counter()
    if method == "ALL512":
        mask = np.ones(N_FEATURES, dtype=bool)
        fit.n_evals += 1
        fit.best_fitness, fit.best_mask = fit.fitness_of_mask(mask), mask
        fit.history.append(fit.best_fitness)
    elif method == "RANDOM":
        rng = np.random.default_rng(seed)
        first = rng.random(N_FEATURES)
        capped(first)
        for _ in range(budget - 1):
            capped(rng.random(N_FEATURES))
        mask = fit.best_mask
        if k:
            # No-search control: the first random k-subset drawn, scored once on the outer test fold.
            single = score_test(fit.to_mask(first))
            extra = {f"single_test_{m}": round(float(single[m]), 5) for m in METRICS}
    else:
        model = ALGORITHMS[method]((budget - pop_size) // pop_size, pop_size)
        try:
            model.solve(FSProblem(capped), seed=seed)
        except BudgetExhausted:
            pass
        mask = fit.best_mask
    runtime = time.perf_counter() - t

    test = score_test(mask)
    return {"method": method, "repeat": repeat, "fold": fold, "seed": seed, "budget": budget, "alpha": alpha,
            "backend": backend, **({"k": k} if k else {}), **extra,
            "n_train": int(len(ytr)), "n_test": int(len(yte)),
            "n_evals": fit.n_evals, "n_fits": fit.n_fits, "runtime_s": round(runtime, 2),
            "best_fitness": round(float(fit.best_fitness), 6), "inner_cv_acc": round(fit.cv_acc(mask), 5),
            "n_features": int(mask.sum()), "pct_features": round(100 * mask.sum() / N_FEATURES, 2),
            **{f"test_{m}": round(float(test[m]), 5) for m in METRICS}, "test_confusion": test["confusion"],
            "mask_hex": np.packbits(mask).tobytes().hex(),
            "history": [round(float(h), 6) for h in fit.history]}


def main(args):
    out = OUT[args.backend]
    if args.k:
        out = out.replace(os.sep + "runs.jsonl", f"_k{args.k}" + os.sep + "runs.jsonl")
        if args.alpha != 1.0:
            print(f"--k {args.k}: subset size is fixed, so alpha is set to 1.0 (fitness = 1 - inner-CV ACC)")
            args.alpha = 1.0
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = set()
    if os.path.exists(out):
        with open(out) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["method"], r["repeat"], r["fold"], r["budget"], r["alpha"]))
    jobs = [(m, rep, fold, args.budget, args.pop_size, args.alpha, args.backend, args.k)
            for m in args.methods for rep in range(args.repeats) for fold in range(OUTER_FOLDS)
            if (m, rep, fold, args.budget, args.alpha) not in done]
    print(f"{len(jobs)} jobs to run ({len(done)} already done), backend {args.backend}, "
          f"{args.workers} workers -> {out}", flush=True)
    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_job, jobs), 1):
            with open(out, "a") as f:
                f.write(json.dumps(r) + "\n")
            flag = "" if r["method"] == "ALL512" or r["n_evals"] == args.budget else f"  !! {r['n_evals']} evals"
            print(f"[{i}/{len(jobs)}] {r['method']:6} rep {r['repeat']} fold {r['fold']}: inner-CV {r['inner_cv_acc']:.4f} "
                  f"| {r['n_features']:3d} feats | test acc {r['test_acc']:.4f} auc {r['test_auc']:.4f} "
                  f"| {r['runtime_s']:.0f}s | elapsed {(time.perf_counter() - t0) / 60:.1f} min{flag}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Nested-CV feature-selection benchmark on ImageNet ResNet-18 features")
    p.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    p.add_argument("--repeats", type=int, default=6)
    p.add_argument("--budget", type=int, default=1000)
    p.add_argument("--pop_size", type=int, default=20)
    p.add_argument("--alpha", type=float, default=0.99)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--backend", default="gpu", choices=["gpu", "cpu"])
    p.add_argument("--k", type=int, default=None,
                   help="keep exactly k features (top-k decoding, alpha forced to 1.0); results go to *_k<K>/")
    main(p.parse_args())
