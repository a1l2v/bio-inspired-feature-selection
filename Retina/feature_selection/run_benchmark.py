"""RetinaMNIST feature-selection benchmark: GA and GWO against random search and all 512 features.

Same protocol as feature_selection/run_benchmark.py (BreastMNIST Study A), adapted to 5 classes; the design
is fixed in Retina/README.md. Fitness = 0.99 * (1 - val ACC) + 0.01 * k / 512 on the 120 validation images,
threshold encoding (> 0.5), exactly 1000 evaluations per run, SVM fit on train, test scored once per run.
The mealpy wrappers (FSProblem, BudgetExhausted, algorithm constructors) are imported from the BreastMNIST
runner so the search code is identical.

Classifier (--backend): gpu (default) = one-vs-rest least-squares SVM on the GPU (gpu_lssvm.py), RBF,
gamma "scale", C = 1, one linear solve per evaluation; cpu = scikit-learn SVC (the original protocol).
Results go to results_gpu_lssvm/ or results_cpu_svc/. Jobs run in parallel worker processes that share the
GPU; finished jobs are skipped on re-launch.

Usage (from project root):
    .venv\\Scripts\\python.exe Retina\\feature_selection\\run_benchmark.py                   # ALL512, RANDOM, GA, GWO
    .venv\\Scripts\\python.exe Retina\\feature_selection\\run_benchmark.py --methods GA --workers 10
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
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RETINA = os.path.dirname(HERE)
ROOT = os.path.dirname(RETINA)
sys.path.insert(0, HERE)                                       # gpu_lssvm
sys.path.insert(0, os.path.join(ROOT, "feature_selection"))   # for the BreastMNIST runner's own imports
sys.path.insert(0, os.path.join(RETINA, "svm_baseline"))
from svm_all_features import METRICS, SVM_PARAMS, evaluate  # noqa: E402

# Same file name as this script, so load the BreastMNIST runner by path under another module name.
_spec = importlib.util.spec_from_file_location("breast_run_benchmark",
                                               os.path.join(ROOT, "feature_selection", "run_benchmark.py"))
_breast = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_breast)
BREAST_ALGORITHMS, BudgetExhausted, FSProblem = _breast.ALGORITHMS, _breast.BudgetExhausted, _breast.FSProblem

FEATURE_SETS = {i: os.path.join(RETINA, "features", f"resnet18_28_{i}.npz") for i in (1, 2, 3)}
OUT = {"gpu": os.path.join(HERE, "results_gpu_lssvm", "runs.jsonl"),
       "cpu": os.path.join(HERE, "results_cpu_svc", "runs.jsonl")}
N_FEATURES = 512
ALGORITHMS = {m: BREAST_ALGORITHMS[m] for m in ("GA", "GWO")}
METHODS = ["ALL512", "RANDOM", "GA", "GWO"]


class Fitness:
    def __init__(self, d, alpha, backend="gpu"):
        sc = StandardScaler().fit(d["X_train"])
        self.Xtr, self.Xva, self.Xte = (sc.transform(d[f"X_{s}"]) for s in ("train", "val", "test"))
        self.ytr, self.yva, self.yte = d["y_train"], d["y_val"], d["y_test"]
        self.alpha, self.backend = alpha, backend
        if backend == "gpu":
            import torch
            from gpu_lssvm import LSSVMOvR
            self.torch, self.Model = torch, LSSVMOvR
            dev = torch.device("cuda")
            self.t = {s: torch.tensor(getattr(self, f"X{s}"), dtype=torch.float32, device=dev) for s in ("tr", "va", "te")}
            self.ytr_t = torch.tensor(self.ytr, device=dev)
        self._cache, self.n_evals, self.n_fits = {}, 0, 0
        self.best_fitness, self.best_mask, self.history = np.inf, None, []

    def _gpu_model(self, mask):
        m = self.torch.as_tensor(mask, device=self.ytr_t.device)
        return self.Model(5).fit(self.t["tr"][:, m], self.ytr_t), m

    def val_acc(self, mask):
        key = np.packbits(mask).tobytes()
        if key not in self._cache:
            if self.backend == "gpu":
                clf, m = self._gpu_model(mask)
                pred = clf.predict(self.t["va"][:, m]).cpu().numpy()
            else:
                pred = SVC(**SVM_PARAMS).fit(self.Xtr[:, mask], self.ytr).predict(self.Xva[:, mask])
            self._cache[key] = float((pred == self.yva).mean())
            self.n_fits += 1
        return self._cache[key]

    def fitness_of_mask(self, mask):
        k = int(mask.sum())
        if k == 0:
            return 1.0
        return self.alpha * (1.0 - self.val_acc(mask)) + (1.0 - self.alpha) * k / N_FEATURES

    def __call__(self, x):
        mask = np.asarray(x, dtype=float) > 0.5
        f = self.fitness_of_mask(mask)
        self.n_evals += 1
        if f < self.best_fitness:
            self.best_fitness, self.best_mask = f, mask.copy()
        self.history.append(self.best_fitness)
        return f

    def evaluate_test(self, mask):
        if self.backend == "gpu":
            clf, m = self._gpu_model(mask)
            score = clf.decision_function(self.t["te"][:, m]).cpu().numpy()
            return evaluate(self.yte, score.argmax(1), score)
        clf = SVC(**SVM_PARAMS).fit(self.Xtr[:, mask], self.ytr)
        return evaluate(self.yte, clf.predict(self.Xte[:, mask]), clf.decision_function(self.Xte[:, mask]))


def run_job(job):
    method, fs, run, budget, pop_size, alpha, backend = job
    seed = 100 * fs + run
    fit = Fitness(np.load(FEATURE_SETS[fs]), alpha, backend)

    def capped(x):
        if fit.n_evals >= budget:
            raise BudgetExhausted
        return fit(x)

    t = time.perf_counter()
    if method == "ALL512":
        fit(np.ones(N_FEATURES))
    elif method == "RANDOM":
        rng = np.random.default_rng(seed)
        for _ in range(budget):
            capped(rng.random(N_FEATURES))
    else:
        model = ALGORITHMS[method]((budget - pop_size) // pop_size, pop_size)
        try:
            model.solve(FSProblem(capped), seed=seed)
        except BudgetExhausted:
            pass
    runtime = time.perf_counter() - t
    mask = fit.best_mask
    test = fit.evaluate_test(mask)
    return {"method": method, "feature_set": fs, "run": run, "seed": seed, "budget": budget, "alpha": alpha,
            "backend": backend, "n_evals": fit.n_evals, "n_fits": fit.n_fits, "runtime_s": round(runtime, 2),
            "best_fitness": round(float(fit.best_fitness), 6), "val_acc": round(fit.val_acc(mask), 5),
            "n_features": int(mask.sum()), "pct_features": round(100 * mask.sum() / N_FEATURES, 2),
            **{f"test_{m}": round(float(test[m]), 5) for m in METRICS},
            "mask_hex": np.packbits(mask).tobytes().hex(), "history": [round(float(h), 6) for h in fit.history]}


def main(args):
    out = OUT[args.backend]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = set()
    if os.path.exists(out):
        with open(out) as f:
            done = {(r["method"], r["feature_set"], r["run"]) for r in map(json.loads, f)}
    jobs = [(m, fs, run, args.budget, args.pop_size, args.alpha, args.backend)
            for m in args.methods for fs in (1, 2, 3) for run in range(1 if m == "ALL512" else args.runs)
            if (m, fs, run) not in done]
    print(f"{len(jobs)} jobs to run ({len(done)} already done), backend {args.backend}, {args.workers} workers "
          f"-> {out}", flush=True)
    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_job, jobs), 1):
            with open(out, "a") as f:
                f.write(json.dumps(r) + "\n")
            flag = "" if r["method"] == "ALL512" or r["n_evals"] == args.budget else f"  !! {r['n_evals']} evals"
            print(f"[{i}/{len(jobs)}] {r['method']:6} set {r['feature_set']} run {r['run']}: val {r['val_acc']:.4f} "
                  f"| {r['n_features']:3d} feats | test acc {r['test_acc']:.4f} auc {r['test_auc']:.4f} "
                  f"| {r['runtime_s']:.0f}s | elapsed {(time.perf_counter() - t0) / 60:.1f} min{flag}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="RetinaMNIST feature-selection benchmark (GA, GWO vs controls)")
    p.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--budget", type=int, default=1000)
    p.add_argument("--pop_size", type=int, default=20)
    p.add_argument("--alpha", type=float, default=0.99)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--backend", default="gpu", choices=["gpu", "cpu"],
                   help="gpu: one-vs-rest LS-SVM on the GPU (default); cpu: scikit-learn SVC")
    main(p.parse_args())
