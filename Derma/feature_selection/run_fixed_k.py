"""DermaMNIST Studies D1 / D2: bio-inspired feature selection at a fixed budget of 50 radiomics features (GPU).

Design fixed in Derma/README.md ("Study D1") before any test result:
    per run  seed = run (0-29); scaler on train + val; fitness pool = stratified 3,000 of the 7,007 training
             images (random_state = seed); fitness = 1 - 5-fold CV ACC on the pool (random_state = seed),
             7-class GPU LS-SVM; decoding = top 50 of x in [0, 1]^1820
    budget   exactly 1,000 evaluations, population 20, hard cap
    test     LS-SVM on train + val (8,010) with the selected features, scored once on the 2,005 test images
    methods  ALL1820 (reference), FILTER (ANOVA-F top 50 on the pool), RANDOM, GA (mealpy BaseGA), CSO

Study D2 (Derma/README.md, "Study D2"): identical except the classifier, --classifier knn (5-NN, gpu_knn.py) in
both the fitness and the test model. The default (lssvm) reproduces D1 unchanged.

Results -> Derma/feature_selection/results_d1/runs.jsonl (D1) or results_d2/runs.jsonl (D2), resumable; worker
processes share the GPU.

Usage (from project root):
    .venv\\Scripts\\python.exe Derma\\feature_selection\\run_fixed_k.py --workers 4 [--classifier knn]
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.feature_selection import f_classif  # noqa: E402
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DERMA = os.path.dirname(HERE)
ROOT = os.path.dirname(DERMA)
sys.path.insert(0, os.path.join(ROOT, "Retina", "feature_selection"))
_spec = importlib.util.spec_from_file_location("breast_run_benchmark", os.path.join(ROOT, "feature_selection", "run_benchmark.py"))
_breast = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_breast)
BudgetExhausted = _breast.BudgetExhausted

FEATS = os.path.join(DERMA, "features", "radiomics_128.npz")
OUT = {"lssvm": os.path.join(HERE, "results_d1", "runs.jsonl"), "knn": os.path.join(HERE, "results_d2", "runs.jsonl")}
N_CLASSES, FOLDS, POOL, K = 7, 5, 3000, 50
METHODS = ["ALL1820", "FILTER", "RANDOM", "GA", "CSO"]
METRICS = ("acc", "auc", "macro_f1", "bal_acc")
CSO_PHI = 0.1


def evaluate(y, score):
    pred = score.argmax(1)
    return {"acc": accuracy_score(y, pred),
            "auc": float(np.mean([roc_auc_score(y == c, score[:, c]) for c in range(N_CLASSES)])),
            "macro_f1": f1_score(y, pred, average="macro", labels=list(range(N_CLASSES)), zero_division=0),
            "bal_acc": balanced_accuracy_score(y, pred)}


def topk_mask(x, k=K):
    mask = np.zeros(len(x), dtype=bool)
    mask[np.argsort(np.asarray(x, dtype=float), kind="stable")[-k:]] = True
    return mask


class Fitness:
    def __init__(self, scorer):
        self.scorer, self._cache = scorer, {}
        self.n_evals, self.n_fits, self.best_fitness, self.best_mask, self.history = 0, 0, np.inf, None, []

    def cv_acc(self, mask):
        key = np.packbits(mask).tobytes()
        if key not in self._cache:
            self._cache[key] = self.scorer.accuracy(mask)
            self.n_fits += 1
        return self._cache[key]

    def of_mask(self, mask):
        f = 1.0 - self.cv_acc(mask)
        self.n_evals += 1
        if f < self.best_fitness:
            self.best_fitness, self.best_mask = f, mask.copy()
        self.history.append(self.best_fitness)
        return f

    def __call__(self, x):
        return self.of_mask(topk_mask(x))


def run_cso(fn, dim, pop_size, rng, phi=CSO_PHI):
    """Competitive Swarm Optimiser (Cheng & Jin 2015) on [0, 1]^dim; stops by raising BudgetExhausted."""
    assert pop_size % 2 == 0
    X = rng.random((pop_size, dim))
    V = np.zeros_like(X)
    fit = np.array([fn(x) for x in X])
    while True:
        mean = X.mean(0)
        perm = rng.permutation(pop_size)
        for i, j in zip(perm[0::2], perm[1::2]):
            w, l = (i, j) if fit[i] <= fit[j] else (j, i)
            r1, r2, r3 = rng.random(dim), rng.random(dim), rng.random(dim)
            V[l] = r1 * V[l] + r2 * (X[w] - X[l]) + phi * r3 * (mean - X[l])
            X[l] = np.clip(X[l] + V[l], 0.0, 1.0)
            fit[l] = fn(X[l])


def run_job(job):
    import torch
    from gpu_knn import BatchedFoldKNN, knn_scores
    from gpu_lssvm import BatchedFoldLSSVMOvR, LSSVMOvR
    from mealpy import GA, FloatVar, Problem

    torch.backends.cuda.preferred_linalg_library("cusolver")
    warnings.filterwarnings("ignore")
    method, run, budget, pop_size, mem_fraction, classifier = job
    # each worker gets a hard share of the 8 GB card: running out raises an error instead of letting Windows
    # spill GPU memory into system RAM (which stalled the first launch with 8 workers)
    torch.cuda.set_per_process_memory_fraction(mem_fraction)
    torch.cuda.reset_peak_memory_stats()
    seed = run
    d = np.load(FEATS)
    Xtv = np.vstack([d["X_train"], d["X_val"]])
    ytv = np.concatenate([d["y_train"], d["y_val"]])
    sc = StandardScaler().fit(Xtv)
    Xtv, Xte, yte = sc.transform(Xtv).astype(np.float32), sc.transform(d["X_test"]).astype(np.float32), d["y_test"]
    dim, n_train = Xtv.shape[1], len(d["y_train"])
    pool, _ = train_test_split(np.arange(n_train), train_size=POOL, stratify=ytv[:n_train], random_state=seed)
    Xp, yp = Xtv[pool], ytv[pool]
    splits = list(StratifiedKFold(FOLDS, shuffle=True, random_state=seed).split(Xp, yp))
    dev = torch.device("cuda")
    scorer = BatchedFoldKNN if classifier == "knn" else BatchedFoldLSSVMOvR
    fit = Fitness(scorer(torch.tensor(Xp, device=dev), torch.tensor(yp, device=dev), splits, N_CLASSES))

    def capped(x):
        if fit.n_evals >= budget:
            raise BudgetExhausted
        return fit(x)

    class FSProblem(Problem):
        def __init__(self, fn):
            self.fn = fn
            super().__init__(bounds=FloatVar(lb=(0.0,) * dim, ub=(1.0,) * dim), minmax="min", log_to=None)
            self._n_objs, self.obj_weights = 1, np.ones(1)   # skip mealpy's uncounted probe evaluation

        def obj_func(self, x):
            return self.fn(x)

    t = time.perf_counter()
    try:
        if method == "ALL1820":
            fit.of_mask(np.ones(dim, dtype=bool))
        elif method == "FILTER":
            F, _ = f_classif(Xp, yp)
            fit.of_mask(topk_mask(np.nan_to_num(F, nan=-np.inf)))
        elif method == "RANDOM":
            rng = np.random.default_rng(seed)
            for _ in range(budget):
                capped(rng.random(dim))
        elif method == "CSO":
            run_cso(capped, dim, pop_size, np.random.default_rng(seed))
        else:
            GA.BaseGA((budget - pop_size) // pop_size, pop_size).solve(FSProblem(capped), seed=seed)
    except BudgetExhausted:
        pass
    runtime = time.perf_counter() - t

    mask = fit.best_mask
    m = torch.as_tensor(mask, device=dev)
    Xt, Xe = torch.tensor(Xtv, device=dev)[:, m], torch.tensor(Xte, device=dev)[:, m]
    yt = torch.tensor(ytv, device=dev)
    if classifier == "knn":
        score = knn_scores(Xt, yt, Xe, N_CLASSES).cpu().numpy()
    else:
        score = LSSVMOvR(N_CLASSES).fit(Xt, yt).decision_function(Xe).cpu().numpy()
    test = evaluate(yte, score)
    out = {"classifier": classifier, "method": method, "run": run, "seed": seed, "budget": budget, "n_evals": fit.n_evals,
           "n_fits": fit.n_fits, "runtime_s": round(runtime, 2), "cv_acc": round(fit.cv_acc(mask), 5),
           "n_features": int(mask.sum()), **{f"test_{k}": round(float(test[k]), 5) for k in METRICS},
           "gpu_peak_mib": round(torch.cuda.max_memory_allocated() / 2 ** 20),
           "selected": np.flatnonzero(mask).tolist() if mask.sum() <= K else None,
           "history": [round(float(h), 6) for h in fit.history]}
    del fit, Xt, Xe, m
    torch.cuda.empty_cache()
    return out


def main(args):
    out_path = OUT[args.classifier]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            done = {(r["method"], r["run"]) for r in map(json.loads, f)}
    jobs = [(m, run, args.budget, args.pop_size, 0.9 / args.workers, args.classifier) for m in args.methods
            for run in range(1 if m == "ALL1820" else args.runs) if (m, run) not in done]
    print(f"{len(jobs)} jobs to run ({len(done)} already done), {args.workers} workers, {args.classifier} -> {out_path}", flush=True)
    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_job, jobs), 1):
            with open(out_path, "a") as f:
                f.write(json.dumps(r) + "\n")
            searched = r["method"] in ("RANDOM", "GA", "CSO")
            flag = f"  !! {r['n_evals']} evals" if searched and r["n_evals"] != args.budget else ""
            print(f"[{i}/{len(jobs)}] {r['method']:7} run {r['run']:2d}: CV {r['cv_acc']:.4f} | {r['n_features']:4d} feats "
                  f"| test acc {r['test_acc']:.4f} auc {r['test_auc']:.4f} | {r['runtime_s']:.0f}s "
                  f"| elapsed {(time.perf_counter() - t0) / 60:.1f} min{flag}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="DermaMNIST Studies D1 / D2: fixed-k bio-inspired feature selection on the GPU")
    p.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--budget", type=int, default=1000)
    p.add_argument("--pop_size", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--classifier", default="lssvm", choices=["lssvm", "knn"])
    main(p.parse_args())
