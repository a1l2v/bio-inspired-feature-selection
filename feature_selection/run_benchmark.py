"""Benchmark bio-inspired feature selection (PSO, GA, GWO, WOA, BPSO, VBPSO) against random search.

Every method gets exactly the same budget of fitness evaluations (default 20 agents x 50 = 1000,
i.e. 20 initial + 49 iterations x 20), the same fitness (feature_selection/fitness.py, validation only),
the same SVM, and the same seeds (seed depends on feature set and run, not on method -> paired runs).
The test set is scored once per run, on the best mask found.

Budget enforcement (mealpy 3.x):
    - mealpy normally makes one probe call to obj_func to infer the number of objectives; FSProblem
      declares a single objective up front so every counted evaluation is a real search step.
    - mealpy checks termination only at the end of an epoch and GWO/WOA/PSO schedules depend on
      epoch / max_epoch, so `epoch` is set to the true iteration count and the wrapper raises
      BudgetExhausted at exactly `budget` evaluations as a hard cap.
Algorithm hyper-parameters are mealpy defaults (OriginalPSO, BaseGA, OriginalGWO, OriginalWOA).

BPSO (binary PSO, Kennedy & Eberhart 1997) is not in mealpy and is implemented here (run_bpso):
particles are 0/1 masks, velocities follow the standard PSO update, and each bit is set to 1 with
probability sigmoid(v) (S-shaped transfer). Settings as commonly used for feature selection
(Mirjalili & Lewis 2013): c1 = c2 = 2, inertia w linearly 0.9 -> 0.4, |v| <= 6, v0 = 0, random 0/1 start.
VBPSO is the same swarm with the V-shaped transfer from that paper (V2): each bit is FLIPPED with
probability |tanh(v)| and otherwise kept, so particles stay put until the swarm pulls them.

Results are appended to feature_selection/results/runs.jsonl, one line per (method, feature set, run);
finished runs are skipped on re-launch, so the benchmark can be resumed.

Equal-size mode (--k K, written to --out results_fixed_k): every method selects exactly K features
(agent -> its K largest coordinates), fitness = 1 - val ACC. Random search additionally scores its first
random K-subset on test ("single_test_*"), the no-search control.

Usage (from project root):
    .venv\\Scripts\\python.exe feature_selection\\run_benchmark.py                          # everything
    .venv\\Scripts\\python.exe feature_selection\\run_benchmark.py --methods GWO --runs 10
    .venv\\Scripts\\python.exe feature_selection\\run_benchmark.py --budget 100 --runs 1 --out results_smoke
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from mealpy import GA, GWO, PSO, WOA, FloatVar, Problem

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fitness import N_FEATURES, FeatureSelectionFitness  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
FEATURE_SETS = {i: os.path.join(ROOT, "features", f"resnet18_28_{i}.npz") for i in (1, 2, 3)}
METRICS = ("acc", "auc", "macro_f1", "malignant_recall", "specificity")

ALGORITHMS = {
    "PSO": lambda epoch, pop: PSO.OriginalPSO(epoch=epoch, pop_size=pop),
    "GA": lambda epoch, pop: GA.BaseGA(epoch=epoch, pop_size=pop),
    "GWO": lambda epoch, pop: GWO.OriginalGWO(epoch=epoch, pop_size=pop),
    "WOA": lambda epoch, pop: WOA.OriginalWOA(epoch=epoch, pop_size=pop),
}
METHODS = ["RANDOM"] + list(ALGORITHMS) + ["BPSO", "VBPSO"]
BPSO_PARAMS = {"c1": 2.0, "c2": 2.0, "w_max": 0.9, "w_min": 0.4, "v_max": 6.0}


class BudgetExhausted(Exception):
    pass


class FSProblem(Problem):
    def __init__(self, fn, **kwargs):
        self.fn = fn
        super().__init__(bounds=FloatVar(lb=(0.0,) * N_FEATURES, ub=(1.0,) * N_FEATURES),
                         minmax="min", log_to=None, **kwargs)
        self._n_objs, self.obj_weights = 1, np.ones(1)   # skip mealpy's uncounted probe evaluation

    def obj_func(self, x):
        return self.fn(x)


def run_bpso(fn, budget, pop_size, rng, c1, c2, w_max, w_min, v_max, transfer="s"):
    """Binary PSO. transfer "s": bit = 1 with p = sigmoid(v); "v": flip bit with p = |tanh(v)|.
    `fn` takes a 0/1 vector; stops by raising BudgetExhausted."""
    iters = (budget - pop_size) // pop_size
    X = (rng.random((pop_size, N_FEATURES)) < 0.5).astype(float)
    V = np.zeros_like(X)
    fit_x = np.array([fn(x) for x in X])
    P, fit_p = X.copy(), fit_x.copy()
    g = P[fit_p.argmin()].copy()
    for t in range(1, iters + 1):
        w = w_max - (w_max - w_min) * t / iters
        r1, r2 = rng.random(X.shape), rng.random(X.shape)
        V = np.clip(w * V + c1 * r1 * (P - X) + c2 * r2 * (g - X), -v_max, v_max)
        if transfer == "s":
            X = (rng.random(X.shape) < 1.0 / (1.0 + np.exp(-V))).astype(float)
        else:
            flip = rng.random(X.shape) < np.abs(np.tanh(V))
            X = np.where(flip, 1.0 - X, X)
        for i in range(pop_size):
            f = fn(X[i])
            if f < fit_p[i]:
                P[i], fit_p[i] = X[i], f
        g = P[fit_p.argmin()].copy()


def run_one(method, feature_file, seed, budget, pop_size, alpha, metric, k=None):
    fit = FeatureSelectionFitness(feature_file, alpha=alpha, metric=metric,
                                  **({"transfer": "topk", "k": k} if k else {}))

    def capped(x):
        if fit.n_evals >= budget:
            raise BudgetExhausted
        return fit(x)

    extra = {}
    t = time.perf_counter()
    if method == "RANDOM":
        rng = np.random.default_rng(seed)
        first = rng.random(N_FEATURES)
        capped(first)
        for _ in range(budget - 1):
            capped(rng.random(N_FEATURES))
        if k:
            # "No search" control: the first random k-subset drawn, scored once on test.
            single = fit.evaluate_test(fit.to_mask(first))
            extra = {f"single_test_{m}": round(single[m], 5) for m in METRICS}
    elif method in ("BPSO", "VBPSO"):
        try:
            run_bpso(capped, budget, pop_size, np.random.default_rng(seed), **BPSO_PARAMS,
                     transfer="s" if method == "BPSO" else "v")
        except BudgetExhausted:
            pass
    else:
        epochs = (budget - pop_size) // pop_size
        model = ALGORITHMS[method](epochs, pop_size)
        try:
            model.solve(FSProblem(capped), seed=seed)
        except BudgetExhausted:
            pass
    runtime = time.perf_counter() - t

    test = fit.evaluate_test(fit.best_mask)
    return {
        "n_evals": fit.n_evals, "n_fits": fit.n_fits, "runtime_s": round(runtime, 2),
        "best_fitness": round(fit.best_fitness, 6),
        "val_score": round(fit._val_score(fit.best_mask), 5),
        "n_features": test["n_features"], "pct_features": round(test["pct_features"], 2),
        **{f"test_{m}": round(test[m], 5) for m in METRICS},
        "test_confusion": test["confusion"],
        "mask_hex": np.packbits(fit.best_mask).tobytes().hex(),
        "history": [round(h, 6) for h in fit.history],
        **extra,
    }


def main(args):
    out_dir = os.path.join(HERE, args.out)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "runs.jsonl")

    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["method"], r["feature_set"], r["run"], r["budget"], r["alpha"], r["metric"],
                          r.get("k")))

    if args.k and args.alpha != 1.0:
        print(f"--k {args.k}: subset size is fixed, so alpha is set to 1.0 (fitness = 1 - val {args.metric})")
        args.alpha = 1.0
    jobs = [(m, s, r) for m in args.methods for s in args.feature_sets for r in range(args.runs)]
    print(f"{len(jobs)} runs planned, budget {args.budget} evals each, alpha {args.alpha}, metric {args.metric}"
          f"{f', exactly {args.k} features' if args.k else ''} -> {out_path}", flush=True)
    for i, (method, fs, run) in enumerate(jobs, 1):
        if (method, fs, run, args.budget, args.alpha, args.metric, args.k) in done:
            print(f"[{i}/{len(jobs)}] {method} set {fs} run {run}: already done, skipped", flush=True)
            continue
        seed = 100 * fs + run
        res = run_one(method, FEATURE_SETS[fs], seed, args.budget, args.pop_size, args.alpha, args.metric, args.k)
        record = {"method": method, "feature_set": fs, "run": run, "seed": seed, "budget": args.budget,
                  "pop_size": args.pop_size, "alpha": args.alpha, "metric": args.metric,
                  **({"k": args.k} if args.k else {}), **res}
        with open(out_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        flag = "" if res["n_evals"] == args.budget else f"  !! used {res['n_evals']} evals"
        print(f"[{i}/{len(jobs)}] {method:6} set {fs} run {run}: fitness {res['best_fitness']:.4f} "
              f"val {res['val_score']:.4f} | {res['n_features']:3d} feats ({res['pct_features']:4.1f}%) | "
              f"test acc {res['test_acc']:.4f} auc {res['test_auc']:.4f} mal.rec {res['test_malignant_recall']:.4f} "
              f"| {res['runtime_s']:.1f}s{flag}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Feature-selection benchmark: PSO, GA, GWO, WOA vs random search")
    p.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    p.add_argument("--feature_sets", nargs="+", type=int, default=[1, 2, 3], choices=[1, 2, 3])
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--budget", type=int, default=1000)
    p.add_argument("--pop_size", type=int, default=20)
    p.add_argument("--alpha", type=float, default=0.99)
    p.add_argument("--metric", default="acc", choices=["acc", "macro_f1"])
    p.add_argument("--k", type=int, default=None,
                   help="select exactly k features (top-k decoding, alpha forced to 1.0); for equal-size comparisons")
    p.add_argument("--out", default="results")
    main(p.parse_args())
