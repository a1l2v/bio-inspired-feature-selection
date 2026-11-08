"""Fitness function for bio-inspired feature selection on ResNet-18 (28) BreastMNIST features.

    fitness = alpha * error_rate + (1 - alpha) * (n_selected / 512)        (minimise)

    error_rate = 1 - accuracy (default) or 1 - macro-F1, on the VALIDATION set only.

Classifier protocol is imported from svm_baseline/svm_all_features.py so every method uses exactly the
same SVM: StandardScaler fit on train, SVC(rbf, C=1, gamma="scale"), fit on train.
StandardScaler works per column, so the full matrices are scaled once and columns are sliced per mask;
this is identical to refitting the scaler on each subset.

Encoding: an agent is a vector of 512 reals in [0, 1], turned into a keep/drop mask by
    "threshold": keep feature j if x_j > 0.5                     (deterministic, default)
    "sigmoid":   keep feature j if U(0,1) < 1 / (1 + exp(-10 (x_j - 0.5)))   (stochastic S-shaped)
    "topk":      keep exactly the k features with the largest x_j          (fixed subset size; use with
                 alpha = 1, since the size term is then constant)
An empty mask gets the worst possible fitness, 1.0.

The test set is never touched by `__call__`; use `evaluate_test(mask)` once, after the search.

Usage:
    from fitness import FeatureSelectionFitness
    fit = FeatureSelectionFitness("features/resnet18_28_1.npz", alpha=0.99)
    f = fit(agent_vector)          # what the optimiser minimises
    fit.n_evals, fit.history       # budget used, best-so-far fitness per evaluation
    fit.evaluate_test(best_mask)   # full metrics on test, once per run

Self-check (from project root):
    .venv\\Scripts\\python.exe feature_selection\\fitness.py
"""

import os
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "svm_baseline"))
from svm_all_features import SVM_PARAMS, evaluate  # noqa: E402  (shared classifier + metrics)

N_FEATURES = 512


class FeatureSelectionFitness:
    def __init__(self, feature_file, alpha=0.99, metric="acc", transfer="threshold", seed=None, k=None):
        assert 0 < alpha <= 1
        assert metric in ("acc", "macro_f1")
        assert transfer in ("threshold", "sigmoid", "topk")
        assert (transfer == "topk") == (k is not None), "k is required for, and only for, transfer='topk'"
        self.alpha, self.metric, self.transfer, self.k = alpha, metric, transfer, k
        self.rng = np.random.default_rng(seed)

        d = np.load(feature_file)
        scaler = StandardScaler().fit(d["X_train"])
        self.X_train = scaler.transform(d["X_train"])
        self.X_val = scaler.transform(d["X_val"])
        self.X_test = scaler.transform(d["X_test"])
        self.y_train, self.y_val, self.y_test = d["y_train"], d["y_val"], d["y_test"]
        assert self.X_train.shape[1] == N_FEATURES

        self._cache = {}
        self.n_evals = 0           # calls made by the optimiser (the budget)
        self.n_fits = 0            # SVMs actually trained (cache misses)
        self.best_fitness = np.inf
        self.best_mask = None
        self.history = []          # best-so-far fitness after each evaluation (convergence curve)

    def to_mask(self, x):
        x = np.asarray(x, dtype=float)
        if self.transfer == "threshold":
            return x > 0.5
        if self.transfer == "topk":
            mask = np.zeros(x.shape, dtype=bool)
            mask[np.argsort(x, kind="stable")[-self.k:]] = True
            return mask
        return self.rng.random(x.shape) < 1.0 / (1.0 + np.exp(-10.0 * (x - 0.5)))

    def _val_score(self, mask):
        key = np.packbits(mask).tobytes()
        if key not in self._cache:
            clf = SVC(**SVM_PARAMS).fit(self.X_train[:, mask], self.y_train)
            pred = clf.predict(self.X_val[:, mask])
            self._cache[key] = (accuracy_score(self.y_val, pred) if self.metric == "acc"
                                else f1_score(self.y_val, pred, average="macro"))
            self.n_fits += 1
        return self._cache[key]

    def fitness_of_mask(self, mask):
        mask = np.asarray(mask, dtype=bool)
        k = int(mask.sum())
        if k == 0:
            return 1.0
        error = 1.0 - self._val_score(mask)
        return self.alpha * error + (1.0 - self.alpha) * k / N_FEATURES

    def __call__(self, x):
        mask = self.to_mask(x)
        f = self.fitness_of_mask(mask)
        self.n_evals += 1
        if f < self.best_fitness:
            self.best_fitness, self.best_mask = f, mask.copy()
        self.history.append(self.best_fitness)
        return f

    def evaluate_test(self, mask):
        """Fit on train with the selected features, score on test. Call once per run, after the search."""
        mask = np.asarray(mask, dtype=bool)
        clf = SVC(**SVM_PARAMS).fit(self.X_train[:, mask], self.y_train)
        X = self.X_test[:, mask]
        res = evaluate(self.y_test, clf.predict(X), clf.decision_function(X))
        res["n_features"] = int(mask.sum())
        res["pct_features"] = 100.0 * mask.sum() / N_FEATURES
        return res


if __name__ == "__main__":
    path = os.path.join(ROOT, "features", "resnet18_28_1.npz")
    fit = FeatureSelectionFitness(path, alpha=0.99)

    # 1) All features must reproduce the SVM baseline (val ACC 0.8590 for resnet18_28_1).
    all_on = np.ones(N_FEATURES)
    f_all = fit(all_on)
    acc_all = fit._val_score(np.ones(N_FEATURES, bool))
    print(f"all 512 features: val acc {acc_all:.4f}  fitness {f_all:.5f}  "
          f"(expected 0.99*(1-{acc_all:.4f}) + 0.01*1 = {0.99 * (1 - acc_all) + 0.01:.5f})")

    # 2) Edge cases.
    print(f"empty mask fitness: {fit(np.zeros(N_FEATURES)):.1f}")
    print(f"cache: repeat of all-features call -> fits {fit.n_fits}, evals before {fit.n_evals}", end="")
    fit(all_on)
    print(f", after {fit.n_evals} evals / {fit.n_fits} fits")

    # 3) Timing on random agents (~50% of features each), the typical cost per evaluation.
    rng = np.random.default_rng(0)
    agents = rng.random((200, N_FEATURES))
    t = time.perf_counter()
    vals = [fit(a) for a in agents]
    per = (time.perf_counter() - t) / len(agents)
    print(f"200 random agents: {per * 1000:.1f} ms/eval | fitness min {min(vals):.4f} median {np.median(vals):.4f} "
          f"| best so far {fit.best_fitness:.4f} with {int(fit.best_mask.sum())} features")

    # 4) Test is separate and not counted in the budget.
    n_before = fit.n_evals
    res = fit.evaluate_test(fit.best_mask)
    assert fit.n_evals == n_before
    print(f"best random mask on test: acc {res['acc']:.4f} auc {res['auc']:.4f} "
          f"malignant recall {res['malignant_recall']:.4f} ({res['n_features']} features, {res['pct_features']:.1f}%)")

    budget = 20 * 50
    print(f"\nbudget per run (20 agents x 50 iters) = {budget} evals ~ {budget * per:.0f} s; "
          f"10 runs x 3 feature sets x 5 methods (4 algorithms + random search) ~ {150 * budget * per / 60:.0f} min")
