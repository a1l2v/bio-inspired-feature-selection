"""Multi-class GPU least-squares SVM (one-vs-rest) with an RBF kernel, in PyTorch.

Extends imagenet_nested_cv/gpu_lssvm.py (binary) to C classes. All C one-vs-rest problems share the kernel
and the system matrix, so training is ONE linear solve with C right-hand sides:

    [ 0   1^T       ] [b_1 .. b_C]   [0 ... 0]
    [ 1   K + I / C ] [a_1 .. a_C] = [T      ],   T[i, c] = +1 if y_i == c else -1

    scores f_c(x) = sum_i a_ic K(x, x_i) + b_c;  predict argmax_c f_c(x)

K(x, z) = exp(-gamma ||x - z||^2), gamma = 1 / (k * Var(X_train)) ("scale"), regularisation C = 1.
"""

import numpy as np
import torch

C_REG = 1.0


def sq_dists(A, B):
    return ((A * A).sum(1, keepdim=True) + (B * B).sum(1) - 2.0 * A @ B.T).clamp_min_(0.0)


class LSSVMOvR:
    def __init__(self, n_classes):
        self.n_classes = n_classes

    @torch.no_grad()
    def fit(self, X, y):
        n = X.shape[0]
        self.X, self.gamma = X, 1.0 / (X.shape[1] * X.var(unbiased=False))
        A = torch.empty(n + 1, n + 1, device=X.device, dtype=X.dtype)
        A[0, 0], A[0, 1:], A[1:, 0] = 0.0, 1.0, 1.0
        A[1:, 1:] = torch.exp(-self.gamma * sq_dists(X, X))
        A[1:, 1:].diagonal().add_(1.0 / C_REG)
        rhs = torch.zeros(n + 1, self.n_classes, device=X.device, dtype=X.dtype)
        rhs[1:] = -1.0
        rhs[1 + torch.arange(n, device=X.device), y] = 1.0
        sol = torch.linalg.solve(A, rhs)
        self.b, self.alpha = sol[0], sol[1:]
        return self

    @torch.no_grad()
    def decision_function(self, Z):
        return torch.exp(-self.gamma * sq_dists(Z, self.X)) @ self.alpha + self.b

    def predict(self, Z):
        return self.decision_function(Z).argmax(1)


class BatchedFoldLSSVMOvR:
    """Cross-validated accuracy for a feature mask: all folds and all classes in ONE batched solve.

    X (n x d) and y on the device; `splits` = list of (train_idx, val_idx). Fold systems of unequal size are
    zero-padded; padding rows get an identity diagonal and zero targets, which leaves the real solution
    unchanged. gamma = "scale" is computed on each fold's own training rows.
    """

    def __init__(self, X, y, splits, n_classes):
        self.X, self.y, self.C = X, y, n_classes
        dev = X.device
        self.F, self.m = len(splits), max(len(a) for a, _ in splits)
        self.tr = torch.zeros(self.F, self.m, dtype=torch.long, device=dev)
        self.valid = torch.zeros(self.F, self.m, dtype=X.dtype, device=dev)
        for f, (a, _) in enumerate(splits):
            self.tr[f, :len(a)] = torch.as_tensor(a, device=dev)
            self.valid[f, :len(a)] = 1.0
        self.va = [torch.as_tensor(b, device=dev) for _, b in splits]
        T = -torch.ones(len(y), n_classes, device=dev, dtype=X.dtype)
        T[torch.arange(len(y), device=dev), y] = 1.0
        self.T = T
        self.eye = torch.eye(self.m, device=dev, dtype=X.dtype)

    @torch.no_grad()
    def accuracy(self, mask):
        Xm = self.X[:, torch.as_tensor(mask, device=self.X.device)]
        D = sq_dists(Xm, Xm)
        v = self.valid
        Xtr = Xm[self.tr] * v[..., None]
        cnt = v.sum(1)
        mean = Xtr.sum(1) / cnt[:, None]
        var = (((Xtr - mean[:, None]) ** 2) * v[..., None]).sum((1, 2)) / (cnt * Xm.shape[1])
        gamma = 1.0 / (Xm.shape[1] * var)
        K = torch.exp(-gamma[:, None, None] * D[self.tr[:, :, None], self.tr[:, None, :]]) * (v[:, :, None] * v[:, None, :])
        A = torch.zeros(self.F, self.m + 1, self.m + 1, device=Xm.device, dtype=Xm.dtype)
        A[:, 0, 1:], A[:, 1:, 0] = v, v
        A[:, 1:, 1:] = K + self.eye * (v / C_REG + (1.0 - v))[:, :, None]
        rhs = torch.zeros(self.F, self.m + 1, self.C, device=Xm.device, dtype=Xm.dtype)
        rhs[:, 1:] = self.T[self.tr] * v[..., None]
        sol = torch.linalg.solve(A, rhs)                        # F x (m+1) x C, one batched solve
        b, alpha = sol[:, 0], sol[:, 1:] * v[..., None]
        correct = 0
        for f in range(self.F):
            va = self.va[f]
            score = torch.exp(-gamma[f] * D[va][:, self.tr[f]]) @ alpha[f] + b[f]
            correct += int((score.argmax(1) == self.y[va]).sum())
        return correct / len(self.y)


def numpy_reference(Xtr, ytr, Xev, n_classes):
    """Plain NumPy float64 version, used only to validate the GPU code; returns scores (n_eval x C)."""
    Xtr, Xev = Xtr.astype(np.float64), Xev.astype(np.float64)
    gamma = 1.0 / (Xtr.shape[1] * Xtr.var())
    d = lambda A, B: ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)  # noqa: E731
    n = len(ytr)
    A = np.zeros((n + 1, n + 1))
    A[0, 1:], A[1:, 0] = 1.0, 1.0
    A[1:, 1:] = np.exp(-gamma * d(Xtr, Xtr)) + np.eye(n) / C_REG
    T = -np.ones((n, n_classes))
    T[np.arange(n), ytr] = 1.0
    sol = np.linalg.solve(A, np.vstack([np.zeros((1, n_classes)), T]))
    return np.exp(-gamma * d(Xev, Xtr)) @ sol[1:] + sol[0]
