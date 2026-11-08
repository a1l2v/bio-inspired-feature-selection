"""GPU least-squares SVM (LS-SVM / kernel ridge classifier) with an RBF kernel, in PyTorch.

Why: scikit-learn's SVC is CPU-only and iterative. The LS-SVM (Suykens & Vandewalle 1999) uses the same
RBF kernel but replaces the hinge loss with a squared loss, so training is one linear solve:

    [ 0   1^T       ] [b]   [0]
    [ 1   K + I / C ] [a] = [t],      t = +1 for label 1 (normal/benign), -1 for label 0 (malignant)

    decision f(x) = sum_i a_i K(x, x_i) + b,   predict label 1 if f(x) > 0

Settings mirror the SVC protocol: K(x, z) = exp(-gamma ||x - z||^2) with gamma = 1 / (k * Var(X_train)) over
the selected k features ("scale", computed on each fit's own training rows), C = 1.

All inner folds are solved in one batched call: fold systems of unequal size are zero-padded and the
padding rows get an identity diagonal and zero target, which leaves the real solution unchanged.
"""

import numpy as np
import torch

C = 1.0


def sq_dists(A, B):
    """Squared Euclidean distances, n_a x n_b, clamped at 0 for float32 round-off."""
    return ((A * A).sum(1, keepdim=True) + (B * B).sum(1) - 2.0 * A @ B.T).clamp_min_(0.0)


def scale_gamma(X):
    return 1.0 / (X.shape[1] * X.var(unbiased=False))


class GPULSSVM:
    """Single fit on (X, y) tensors already on the device; y holds labels 0/1."""

    def fit(self, X, y):
        self.X, self.gamma = X, scale_gamma(X)
        n = X.shape[0]
        A = torch.empty(n + 1, n + 1, device=X.device, dtype=X.dtype)
        A[0, 0], A[0, 1:], A[1:, 0] = 0.0, 1.0, 1.0
        A[1:, 1:] = torch.exp(-self.gamma * sq_dists(X, X)) + torch.eye(n, device=X.device, dtype=X.dtype) / C
        rhs = torch.cat([torch.zeros(1, device=X.device, dtype=X.dtype), 2.0 * y.to(X.dtype) - 1.0])
        sol = torch.linalg.solve(A, rhs)
        self.b, self.alpha = sol[0], sol[1:]
        return self

    def decision_function(self, Z):
        return torch.exp(-self.gamma * sq_dists(Z, self.X)) @ self.alpha + self.b

    def predict(self, Z):
        return (self.decision_function(Z) > 0).long()


class BatchedFoldLSSVM:
    """Inner-CV accuracy for a feature mask, all folds in one batched solve.

    Built once per outer fold: X (n x 512) and y on the device, `splits` = list of (train_idx, val_idx).
    """

    def __init__(self, X, y, splits):
        self.X, self.y = X, y
        self.n = len(y)
        self.folds = len(splits)
        self.m = max(len(a) for a, _ in splits)                       # padded training size
        dev = X.device
        self.tr = torch.zeros(self.folds, self.m, dtype=torch.long, device=dev)
        self.tr_valid = torch.zeros(self.folds, self.m, dtype=torch.bool, device=dev)
        self.va = [torch.as_tensor(b, device=dev) for _, b in splits]
        for f, (a, _) in enumerate(splits):
            self.tr[f, :len(a)] = torch.as_tensor(a, device=dev)
            self.tr_valid[f, :len(a)] = True
        self.t = 2.0 * y.to(X.dtype) - 1.0
        self.eye = torch.eye(self.m, device=dev, dtype=X.dtype)

    @torch.no_grad()
    def accuracy(self, mask):
        Xm = self.X[:, torch.as_tensor(mask, device=self.X.device)]
        D = sq_dists(Xm, Xm)                                            # n x n, shared by all folds
        F, m = self.folds, self.m
        valid = self.tr_valid.to(Xm.dtype)                              # F x m
        Xtr = Xm[self.tr] * valid[..., None]                            # F x m x k (padding rows zeroed)
        cnt = valid.sum(1)                                              # F
        mean = Xtr.sum(1) / cnt[:, None]                                # F x k
        var = (((Xtr - mean[:, None]) ** 2) * valid[..., None]).sum((1, 2)) / (cnt * Xm.shape[1])
        gamma = 1.0 / (Xm.shape[1] * var)                               # F, sklearn "scale" per training set

        Dtr = D[self.tr[:, :, None], self.tr[:, None, :]]               # F x m x m
        pair = valid[:, :, None] * valid[:, None, :]
        K = torch.exp(-gamma[:, None, None] * Dtr) * pair
        A = torch.zeros(F, m + 1, m + 1, device=Xm.device, dtype=Xm.dtype)
        A[:, 0, 1:] = valid
        A[:, 1:, 0] = valid
        A[:, 1:, 1:] = K + self.eye / C * valid[:, :, None] + self.eye * (1.0 - valid)[:, :, None]
        rhs = torch.zeros(F, m + 1, device=Xm.device, dtype=Xm.dtype)
        rhs[:, 1:] = self.t[self.tr] * valid
        sol = torch.linalg.solve(A, rhs)                                # one batched solve for all folds
        b, alpha = sol[:, 0], sol[:, 1:] * valid

        correct = 0
        for f in range(F):
            va = self.va[f]
            Kva = torch.exp(-gamma[f] * D[va][:, self.tr[f]])          # n_val x m
            pred = (Kva @ alpha[f] + b[f]) > 0
            correct += int((pred.long() == self.y[va]).sum())
        return correct / self.n


def cpu_reference_accuracy(X, y, splits, mask):
    """Plain NumPy float64 LS-SVM, one fold at a time - used only to validate the GPU code."""
    Xm = X[:, mask].astype(np.float64)
    correct = 0
    for a, b in splits:
        Xa = Xm[a]
        gamma = 1.0 / (Xa.shape[1] * Xa.var())
        D = ((Xa[:, None, :] - Xa[None, :, :]) ** 2).sum(-1)
        n = len(a)
        A = np.zeros((n + 1, n + 1))
        A[0, 1:], A[1:, 0] = 1.0, 1.0
        A[1:, 1:] = np.exp(-gamma * D) + np.eye(n) / C
        sol = np.linalg.solve(A, np.r_[0.0, 2.0 * y[a] - 1.0])
        Kv = np.exp(-gamma * ((Xm[b][:, None, :] - Xa[None, :, :]) ** 2).sum(-1))
        correct += int((((Kv @ sol[1:] + sol[0]) > 0).astype(int) == y[b]).sum())
    return correct / len(y)
