"""k-nearest-neighbour classifier on the GPU, matching scikit-learn's KNeighborsClassifier(n_neighbors=5).

Euclidean distance, uniform votes; prediction = class with the most votes, ties to the lower class index
(torch.argmax returns the first maximum, as scipy's mode does in scikit-learn); class scores = vote fractions
(scikit-learn's predict_proba).
"""

import torch

KNN_K = 5


@torch.no_grad()
def knn_scores(Xtr, ytr, Xev, n_classes, k=KNN_K):
    """Vote fractions (n_eval x n_classes) of the k nearest training rows."""
    nn = ytr[torch.cdist(Xev, Xtr).topk(k, largest=False).indices]
    return torch.nn.functional.one_hot(nn, n_classes).sum(1).float() / k


class BatchedFoldKNN:
    """Cross-validated accuracy for a feature mask: one pool distance matrix per mask, sliced per fold."""

    def __init__(self, X, y, splits, n_classes, k=KNN_K):
        self.X, self.y, self.C, self.k = X, y, n_classes, k
        self.splits = [(torch.as_tensor(a, device=X.device), torch.as_tensor(b, device=X.device)) for a, b in splits]

    @torch.no_grad()
    def accuracy(self, mask):
        Xm = self.X[:, torch.as_tensor(mask, device=self.X.device)]
        D = torch.cdist(Xm, Xm)
        correct = 0
        for tr, va in self.splits:
            nn = self.y[tr][D[va][:, tr].topk(self.k, largest=False).indices]
            votes = torch.nn.functional.one_hot(nn, self.C).sum(1)
            correct += int((votes.argmax(1) == self.y[va]).sum())
        return correct / len(self.y)
