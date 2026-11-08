"""Extract 512-dim ResNet-18 (28) features for RetinaMNIST using the released MedMNIST weights.

Same procedure as scripts/extract_features.py (BreastMNIST): for each released checkpoint, every
train / val / test image goes through the network in eval mode and the global-average-pool output
(input to the final linear layer) is saved. RetinaMNIST is natively RGB (3 channels), so no channel
copying is needed; preprocessing is ToTensor + Normalize(mean=.5, std=.5), as in the official script.

Check: the saved features are pushed through the checkpoint's own final layer and scored with
medmnist.Evaluator; the test AUC / ACC must equal the official re-evaluation (Retina/baseline/SOURCE.md).

Outputs in Retina/features/, per checkpoint:
    <name>.npz   X_train, X_val, X_test (float32, N x 512); y_train, y_val, y_test (int64, grades 0-4);
                 prob_train, prob_val, prob_test (float32, N x 5, softmax of the original head)
    <name>.json  provenance and check metrics

Usage (from project root):
    .venv\\Scripts\\python.exe Retina\\scripts\\extract_features.py
"""

import hashlib
import json
import os
import sys

import medmnist
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from medmnist import Evaluator, RetinaMNIST

RETINA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(RETINA)
sys.path.insert(0, os.path.join(ROOT, "baseline"))
from models import ResNet18  # noqa: E402  (official MedMNIST model definition)

DATA_ROOT = os.path.join(RETINA, "dataset")
WEIGHTS = [os.path.join(RETINA, "baseline", "weights", f"resnet18_28_{i}.pth") for i in (1, 2, 3)]
OUT_DIR = os.path.join(RETINA, "features")
SPLITS = ("train", "val", "test")
N_CLASSES = 5


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def extract(model, loader, device):
    captured = {}
    hook = model.avgpool.register_forward_hook(
        lambda _m, _inp, out: captured.__setitem__("feat", torch.flatten(out, 1)))
    feats, logits, labels = [], [], []
    for inputs, targets in loader:
        out = model(inputs.to(device))
        feats.append(captured["feat"].cpu())
        logits.append(out.cpu())
        labels.append(targets.view(-1))
    hook.remove()
    return torch.cat(feats), torch.cat(logits), torch.cat(labels)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])])
    loaders = {s: data.DataLoader(RetinaMNIST(split=s, transform=transform, root=DATA_ROOT, size=28),
                                  batch_size=128, shuffle=False) for s in SPLITS}

    for path in WEIGHTS:
        name = os.path.splitext(os.path.basename(path))[0]
        model = ResNet18(in_channels=3, num_classes=N_CLASSES)
        model.load_state_dict(torch.load(path, map_location=device)["net"], strict=True)
        model = model.to(device).eval()

        arrays, metrics = {}, {}
        print(f"==> {name}")
        for s in SPLITS:
            feats, logits, labels = extract(model, loaders[s], device)
            with torch.no_grad():
                diff = (model.linear(feats.to(device)).cpu() - logits).abs().max().item()
            assert diff < 1e-4, f"{s}: linear(features) differs from model output by {diff}"
            prob = torch.softmax(logits, dim=1).numpy()
            auc, acc = Evaluator("retinamnist", s, size=28, root=DATA_ROOT).evaluate(prob)
            metrics[s] = {"n": len(labels), "auc": round(float(auc), 5), "acc": round(float(acc), 5)}
            print(f"    {s:5}  X {tuple(feats.shape)}  auc {auc:.5f}  acc {acc:.5f}")
            arrays[f"X_{s}"] = feats.numpy().astype(np.float32)
            arrays[f"y_{s}"] = labels.numpy().astype(np.int64)
            arrays[f"prob_{s}"] = prob.astype(np.float32)

        np.savez_compressed(os.path.join(OUT_DIR, f"{name}.npz"), **arrays)
        meta = {"weights": os.path.relpath(path, ROOT).replace("\\", "/"), "weights_md5": md5(path),
                "data": "Retina/dataset/retinamnist.npz", "data_md5": md5(os.path.join(DATA_ROOT, "retinamnist.npz")),
                "layer": "avgpool output (input to final linear layer), 512-dim, post-ReLU",
                "preprocessing": "native RGB, ToTensor, Normalize(mean=.5, std=.5), 28x28",
                "labels": "diabetic-retinopathy grade 0-4 (ordinal)",
                "torch": torch.__version__, "medmnist": medmnist.__version__,
                "metrics_from_original_head": metrics}
        with open(os.path.join(OUT_DIR, f"{name}.json"), "w") as f:
            json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
