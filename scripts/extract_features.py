"""Extract 512-dim ResNet-18 (28) features for BreastMNIST using the released MedMNIST weights.

For each weights file, every train / val / test image is passed through the network in eval mode
and the global-average-pool output (the input to the final linear layer) is saved.

Preprocessing matches the official baseline (baseline/train_and_eval_pytorch.py with --as_rgb):
grayscale -> 3-channel RGB, ToTensor, Normalize(mean=.5, std=.5), no resize.

Sanity check: the saved features are pushed through the model's own final layer and scored with
medmnist.Evaluator; the test AUC / ACC must equal the released-weights evaluation in baseline/SOURCE.md.

Outputs, per weights file, in features/:
    <name>.npz   X_train, X_val, X_test (float32, N x 512), y_train, y_val, y_test (int64, N),
                 prob_train, prob_val, prob_test (float32, N x 2, softmax of the original head)
    <name>.json  provenance and sanity-check metrics

Usage (from project root):
    .venv\\Scripts\\python.exe scripts\\extract_features.py
    .venv\\Scripts\\python.exe scripts\\extract_features.py --weights baseline/weights/resnet18_28_1.pth
"""

import argparse
import hashlib
import json
import os
import sys

import medmnist
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from medmnist import BreastMNIST, Evaluator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "baseline"))
from models import ResNet18  # noqa: E402  (official MedMNIST model definition)

DATA_ROOT = os.path.join(ROOT, "dataset")
DEFAULT_WEIGHTS = [os.path.join(ROOT, "baseline", "weights", f"resnet18_28_{i}.pth") for i in (1, 2, 3)]
SPLITS = ("train", "val", "test")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(weights_path, device):
    model = ResNet18(in_channels=3, num_classes=2)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state["net"], strict=True)
    return model.to(device).eval()


@torch.no_grad()
def extract(model, loader, device):
    """Return (features N x 512, logits N x 2, labels N) for one split, in dataset order."""
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


def main(weights_list, out_dir, batch_size, device):
    os.makedirs(out_dir, exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])])
    loaders = {
        split: data.DataLoader(
            BreastMNIST(split=split, transform=transform, as_rgb=True, root=DATA_ROOT, size=28),
            batch_size=batch_size, shuffle=False)
        for split in SPLITS
    }

    for weights_path in weights_list:
        name = os.path.splitext(os.path.basename(weights_path))[0]
        print(f"==> {name}")
        model = load_model(weights_path, device)

        arrays, metrics = {}, {}
        for split in SPLITS:
            feats, logits, labels = extract(model, loaders[split], device)

            # The features must be exactly what the final layer sees.
            with torch.no_grad():
                relogits = model.linear(feats.to(device)).cpu()
            head_diff = (relogits - logits).abs().max().item()
            assert head_diff < 1e-4, f"{split}: linear(features) differs from model output by {head_diff}"

            prob = torch.softmax(logits, dim=1).numpy()
            auc, acc = Evaluator("breastmnist", split, size=28, root=DATA_ROOT).evaluate(prob)
            metrics[split] = {"n": len(labels), "auc": round(float(auc), 5), "acc": round(float(acc), 5),
                              "head_max_abs_diff": head_diff}
            print(f"    {split:5}  X {tuple(feats.shape)}  auc {auc:.5f}  acc {acc:.5f}")

            arrays[f"X_{split}"] = feats.numpy().astype(np.float32)
            arrays[f"y_{split}"] = labels.numpy().astype(np.int64)
            arrays[f"prob_{split}"] = prob.astype(np.float32)

        npz_path = os.path.join(out_dir, f"{name}.npz")
        np.savez_compressed(npz_path, **arrays)

        meta = {
            "weights": os.path.relpath(weights_path, ROOT).replace("\\", "/"),
            "weights_md5": md5(weights_path),
            "data": os.path.relpath(os.path.join(DATA_ROOT, "breastmnist.npz"), ROOT).replace("\\", "/"),
            "data_md5": md5(os.path.join(DATA_ROOT, "breastmnist.npz")),
            "layer": "avgpool output (input to final linear layer), 512-dim, post-ReLU",
            "preprocessing": "as_rgb=True, ToTensor, Normalize(mean=.5, std=.5), 28x28",
            "label_map": medmnist.INFO["breastmnist"]["label"],
            "positive_class_note": "label 0 = malignant; use pos_label=0 for malignant recall",
            "torch": torch.__version__,
            "medmnist": medmnist.__version__,
            "metrics_from_original_head": metrics,
        }
        with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"    saved {os.path.relpath(npz_path, ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 512-dim ResNet-18 (28) features for BreastMNIST")
    parser.add_argument("--weights", nargs="+", default=DEFAULT_WEIGHTS, help="weights files ({'net': state_dict})")
    parser.add_argument("--out_dir", default=os.path.join(ROOT, "features"))
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    main(args.weights, args.out_dir, args.batch_size, torch.device(args.device))
