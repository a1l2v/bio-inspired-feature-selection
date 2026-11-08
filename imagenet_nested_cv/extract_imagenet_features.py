"""Extract 512-dim features for all 780 BreastMNIST images with a frozen ImageNet-pretrained ResNet-18.

Option 2 of the improvement plan: unlike the released MedMNIST weights, this network has never seen
BreastMNIST, so no image's features are memorised and every image can be used for honest nested CV.

Preprocessing: 28x28 grayscale -> [0, 1] -> 3 channels -> bilinear resize to 224x224 -> ImageNet
normalisation. Features = global-average-pool output (input to the ImageNet classifier), eval mode.

Output: features/imagenet_resnet18.npz
    X      float32, 780 x 512, in the order train, val, test of dataset/breastmnist.npz
    y      int64, 780 (label 0 = malignant)
    split  str, 780 ("train" / "val" / "test"; the official split, kept for reference only)
and features/imagenet_resnet18.json with provenance.

Usage (from project root):
    .venv\\Scripts\\python.exe imagenet_nested_cv\\extract_imagenet_features.py
"""

import hashlib
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision.models import ResNet18_Weights, resnet18

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "dataset", "breastmnist.npz")
OUT = os.path.join(ROOT, "features", "imagenet_resnet18")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def extract(model, images, device, batch_size=64):
    feats = []
    for i in range(0, len(images), batch_size):
        x = torch.from_numpy(images[i:i + batch_size]).float().div(255.0).unsqueeze(1)   # N x 1 x 28 x 28
        x = F.interpolate(x.repeat(1, 3, 1, 1), size=(224, 224), mode="bilinear", align_corners=False)
        x = ((x - MEAN) / STD).to(device)
        feats.append(model(x).cpu())
    return torch.cat(feats).numpy().astype(np.float32)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()           # output = 512-dim average-pool features
    model = model.to(device).eval()

    d = np.load(DATA)
    splits = ("train", "val", "test")
    images = np.concatenate([d[f"{s}_images"] for s in splits])
    y = np.concatenate([d[f"{s}_labels"].ravel() for s in splits]).astype(np.int64)
    split = np.concatenate([[s] * len(d[f"{s}_labels"]) for s in splits])

    X = extract(model, images, device)
    again = extract(model, images[:64], device)
    assert np.allclose(X[:64], again, atol=1e-5), "extraction is not deterministic"
    assert X.shape == (780, 512) and np.isfinite(X).all()

    np.savez_compressed(OUT + ".npz", X=X, y=y, split=split)
    with open(DATA, "rb") as f:
        data_md5 = hashlib.md5(f.read()).hexdigest()
    meta = {
        "extractor": "torchvision resnet18, ResNet18_Weights.IMAGENET1K_V1, frozen, eval mode, fc removed",
        "weights_url": weights.url,
        "never_trained_on_breastmnist": True,
        "preprocessing": "uint8/255, 1->3 channels, bilinear resize 28->224, ImageNet mean/std",
        "data": "dataset/breastmnist.npz", "data_md5": data_md5,
        "n_images": int(len(y)), "n_malignant_label0": int((y == 0).sum()),
        "torch": torch.__version__, "torchvision": torchvision.__version__, "device": str(device),
    }
    with open(OUT + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved {os.path.relpath(OUT, ROOT)}.npz: X {X.shape}, malignant {int((y == 0).sum())} / {len(y)}, "
          f"feature range [{X.min():.3f}, {X.max():.3f}], all-zero features {int((X.max(0) == 0).sum())}")


if __name__ == "__main__":
    main()
