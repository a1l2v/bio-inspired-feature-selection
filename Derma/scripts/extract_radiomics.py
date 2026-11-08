"""Extract handcrafted radiomics / texture / colour features from DermaMNIST 128x128 RGB skin-lesion images.

Grey features: the full PneumoniaMNIST set (Pneumonia/scripts/extract_radiomics.py, 1,359 features) computed
on the luminance image (skimage rgb2gray, scaled to uint8). Colour features added here:

    col_*       first-order statistics (20) of 9 channels: R, G, B, H, S, V, L*, a*, b* (180)
    chist_*     16-bin histogram of the same 9 channels over each channel's fixed range (144)
    cgrid_*     mean and std of H, S, a*, b* in each 4 x 4 grid cell (128)
    cr_*        centre (inner 64 x 64) vs surrounding ring: mean of L*, a*, b* in each and the difference (9)

Nothing is learned from the labels. NaN is replaced by 0.

Output: Derma/features/radiomics_128.npz  X_train, X_val, X_test (float32), y_train, y_val, y_test
        Derma/features/radiomics_128.json feature names, group counts, provenance

Usage (from project root):
    .venv\\Scripts\\python.exe Derma\\scripts\\extract_radiomics.py --workers 16
"""

import argparse
import hashlib
import importlib.util
import json
import os
import time
from multiprocessing import Pool

import numpy as np
from skimage.color import rgb2gray, rgb2hsv, rgb2lab

DERMA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(DERMA)
_spec = importlib.util.spec_from_file_location("pneu_radiomics", os.path.join(ROOT, "Pneumonia", "scripts", "extract_radiomics.py"))
_p = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_p)

DATA = os.path.join(DERMA, "dataset", "dermamnist_128.npz")
OUT_DIR = os.path.join(DERMA, "features")
SPLITS = ("train", "val", "test")
# channel name -> (index into the stacked 9-channel image, histogram range)
CHANNELS = {"R": (0, (0, 1)), "G": (1, (0, 1)), "B": (2, (0, 1)), "H": (3, (0, 1)), "S": (4, (0, 1)),
            "V": (5, (0, 1)), "L": (6, (0, 100)), "a": (7, (-128, 127)), "b": (8, (-128, 127))}


def features(img):
    rgb = img.astype(np.float64) / 255.0
    grey = np.clip(np.round(rgb2gray(rgb) * 255), 0, 255).astype(np.uint8)
    out = {f"grey_{k}": v for k, v in _p.features(grey).items()}
    ch = np.concatenate([rgb, rgb2hsv(rgb), rgb2lab(rgb)], axis=2)
    for nm, (i, rng) in CHANNELS.items():
        _p.first_order(ch[..., i], out, f"col_{nm}")
        h, _ = np.histogram(ch[..., i], bins=16, range=rng)
        out.update({f"chist_{nm}_b{j}": v / ch[..., i].size for j, v in enumerate(h)})
    for nm in ("H", "S", "a", "b"):
        c = ch[..., CHANNELS[nm][0]]
        for r in range(4):
            for cc in range(4):
                cell = c[32 * r: 32 * r + 32, 32 * cc: 32 * cc + 32]
                out[f"cgrid_{nm}_r{r}c{cc}_mean"], out[f"cgrid_{nm}_r{r}c{cc}_std"] = cell.mean(), cell.std()
    centre = np.zeros((128, 128), bool)
    centre[32:96, 32:96] = True
    for nm in ("L", "a", "b"):
        c = ch[..., CHANNELS[nm][0]]
        out[f"cr_{nm}_centre"], out[f"cr_{nm}_ring"] = c[centre].mean(), c[~centre].mean()
        out[f"cr_{nm}_diff"] = out[f"cr_{nm}_centre"] - out[f"cr_{nm}_ring"]
    return out


def vec(img):
    return np.nan_to_num(np.array(list(features(img).values()), dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def group_of(name):
    parts = name.split("_")
    return f"grey_{parts[1]}" if parts[0] == "grey" else parts[0]


def main(workers):
    t0 = time.perf_counter()
    d = np.load(DATA)
    names = list(features(d["train_images"][0]).keys())
    groups = {}
    for n in names:
        groups[group_of(n)] = groups.get(group_of(n), 0) + 1
    print(f"{len(names)} features per image: {groups}", flush=True)
    res = {}
    with Pool(workers) as pool:
        for s in SPLITS:
            X = np.stack(pool.map(vec, d[f"{s}_images"], chunksize=32)).astype(np.float32)
            res[f"X_{s}"], res[f"y_{s}"] = X, d[f"{s}_labels"].ravel().astype(np.int64)
            print(f"{s}: {X.shape}, {time.perf_counter() - t0:.0f}s", flush=True)
    const = int((np.vstack([res["X_train"], res["X_val"]]).std(0) == 0).sum())
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(OUT_DIR, "radiomics_128.npz"), **res)
    with open(DATA, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    meta = {"dataset": "dermamnist_128.npz", "dataset_md5": md5, "n_features": len(names), "groups": groups,
            "constant_on_train_val": const, "shapes": {k: list(v.shape) for k, v in res.items()},
            "runtime_s": round(time.perf_counter() - t0, 1), "feature_names": names}
    with open(os.path.join(OUT_DIR, "radiomics_128.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"saved {len(names)} features ({const} constant on train+val) in {meta['runtime_s']}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Handcrafted radiomics + colour features for DermaMNIST 128x128")
    p.add_argument("--workers", type=int, default=16)
    main(p.parse_args().workers)
