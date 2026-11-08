"""Extract handcrafted radiomics / texture features from PneumoniaMNIST 128x128 chest X-rays.

Every image (uint8, 128 x 128, grey) gives the same fixed, ordered feature vector. Nothing is learned from
the labels, so extraction cannot leak test information. Groups (feature counts in brackets):

    fo_*        first-order statistics of the whole image (20)
    hist_*      32-bin intensity histogram, fixed range 0-255 (32)
    prof_*      mean intensity of 16 horizontal and 16 vertical bands (32)
    grid_*      first-order subset (8 stats) in each cell of a 4 x 4 grid (128)
    glcm_*      GLCM, 32 grey levels, distances 1/2/4/8 x angles 0/45/90/135, 9 properties (144),
                plus the angle-average per distance (36)
    qglcm_*     angle-averaged GLCM, distances 1/4, in each 64 x 64 quadrant (72)
    lbp_*       uniform LBP histograms P8R1 / P16R2 / P24R3 (54), and P8R1 in each 4 x 4 grid cell (160)
    hog_*       HOG, 9 orientations, 32 x 32 cells, 2 x 2 blocks (324)
    wav_*       first-order statistics of 7 Haar wavelet sub-bands, 2 levels (140)
    log_*       first-order statistics after Laplacian of Gaussian, sigma 1/2/3/5 (80)
    gabor_*     5 statistics of the Gabor magnitude, 4 frequencies x 4 orientations (80)
    edge_*      Sobel-magnitude first-order (20), Canny edge density global + 4 x 4 grid (17)
    fft_*       log power in 16 radial and 4 angular bands of the 2-D spectrum (20)

NaN (e.g. skewness of a constant cell) is replaced by 0.

Output: Pneumonia/features/radiomics_128.npz  X_train, X_val, X_test (float32), y_train, y_val, y_test
        Pneumonia/features/radiomics_128.json feature names, group counts, provenance

Usage (from project root):
    .venv\\Scripts\\python.exe Pneumonia\\scripts\\extract_radiomics.py --workers 16
"""

import argparse
import hashlib
import json
import os
import time
from multiprocessing import Pool

import numpy as np
from scipy import ndimage, signal, stats
from skimage.feature import canny, graycomatrix, graycoprops, hog, local_binary_pattern
from skimage.filters import gabor_kernel

PNEU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PNEU, "dataset", "pneumoniamnist_128.npz")
OUT_DIR = os.path.join(PNEU, "features")
SPLITS = ("train", "val", "test")
GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation", "mean", "variance", "entropy"]
ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
PCTS = [1, 5, 10, 25, 50, 75, 90, 95, 99]


def hist_entropy(v, bins=32):
    p, _ = np.histogram(v, bins=bins)
    p = p[p > 0] / v.size
    return float(-(p * np.log2(p)).sum()), float((p ** 2).sum())


def first_order(v, out, name):
    v = v.ravel()
    q = np.percentile(v, PCTS)
    ent, uni = hist_entropy(v)
    vals = [v.mean(), v.std(), stats.skew(v), stats.kurtosis(v), v.min(), v.max(), *q, q[6] - q[2],
            np.abs(v - v.mean()).mean(), np.sqrt((v ** 2).mean()), ent, uni]
    keys = ["mean", "std", "skew", "kurt", "min", "max", *[f"p{p}" for p in PCTS], "iqr", "mad", "rms", "entropy",
            "uniformity"]
    out.update({f"{name}_{k}": x for k, x in zip(keys, vals)})


def small_first_order(v, out, name):
    v = v.ravel()
    p10, p50, p90 = np.percentile(v, [10, 50, 90])
    out.update({f"{name}_mean": v.mean(), f"{name}_std": v.std(), f"{name}_skew": stats.skew(v),
                f"{name}_kurt": stats.kurtosis(v), f"{name}_p10": p10, f"{name}_p50": p50, f"{name}_p90": p90,
                f"{name}_entropy": hist_entropy(v)[0]})


def glcm(q, distances, out, name, per_angle):
    m = graycomatrix(q, distances, ANGLES, levels=32, symmetric=True, normed=True)
    for prop in GLCM_PROPS:
        v = graycoprops(m, prop)                                 # distances x angles
        for i, d in enumerate(distances):
            if per_angle:
                for j, a in enumerate((0, 45, 90, 135)):
                    out[f"{name}_{prop}_d{d}_a{a}"] = v[i, j]
            out[f"{name}_{prop}_d{d}_avg"] = v[i].mean()


def lbp_hist(img, P, R):
    codes = local_binary_pattern(img, P, R, method="uniform")
    return np.bincount(codes.astype(int).ravel(), minlength=P + 2)[: P + 2] / codes.size


def haar(x):
    a, b, c, d = x[0::2, 0::2], x[0::2, 1::2], x[1::2, 0::2], x[1::2, 1::2]
    return (a + b + c + d) / 2, (a - b + c - d) / 2, (a + b - c - d) / 2, (a - b - c + d) / 2


GABOR = [(f, t, np.real(k), np.imag(k)) for f in (0.05, 0.1, 0.2, 0.4) for t in range(4)
         for k in [gabor_kernel(f, theta=t * np.pi / 4)]]


def features(img):
    x = img.astype(np.float64)
    q = (img // 8).astype(np.uint8)                              # 32 grey levels
    out = {}
    first_order(x, out, "fo")
    h = np.bincount((img // 8).ravel(), minlength=32) / img.size
    out.update({f"hist_b{i}": h[i] for i in range(32)})
    for i in range(16):
        out[f"prof_row{i}"] = x[8 * i: 8 * i + 8].mean()
        out[f"prof_col{i}"] = x[:, 8 * i: 8 * i + 8].mean()
    for r in range(4):
        for c in range(4):
            small_first_order(x[32 * r: 32 * r + 32, 32 * c: 32 * c + 32], out, f"grid_r{r}c{c}")
    glcm(q, [1, 2, 4, 8], out, "glcm", per_angle=True)
    for r in range(2):
        for c in range(2):
            glcm(q[64 * r: 64 * r + 64, 64 * c: 64 * c + 64], [1, 4], out, f"qglcm_r{r}c{c}", per_angle=False)
    for P, R in ((8, 1), (16, 2), (24, 3)):
        out.update({f"lbp_P{P}R{R}_b{i}": v for i, v in enumerate(lbp_hist(img, P, R))})
    codes = local_binary_pattern(img, 8, 1, method="uniform").astype(int)
    for r in range(4):
        for c in range(4):
            cell = codes[32 * r: 32 * r + 32, 32 * c: 32 * c + 32]
            hc = np.bincount(cell.ravel(), minlength=10)[:10] / cell.size
            out.update({f"lbp_grid_r{r}c{c}_b{i}": v for i, v in enumerate(hc)})
    out.update({f"hog_{i}": v for i, v in enumerate(hog(img, orientations=9, pixels_per_cell=(32, 32),
                                                          cells_per_block=(2, 2), feature_vector=True))})
    a1, h1, v1, d1 = haar(x)
    a2, h2, v2, d2 = haar(a1)
    for nm, band in (("L1_H", h1), ("L1_V", v1), ("L1_D", d1), ("L2_A", a2), ("L2_H", h2), ("L2_V", v2), ("L2_D", d2)):
        first_order(band, out, f"wav_{nm}")
    for s in (1, 2, 3, 5):
        first_order(ndimage.gaussian_laplace(x, s), out, f"log_s{s}")
    xc = x - x.mean()
    for f, t, kr, ki in GABOR:
        mag = np.hypot(signal.fftconvolve(xc, kr, mode="same"), signal.fftconvolve(xc, ki, mode="same")).ravel()
        nm = f"gabor_f{f}_o{45 * t}"
        out.update({f"{nm}_mean": mag.mean(), f"{nm}_std": mag.std(), f"{nm}_skew": stats.skew(mag),
                    f"{nm}_kurt": stats.kurtosis(mag), f"{nm}_energy": (mag ** 2).mean()})
    first_order(np.hypot(ndimage.sobel(x, 0), ndimage.sobel(x, 1)), out, "edge_sobel")
    e = canny(x / 255.0, sigma=2)
    out["edge_canny_density"] = e.mean()
    for r in range(4):
        for c in range(4):
            out[f"edge_canny_r{r}c{c}"] = e[32 * r: 32 * r + 32, 32 * c: 32 * c + 32].mean()
    P = np.abs(np.fft.fftshift(np.fft.fft2(xc))) ** 2
    yy, xx = np.indices(P.shape) - 64
    rad, ang = np.hypot(yy, xx), np.mod(np.arctan2(yy, xx), np.pi)
    for i in range(16):
        sel = (rad >= 4 * i) & (rad < 4 * i + 4)
        out[f"fft_rad{i}"] = np.log1p(P[sel].mean())
    for i in range(4):
        sel = (ang >= i * np.pi / 4) & (ang < (i + 1) * np.pi / 4) & (rad > 0)
        out[f"fft_ang{i}"] = np.log1p(P[sel].mean())
    return out


def vec(img):
    return np.nan_to_num(np.array(list(features(img).values()), dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def main(workers):
    t0 = time.perf_counter()
    d = np.load(DATA)
    names = list(features(d["train_images"][0]).keys())
    groups = {}
    for n in names:
        g = n.split("_")[0]
        groups[g] = groups.get(g, 0) + 1
    print(f"{len(names)} features per image: {groups}", flush=True)
    res = {}
    with Pool(workers) as pool:
        for s in SPLITS:
            imgs = d[f"{s}_images"]
            X = np.stack(pool.map(vec, imgs, chunksize=32)).astype(np.float32)
            res[f"X_{s}"], res[f"y_{s}"] = X, d[f"{s}_labels"].ravel().astype(np.int64)
            print(f"{s}: {X.shape}, {time.perf_counter() - t0:.0f}s", flush=True)
    const = int((np.vstack([res["X_train"], res["X_val"]]).std(0) == 0).sum())
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(OUT_DIR, "radiomics_128.npz"), **res)
    with open(DATA, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    meta = {"dataset": "pneumoniamnist_128.npz", "dataset_md5": md5, "n_features": len(names), "groups": groups,
            "constant_on_train_val": const, "shapes": {k: list(v.shape) for k, v in res.items()},
            "runtime_s": round(time.perf_counter() - t0, 1), "feature_names": names}
    with open(os.path.join(OUT_DIR, "radiomics_128.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"saved {len(names)} features ({const} constant on train+val) in {meta['runtime_s']}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Handcrafted radiomics features for PneumoniaMNIST 128x128")
    p.add_argument("--workers", type=int, default=16)
    main(p.parse_args().workers)
