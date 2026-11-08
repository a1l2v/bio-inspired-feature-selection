# PneumoniaMNIST — handcrafted radiomics features + bio-inspired feature selection

## Why this study

Every earlier study (BreastMNIST A–D, RetinaMNIST R1–R2) used features from a network trained on the task.
Those features were redundant: random 10–20 features matched all 512, so no subset could beat the full set.
Wrapper feature selection is expected to help when many features are irrelevant or noisy, which is the
typical case for handcrafted radiomics features. This study tests that setting.

## Data

`dataset/pneumoniamnist_128.npz` from MedMNIST+ (Zenodo record 10519652), MD5 `05b46931834c231683c68f40c47b2971`
(matches the medmnist 3.0.2 registry). Chest X-rays, 128 × 128 grey, binary (0 = normal, 1 = pneumonia).

| Split | Images | Normal | Pneumonia |
|---|---|---|---|
| Train | 4,708 | 1,214 | 3,494 |
| Val | 524 | 135 | 389 |
| Test | 624 | 234 | 390 |

## Features

`scripts/extract_radiomics.py` → `features/radiomics_128.npz` (+ `.json` with names and provenance).
1,359 features per image, no labels used, 81 s on 16 CPU workers; 1 feature is constant on train + val.

| Group | n | Content |
|---|---|---|
| fo | 20 | first-order statistics |
| hist | 32 | 32-bin intensity histogram |
| prof | 32 | 16 row-band and 16 column-band means |
| grid | 128 | 8 first-order statistics in each 4 × 4 grid cell |
| glcm | 180 | GLCM (32 levels, d 1/2/4/8, 4 angles, 9 properties) + angle averages |
| qglcm | 72 | angle-averaged GLCM per 64 × 64 quadrant |
| lbp | 214 | uniform LBP P8R1/P16R2/P24R3, and P8R1 per grid cell |
| hog | 324 | HOG, 9 orientations, 32 × 32 cells |
| wav | 140 | first-order statistics of 7 Haar sub-bands (2 levels) |
| log | 80 | first-order statistics after LoG, σ 1/2/3/5 |
| gabor | 80 | Gabor magnitude statistics, 4 frequencies × 4 orientations |
| edge | 37 | Sobel magnitude statistics, Canny edge density |
| fft | 20 | radial and angular log-power bands |

No radiomics library was installed (pyradiomics, PyWavelets), so the features are built from scikit-image
and SciPy; GLRLM / GLSZM matrices are not included.

## Pre-check (training images only; val and test not touched)

`feature_selection/precheck.py` → `feature_selection/precheck/`. 5-fold stratified CV on the 4,708 training
images; GPU LS-SVM (RBF, gamma "scale", C = 1; primary) and 5-NN (secondary). float32 and float64 LS-SVM
predictions agree 100 % on fold 0. Runtime 32 s, GPU utilisation mean 62 %.

Decision rule, fixed in the script before it ran:
- H1 (features not interchangeable): random subsets of ≤ 100 features ≥ 1 point below all features (SVM).
- H2 (headroom): the best filter top-k or best single group ≥ 1 point above all features (SVM). GO only if H2.

Results (CV ACC; majority class 0.742):

| | SVM | KNN |
|---|---|---|
| All 1,359 features | **0.9741** | 0.9475 |
| Random 25 / 100 / 400 (mean of 20) | 0.939 / 0.963 / 0.972 | 0.919 / 0.937 / 0.947 |
| ANOVA-F top 25 / 100 / 400 / 680 | 0.906 / 0.927 / 0.963 / 0.972 | 0.903 / 0.915 / 0.938 / 0.946 |
| Best single group | grid (128): 0.9711 | **grid (128): 0.9552** |

- **H1 holds (+2.1 points):** unlike the deep features, these features are not interchangeable; small random
  subsets lose accuracy.
- **H2 fails (−0.2 points): NO GO as pre-specified.** No subset found beats all features with the SVM. The RBF
  SVM already reaches 97.4 % CV accuracy and is not hurt by the irrelevant features.
- The ANOVA filter is *worse* than random subsets at every k: it picks correlated, redundant features.
- KNN is hurt by irrelevant features: the grid group alone beats all features by 0.8 points, and HOG alone
  drops KNN to 0.896 while the SVM keeps 0.965. This is below the 1-point threshold and KNN was not decisive.

Known caveat for any later test result: PneumoniaMNIST's test split is harder than train / val (the paper's
ResNet-18 (28) reaches only 0.854 test ACC), so CV accuracy on train overstates test accuracy.
