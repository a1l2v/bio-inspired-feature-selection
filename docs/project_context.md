# Project Context — Bio-Inspired Feature Selection on MedMNIST (BreastMNIST)

> **Status:** All plan steps complete (Section 11) **Last updated:** 31 Oct 2026
>
> Features for every algorithm come from the paper's **released** ResNet-18 (28) weights (`baseline/weights/`), not from our retrained models; the retraining runs only check reproducibility.

---

## 1. Project goal

Reproduce the official MedMNIST v2 ResNet-18 baseline on **BreastMNIST**, then apply **bio-inspired optimization algorithms** for **feature selection** on the ResNet-18 deep features, and benchmark whether they improve accuracy / AUC while using fewer features.

---

## 2. Base paper

| Item | Detail |
| --- | --- |
| Title | MedMNIST v2 – A large-scale lightweight benchmark for 2D and 3D biomedical image classification |
| Paper link | [https://doi.org/10.1038/s41597-022-01721-8](https://doi.org/10.1038/s41597-022-01721-8) |
| GitHub (data API) | [https://github.com/MedMNIST/MedMNIST](https://github.com/MedMNIST/MedMNIST) |
| GitHub (experiments code) | [https://github.com/MedMNIST/experiments](https://github.com/MedMNIST/experiments) |

---

## 3. Other resources

| What | Link |
| --- | --- |
| Project website | [https://medmnist.com/](https://medmnist.com/) |
| Dataset files (Zenodo) | [https://doi.org/10.5281/zenodo.5208230](https://doi.org/10.5281/zenodo.5208230) |
| Original BUSI dataset paper (source of BreastMNIST) | [https://doi.org/10.1016/j.dib.2019.104863](https://doi.org/10.1016/j.dib.2019.104863) |
| mealpy (Python library with PSO, GA, GWO, WOA, etc.) — verify link | [https://github.com/thieu1995/mealpy](https://github.com/thieu1995/mealpy) |

Install the data API:

```bash
pip install medmnist
```

---

## 4. Dataset — BreastMNIST

| Property | Value |
| --- | --- |
| Source | Al-Dhabyani et al., "Dataset of breast ultrasound images" (BUSI), *Data in Brief* 28, 2020 |
| Modality | Breast ultrasound |
| Original images | 780 images, originally 1×500×500, 3 classes (normal, benign, malignant) |
| MedMNIST format | **1×28×28 grayscale** (cubic spline resize) |
| Task | **Binary classification** — MedMNIST merges *normal + benign* into one class vs *malignant* |
| Total samples | 780 |
| Split | **546 train / 78 validation / 156 test** (official 7:1:2) |
| File | `breastmnist.npz` with keys `train_images`, `train_labels`, `val_images`, `val_labels`, `test_images`, `test_labels` (uint8) |

### Class balance (approximate, from original BUSI counts)

- Benign 437 + Normal 133 = **570** (non-malignant)
- Malignant = **210**
- → Imbalanced (\~73% vs \~27%). **Report macro-F1 and malignant sensitivity, not just accuracy.**

### Label mapping (verified from downloaded file)

- **Label 0 = malignant**, **label 1 = normal + benign** (malignant is *not* the positive class by default).
- Per split (malignant / non-malignant): train 147 / 399, val 21 / 57, test 42 / 114 → total 210 / 570.
- **Malignant recall:** use `pos_label=0` (e.g. `recall_score(y, y_pred, pos_label=0)`).
- **Malignant AUC:** score with the probability of class 0, or equivalently use class-1 probability with labels flipped. sklearn defaults treat 1 as positive.

### Local copy

- `dataset/breastmnist.npz` — downloaded from Zenodo record 10519652, MD5 `750601b1f35ba3300ea97c75c52ff8f6` (matches official).
- Load without re-downloading: `BreastMNIST(split="train", root="dataset")`.

### Loading example

```python
from medmnist import BreastMNIST
train_ds = BreastMNIST(split="train", download=True)
val_ds   = BreastMNIST(split="val",   download=True)
test_ds  = BreastMNIST(split="test",  download=True)
```

---

## 5. What the paper did — ResNet-18

- Trained **ResNet-18** as a baseline on every MedMNIST dataset.
- No feature selection, no hyperparameter search, no bio-inspired algorithm — all settings fixed by hand.
- Each experiment repeated **at least 3 times**; mean reported.
- Metrics: **AUC** and **ACC**.

### ResNet-18 results on BreastMNIST (paper Table 3)

| Method | AUC | ACC |
| --- | --- | --- |
| **ResNet-18 (28×28)** ← our baseline | **0.901** | **0.863** |
| ResNet-18 (224×224, upsampled) | 0.891 | 0.833 |

**Note:** on 156 test images, 1 image ≈ 0.64% accuracy. Small gains must be backed by repeated runs + statistics.

---

## 6. Model — ResNet-18 at 28×28

- Produces **512 features per image** (global average pooling layer) → this is the feature vector the algorithms select from.

### Official training settings (reproduce exactly)

| Setting | Value |
| --- | --- |
| Loss | Cross-entropy |
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| Epochs | 100 |
| LR schedule | ×0.1 at epoch 50 and epoch 75 |
| Batch size | 128 |
| Model selection | Early stopping on validation set |
| Input | Grayscale copied to 3 channels |
| Runs | At least 3 seeds, report mean |

**Reproduction target:** within \~1% of ACC 0.863 / AUC 0.901.

---

## 7. Method — bio-inspired feature selection

### Pipeline

1. Train ResNet-18 (28) with official settings (= reproduction).
2. Remove the final classification layer; extract the **512-dim feature vector** for every train / val / test image. Save to disk once.
3. Baseline: train a classifier (SVM, or KNN / logistic regression) on **all 512 features**.
4. For each bio-inspired algorithm: search for the best **binary mask** over the 512 features.
5. Train the classifier on the selected features → evaluate **once** on the test set.
6. Repeat each algorithm **10 times** (different seeds); report mean ± std.

### Encoding

- Each agent (particle / wolf / whale / chromosome) = vector of 512 values.
- Converted to a binary mask (1 = keep feature, 0 = drop) using a threshold (> 0.5) or a transfer function (S-shaped sigmoid or V-shaped).

### Fitness function (minimize)

```
fitness = α × error_rate + (1 − α) × (number_of_selected_features / 512)
```

- `error_rate` = 1 − accuracy (or 1 − macro-F1) on the **validation set only**.
- **α = 0.99** (main setting): accuracy dominates, feature reduction is a tie-breaker. Typically keeps \~20–50% of features (≈100–250 of 512).
- Lower α (e.g. 0.9) → stronger reduction (\~5–15%), possible small accuracy drop. *(Expectation not met here — see step 10: α ∈ [0.90, 0.99] barely changes the masks found.)*
- **Ablation:** run α ∈ {0.90, 0.95, 0.99} to show the trade-off.
- The **percentage of features kept is decided by the algorithm**, not fixed in advance. Report it as a result.

---

## 8. Bio-inspired algorithms (final 4: PSO, GA, GWO, WOA — decided 31 Oct 2026; BPSO S- and V-shaped added as extras)

### Main candidates (recommended)

| Algorithm | Inspiration | Why include |
| --- | --- | --- |
| **PSO** – Particle Swarm Optimization | Bird flocking | Classic, most cited, simple |
| **GA** – Genetic Algorithm | Natural evolution | Classic evolutionary baseline |
| **GWO** – Grey Wolf Optimizer | Wolf pack hunting | Very popular for feature selection in medical imaging |
| **WOA** – Whale Optimization Algorithm | Humpback whale bubble-net hunting | Popular, frequently compared |

### Alternatives

| Algorithm | Inspiration | Why include |
| --- | --- | --- |
| **HHO** – Harris Hawks Optimization | Cooperative hawk hunting | Widely used binary versions for feature selection; strong recent results |
| **SSA** – Salp Swarm Algorithm | Salp chain movement | Very common in feature selection studies |
| **DE** – Differential Evolution | Evolutionary mutation / crossover | Strong, robust optimizer; frequent feature selection baseline |
| **ACO** – Ant Colony Optimization | Ant foraging trails | Naturally suited to discrete (select / drop) problems |

---

## 9. Fair benchmark protocol

| Rule | Value |
| --- | --- |
| Equal budget | Same population and iterations for every algorithm, e.g. **20 agents × 50 iterations** |
| Fitness data | Validation set only (never test) |
| Test set | Used **once** per run, at the end |
| Runs | **10 independent runs** per algorithm (different seeds) |
| Report | Mean ± std |
| Classifier | Same for all (e.g. SVM), same settings |
| Control baseline | **Random search** with the same budget (non-bio) — shows gains come from the algorithm, not just from searching |

### Metrics to report

- Accuracy (ACC)
- AUC
- Macro-F1
- Malignant sensitivity (recall)
- Number and % of features selected
- Runtime per run

### Statistics and plots

- **Wilcoxon signed-rank test**: each algorithm vs baseline (and vs each other)
- **Convergence curves**: fitness vs iteration for all algorithms
- Bar chart: accuracy vs number of features

---

## 10. Results table template

| Method | ACC (mean ± std) | AUC | Macro-F1 | Malignant recall | # Features (%) | Runtime |
| --- | --- | --- | --- | --- | --- | --- |
| ResNet-18 (28) — paper | 0.863 | 0.901 | – | – | 512 (100%) | – |
| ResNet-18 (28) — released weights, re-evaluated | 0.863 ± 0.015 | 0.901 ± 0.015 | 0.821 ± 0.017 | 0.698 ± 0.014 | 512 (100%) | – |
| ResNet-18 (28) — our reproduction (10 runs, retrained) | 0.837 ± 0.050 | 0.899 ± 0.020 |  |  | 512 (100%) | ~1.2 min/run (RTX 4060) |
| All 512 features + SVM (released-weights features, 3 sets) | 0.859 ± 0.013 | 0.885 ± 0.010 | 0.813 ± 0.014 | 0.675 ± 0.028 | 512 (100%) | < 0.1 s |
| Random search + SVM (30 runs) | 0.858 ± 0.009 | 0.886 ± 0.008 | 0.813 ± 0.010 | 0.679 ± 0.020 | 230 (45.0%) | 4.7 s |
| PSO + SVM (30 runs) | 0.858 ± 0.010 | 0.886 ± 0.006 | 0.813 ± 0.011 | 0.679 ± 0.021 | 191 (37.2%) | 4.5 s |
| GA + SVM (30 runs) | 0.859 ± 0.011 | 0.887 ± 0.009 | 0.815 ± 0.012 | 0.687 ± 0.018 | 138 (26.9%) | 4.1 s |
| GWO + SVM (30 runs) | 0.856 ± 0.012 | 0.887 ± 0.007 | 0.810 ± 0.014 | 0.683 ± 0.024 | 40 (7.9%) | 3.1 s |
| WOA + SVM (30 runs) | 0.846 ± 0.023 | 0.848 ± 0.033 | 0.796 ± 0.031 | 0.656 ± 0.059 | 3 (0.6%) | 0.8 s |
| BPSO + SVM (30 runs, extra) | 0.859 ± 0.010 | 0.886 ± 0.008 | 0.814 ± 0.010 | 0.683 ± 0.014 | 210 (41.0%) | 4.6 s |
| BPSO V-shaped + SVM (30 runs, extra) | 0.857 ± 0.008 | 0.886 ± 0.009 | 0.812 ± 0.009 | 0.678 ± 0.020 | 193 (37.7%) | 4.4 s |

30 runs = 3 released-weights feature sets × 10 seeds; budget exactly 1000 evals/run (20 × 50); α = 0.99, error = 1 − val ACC; per-run records in `feature_selection/results/runs.jsonl`.

**First-pass findings (31 Oct 2026, Wilcoxon on 30 paired runs, uncorrected):**

- No method improves test ACC over all 512 features. The defensible result is **feature reduction at equal accuracy**: GA keeps 27%, GWO 8% with no significant loss.
- GA vs all-512: AUC +0.002 (p = 0.008), malignant recall +0.012 (p = 0.022); vs random search AUC p = 0.066. **Not significant after Holm correction** (step 9).
- No bio-inspired method is significantly better than random search on any metric.
- After Holm (28 tests per metric, with both BPSO variants), the only significant results are WOA's AUC loss (vs all-512 and vs every other method) and small specificity drops vs all-512 for GWO (−0.008) and V-shaped BPSO (−0.004).
- **BPSO (added later, own implementation — not in mealpy):** binary PSO, sigmoid transfer, c1 = c2 = 2, w 0.9 → 0.4, |v| ≤ 6. Behaves like random search: ~41% features, val 0.880, test ACC 0.859 / AUC 0.886; no significant difference from random search or all-512 on any metric. Known weakness of S-shaped BPSO — bits are re-sampled every iteration with p = sigmoid(v) ≈ 0.5, so the swarm never concentrates.
- **V-shaped BPSO (extra, own implementation):** same swarm, bit flipped with p = |tanh(v)| (Mirjalili & Lewis 2013, V2). Barely different: ~38% features (vs 41%), val 0.880, test ACC 0.857 / AUC 0.886; not significantly different from S-shaped BPSO or random search. The transfer function is not the bottleneck: at α = 0.99 most masks tie on val ACC (78 images, 1.3% steps), so the swarm gets almost no signal to move, and the 0.01 feature term only nudges. GWO/WOA shrink masks because their continuous dynamics drive values toward the bounds, not because of the fitness signal.
- **Fitness-signal diagnostic (no test data used; 200–300 random masks, feature set 1):**
  - 5-fold CV on train only: every mask scores 1.000 (train features memorised by the ResNet) → useless.
  - Pooled train+val CV: 0.984–0.987, dominated by train rows → useless.
  - Every accuracy-based fitness (holdout, CV scored on val rows, val 5-fold ×3) gives only 2–6 distinct values over 200 masks.
  - Continuous val scores give many distinct values (AUC 35, hinge loss 294) but they are not reproducible: split-half rank agreement between two halves of val is ρ = +0.14 (AUC) and −0.24 (hinge).
  - Conclusion: random subsets of these 512 features are statistically equivalent on this data; differences between masks are validation noise. No fitness design (CV or finer metric) can be expected to find a genuinely better subset — the defensible claim is feature reduction at equal accuracy.
- **WOA overfits validation:** best val ACC (0.906) with ~3 features, but significantly worse test ACC / AUC / macro-F1 than both baselines (p ≤ 0.016). Val–test gap grows with search aggressiveness: random 0.022, PSO 0.021, GA 0.023, GWO 0.033, WOA 0.060.

---

## 11. Step-by-step plan

1. [x] Set up environment — done locally instead of Colab: `.venv` with CUDA torch, medmnist, RTX 4060 (`mealpy`, `scikit-learn` still to add)
2. [x] Copy official `MedMNIST/experiments` code to `baseline/`, run ResNet-18 (28) on BreastMNIST × 10 runs (see `baseline/SOURCE.md`)
3. [x] Confirm reproduction — released weights match exactly (0.901 / 0.863); retrained AUC 0.899 matches, ACC 0.837 lower but not significant (p = 0.13)
4. [x] Extract and save 512-dim features (train / val / test) — `scripts/extract_features.py`, released weights → `features/resnet18_28_{1,2,3}.npz` (+ `.json` provenance). Keys `X_/y_/prob_{train,val,test}`. Verified: original head on saved features gives the exact released-weights scores; labels match dataset; deterministic; no dead features. Feature scales differ by run (max ≈ 5.7 / 20.8 / 6.3) → standardise (fit on train) before SVM / KNN.
5. [x] Baseline: all features + SVM — `svm_baseline/svm_all_features.py` → `results.csv`, `summary.json`. StandardScaler (train) + SVC(rbf, C=1, gamma="scale"), fixed, fit on train, AUC from decision_function. SVM uses only 24–33 support vectors of 546 (train features near-separable), and roughly matches the original head: same test ACC on 2 of 3 sets, AUC −0.017 on average.
6. [x] Implement fitness function (α = 0.99) — `feature_selection/fitness.py`, class `FeatureSelectionFitness(feature_file, alpha, metric="acc"|"macro_f1", transfer="threshold"|"sigmoid")`. Reuses the baseline SVM + metrics; scales once, slices columns per mask; caches repeated masks; counts evaluations (`n_evals`) and keeps a best-so-far curve (`history`); `evaluate_test(mask)` is separate and not counted. Empty mask → 1.0. Checked: all-ones mask reproduces baseline val ACC 0.8590 (fitness 0.14962). ~4.7 ms/eval → 1000-eval run ≈ 5 s; full benchmark (5 methods × 10 runs × 3 feature sets) ≈ 12 min. One val image = 0.0127 fitness, more than the whole feature term (≤ 0.01), so at α = 0.99 accuracy always wins. Early warning: best of 200 random masks beat all features on val (0.1317 vs 0.1496) but not on test (ACC 0.865 vs 0.872, AUC 0.872 vs 0.875) — val overfitting is likely.
7. [x] Decide final 4 algorithms (Section 8) — **PSO, GA, GWO, WOA** (mealpy 3.0.3 `OriginalPSO`, `BaseGA`, `OriginalGWO`, `OriginalWOA`, default hyper-parameters)
8. [x] Run random search + 4 algorithms × 10 runs, equal budget — `feature_selection/run_benchmark.py` (resumable; hard cap at exactly 1000 evals; mealpy's probe evaluation suppressed; epoch = 49 so GWO/WOA schedules complete; same seeds across methods → paired). All 150 runs used exactly 1000 evals.
9. [x] Build results table, Wilcoxon tests, convergence curves — `feature_selection/analyse_results.py` → `feature_selection/analysis/`: `summary_table.{csv,md}`, `summary_by_set.csv`, `wilcoxon.{csv,md}` (paired, every method vs all-512 + all method pairs, Holm within metric), `convergence.png` (best-so-far fitness vs evaluations per feature set), `features_vs_accuracy.png` (% features vs test ACC and vs val ACC — replaces the planned bar chart; shows val overfitting directly).
10. [x] α ablation (0.90 / 0.95 / 0.99) — random search + PSO, GA, GWO, WOA, 30 runs per α, same seeds → `feature_selection/analysis/alpha_ablation.{csv,md}`. **α has almost no effect:** at 0.95, every method except WOA (26/30) returns exactly the same mask as at 0.99 in 30/30 runs; at 0.90, random search and PSO 30/30, GA 29/30, GWO 27/30, WOA 21/30. % features kept stays ~45 / 37 / 27 / 8 / 0.5. The expected 5–15% at α = 0.90 does not happen. Reason: one val image is worth ~125 features at α = 0.95 and ~59 at α = 0.90; the searches almost never face a trade that large, so rankings (and paths) don't change. Each algorithm's feature count is set by its search dynamics, not by α. Main tables and tests use α = 0.99 only.
12. [x] **Improvement attempt, option 1 — equal-size comparison (negative result).** `run_benchmark.py --k K --out results_fixed_k` (top-k decoding: every method keeps exactly k features; fitness = 1 − val ACC; random search also scores its first random k-subset as a no-search control) → `analyse_fixed_k.py` → `feature_selection/analysis_fixed_k/`. 450 runs (k ∈ {10, 20, 40} × RANDOM, PSO, GA, GWO, WOA × 30), all exactly 1000 evals. Design fixed before running; test used once per run.
    - **No algorithm beats random search at any k** on any test metric (smallest raw p = 0.16, Holm p = 1.0).
    - No algorithm beats all 512 features; the only significant results are specificity *losses* vs all-512 (−0.006 to −0.010, 10 of 12 algorithm×k cells).
    - **A single random 10-feature subset with no search gets test ACC 0.855** (20: 0.857, 40: 0.857) vs 0.859 for all 512. The released-weight features are so redundant that almost any 10 of them carry the full signal, so there is nothing for selection to find. Validation ACC rises to 0.91 under search but test stays at 0.855–0.861 → pure validation fitting.
    - Implication: on these features no selection method can be shown better; next step is option 2 (frozen ImageNet ResNet-18 features + nested CV over all 780 images), where the feature space is not pre-tuned to this task.
13. [x] **Improvement attempt, option 2 — ImageNet features + nested CV (negative result).** GPU run: 180 jobs (6 methods × 30 outer folds), all searches exactly 1000 evals, 18.8 min on the RTX 4060 (GPU utilisation mean 91%, median 96%; peak 1.9 GB, 39 W) → `imagenet_nested_cv/results_gpu_lssvm/`, analysis `imagenet_nested_cv/analysis_gpu/`.
    - Searches raise inner-CV ACC well above all 512 (GA 0.879, GWO 0.880, PSO 0.868 vs 0.842; random search 0.858), but outer-test ACC is flat: 0.845–0.849 for every method vs 0.846 for all 512.
    - **No comparison significant after Holm on the primary (corrected) test**, ACC or AUC, algorithm vs random search or vs all 512. Smallest corrected p for an algorithm better than random search: PSO ACC, p = 0.80.
    - AUC is *lower* than all 512 for every method (−0.004 to −0.010); significant only on the optimistic Wilcoxon test (GA, GWO, PSO vs all 512, Holm p ≤ 0.021), not on the corrected test.
    - The ~3.7-point inner-CV gain does not transfer: with 1000 evaluations over ~624 images the search fits the inner-CV partition (selection bias), even though the fitness signal itself is reproducible.
    - Design as fixed before scoring:
    - **Features:** frozen torchvision ResNet-18 (`IMAGENET1K_V1`), never trained on BreastMNIST; 28×28 → 3 channels → bilinear 224×224 → ImageNet normalisation; 512-dim average-pool output. `imagenet_nested_cv/extract_imagenet_features.py` → `features/imagenet_resnet18.npz` (780 × 512).
    - **Evaluation:** nested CV over all 780 images. Outer: stratified 5-fold × 6 repeats = 30 outer folds (`random_state = repeat`). Inner fitness: stratified 5-fold CV accuracy on the outer-train part (~624 images), partition fixed per outer fold and shared by all methods. StandardScaler fit on outer-train.
    - **Fitness:** 0.99 × (1 − inner-CV ACC) + 0.01 × k/512, threshold encoding (> 0.5), 1000 evaluations (20 × 50), same SVM (RBF, C = 1, γ = "scale").
    - **Methods:** ALL512 (no selection), RANDOM, PSO, GA, GWO, WOA (mealpy defaults); seed = 1000 × repeat + fold, identical across methods → paired.
    - **Hypotheses (primary metrics test ACC and AUC):** each algorithm > RANDOM; each algorithm > ALL512; also RANDOM vs ALL512. 9 comparisons per metric.
    - **Tests:** primary = corrected resampled t-test (Nadeau & Bengio 2003; variance × (1/J + n_test/n_train), J = 30, ratio 156/624), two-sided; secondary = Wilcoxon signed-rank. Holm within each metric. Macro-F1, malignant recall, specificity reported as secondary.
    - **Pre-check (outer-train of one fold only, no test data):** all 512 inner-CV ACC 0.841; random 50% masks rank-agree across two inner partitions ρ = +0.47 (released features: ≈ 0); random 20-feature masks ρ = +0.89 and lose 5.6 points (0.785) → fitness signal is real and selection has room.
    - **Design amendment (1 Nov 2026) — classifier changed to a GPU LS-SVM, everything re-run.** Reason: speed and the user's request to use the GPU; scikit-learn SVC is CPU-only. **Disclosure:** made after 120 CPU-SVC jobs had been scored (ALL512, RANDOM, PSO, GA; test ACC 0.850 / 0.844 / 0.846 / 0.848), so it is not blind to those partial results. Those runs (plus 8 GWO folds completed before the run was stopped) are kept unchanged in `imagenet_nested_cv/results_cpu_svc/` and are not mixed with the new ones.
      - New classifier: least-squares SVM (kernel ridge classifier), RBF, γ = "scale" per training set, C = 1, bias term; one batched linear solve for all inner folds on the GPU (`imagenet_nested_cv/gpu_lssvm.py`). Every method, including ALL512, uses it.
      - Validation (outer-train of one fold, no test data): GPU result identical to a NumPy float64 reference on 12 masks; inner-CV ACC vs sklearn SVC 0.835 vs 0.841 (all 512), 0.793 vs 0.791 (20-feature masks); mask rankings agree ρ = +0.91; 9.8 ms vs ~106 ms per evaluation.
      - Everything else (folds, seeds, fitness, budget, methods, hypotheses, tests) unchanged.
14. [x] **Improvement attempt, option 2b — fixed k = 20 on ImageNet features (negative result).** 150 GPU jobs, all exactly 1000 evals → `imagenet_nested_cv/results_gpu_lssvm_k20/`, `imagenet_nested_cv/analysis_gpu_k20/`. Wall time 87.4 min, but with a ~75-min gap between jobs 30 and 60 (the machine most likely slept); the GPU log covers 872 s of active running at 96% mean utilisation.
    - Test ACC: single random 20-subset 0.793, random search 0.813, PSO 0.816, GA 0.815, GWO 0.816, WOA 0.809, all 512 0.846. AUC: 0.799 / 0.815 / 0.814 / 0.818 / 0.821 / 0.818 vs 0.885.
    - **No algorithm beats random search** (corrected p ≥ 0.78) — again GA raises inner-CV ACC (0.868 vs 0.837) without any test gain.
    - **Algorithms beat a single unsearched random 20-subset by ~2 points, but not significantly** on the primary corrected test (p = 0.29–0.50; Wilcoxon Holm p ≥ 0.051). Random search gets the same gain, so it comes from searching, not the algorithms.
    - Every 20-feature method is significantly *worse* than all 512 on AUC (−0.064 to −0.071, Holm p ≤ 0.008); WOA also on ACC and macro-F1.
    - Design as fixed before scoring:
    - Same ImageNet features, outer folds (5 × 6 = 30), inner 5-fold CV, GPU LS-SVM, seeds (1000 × repeat + fold) as step 13.
    - Every method keeps exactly k = 20 features (top-k decoding); fitness = 1 − inner-CV ACC; 1000 evaluations. k = 20 chosen from the step-13 pre-check (outer-train only): random 20-feature subsets lose ~5.6 points and rank-agree ρ = 0.89 across inner partitions. Only one k, to limit multiplicity.
    - Methods: RANDOM, PSO, GA, GWO, WOA. References: SINGLE = the first random 20-subset drawn by RANDOM (no search), scored on the outer test; ALL512 = reused from step 13 (identical deterministic computation on the same folds).
    - Hypotheses (primary metrics test ACC, AUC): each algorithm vs RANDOM, vs SINGLE, vs ALL512 (12 comparisons per metric). Primary test: corrected resampled t-test; secondary Wilcoxon; Holm within each metric.
    - Results → `imagenet_nested_cv/results_gpu_lssvm_k20/`, analysis → `imagenet_nested_cv/analysis_gpu_k20/`.
11. [x] Write up findings — `docs/findings_writeup.md`, rewritten 1 Nov 2026 as one paper covering Studies A–D (A = steps 1–10, B = step 12, C = step 13, D = step 14): controlled negative result, synthesis table, disclosed design amendment, 15 references. Numbers cross-checked against all analysis outputs. Mealpy citation volume/pages still to verify.
15. [x] **Other datasets.** RetinaMNIST R1–R2 (deep features, negative) → `Retina/README.md`; PneumoniaMNIST and DermaMNIST radiomics pre-checks → `Pneumonia/README.md`, `Derma/README.md`; **DermaMNIST Study D1 (fixed 50 radiomics features): GA and CSO significantly beat random search and the ANOVA filter on test ACC and AUC (all 8 primary tests, Holm p ≤ 1e-05)** → `Derma/README.md`.
16. [x] **Project write-up** — `docs/project_writeup.md` (2 Nov 2026): all four datasets, deep-feature nulls, pre-checks, D1 positive result with post-hoc test-set bootstrap, 25 references (MedMNIST+ and mealpy citations to verify).
17. [x] **DermaMNIST Study D2 (KNN, requested 2 Nov 2026):** same design as D1 with 5-NN; all 8 primary tests significant again; GA matches all 1,820 features on ACC with 50; absolute scores ~5 (ACC) and ~12 (AUC) points below the SVM → `Derma/README.md`, write-up §6.4.

---

## 12. Pitfalls to avoid

- **Data leakage:** never use the test set inside the fitness function.
- **Single-run claims:** with 156 test images, always repeat runs and test significance.
- **Unequal budgets:** every algorithm must get the same number of fitness evaluations.
- **Accuracy only:** the dataset is imbalanced — report macro-F1 and malignant recall.
- **No non-bio baseline:** without random search, reviewers can't tell if the algorithm matters.
- **Newer results exist:** later papers report higher BreastMNIST accuracy than the original baseline; frame the claim as improving the official ResNet-18 baseline.

---

## 13. Note on terminology

PSO, GA, GWO, WOA, etc. are **bio-inspired / nature-inspired metaheuristic algorithms**, not *bioinformatics* (genomic / protein data analysis). Use "bio-inspired" or "metaheuristic" in the write-up.