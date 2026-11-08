# DermaMNIST — handcrafted radiomics + colour features, bio-inspired feature selection

Follows the PneumoniaMNIST pre-check (`Pneumonia/README.md`), which failed its GO rule because the RBF SVM
was already at 97.4 % CV accuracy. DermaMNIST was chosen as a harder task (ResNet-18 (28) test ACC 0.735 in
the MedMNIST v2 paper) with colour, where feature usefulness should vary more.

## Data

`dataset/dermamnist_128.npz` from MedMNIST+ (Zenodo record 10519652), MD5 `2defd784463fa5243564e855ed717de1`
(matches the medmnist 3.0.2 registry; downloaded in 8 parallel byte ranges and concatenated). Dermatoscopic
images (HAM10000), 128 × 128 RGB, 7 classes.

| Split | Images | AKIEC | BCC | BKL | DF | MEL | NV | VASC |
|---|---|---|---|---|---|---|---|---|
| Train | 7,007 | 228 | 359 | 769 | 80 | 779 | 4,693 | 99 |
| Val | 1,003 | 33 | 52 | 110 | 12 | 111 | 671 | 14 |
| Test | 2,005 | 66 | 103 | 220 | 23 | 223 | 1,341 | 29 |

## Features

`scripts/extract_radiomics.py` → `features/radiomics_128.npz` (+ `.json`). 1,820 features per image, no labels
used, 168 s on 16 CPU workers; 16 are constant on train + val (empty histogram bins).

- 1,359 grey features: the full PneumoniaMNIST set on the luminance image (`grey_*` groups).
- `col` (180): first-order statistics of R, G, B, H, S, V, L*, a*, b*.
- `chist` (144): 16-bin histograms of the same 9 channels.
- `cgrid` (128): mean and std of H, S, a*, b* in each 4 × 4 grid cell.
- `cr` (9): centre (inner 64 × 64) vs surrounding ring, mean of L*, a*, b* and the difference.

## Pre-check (training images only; val and test not touched)

Same script and decision rule as PneumoniaMNIST (`Pneumonia/feature_selection/precheck.py`, generalised with
`--features --out --n_classes`; re-running it on PneumoniaMNIST reproduces all 98 recorded values exactly).
Output in `feature_selection/precheck/`. Runtime 49 s, GPU utilisation mean 91 % (median 99 %), peak 4.9 GB.
float32 and float64 LS-SVM predictions agree 100 % on fold 0.

Results (5-fold CV ACC on 7,007 training images; majority class 0.670):

| | SVM | KNN |
|---|---|---|
| All 1,820 features | **0.7842** | **0.7257** |
| Random 25 / 50 / 100 / 200 (mean of 20) | 0.729 / 0.747 / 0.759 / 0.770 | 0.677 / 0.697 / 0.709 / 0.717 |
| Random 50, best of 20 draws | 0.755 | — |
| ANOVA-F top 25 / 50 / 100 / 200 | 0.732 / 0.756 / 0.768 / 0.782 | 0.688 / 0.708 / 0.718 / 0.725 |
| Best single group | col (180): 0.762 | col (180): 0.711 |

- **H1 holds (+4.0 points):** small random subsets lose clearly; the features are far from interchangeable.
- **H2 fails (−0.3 points): NO GO as pre-specified.** No subset beats all 1,820 features in CV, for the SVM or KNN.
- Colour carries the most signal: the colour statistics alone (180) reach 0.762; the best grey group 0.712.
- Radiomics + SVM is competitive here: 0.784 CV ACC on train, against the paper's ResNet-18 (28) test ACC 0.735
  (not directly comparable: CV on train vs the test set).

## What the two pre-checks show

On both datasets, the RBF SVM (gamma "scale", standardised features) is not hurt by adding weaker features,
so the full feature set is the best set and selection cannot raise accuracy above it. Selection *does* matter
at a fixed, small feature budget: at k = 50, random subsets score 0.747, the ANOVA filter 0.756 and all
features 0.784, a 3–4 point gap that a search could close. That is the case for a fixed-k study.

## Study D1 — fixed 50-feature budget (design fixed before any test result)

**Question.** With exactly 50 of the 1,820 features allowed, do bio-inspired searches choose better features
than random search and a standard filter, on the test set?

**Design check** (`feature_selection/fitness_check.py`, training images only): one fitness evaluation takes
64 ms on the GPU. Over 60 fifty-feature subsets, the pool CV accuracy ranks subsets like accuracy on the
4,007 training images outside the pool: Spearman ρ = 0.83 overall, 0.72 among uniform-random subsets. On
RetinaMNIST a related check (agreement between two CV partitions of the same images) gave only −0.06 to +0.33, and that study was null.

**Per run** (30 runs, seed = run index 0–29; every method in a run shares the seed, pool and folds):
- Standardise all features with a scaler fitted on train + val (8,010 images; no labels, no test).
- Fitness pool: stratified random 3,000 of the 7,007 training images (random_state = seed).
- Fitness: 1 − 5-fold stratified CV accuracy on the pool (random_state = seed), 7-class GPU LS-SVM
  (RBF, gamma "scale", C = 1). No size term: the size is fixed.
- Decoding: continuous position x ∈ [0, 1]^1820 → the 50 features with the largest x_j (stable sort).
- Budget: exactly 1,000 fitness evaluations, population 20, hard cap.
- Test: LS-SVM trained on train + val (8,010) with the selected 50 features, scored once on the 2,005 test images.

**Methods**
- ALL1820 (reference, deterministic, one model): all features.
- FILTER: ANOVA-F top 50 computed on the run's pool (1 evaluation, no search).
- RANDOM: 1,000 uniform random 50-subsets; keep the best fitness.
- GA: mealpy BaseGA, 49 generations × 20, default operators.
- CSO: Competitive Swarm Optimiser (Cheng & Jin 2015), φ = 0.1, same implementation as Retina R2,
  generalised to 1,820 dimensions.

**Metrics.** Primary: test ACC and test AUC (macro one-vs-rest, from the LS-SVM scores). Secondary: macro-F1
and balanced accuracy.

**Primary hypotheses** (per primary metric, two-sided paired Wilcoxon over the 30 runs, Holm over 4 tests):
GA vs RANDOM, CSO vs RANDOM, GA vs FILTER, CSO vs FILTER.

**Secondary** (reported, not corrected): GA and CSO vs ALL1820 (each run against the single ALL1820 value),
GA vs CSO, FILTER vs RANDOM, and all comparisons on macro-F1 and balanced accuracy.

**Expected.** At k = 50, training CV shows random 0.747, filter 0.756 and all features 0.784, so the searches
have up to ~3–4 points to gain over the controls. Matching ALL1820 is not expected and is not a hypothesis.

### Study D1 — results

`feature_selection/run_fixed_k.py` → `results_d1/runs.jsonl`; `feature_selection/analyse_d1.py` → `analysis_d1/`.
121 runs (ALL1820 1, FILTER 30, RANDOM 30, GA 30, CSO 30); every search used exactly 1,000 evaluations.
121 minutes; GPU utilisation mean 99 % (median 100 %), peak 6.4 GB, mean 58 W.

**Launch incident (disclosed).** The first launch used 8 worker processes. At about 1.5 GB per worker the card
ran out of its 8 GB, Windows spilled GPU memory into system RAM, and the run stalled (5 jobs in 33 minutes,
100 % "utilisation" at 21 W). It was stopped. The 5 finished results (ALL1820 and FILTER runs 7–10; the
selection and model are deterministic per seed) were kept, and the runner was given a hard per-worker GPU
memory cap, a cache release after each job and 4 workers. Nothing in the design changed. One more FILTER/GA
smoke test at a 40-evaluation budget was run to measure memory (not recorded).

Test results (mean ± std over 30 runs; ALL1820 is one model):

| Method | Pool CV ACC | Features | Test ACC | Test AUC | Macro-F1 | Bal. ACC |
|---|---|---|---|---|---|---|
| All 1,820 features | 0.760 | 1,820 | 0.803 | 0.954 | 0.541 | 0.480 |
| ANOVA filter top 50 | 0.741 ± 0.006 | 50 | 0.765 ± 0.004 | 0.928 ± 0.004 | 0.496 ± 0.016 | 0.442 ± 0.015 |
| Random search | 0.752 ± 0.004 | 50 | 0.770 ± 0.006 | 0.925 ± 0.007 | 0.458 ± 0.034 | 0.410 ± 0.026 |
| **GA** | 0.776 ± 0.005 | 50 | **0.783 ± 0.006** | **0.936 ± 0.006** | 0.516 ± 0.037 | 0.462 ± 0.032 |
| **CSO** | 0.770 ± 0.005 | 50 | **0.782 ± 0.006** | **0.935 ± 0.006** | **0.521 ± 0.037** | **0.464 ± 0.032** |

**Primary hypotheses — all 8 supported** (two-sided paired Wilcoxon, Holm over 4 tests per metric):

| Metric | Comparison | Mean diff | Better / tied / worse | p | Holm p |
|---|---|---|---|---|---|
| ACC | GA vs random search | +0.0138 | 27 / 0 / 3 | 5.0e-06 | 9.9e-06 |
| ACC | CSO vs random search | +0.0127 | 27 / 0 / 3 | 5.2e-06 | 9.9e-06 |
| ACC | GA vs filter | +0.0188 | 30 / 0 / 0 | 1.7e-06 | 6.9e-06 |
| ACC | CSO vs filter | +0.0177 | 29 / 1 / 0 | 1.8e-06 | 6.9e-06 |
| AUC | GA vs random search | +0.0116 | 26 / 0 / 4 | 2.0e-06 | 6.0e-06 |
| AUC | CSO vs random search | +0.0104 | 26 / 0 / 4 | 5.7e-07 | 2.3e-06 |
| AUC | GA vs filter | +0.0083 | 25 / 0 / 5 | 6.0e-06 | 6.0e-06 |
| AUC | CSO vs filter | +0.0071 | 27 / 0 / 3 | 2.8e-06 | 6.0e-06 |

Secondary (uncorrected):
- Macro-F1 and balanced accuracy: GA and CSO beat random search (+5–6 points, p ≤ 1e-05) and the filter
  (+2 points, p ≤ 0.013).
- GA vs CSO: no difference on any metric (p ≥ 0.48).
- Against all 1,820 features, both searches are lower in every run: ACC −2.0 to −2.1 points, AUC −1.7 to
  −1.9 points, using 2.7 % of the features. Matching ALL1820 was not a hypothesis.
- Filter vs random search: the filter has lower ACC (−0.5 points, p = 0.002) but higher macro-F1 and balanced
  accuracy (+3–4 points).

**Conclusion.** At a fixed budget of 50 handcrafted radiomics features, GA and CSO choose features that give
significantly higher test accuracy and AUC than both random search and the ANOVA filter, consistently across
30 independent runs. The difference from the earlier null studies is the setting, predicted by the pre-checks:
these features are not interchangeable (random 50 lose 4 points), and the fitness ranks subsets reliably
(ρ = 0.83). Neither search reaches the accuracy of all 1,820 features.

### Study D1 — post-hoc robustness check (not part of the design)

`feature_selection/bootstrap_d1.py` → `analysis_d1/bootstrap.{json,md}`. The 30 runs share one test set, so the
Wilcoxon tests do not cover test-set sampling noise. Re-fitting every run's saved 50 features reproduced the stored
test scores (max deviation 4.9e-06). Over 2,000 paired bootstrap resamples of the 2,005 test images, every
GA / CSO vs RANDOM / FILTER difference in mean ACC and AUC has a 95 % interval above zero. The weakest are AUC vs
the filter: GA +0.0083 [+0.0017, +0.0150], CSO +0.0071 [+0.0004, +0.0134]. Figure: `analysis_d1/test_scores.png`
(`plot_d1.py`).

## Study D2 — the same 50-feature study with a KNN classifier (design fixed before any D2 result)

**Why.** Requested after D1: replace the SVM with k-nearest neighbours. Stated in advance, from the D1
pre-check (training CV): KNN is about 5 points less accurate than the SVM at every feature-set size (all
features 0.726 vs 0.784; random 50 0.697 vs 0.747), its AUC uses only 6 possible vote levels per class, and the
gap between random 50 and all features is smaller (2.9 vs 3.7 points). **Expected: lower ACC and AUC than D1;
whether the GA / CSO margins over the controls are larger is uncertain.** D1 remains the primary result.

**Design.** Identical to D1 (same 30 seeds, pools, folds, scaler, decoding, budget, methods, hypotheses, Holm
family and metrics) except the classifier, in both the fitness and the final test model:
- **KNN:** k = 5, Euclidean distance on the standardised selected features, uniform votes. The prediction is
  the class with the most votes, ties going to the lower class index (as in scikit-learn's
  `KNeighborsClassifier`). Class scores for AUC are the vote fractions (scikit-learn's `predict_proba`).
- **Fitness:** 1 − 5-fold CV accuracy of this KNN on the run's 3,000-image pool, computed on the GPU.
- **Test:** KNN with the selected features, neighbours drawn from train + val (8,010 images), scored once on
  the 2,005 test images.
- Primary hypotheses (test ACC and AUC, Holm over 4 per metric): GA vs RANDOM, CSO vs RANDOM, GA vs FILTER,
  CSO vs FILTER. Secondary as in D1.

Outputs: `feature_selection/results_d2/`, `feature_selection/analysis_d2/`.

### Study D2 — results

`run_fixed_k.py --classifier knn --workers 8` → `results_d2/runs.jsonl`; `analyse_d1.py --study d2` →
`analysis_d2/`. 121 runs, every search exactly 1,000 evaluations; 7 minutes, GPU utilisation mean 89 % (median
99 %), peak 2.0 GB. Before the run, the GPU KNN was checked against scikit-learn's `KNeighborsClassifier`
(identical CV accuracy on 5 masks; identical vote fractions train → val), and the LS-SVM path re-ran a D1 job
identically. No test data were used in these checks.

| Method | Pool CV ACC | Features | Test ACC | Test AUC | Macro-F1 | Bal. ACC |
|---|---|---|---|---|---|---|
| All 1,820 features | 0.697 | 1,820 | 0.732 | 0.834 | 0.420 | 0.426 |
| ANOVA filter top 50 | 0.690 ± 0.007 | 50 | 0.714 ± 0.005 | 0.806 ± 0.010 | 0.420 ± 0.015 | 0.401 ± 0.012 |
| Random search | 0.707 ± 0.004 | 50 | 0.718 ± 0.008 | 0.795 ± 0.014 | 0.393 ± 0.025 | 0.384 ± 0.019 |
| **GA** | 0.733 ± 0.006 | 50 | **0.733 ± 0.009** | 0.813 ± 0.016 | **0.432 ± 0.034** | 0.415 ± 0.027 |
| **CSO** | 0.724 ± 0.009 | 50 | 0.731 ± 0.010 | **0.815 ± 0.013** | 0.429 ± 0.028 | 0.413 ± 0.021 |

**Primary hypotheses — all 8 supported** (Holm over 4 per metric):

| Metric | Comparison | Mean diff | Better / tied / worse | p | Holm p |
|---|---|---|---|---|---|
| ACC | GA vs random search | +0.0147 | 29 / 0 / 1 | 2.3e-06 | 8.5e-06 |
| ACC | CSO vs random search | +0.0133 | 26 / 0 / 4 | 1.6e-05 | 1.6e-05 |
| ACC | GA vs filter | +0.0185 | 29 / 0 / 1 | 2.1e-06 | 8.5e-06 |
| ACC | CSO vs filter | +0.0171 | 28 / 0 / 2 | 2.6e-06 | 8.5e-06 |
| AUC | GA vs random search | +0.0183 | 25 / 0 / 5 | 1.8e-05 | 5.5e-05 |
| AUC | CSO vs random search | +0.0197 | 25 / 0 / 5 | 6.9e-06 | 2.8e-05 |
| AUC | GA vs filter | +0.0072 | 22 / 0 / 8 | 0.035 | 0.035 |
| AUC | CSO vs filter | +0.0085 | 19 / 0 / 11 | 0.015 | 0.031 |

Secondary (uncorrected):
- **Against all 1,820 features, GA and CSO now match the full model on ACC** (+0.0004, p = 0.92; −0.0010,
  p = 0.56) and on macro-F1 (+0.010 to +0.012, p ≥ 0.12), with 2.7 % of the features. AUC stays lower
  (−0.020 to −0.021, p < 1e-05).
- GA vs CSO: no difference (p ≥ 0.53).
- Macro-F1 and balanced accuracy: GA and CSO beat random search (+3–4 points, p ≤ 1e-05); against the filter,
  balanced accuracy +1.3–1.4 points (p ≤ 0.021), macro-F1 not significant.

**D2 vs D1 (descriptive; different studies, not a pre-specified test).** As stated in advance, KNN is much
weaker than the SVM in absolute terms: GA test ACC 0.733 vs 0.783 and AUC 0.813 vs 0.936. The margins over the
controls are similar: ACC vs random search +1.5 vs +1.4 points, vs the filter +1.8 vs +1.9; AUC vs random search
larger (+1.8 vs +1.2), AUC vs the filter weaker (+0.7, Holm p = 0.035, vs +0.8, Holm p = 6e-06).
