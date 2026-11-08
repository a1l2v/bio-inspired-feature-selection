# Retina — bio-inspired feature selection on RetinaMNIST

Same pipeline as the BreastMNIST Study A (`../docs/findings_writeup.md` §4), moved to a larger and harder
dataset. Chosen because BreastMNIST (780 images, 78 validation, 156 test) was too small to show differences.

## Data

RetinaMNIST (MedMNIST v2): retinal fundus photographs, 28×28 RGB, diabetic-retinopathy grade 0–4
(ordinal, 5 classes). `dataset/retinamnist.npz`, MD5 `bd4c0672f1bba3e3a89f0e4e876791e4` (matches official).

| Split | Images | Grade 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- | --- |
| Train | 1080 | 486 | 128 | 206 | 194 | 66 |
| Validation | 120 | 54 | 12 | 28 | 20 | 6 |
| Test | 400 | 174 | 46 | 92 | 68 | 20 |

## Released weights and reproduction

`baseline/weights/resnet18_28_{1,2,3}.pth` — the three ResNet-18 (28) members of `weights_retinamnist.zip`
(Zenodo 10.5281/zenodo.7782113), downloaded by HTTP range requests, CRC-checked. MD5:
`f0a50a74bbbf2174b27c1f7918ea9e2b`, `ba1562f7dc0d96db49caecdaddd62af4`, `de3675c000b80d898723f4e49bd11f55`.

Re-evaluated with the official `../baseline/train_and_eval_pytorch.py --data_flag retinamnist --num_epochs 0`
(outputs in `baseline/output_eval/`):

| Weights | Train AUC / ACC | Val AUC / ACC | Test AUC | Test ACC |
| --- | --- | --- | --- | --- |
| resnet18_28_1 | 0.783 / 0.523 | 0.799 / 0.608 | 0.7222 | 0.5325 |
| resnet18_28_2 | 0.801 / 0.520 | 0.836 / 0.533 | 0.7195 | 0.5300 |
| resnet18_28_3 | 0.797 / 0.554 | 0.831 / 0.575 | 0.7092 | 0.5100 |
| **Mean** | | | **0.717** | **0.524** |
| Paper (Table 3) | | | 0.717 | 0.524 |

Exact match. Unlike BreastMNIST (train ACC ≈ 1.000), the network does **not** memorise its training images
(train ACC ≈ 0.52–0.55), so training features are informative for the SVM.

## Pipeline

| Step | Script | Output |
| --- | --- | --- |
| Features | `scripts/extract_features.py` | `features/resnet18_28_{1,2,3}.npz` (checked: original head on saved features = official scores) |
| All-512 SVM | `svm_baseline/svm_all_features.py` | `svm_baseline/results.csv`, `summary.json` |
| Benchmark | `feature_selection/run_benchmark.py` | `feature_selection/results/runs.jsonl` |
| Analysis | `feature_selection/analyse_results.py` | `feature_selection/analysis/` |

All-512 SVM baseline (test, mean ± std over 3 checkpoints): ACC 0.517 ± 0.014, AUC 0.665 ± 0.013,
macro-F1 0.339 ± 0.013, QWK 0.548 ± 0.013. ResNet-18 head: ACC 0.524, AUC 0.717, macro-F1 0.318, QWK 0.512.

## Benchmark design — fixed 1 Nov 2026, before any feature-selection test result was scored

- **Methods:** GA (mealpy `BaseGA`) — best main algorithm on BreastMNIST test; GWO (mealpy `OriginalGWO`) —
  strongest optimiser on BreastMNIST without collapsing (expected to do well here). Controls: RANDOM
  (random search, same budget) and ALL512 (SVM on all 512 features). mealpy defaults.
- **Fitness:** 0.99 × (1 − val ACC) + 0.01 × k/512 on the 120 validation images; threshold encoding (> 0.5);
  SVM = StandardScaler (train) + SVC(rbf, C = 1, γ = "scale"), fit on train. Exactly 1000 evaluations
  (20 × 50) per run.
- **Runs:** 3 feature sets (released checkpoints) × 10 seeds = 30 per method; seed = 100 × set + run, shared
  across methods → paired. Test scored once per run on the best mask.
- **Hypotheses (primary metrics test ACC and AUC):** GA > RANDOM, GWO > RANDOM, GA > ALL512, GWO > ALL512
  (4 comparisons per metric). Two-sided paired Wilcoxon signed-rank, Holm within each metric.
  Macro-F1 and QWK reported as secondary.
- **Amendment (1 Nov 2026, before any search-method test result existed): classifier → GPU LS-SVM.**
  At the user's request to use the GPU, the SVM is replaced by a one-vs-rest least-squares SVM on the GPU
  (`feature_selection/gpu_lssvm.py`; RBF, γ = "scale", C = 1, one solve per evaluation). Every method,
  ALL512 included, uses it. A first CPU-SVC launch was stopped after only the 3 ALL512 jobs had finished
  (identical to the SVC baseline above, nothing else scored); kept unread in
  `feature_selection/results_cpu_svc_partial/`. Validation (validation data only): GPU result identical to a
  NumPy float64 reference (6 masks, max score difference 1e-5); val ACC with all 512 features 0.567 vs SVC
  0.575; 6.3 ms vs ~108 ms per evaluation. Over random 50% masks both classifiers give only 5 distinct
  val-ACC values and rank masks almost independently (ρ = +0.04), i.e. the 120-image fitness is mostly ties
  and noise for either classifier. Everything else in the design is unchanged.
- **Known limitation, stated in advance:** the validation set is 120 images (1 image = 0.83% ACC; only
  6 of grade 4), so the fitness is coarse, as on BreastMNIST.

## Results (GPU LS-SVM, 1 Nov 2026) — negative

93 jobs (ALL512 × 3, RANDOM / GA / GWO × 30), all searches exactly 1000 evaluations; 6.8 min on the RTX 4060,
GPU utilisation mean 99% (median 100%), peak 1.5 GB, 61 W. Outputs: `feature_selection/results_gpu_lssvm/`,
`feature_selection/analysis_gpu/`.

| Method | Val ACC | # Features | Test ACC | Test AUC | Macro-F1 | QWK |
| --- | --- | --- | --- | --- | --- | --- |
| ResNet-18 head (paper weights) | — | 512 | 0.524 | 0.717 | 0.318 | 0.512 |
| All 512 features (LS-SVM) | 0.547 | 512 | 0.515 ± 0.018 | 0.713 ± 0.007 | 0.361 | 0.495 |
| Random search | 0.587 | 242 | 0.514 ± 0.017 | 0.713 ± 0.007 | 0.360 | 0.496 |
| GA | 0.615 | 199 | 0.516 ± 0.012 | 0.714 ± 0.007 | 0.360 | 0.496 |
| GWO | 0.624 | 113 | 0.512 ± 0.014 | 0.713 ± 0.008 | 0.354 | 0.493 |

- **No improvement in test ACC or AUC.** GA vs random search: ACC +0.002 (p = 0.39), AUC +0.0005 (p = 0.26);
  GA vs all 512: ACC +0.001 (p = 0.82), AUC +0.0005 (p = 0.25). GWO is level or slightly lower on both.
  Nothing significant after Holm on the primary metrics.
- Only significant result: **GWO macro-F1 −0.008 vs all 512** (Holm p = 0.037) — a small loss.
- Same pattern as BreastMNIST: GA and GWO raise validation ACC by 7–8 points (0.547 → 0.615 / 0.624) and
  random search by 4, but none of it reaches the test set — the search fits the 120 validation images.
- What does hold: **feature reduction at equal accuracy** — GWO keeps 22% (113 of 512) and GA 39% with test
  ACC and AUC unchanged; random search keeps 47%.

## Study R2 — HHO and CSO with a 1,200-image CV fitness. Design fixed 1 Nov 2026, before any R2 test result

Rationale: the R1 bottleneck was the 120-image validation fitness. Unlike BreastMNIST, the RetinaMNIST network
did not memorise its training images (train ACC ≈ 0.52), so train + validation can be pooled for an honest
cross-validated fitness.

- **Fitness:** 0.99 × (1 − 5-fold stratified CV ACC on train + val pooled, 1,200 images) + 0.01 × k/512;
  threshold encoding (> 0.5); CV partition `random_state = seed`, shared by all methods. StandardScaler fit on
  the pooled 1,200. Classifier: 5-class GPU LS-SVM (`gpu_lssvm.py`, `BatchedFoldLSSVMOvR`: all folds and
  classes in one batched solve; matches a NumPy float64 reference to within 1 of 1,200 images on 5 masks;
  27.6 ms per evaluation).
- **Test:** best mask → LS-SVM trained on all 1,200 → scored once on the 400 test images. ALL512 the same way.
- **Methods:**
  - **HHO** — Harris Hawks Optimisation (Heidari et al. 2019), mealpy `OriginalHHO`, binarised by the same
    threshold. Uses ~37 evaluations per iteration (rapid dives), so epoch = 27 (≈ 1,019 natural evaluations)
    and the hard cap stops it at exactly 1,000, keeping its escape-energy schedule ~96% complete.
  - **CSO** — Competitive Swarm Optimiser (Cheng & Jin 2015; for feature selection Gu, Cheng & Jin 2018),
    own implementation: swarm of 20 split into random pairs each iteration; each loser updates
    v ← r1·v + r2·(x_winner − x_loser) + φ·r3·(x̄ − x_loser), x ← clip(x + v, 0, 1), φ = 0.1; winners pass
    unchanged; 10 evaluations per iteration → 98 iterations.
  - **Controls:** ALL512, RANDOM (same budget). **Secondary:** GA under the same fitness (descriptive only).
- **Budget / runs:** exactly 1,000 evaluations; 3 feature sets × 10 seeds = 30 runs per method; seed =
  100 × set + run, shared → paired.
- **Primary hypotheses (test ACC and AUC):** HHO > RANDOM, CSO > RANDOM, HHO > ALL512, CSO > ALL512
  (4 per metric); two-sided paired Wilcoxon, Holm within metric. GA comparisons reported without entering
  the Holm family.
- **Pre-check (train + val only, feature set 1, no test data):** all 512 → CV ACC 0.535 / 0.544 (two
  partitions). Random ~50% masks: 18 distinct values, rank agreement across partitions ρ = −0.06 (no signal).
  Random ~20% masks: ρ = +0.33. Random 20-feature masks: mean 0.539 (≥ all 512), ρ = +0.27. → The fitness
  carries only a weak signal, and small random subsets already match all 512 features (redundancy, as on
  BreastMNIST). **Stated in advance: a significant test gain is unlikely.**
- Outputs: `feature_selection/results_r2_gpu/`, `feature_selection/analysis_r2/`.

### R2 results (1 Nov 2026) — negative

123 jobs, 46.8 min, GPU utilisation mean 92% (median 98%), peak 2.4 GB, 43 W. **Budget deviation:** 3 of 30 HHO
runs used fewer than 1,000 evaluations (996, 996, 964): HHO's evaluations per iteration vary, and epoch = 27
occasionally ends just short of the cap. All other runs used exactly 1,000.

| Method | CV ACC (fitness) | # Features | Test ACC | Test AUC | Macro-F1 | QWK |
| --- | --- | --- | --- | --- | --- | --- |
| All 512 | 0.549 | 512 | 0.522 ± 0.019 | 0.719 ± 0.007 | 0.364 | 0.508 |
| Random search | 0.563 | 249 | 0.522 ± 0.015 | 0.717 ± 0.007 | 0.364 | 0.508 |
| GA (secondary) | 0.575 | 238 | 0.523 ± 0.013 | 0.718 ± 0.007 | 0.366 | 0.515 |
| HHO | 0.567 | 80 | 0.518 ± 0.015 | 0.716 ± 0.007 | 0.358 | 0.512 |
| CSO | 0.574 | 237 | 0.523 ± 0.016 | 0.717 ± 0.006 | 0.367 | 0.510 |

- **Primary hypotheses: none supported.** CSO vs random search ACC +0.001 (p = 0.31), AUC +0.0003 (p = 0.63);
  HHO is lower than random search on ACC (−0.004, p = 0.064).
- Only significant primary result: **CSO AUC −0.0013 vs all 512** (worse in 24/30 runs, Holm p = 0.017) — tiny loss.
- The larger fitness did help the *fitness*: the CV gain over all 512 is 1.4–2.6 points (vs 7–8 points of
  validation gain in R1), i.e. far less overfitting — but there is still no test gain to find. As the
  pre-check predicted, small random subsets already match all 512 features.
- Secondary, uncorrected: GA QWK +0.007 vs all 512 (p = 0.003) and vs random search (p = 0.013); HHO below GA
  on ACC (p = 0.024) and macro-F1 (p = 0.007). These are exploratory only and would not survive correction
  across all secondary tests.
