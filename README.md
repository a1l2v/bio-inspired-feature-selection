# Bio-Inspired Feature Selection for Medical Image Classification

When do nature-inspired algorithms such as the Genetic Algorithm (GA) and the Competitive Swarm Optimiser (CSO)
actually improve a medical image classifier? This project answers that under controlled conditions on four
[MedMNIST](https://medmnist.com/) datasets: equal evaluation budgets, a random-search control, paired seeds, a
test set scored once, and Wilcoxon signed-rank tests with Holm correction.

Bio Inspired Algorithms AAT, Department of CSE (Data Science), B.M.S. College of Engineering.
Alvin S Thottappilly (1BM23CD005), Chinmayee S Bharadwaj (1BM23CD012), Taran Thimmaiah (1BM23CD065).
Guide: Manoj Kumar, Assistant Professor.

## Main result

On **DermaMNIST** (10,015 skin-lesion images, 7 classes), with **1,820 handcrafted radiomics features** and a
budget of **exactly 50 features**, GA and CSO choose significantly better features than random search and an
ANOVA filter (SVM classifier, mean of 30 runs on the 2,005 test images):

| Method | Features | Test accuracy | Test AUC |
|---|---|---|---|
| All features (reference) | 1,820 | 0.803 | 0.954 |
| ANOVA filter | 50 | 0.765 | 0.928 |
| Random search | 50 | 0.770 | 0.925 |
| **GA** | 50 | **0.783** | **0.936** |
| **CSO** | 50 | **0.782** | **0.935** |

All 8 primary tests (GA and CSO vs random search and the filter, on accuracy and AUC) are significant with both an
SVM (Holm p ≤ 1e-05) and a KNN classifier (Holm p ≤ 0.035). On deep ResNet-18 features (BreastMNIST, RetinaMNIST),
six controlled studies found no significant gain: those features are too redundant for selection to help.

## Repository layout

| Folder | Contents |
|---|---|
| `baseline/` | Official MedMNIST ResNet-18 training/evaluation script and model definition; reproduction notes |
| `scripts/`, `features/`, `svm_baseline/` | BreastMNIST deep-feature extraction and all-features SVM baseline |
| `feature_selection/` | BreastMNIST Studies A–B: PSO, GA, GWO, WOA, binary PSO; results and analysis |
| `imagenet_nested_cv/` | BreastMNIST Studies C–D: ImageNet features, nested CV, GPU least-squares SVM |
| `Retina/` | RetinaMNIST Studies R1–R2 (GA, GWO, HHO, CSO); see `Retina/README.md` |
| `Pneumonia/` | PneumoniaMNIST radiomics extractor and pre-check; see `Pneumonia/README.md` |
| `Derma/` | DermaMNIST radiomics features, Studies D1 (SVM) and D2 (KNN); see `Derma/README.md` |
| `docs/` | Project plan and log, full write-ups (`project_writeup.md`, `findings_writeup.md`) |
| `presentation/` | Presentation deck and its build script |
| `report/` | AAT project report (LaTeX source) |

## Reproducing the main study

Requirements: Python 3.11, PyTorch with CUDA, scikit-learn, scikit-image, SciPy, mealpy 3.0.3, medmnist 3.0.2.
Datasets, feature files and model weights are not stored in the repository; the scripts download or regenerate them.

```bash
# 1. Download dermamnist_128.npz from MedMNIST+ (Zenodo record 10519652) into Derma/dataset/
# 2. Extract the 1,820 radiomics features
python Derma/scripts/extract_radiomics.py --workers 16
# 3. Run Study D1 (SVM) and D2 (KNN): 30 runs each of FILTER, RANDOM, GA, CSO, plus all features
python Derma/feature_selection/run_fixed_k.py --workers 4
python Derma/feature_selection/run_fixed_k.py --classifier knn --workers 8
# 4. Analyse
python Derma/feature_selection/analyse_d1.py
python Derma/feature_selection/analyse_d1.py --study d2
```

Released MedMNIST v2 ResNet-18 weights (for the deep-feature studies) are on Zenodo (doi:10.5281/zenodo.7782113).

## References

- J. Yang et al., "MedMNIST v2 – A large-scale lightweight benchmark for 2D and 3D biomedical image classification,"
  *Scientific Data* 10, 41 (2023).
- R. Cheng and Y. Jin, "A competitive swarm optimizer for large scale optimization," *IEEE Trans. Cybernetics* 45,
  191–204 (2015).
- N. Van Thieu and S. Mirjalili, "MEALPY: An open-source library for latest meta-heuristic algorithms in Python,"
  *Journal of Systems Architecture* (2023).
