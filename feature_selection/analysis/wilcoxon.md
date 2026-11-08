# Paired Wilcoxon signed-rank tests on test metrics

Holm correction within each metric (7 vs all-512 + 21 method pairs = 28 tests).

## Significant after Holm (p < 0.05)

- WOA vs ALL512, AUC: diff -0.0374, p = 0.0000, Holm p = 0.0001
- RANDOM vs WOA, AUC: diff +0.0381, p = 0.0000, Holm p = 0.0000
- PSO vs WOA, AUC: diff +0.0384, p = 0.0000, Holm p = 0.0000
- GA vs WOA, AUC: diff +0.0396, p = 0.0000, Holm p = 0.0001
- GWO vs WOA, AUC: diff +0.0395, p = 0.0000, Holm p = 0.0000
- WOA vs BPSO, AUC: diff -0.0384, p = 0.0000, Holm p = 0.0000
- WOA vs VBPSO, AUC: diff -0.0383, p = 0.0000, Holm p = 0.0000
- GWO vs ALL512, Specificity: diff -0.0076, p = 0.0006, Holm p = 0.0155
- VBPSO vs ALL512, Specificity: diff -0.0035, p = 0.0018, Holm p = 0.0473

## Each method vs all 512 features + SVM

| Metric | Method | Mean diff | Wins | Ties | Losses | p | Holm p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| acc | RANDOM | -0.00107 | 5 | 15 | 10 | 0.41413 | 1.0 |
| acc | PSO | -0.00085 | 3 | 19 | 8 | 0.26455 | 1.0 |
| acc | GA | 0.0 | 8 | 14 | 8 | 0.80247 | 1.0 |
| acc | GWO | -0.00342 | 7 | 9 | 14 | 0.17171 | 1.0 |
| acc | WOA | -0.01303 | 6 | 5 | 19 | 0.01031 | 0.26702 |
| acc | BPSO | -0.00043 | 7 | 15 | 8 | 0.84238 | 1.0 |
| acc | VBPSO | -0.00171 | 3 | 17 | 10 | 0.12872 | 1.0 |
| auc | RANDOM | 0.00077 | 17 | 1 | 12 | 0.22085 | 1.0 |
| auc | PSO | 0.00104 | 17 | 2 | 11 | 0.13316 | 1.0 |
| auc | GA | 0.00225 | 22 | 0 | 8 | 0.00772 | 0.16218 |
| auc | GWO | 0.0021 | 17 | 0 | 13 | 0.22886 | 1.0 |
| auc | WOA | -0.03736 | 2 | 0 | 28 | 1e-05 | 0.00013 |
| auc | BPSO | 0.00107 | 18 | 0 | 12 | 0.10407 | 1.0 |
| auc | VBPSO | 0.0009 | 17 | 1 | 12 | 0.19496 | 1.0 |
| macro_f1 | RANDOM | -0.00053 | 6 | 14 | 10 | 0.49093 | 1.0 |
| macro_f1 | PSO | -0.0004 | 4 | 18 | 8 | 0.36341 | 1.0 |
| macro_f1 | GA | 0.00158 | 8 | 14 | 8 | 0.75461 | 1.0 |
| macro_f1 | GWO | -0.00263 | 7 | 9 | 14 | 0.29248 | 1.0 |
| macro_f1 | WOA | -0.01678 | 10 | 1 | 19 | 0.00899 | 0.23377 |
| macro_f1 | BPSO | 0.00062 | 8 | 14 | 8 | 0.72283 | 1.0 |
| macro_f1 | VBPSO | -0.00144 | 4 | 16 | 10 | 0.18199 | 1.0 |
| malignant_recall | RANDOM | 0.00476 | 6 | 23 | 1 | 0.13844 | 1.0 |
| malignant_recall | PSO | 0.00397 | 4 | 24 | 2 | 0.51724 | 1.0 |
| malignant_recall | GA | 0.0119 | 8 | 22 | 0 | 0.02197 | 0.54688 |
| malignant_recall | GWO | 0.00794 | 8 | 14 | 8 | 0.67526 | 1.0 |
| malignant_recall | WOA | -0.01825 | 8 | 7 | 15 | 0.095 | 1.0 |
| malignant_recall | BPSO | 0.00794 | 8 | 22 | 0 | 0.02188 | 0.54688 |
| malignant_recall | VBPSO | 0.00317 | 4 | 25 | 1 | 0.34643 | 1.0 |
| specificity | RANDOM | -0.00322 | 0 | 20 | 10 | 0.00635 | 0.15873 |
| specificity | PSO | -0.00263 | 0 | 22 | 8 | 0.0218 | 0.43596 |
| specificity | GA | -0.00439 | 0 | 20 | 10 | 0.00644 | 0.15873 |
| specificity | GWO | -0.00761 | 0 | 16 | 14 | 0.00055 | 0.01551 |
| specificity | WOA | -0.01111 | 4 | 9 | 17 | 0.00284 | 0.07375 |
| specificity | BPSO | -0.00351 | 0 | 20 | 10 | 0.00635 | 0.15873 |
| specificity | VBPSO | -0.00351 | 0 | 18 | 12 | 0.00175 | 0.04733 |

All pairwise comparisons: `wilcoxon.csv`.
