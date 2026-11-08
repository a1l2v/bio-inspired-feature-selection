# DermaMNIST Study D1 (LS-SVM) — paired Wilcoxon tests, Holm over 4 primary tests per primary metric

## Primary hypotheses significant after Holm

- ACC: GA vs RANDOM: diff +0.0138 (27/0/3), p = 0.00e+00, Holm p = 1.00e-05
- ACC: CSO vs RANDOM: diff +0.0127 (27/0/3), p = 1.00e-05, Holm p = 1.00e-05
- ACC: GA vs FILTER: diff +0.0188 (30/0/0), p = 0.00e+00, Holm p = 1.00e-05
- ACC: CSO vs FILTER: diff +0.0177 (29/1/0), p = 0.00e+00, Holm p = 1.00e-05
- AUC: GA vs RANDOM: diff +0.0116 (26/0/4), p = 0.00e+00, Holm p = 1.00e-05
- AUC: CSO vs RANDOM: diff +0.0104 (26/0/4), p = 0.00e+00, Holm p = 0.00e+00
- AUC: GA vs FILTER: diff +0.0083 (25/0/5), p = 1.00e-05, Holm p = 1.00e-05
- AUC: CSO vs FILTER: diff +0.0071 (27/0/3), p = 0.00e+00, Holm p = 1.00e-05

| Metric | Family | Method | vs | Mean diff | Better | Tied | Worse | p | Holm p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| acc | primary | GA | RANDOM | 0.01378 | 27 | 0 | 3 | 0.0 | 1e-05 |
| acc | primary | CSO | RANDOM | 0.01272 | 27 | 0 | 3 | 1e-05 | 1e-05 |
| acc | primary | GA | FILTER | 0.01877 | 30 | 0 | 0 | 0.0 | 1e-05 |
| acc | primary | CSO | FILTER | 0.01771 | 29 | 1 | 0 | 0.0 | 1e-05 |
| acc | secondary (not corrected) | GA | ALL1820 | -0.02008 | 0 | 0 | 30 | 0.0 | – |
| acc | secondary (not corrected) | CSO | ALL1820 | -0.02115 | 0 | 0 | 30 | 0.0 | – |
| acc | secondary (not corrected) | GA | CSO | 0.00106 | 16 | 2 | 12 | 0.48433 | – |
| acc | secondary (not corrected) | FILTER | RANDOM | -0.00499 | 6 | 0 | 24 | 0.00225 | – |
| auc | primary | GA | RANDOM | 0.01161 | 26 | 0 | 4 | 0.0 | 1e-05 |
| auc | primary | CSO | RANDOM | 0.01037 | 26 | 0 | 4 | 0.0 | 0.0 |
| auc | primary | GA | FILTER | 0.0083 | 25 | 0 | 5 | 1e-05 | 1e-05 |
| auc | primary | CSO | FILTER | 0.00706 | 27 | 0 | 3 | 0.0 | 1e-05 |
| auc | secondary (not corrected) | GA | ALL1820 | -0.01739 | 0 | 0 | 30 | 0.0 | – |
| auc | secondary (not corrected) | CSO | ALL1820 | -0.01863 | 0 | 0 | 30 | 0.0 | – |
| auc | secondary (not corrected) | GA | CSO | 0.00124 | 16 | 0 | 14 | 0.55611 | – |
| auc | secondary (not corrected) | FILTER | RANDOM | 0.00331 | 21 | 0 | 9 | 0.06989 | – |
| macro_f1 | secondary (not corrected) | GA | RANDOM | 0.0583 | 25 | 0 | 5 | 1e-05 | – |
| macro_f1 | secondary (not corrected) | CSO | RANDOM | 0.06363 | 28 | 0 | 2 | 0.0 | – |
| macro_f1 | secondary (not corrected) | GA | FILTER | 0.02023 | 21 | 0 | 9 | 0.01283 | – |
| macro_f1 | secondary (not corrected) | CSO | FILTER | 0.02556 | 21 | 0 | 9 | 0.00186 | – |
| macro_f1 | secondary (not corrected) | GA | ALL1820 | -0.02535 | 8 | 0 | 22 | 0.00113 | – |
| macro_f1 | secondary (not corrected) | CSO | ALL1820 | -0.02002 | 9 | 0 | 21 | 0.00814 | – |
| macro_f1 | secondary (not corrected) | GA | CSO | -0.00533 | 17 | 0 | 13 | 0.74565 | – |
| macro_f1 | secondary (not corrected) | FILTER | RANDOM | 0.03807 | 25 | 0 | 5 | 1e-05 | – |
| bal_acc | secondary (not corrected) | GA | RANDOM | 0.0516 | 25 | 0 | 5 | 0.0 | – |
| bal_acc | secondary (not corrected) | CSO | RANDOM | 0.0538 | 28 | 0 | 2 | 0.0 | – |
| bal_acc | secondary (not corrected) | GA | FILTER | 0.02037 | 19 | 0 | 11 | 0.00538 | – |
| bal_acc | secondary (not corrected) | CSO | FILTER | 0.02256 | 20 | 0 | 10 | 0.00434 | – |
| bal_acc | secondary (not corrected) | GA | ALL1820 | -0.01816 | 9 | 0 | 21 | 0.0093 | – |
| bal_acc | secondary (not corrected) | CSO | ALL1820 | -0.01597 | 10 | 0 | 20 | 0.01546 | – |
| bal_acc | secondary (not corrected) | GA | CSO | -0.00219 | 16 | 0 | 14 | 0.77657 | – |
| bal_acc | secondary (not corrected) | FILTER | RANDOM | 0.03124 | 24 | 0 | 6 | 1e-05 | – |
