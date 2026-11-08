# DermaMNIST Study D2 (5-NN) — paired Wilcoxon tests, Holm over 4 primary tests per primary metric

## Primary hypotheses significant after Holm

- ACC: GA vs RANDOM: diff +0.0147 (29/0/1), p = 0.00e+00, Holm p = 1.00e-05
- ACC: CSO vs RANDOM: diff +0.0133 (26/0/4), p = 2.00e-05, Holm p = 2.00e-05
- ACC: GA vs FILTER: diff +0.0185 (29/0/1), p = 0.00e+00, Holm p = 1.00e-05
- ACC: CSO vs FILTER: diff +0.0171 (28/0/2), p = 0.00e+00, Holm p = 1.00e-05
- AUC: GA vs RANDOM: diff +0.0183 (25/0/5), p = 2.00e-05, Holm p = 5.00e-05
- AUC: CSO vs RANDOM: diff +0.0197 (25/0/5), p = 1.00e-05, Holm p = 3.00e-05
- AUC: GA vs FILTER: diff +0.0072 (22/0/8), p = 3.45e-02, Holm p = 3.45e-02
- AUC: CSO vs FILTER: diff +0.0085 (19/0/11), p = 1.55e-02, Holm p = 3.09e-02

| Metric | Family | Method | vs | Mean diff | Better | Tied | Worse | p | Holm p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| acc | primary | GA | RANDOM | 0.01471 | 29 | 0 | 1 | 0.0 | 1e-05 |
| acc | primary | CSO | RANDOM | 0.01332 | 26 | 0 | 4 | 2e-05 | 2e-05 |
| acc | primary | GA | FILTER | 0.01849 | 29 | 0 | 1 | 0.0 | 1e-05 |
| acc | primary | CSO | FILTER | 0.01709 | 28 | 0 | 2 | 0.0 | 1e-05 |
| acc | secondary (not corrected) | GA | ALL1820 | 0.00036 | 15 | 0 | 15 | 0.91805 | – |
| acc | secondary (not corrected) | CSO | ALL1820 | -0.00103 | 14 | 0 | 16 | 0.56462 | – |
| acc | secondary (not corrected) | GA | CSO | 0.0014 | 16 | 0 | 14 | 0.53043 | – |
| acc | secondary (not corrected) | FILTER | RANDOM | -0.00377 | 9 | 2 | 19 | 0.03587 | – |
| auc | primary | GA | RANDOM | 0.0183 | 25 | 0 | 5 | 2e-05 | 5e-05 |
| auc | primary | CSO | RANDOM | 0.01965 | 25 | 0 | 5 | 1e-05 | 3e-05 |
| auc | primary | GA | FILTER | 0.00715 | 22 | 0 | 8 | 0.03454 | 0.03454 |
| auc | primary | CSO | FILTER | 0.0085 | 19 | 0 | 11 | 0.01546 | 0.03092 |
| auc | secondary (not corrected) | GA | ALL1820 | -0.02129 | 2 | 0 | 28 | 0.0 | – |
| auc | secondary (not corrected) | CSO | ALL1820 | -0.01994 | 3 | 0 | 27 | 0.0 | – |
| auc | secondary (not corrected) | GA | CSO | -0.00135 | 12 | 0 | 18 | 0.52911 | – |
| auc | secondary (not corrected) | FILTER | RANDOM | 0.01115 | 21 | 0 | 9 | 0.0008 | – |
| macro_f1 | secondary (not corrected) | GA | RANDOM | 0.0387 | 24 | 0 | 6 | 1e-05 | – |
| macro_f1 | secondary (not corrected) | CSO | RANDOM | 0.03597 | 24 | 0 | 6 | 1e-05 | – |
| macro_f1 | secondary (not corrected) | GA | FILTER | 0.01144 | 16 | 0 | 14 | 0.14028 | – |
| macro_f1 | secondary (not corrected) | CSO | FILTER | 0.00872 | 18 | 0 | 12 | 0.1706 | – |
| macro_f1 | secondary (not corrected) | GA | ALL1820 | 0.01227 | 18 | 0 | 12 | 0.13474 | – |
| macro_f1 | secondary (not corrected) | CSO | ALL1820 | 0.00954 | 19 | 0 | 11 | 0.12413 | – |
| macro_f1 | secondary (not corrected) | GA | CSO | 0.00273 | 15 | 0 | 15 | 0.98383 | – |
| macro_f1 | secondary (not corrected) | FILTER | RANDOM | 0.02725 | 24 | 0 | 6 | 2e-05 | – |
| bal_acc | secondary (not corrected) | GA | RANDOM | 0.03123 | 26 | 0 | 4 | 0.0 | – |
| bal_acc | secondary (not corrected) | CSO | RANDOM | 0.02956 | 27 | 0 | 3 | 0.0 | – |
| bal_acc | secondary (not corrected) | GA | FILTER | 0.01438 | 18 | 0 | 12 | 0.02085 | – |
| bal_acc | secondary (not corrected) | CSO | FILTER | 0.01271 | 20 | 0 | 10 | 0.00761 | – |
| bal_acc | secondary (not corrected) | GA | ALL1820 | -0.01092 | 11 | 0 | 19 | 0.04602 | – |
| bal_acc | secondary (not corrected) | CSO | ALL1820 | -0.01259 | 9 | 0 | 21 | 0.00501 | – |
| bal_acc | secondary (not corrected) | GA | CSO | 0.00167 | 17 | 0 | 13 | 0.88719 | – |
| bal_acc | secondary (not corrected) | FILTER | RANDOM | 0.01685 | 23 | 0 | 7 | 0.00026 | – |
