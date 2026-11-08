# RetinaMNIST — paired Wilcoxon tests (gpu backend), Holm within metric (4 tests)

## Significant after Holm

- AUC: CSO vs ALL512: diff -0.0013 (6/0/24), p = 0.0043, Holm p = 0.0170

| Metric | Family | Method | vs | Mean diff | Better | Tied | Worse | p | Holm p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| acc | primary | HHO | RANDOM | -0.00383 | 11 | 0 | 19 | 0.06373 | 0.19118 |
| acc | primary | CSO | RANDOM | 0.00133 | 16 | 3 | 11 | 0.30794 | 0.53074 |
| acc | primary | HHO | ALL512 | -0.00417 | 10 | 2 | 18 | 0.04451 | 0.17805 |
| acc | primary | CSO | ALL512 | 0.001 | 16 | 5 | 9 | 0.26537 | 0.53074 |
| acc | secondary (not corrected) | GA | RANDOM | 0.00117 | 17 | 2 | 11 | 0.4097 | – |
| acc | secondary (not corrected) | GA | ALL512 | 0.00083 | 16 | 3 | 11 | 0.42632 | – |
| acc | secondary (not corrected) | HHO | GA | -0.005 | 9 | 4 | 17 | 0.02404 | – |
| acc | secondary (not corrected) | CSO | GA | 0.00017 | 12 | 4 | 14 | 1.0 | – |
| auc | primary | HHO | RANDOM | -0.0009 | 16 | 0 | 14 | 0.9193 | 1.0 |
| auc | primary | CSO | RANDOM | 0.00027 | 14 | 0 | 16 | 0.62883 | 1.0 |
| auc | primary | HHO | ALL512 | -0.00241 | 15 | 0 | 15 | 0.27101 | 0.81302 |
| auc | primary | CSO | ALL512 | -0.00125 | 6 | 0 | 24 | 0.00425 | 0.017 |
| auc | secondary (not corrected) | GA | RANDOM | 0.00107 | 19 | 0 | 11 | 0.09986 | – |
| auc | secondary (not corrected) | GA | ALL512 | -0.00044 | 11 | 0 | 19 | 0.29416 | – |
| auc | secondary (not corrected) | HHO | GA | -0.00197 | 13 | 0 | 17 | 0.3184 | – |
| auc | secondary (not corrected) | CSO | GA | -0.00081 | 12 | 0 | 18 | 0.09986 | – |
| macro_f1 | primary | HHO | RANDOM | -0.00534 | 11 | 0 | 19 | 0.07324 | 0.24226 |
| macro_f1 | primary | CSO | RANDOM | 0.00314 | 18 | 0 | 12 | 0.28937 | 0.39615 |
| macro_f1 | primary | HHO | ALL512 | -0.00547 | 10 | 0 | 20 | 0.06057 | 0.24226 |
| macro_f1 | primary | CSO | ALL512 | 0.00301 | 20 | 0 | 10 | 0.19808 | 0.39615 |
| macro_f1 | secondary (not corrected) | GA | RANDOM | 0.00245 | 19 | 1 | 10 | 0.11801 | – |
| macro_f1 | secondary (not corrected) | GA | ALL512 | 0.00232 | 17 | 0 | 13 | 0.14138 | – |
| macro_f1 | secondary (not corrected) | HHO | GA | -0.00779 | 8 | 0 | 22 | 0.00664 | – |
| macro_f1 | secondary (not corrected) | CSO | GA | 0.00069 | 10 | 0 | 20 | 0.37092 | – |
| qwk | primary | HHO | RANDOM | 0.0037 | 18 | 0 | 12 | 0.21293 | 0.6388 |
| qwk | primary | CSO | RANDOM | 0.00242 | 15 | 0 | 15 | 0.58376 | 0.83226 |
| qwk | primary | HHO | ALL512 | 0.00396 | 20 | 0 | 10 | 0.13474 | 0.53894 |
| qwk | primary | CSO | ALL512 | 0.00269 | 15 | 0 | 15 | 0.41613 | 0.83226 |
| qwk | secondary (not corrected) | GA | RANDOM | 0.0066 | 21 | 0 | 9 | 0.01283 | – |
| qwk | secondary (not corrected) | GA | ALL512 | 0.00687 | 23 | 0 | 7 | 0.00316 | – |
| qwk | secondary (not corrected) | HHO | GA | -0.00291 | 16 | 0 | 14 | 0.82358 | – |
| qwk | secondary (not corrected) | CSO | GA | -0.00418 | 11 | 0 | 19 | 0.12797 | – |
