# RetinaMNIST — paired Wilcoxon tests (gpu backend), Holm within metric (4 tests)

## Significant after Holm

- Macro-F1: GWO vs ALL512: diff -0.0078 (10/0/20), p = 0.0093, Holm p = 0.0372

| Metric | Family | Method | vs | Mean diff | Better | Tied | Worse | p | Holm p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| acc | primary | GA | RANDOM | 0.00233 | 15 | 3 | 12 | 0.38728 | 1.0 |
| acc | primary | GWO | RANDOM | -0.00167 | 13 | 2 | 15 | 0.36475 | 1.0 |
| acc | primary | GA | ALL512 | 0.001 | 13 | 2 | 15 | 0.82078 | 1.0 |
| acc | primary | GWO | ALL512 | -0.003 | 11 | 3 | 16 | 0.10595 | 0.4238 |
| auc | primary | GA | RANDOM | 0.00051 | 18 | 0 | 12 | 0.26212 | 0.99747 |
| auc | primary | GWO | RANDOM | -0.00014 | 13 | 0 | 17 | 1.0 | 1.0 |
| auc | primary | GA | ALL512 | 0.00053 | 18 | 0 | 12 | 0.24937 | 0.99747 |
| auc | primary | GWO | ALL512 | -0.00012 | 14 | 0 | 16 | 0.98383 | 1.0 |
| macro_f1 | primary | GA | RANDOM | 0.0004 | 16 | 0 | 14 | 0.95153 | 1.0 |
| macro_f1 | primary | GWO | RANDOM | -0.00626 | 12 | 0 | 18 | 0.08033 | 0.24098 |
| macro_f1 | primary | GA | ALL512 | -0.00116 | 15 | 0 | 15 | 0.544 | 1.0 |
| macro_f1 | primary | GWO | ALL512 | -0.00781 | 10 | 0 | 20 | 0.0093 | 0.0372 |
| qwk | primary | GA | RANDOM | 0.00092 | 16 | 0 | 14 | 0.9354 | 1.0 |
| qwk | primary | GWO | RANDOM | -0.0023 | 12 | 0 | 18 | 0.35988 | 1.0 |
| qwk | primary | GA | ALL512 | 0.00103 | 16 | 0 | 14 | 1.0 | 1.0 |
| qwk | primary | GWO | ALL512 | -0.00218 | 13 | 0 | 17 | 0.55611 | 1.0 |
