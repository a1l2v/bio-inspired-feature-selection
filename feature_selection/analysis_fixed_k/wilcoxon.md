# Equal-size benchmark — paired Wilcoxon tests on test metrics

Holm correction within each (k, metric): 4 algorithms × 3 references = 12 tests.

## Significant after Holm (p < 0.05)

- k = 10: PSO vs ALL512, Specificity: diff -0.0088 (2 better / 9 tied / 19 worse), p = 0.0001, Holm p = 0.0011
- k = 10: GA vs ALL512, Specificity: diff -0.0067 (5 better / 8 tied / 17 worse), p = 0.0017, Holm p = 0.0175
- k = 10: GWO vs ALL512, Specificity: diff -0.0100 (3 better / 8 tied / 19 worse), p = 0.0002, Holm p = 0.0027
- k = 20: PSO vs ALL512, Specificity: diff -0.0067 (3 better / 13 tied / 14 worse), p = 0.0031, Holm p = 0.0342
- k = 20: GWO vs ALL512, Specificity: diff -0.0082 (2 better / 14 tied / 14 worse), p = 0.0019, Holm p = 0.0226
- k = 20: WOA vs ALL512, Specificity: diff -0.0067 (3 better / 13 tied / 14 worse), p = 0.0032, Holm p = 0.0342
- k = 40: PSO vs ALL512, Specificity: diff -0.0070 (1 better / 15 tied / 14 worse), p = 0.0010, Holm p = 0.0101
- k = 40: GA vs ALL512, Specificity: diff -0.0076 (1 better / 14 tied / 15 worse), p = 0.0006, Holm p = 0.0063
- k = 40: GWO vs ALL512, Specificity: diff -0.0082 (0 better / 12 tied / 18 worse), p = 0.0001, Holm p = 0.0007
- k = 40: WOA vs ALL512, Specificity: diff -0.0059 (0 better / 17 tied / 13 worse), p = 0.0010, Holm p = 0.0101

## ACC and AUC, all comparisons

| k | Metric | Method | vs | Mean diff | Better | Tied | Worse | p | Holm p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | acc | PSO | RANDOM | -0.00021 | 10 | 8 | 12 | 0.80429 | 1.0 |
| 10 | acc | PSO | SINGLE | 0.0015 | 14 | 4 | 12 | 0.38608 | 1.0 |
| 10 | acc | PSO | ALL512 | -0.00214 | 9 | 6 | 15 | 0.52244 | 1.0 |
| 10 | acc | GA | RANDOM | -0.00021 | 11 | 6 | 13 | 0.88528 | 1.0 |
| 10 | acc | GA | SINGLE | 0.0015 | 16 | 8 | 6 | 0.16267 | 1.0 |
| 10 | acc | GA | ALL512 | -0.00214 | 11 | 5 | 14 | 0.71007 | 1.0 |
| 10 | acc | GWO | RANDOM | 0.0 | 13 | 4 | 13 | 0.69538 | 1.0 |
| 10 | acc | GWO | SINGLE | 0.00171 | 15 | 3 | 12 | 0.34791 | 1.0 |
| 10 | acc | GWO | ALL512 | -0.00192 | 10 | 5 | 15 | 0.72606 | 1.0 |
| 10 | acc | WOA | RANDOM | -0.0015 | 10 | 12 | 8 | 0.97519 | 1.0 |
| 10 | acc | WOA | SINGLE | 0.00021 | 14 | 3 | 13 | 0.92611 | 1.0 |
| 10 | acc | WOA | ALL512 | -0.00342 | 10 | 8 | 12 | 0.47004 | 1.0 |
| 10 | auc | PSO | RANDOM | -0.00056 | 13 | 4 | 13 | 0.98359 | 1.0 |
| 10 | auc | PSO | SINGLE | 0.00226 | 17 | 0 | 13 | 0.39903 | 1.0 |
| 10 | auc | PSO | ALL512 | -0.00327 | 12 | 0 | 18 | 0.2493 | 1.0 |
| 10 | auc | GA | RANDOM | -0.00077 | 14 | 3 | 13 | 0.8855 | 1.0 |
| 10 | auc | GA | SINGLE | 0.00205 | 20 | 0 | 10 | 0.53042 | 1.0 |
| 10 | auc | GA | ALL512 | -0.00348 | 11 | 0 | 19 | 0.221 | 1.0 |
| 10 | auc | GWO | RANDOM | 0.00088 | 13 | 5 | 12 | 0.74193 | 1.0 |
| 10 | auc | GWO | SINGLE | 0.0037 | 20 | 0 | 10 | 0.15583 | 1.0 |
| 10 | auc | GWO | ALL512 | -0.00184 | 13 | 0 | 17 | 0.4165 | 1.0 |
| 10 | auc | WOA | RANDOM | 0.00189 | 15 | 5 | 10 | 0.29392 | 1.0 |
| 10 | auc | WOA | SINGLE | 0.00471 | 18 | 0 | 12 | 0.16817 | 1.0 |
| 10 | auc | WOA | ALL512 | -0.00083 | 14 | 1 | 15 | 0.5577 | 1.0 |
| 20 | acc | PSO | RANDOM | -0.00577 | 5 | 9 | 16 | 0.01747 | 0.20966 |
| 20 | acc | PSO | SINGLE | -0.0015 | 13 | 6 | 11 | 0.77239 | 1.0 |
| 20 | acc | PSO | ALL512 | -0.00363 | 9 | 10 | 11 | 0.3961 | 1.0 |
| 20 | acc | GA | RANDOM | -0.00235 | 8 | 8 | 14 | 0.20747 | 1.0 |
| 20 | acc | GA | SINGLE | 0.00192 | 15 | 7 | 8 | 0.32701 | 1.0 |
| 20 | acc | GA | ALL512 | -0.00021 | 13 | 10 | 7 | 0.52748 | 1.0 |
| 20 | acc | GWO | RANDOM | -0.00406 | 7 | 6 | 17 | 0.07093 | 0.78027 |
| 20 | acc | GWO | SINGLE | 0.00021 | 14 | 7 | 9 | 0.39198 | 1.0 |
| 20 | acc | GWO | ALL512 | -0.00192 | 12 | 8 | 10 | 0.95056 | 1.0 |
| 20 | acc | WOA | RANDOM | -0.00256 | 6 | 10 | 14 | 0.0853 | 0.85298 |
| 20 | acc | WOA | SINGLE | 0.00171 | 15 | 6 | 9 | 0.3322 | 1.0 |
| 20 | acc | WOA | ALL512 | -0.00043 | 11 | 9 | 10 | 0.88493 | 1.0 |
| 20 | auc | PSO | RANDOM | -0.00212 | 14 | 1 | 15 | 0.28479 | 1.0 |
| 20 | auc | PSO | SINGLE | 0.00165 | 13 | 0 | 17 | 0.5857 | 1.0 |
| 20 | auc | PSO | ALL512 | -0.00097 | 14 | 1 | 15 | 0.56465 | 1.0 |
| 20 | auc | GA | RANDOM | 0.00193 | 14 | 3 | 13 | 0.44656 | 1.0 |
| 20 | auc | GA | SINGLE | 0.00571 | 22 | 0 | 8 | 0.01209 | 0.14512 |
| 20 | auc | GA | ALL512 | 0.00308 | 21 | 0 | 9 | 0.04947 | 0.54421 |
| 20 | auc | GWO | RANDOM | -0.00068 | 12 | 4 | 14 | 0.67319 | 1.0 |
| 20 | auc | GWO | SINGLE | 0.00309 | 20 | 0 | 10 | 0.1588 | 1.0 |
| 20 | auc | GWO | ALL512 | 0.00047 | 16 | 0 | 14 | 0.517 | 1.0 |
| 20 | auc | WOA | RANDOM | 0.00069 | 13 | 3 | 14 | 0.98359 | 1.0 |
| 20 | auc | WOA | SINGLE | 0.00446 | 18 | 0 | 12 | 0.19808 | 1.0 |
| 20 | auc | WOA | ALL512 | 0.00184 | 16 | 0 | 14 | 0.47157 | 1.0 |
| 40 | acc | PSO | RANDOM | 0.00021 | 16 | 5 | 9 | 0.49902 | 1.0 |
| 40 | acc | PSO | SINGLE | 0.00064 | 11 | 15 | 4 | 0.15827 | 1.0 |
| 40 | acc | PSO | ALL512 | -0.0015 | 6 | 13 | 11 | 0.40536 | 1.0 |
| 40 | acc | GA | RANDOM | 0.00064 | 10 | 7 | 13 | 0.83644 | 1.0 |
| 40 | acc | GA | SINGLE | 0.00107 | 7 | 12 | 11 | 0.58225 | 1.0 |
| 40 | acc | GA | ALL512 | -0.00107 | 9 | 9 | 12 | 0.74083 | 1.0 |
| 40 | acc | GWO | RANDOM | -0.00192 | 11 | 6 | 13 | 0.40334 | 1.0 |
| 40 | acc | GWO | SINGLE | -0.0015 | 8 | 12 | 10 | 0.61865 | 1.0 |
| 40 | acc | GWO | ALL512 | -0.00363 | 7 | 8 | 15 | 0.13956 | 1.0 |
| 40 | acc | WOA | RANDOM | -0.00086 | 10 | 7 | 13 | 0.52168 | 1.0 |
| 40 | acc | WOA | SINGLE | -0.00043 | 10 | 12 | 8 | 0.89269 | 1.0 |
| 40 | acc | WOA | ALL512 | -0.00256 | 5 | 14 | 11 | 0.18224 | 1.0 |
| 40 | auc | PSO | RANDOM | 1e-05 | 16 | 0 | 14 | 0.66578 | 1.0 |
| 40 | auc | PSO | SINGLE | 0.00346 | 21 | 0 | 9 | 0.06562 | 0.52496 |
| 40 | auc | PSO | ALL512 | 0.00251 | 21 | 0 | 9 | 0.06563 | 0.52496 |
| 40 | auc | GA | RANDOM | -0.00177 | 10 | 0 | 20 | 0.29417 | 1.0 |
| 40 | auc | GA | SINGLE | 0.00168 | 17 | 1 | 12 | 0.32855 | 1.0 |
| 40 | auc | GA | ALL512 | 0.00074 | 17 | 0 | 13 | 0.4466 | 1.0 |
| 40 | auc | GWO | RANDOM | 0.00118 | 16 | 0 | 14 | 0.24938 | 1.0 |
| 40 | auc | GWO | SINGLE | 0.00464 | 19 | 1 | 10 | 0.04171 | 0.41713 |
| 40 | auc | GWO | ALL512 | 0.00369 | 18 | 0 | 12 | 0.02181 | 0.26175 |
| 40 | auc | WOA | RANDOM | 0.00081 | 17 | 0 | 13 | 0.59279 | 1.0 |
| 40 | auc | WOA | SINGLE | 0.00427 | 18 | 3 | 9 | 0.04274 | 0.41713 |
| 40 | auc | WOA | ALL512 | 0.00332 | 18 | 1 | 11 | 0.03078 | 0.33863 |

All metrics: `wilcoxon.csv`.
