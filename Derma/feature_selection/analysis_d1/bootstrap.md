# Study D1 — post-hoc test-set bootstrap (2,000 paired resamples of the 2,005 test images)

| Comparison | Metric | Observed diff | 95 % interval | Resamples <= 0 |
|---|---|---|---|---|
| GA vs RANDOM | ACC | +0.0138 | [+0.0104, +0.0173] | 0.0000 |
| GA vs RANDOM | AUC | +0.0116 | [+0.0089, +0.0147] | 0.0000 |
| CSO vs RANDOM | ACC | +0.0127 | [+0.0094, +0.0162] | 0.0000 |
| CSO vs RANDOM | AUC | +0.0104 | [+0.0077, +0.0133] | 0.0000 |
| GA vs FILTER | ACC | +0.0188 | [+0.0106, +0.0268] | 0.0000 |
| GA vs FILTER | AUC | +0.0083 | [+0.0017, +0.0150] | 0.0080 |
| CSO vs FILTER | ACC | +0.0177 | [+0.0095, +0.0256] | 0.0000 |
| CSO vs FILTER | AUC | +0.0071 | [+0.0004, +0.0134] | 0.0180 |
