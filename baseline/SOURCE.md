# Provenance — official MedMNIST v2 ResNet-18 (28) baseline

Files in this folder are copied unmodified from the official sources. Nothing here was written by us.

## Code

From [MedMNIST/experiments](https://github.com/MedMNIST/experiments) @ `70b6b3a7ad7afddff1df2a3b735235830fbdb142` (2024-07-11), Apache-2.0 (see `LICENSE`).

| File | Repo path |
| --- | --- |
| `models.py` | `MedMNIST2D/models.py` (ResNet-18 / ResNet-50 for 28×28 input) |
| `train_and_eval_pytorch.py` | `MedMNIST2D/train_and_eval_pytorch.py` |
| `LICENSE` | `LICENSE` |

Dependency (not copied, install instead): `pip install medmnist` — provides `INFO`, `Evaluator`, `BreastMNIST`. Also needs `torch`, `torchvision`, `tensorboardX`, `tqdm`, `Pillow`.

## Pre-trained weights

From Zenodo [10.5281/zenodo.7782113](https://doi.org/10.5281/zenodo.7782113) (record 7782114), file `weights_breastmnist.zip` (1.08 GB, md5 `f3e91872a00cf0ce94186e17ee805156`). Only the three ResNet-18 (28) members were downloaded (via HTTP range requests); each was CRC-checked against the zip's directory.

| File | Bytes | MD5 |
| --- | --- | --- |
| `weights/resnet18_28_1.pth` | 44,737,899 | `e64c22fe24f04bba40ec37958b09de1b` |
| `weights/resnet18_28_2.pth` | 44,737,899 | `d4c9b397e4f186309902ae1556a7c648` |
| `weights/resnet18_28_3.pth` | 44,737,899 | `4bacbc1e8129e6120b03d7d50cbfa360` |

These are the paper's 3 runs (Table 3: BreastMNIST ResNet-18 (28) AUC 0.901, ACC 0.863). Checkpoint format: `{'net': state_dict}`, as loaded by `--model_path`.

## Notes from reading the code

- The released weights were trained with **`--as_rgb`** (conv1 is `[64, 3, 3, 3]`): grayscale copied to 3 channels. The script's default is 1 channel, so `--as_rgb` is required both to load these weights and to reproduce the paper.
- It is the CIFAR-style ResNet (3×3 conv1, stride 1, no max-pool), not torchvision's.
- Model selection = best **validation AUC** across epochs; LR ×0.1 at 50% and 75% of epochs.
- Data is read from `~/.medmnist/breastmnist.npz` (fixed default root); copy `dataset/breastmnist.npz` there.

## Evaluating the released weights

Run from `baseline/` with the project venv:

```
..\.venv\Scripts\python.exe train_and_eval_pytorch.py --data_flag breastmnist --num_epochs 0 --size 28 --gpu_ids 0 --model_flag resnet18 --as_rgb --model_path weights/resnet18_28_1.pth --output_root ./output_eval --run resnet18_28_1
```

Results (31 Oct 2026; torch 2.14.1+cu126, medmnist 3.0.2, RTX 4060):

| Weights | Train AUC / ACC | Val AUC / ACC | Test AUC | Test ACC |
| --- | --- | --- | --- | --- |
| `resnet18_28_1` | 1.000 / 1.000 | 0.957 / 0.872 | 0.8985 | 0.8718 |
| `resnet18_28_2` | 1.000 / 0.998 | 0.963 / 0.897 | 0.9173 | 0.8718 |
| `resnet18_28_3` | 1.000 / 1.000 | 0.958 / 0.872 | 0.8887 | 0.8462 |
| **Mean ± std** |  |  | **0.901 ± 0.015** | **0.863 ± 0.015** |
| Paper (Table 3) |  |  | 0.901 | 0.863 |

Matches the paper exactly. Per-split prediction CSVs are in `output_eval/breastmnist/<timestamp>/`.

## Retraining from scratch (our reproduction)

Same unmodified script, official settings (100 epochs, Adam 1e-3, ×0.1 at 50/75, batch 128, best val AUC), `--as_rgb`. The script sets no random seed, so runs are not bit-reproducible.

```
..\.venv\Scripts\python.exe train_and_eval_pytorch.py --data_flag breastmnist --num_epochs 100 --size 28 --gpu_ids 0 --batch_size 128 --model_flag resnet18 --as_rgb --output_root ./output_train --run repro_run1
```

| Run | Time | Best epoch | Val AUC / ACC | Test AUC | Test ACC | Checkpoint |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.4 min | 30 | 0.972 / 0.923 | 0.9173 | 0.8782 | `output_train/breastmnist/261031_151915/best_model.pth` |
| 2 | 1.5 min | 36 | 0.929 / 0.910 | 0.8878 | 0.8397 | `output_train/breastmnist/261031_152051/best_model.pth` |
| 3 | 2.0 min | 36 | 0.957 / 0.923 | 0.8665 | 0.7756 | `output_train/breastmnist/261031_152245/best_model.pth` |
| 4 | 1.4 min | 36 | 0.971 / 0.910 | 0.9297 | 0.8910 | `output_train/breastmnist/261031_153447/best_model.pth` |
| 5 | 3.0 min | 26 | 0.937 / 0.872 | 0.9029 | 0.8333 | `output_train/breastmnist/261031_153622/best_model.pth` |
| 6 | 0.9 min | 9 | 0.951 / 0.795 | 0.8768 | 0.7244 | `output_train/breastmnist/261031_153923/best_model.pth` |
| 7 | 1.0 min | 23 | 0.942 / 0.846 | 0.8841 | 0.8526 | `output_train/breastmnist/261031_154037/best_model.pth` |
| 8 | 1.0 min | 30 | 0.977 / 0.910 | 0.9170 | 0.8590 | `output_train/breastmnist/261031_154145/best_model.pth` |
| 9 | 1.1 min | 37 | 0.962 / 0.885 | 0.9027 | 0.8462 | `output_train/breastmnist/261031_154341/best_model.pth` |
| 10 | 1.0 min | 52 | 0.957 / 0.885 | 0.9012 | 0.8654 | `output_train/breastmnist/261031_154504/best_model.pth` |
| **Mean ± std (10 runs)** |  |  |  | **0.899 ± 0.020** | **0.837 ± 0.050** |  |
| Median |  |  |  | 0.902 | 0.849 |  |
| 95% CI of mean |  |  |  | [0.884, 0.913] | [0.801, 0.873] |  |
| Paper |  |  |  | 0.901 | 0.863 |  |

**Verdict (10 runs):**

- **AUC reproduces:** 0.899 vs 0.901 (−0.002); one-sample t-test p = 0.71, Wilcoxon p = 0.92.
- **ACC is lower but not significantly:** 0.837 vs 0.863 (−0.027); the paper's value is inside the 95% CI; t-test p = 0.13, Wilcoxon p = 0.13.
- ACC has a heavy low tail: run 6 (val-AUC selection picked epoch 9, train ACC only 0.86) and run 3. Without run 6: 0.849 ± 0.033 (p = 0.24).
- Lesson for later stages: on 78 validation images, selecting by val AUC is noisy, and test ACC swings by ±5% between identical runs. Any feature-selection gain must be judged against this spread.

Full logs: `train_run{1..10}.log` (UTF-16, written by PowerShell 5.1 `Tee-Object`). These retrained models only check reproducibility; the feature-selection pipeline uses only the released weights in `weights/`.
