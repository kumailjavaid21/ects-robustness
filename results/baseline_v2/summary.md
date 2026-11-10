# ECTS v2: checkpoint MiniRocket and selective stopping

Summed dataset-mode processing time: 231.5 seconds; CPU-only, one worker/thread. Official train/test splits were maintained.
Each checkpoint at 20%, 30%, ..., 100% has its own aeon MiniRocket (672 features) and scaled logistic regression (C=0.1, tol=0.001, max_iter=400). C was selected in a small Crop validation pilot and then fixed across all datasets. Fit size is capped at 4,000, validation at about 400, calibration at about 2,000; unused training rows are recorded.
Standard uses the archived whole-series-normalized series. Causal normalizes each observation with running prefix statistics, cancelling prior per-series positive affine normalization up to numerical precision, not unknown non-affine preprocessing.

Fixed and probability settings are chosen on validation; probabilities use thresholds .05, .10, .20, ..., .90, .95, .99, 1.00. Aeon TEASER uses the same checkpoint MiniRocket models, 3-fold out-of-fold logistic predictions on shared unsupervised features (2-fold if needed), and a linear SGD one-class safety classifier. Unsupervised MiniRocket features are fitted on all fit examples before the TEASER folds. TEASER uses the same nine checkpoints.

Our thresholds are selected sequentially on held-out calibration cases still undecided at each checkpoint: use the largest accepted set with empirical error <= alpha. Tied confidence values are handled together. If none qualifies, the checkpoint is disabled. Unaccepted cases are forced to decide at 100%.
**Important:** these empirical checkpoint conditions do not control error at the first crossing. The data-dependent threshold selection, repeated checks, and forced decisions provide no finite-sample or anytime guarantee. Calibration error at the actual stopping point and test forced fractions are saved in JSON.

Small-data contrast: GunPoint and Coffee have only 10 and 6 calibration examples. ECG5000 has 200 calibration examples but its rarest class (only two training examples) is entirely assigned to fit, so calibration does not cover that class.

Cells: accuracy / balanced accuracy / macro-F1 / earliness / combined score. Combined is the harmonic mean of accuracy and (1 - earliness); dataset means are unweighted.

## Standard normalization

| Dataset (train/test x length; fit/val/cal) | Fixed | Probability | TEASER | alpha=.05 | alpha=.10 | alpha=.20 |
|---|---:|---:|---:|---:|---:|---:|
| FordA (3601/1320 x 500; 2301/400/900) | 0.827 / 0.826 / 0.826 / 0.200 / 0.813 | 0.840 / 0.840 / 0.840 / 0.208 / 0.815 | 0.923 / 0.923 / 0.923 / 0.382 / 0.740 | 0.926 / 0.926 / 0.926 / 0.405 / 0.724 | 0.898 / 0.898 / 0.898 / 0.285 / 0.796 | 0.827 / 0.826 / 0.826 / 0.200 / 0.813 |
| ECG5000 (500/4500 x 140; 201/99/200) | 0.921 / 0.483 / 0.477 / 0.200 / 0.856 | 0.920 / 0.476 / 0.473 / 0.203 / 0.854 | 0.938 / 0.522 / 0.563 / 0.377 / 0.749 | 0.928 / 0.483 / 0.490 / 0.219 / 0.848 | 0.921 / 0.481 / 0.476 / 0.204 / 0.854 | 0.921 / 0.481 / 0.476 / 0.204 / 0.854 |
| StarLightCurves (1000/8236 x 1024; 550/200/250) | 0.950 / 0.936 / 0.930 / 0.200 / 0.868 | 0.966 / 0.950 / 0.952 / 0.213 / 0.868 | 0.982 / 0.964 / 0.973 / 0.352 / 0.781 | 0.950 / 0.937 / 0.931 / 0.201 / 0.868 | 0.950 / 0.937 / 0.931 / 0.201 / 0.868 | 0.950 / 0.937 / 0.931 / 0.201 / 0.868 |
| Wafer (1000/6164 x 152; 550/200/250) | 0.995 / 0.993 / 0.988 / 0.204 / 0.885 | 0.995 / 0.993 / 0.988 / 0.204 / 0.885 | 1.000 / 0.999 / 1.000 / 0.354 / 0.785 | 0.997 / 0.996 / 0.992 / 0.207 / 0.883 | 0.997 / 0.996 / 0.992 / 0.207 / 0.883 | 0.997 / 0.996 / 0.992 / 0.207 / 0.883 |
| ElectricDevices (8926/7711 x 96; 4000/401/2000) | 0.494 / 0.373 / 0.362 / 0.208 / 0.609 | 0.614 / 0.530 / 0.542 / 0.327 / 0.642 | 0.604 / 0.505 / 0.519 / 0.422 / 0.591 | 0.690 / 0.616 / 0.624 / 0.566 / 0.533 | 0.666 / 0.583 / 0.592 / 0.453 / 0.600 | 0.608 / 0.515 / 0.530 / 0.336 / 0.634 |
| Crop (7200/16800 x 46; 4000/408/1800) | 0.602 / 0.602 / 0.599 / 0.304 / 0.646 | 0.635 / 0.635 / 0.630 / 0.318 / 0.657 | 0.699 / 0.699 / 0.698 / 0.463 / 0.608 | 0.721 / 0.721 / 0.719 / 0.763 / 0.356 | 0.718 / 0.718 / 0.716 / 0.696 / 0.427 | 0.699 / 0.699 / 0.695 / 0.526 / 0.564 |
| GunPoint (50/150 x 150; 30/10/10) | 0.973 / 0.974 / 0.973 / 0.300 / 0.814 | 0.913 / 0.914 / 0.913 / 0.234 / 0.833 | 0.993 / 0.993 / 0.993 / 0.398 / 0.750 | 0.980 / 0.980 / 0.980 / 0.333 / 0.794 | 0.800 / 0.800 / 0.800 / 0.253 / 0.772 | 0.800 / 0.800 / 0.800 / 0.253 / 0.772 |
| Coffee (28/28 x 286; 16/6/6) | 1.000 / 1.000 / 1.000 / 0.203 / 0.887 | 1.000 / 1.000 / 1.000 / 0.203 / 0.887 | 1.000 / 1.000 / 1.000 / 0.365 / 0.777 | 1.000 / 1.000 / 1.000 / 0.345 / 0.791 | 1.000 / 1.000 / 1.000 / 0.345 / 0.791 | 1.000 / 1.000 / 1.000 / 0.345 / 0.791 |
| **Mean** | 0.845 / 0.773 / 0.770 / 0.227 / 0.797 (n=8) | 0.860 / 0.792 / 0.792 / 0.239 / 0.805 (n=8) | 0.892 / 0.826 / 0.834 / 0.389 / 0.722 (n=8) | 0.899 / 0.832 / 0.833 / 0.380 / 0.725 (n=8) | 0.869 / 0.802 / 0.801 / 0.331 / 0.749 (n=8) | 0.850 / 0.782 / 0.781 / 0.284 / 0.773 (n=8) |

### Empirical reliability

| Dataset | alpha=.05 error / forced | alpha=.10 error / forced | alpha=.20 error / forced |
|---|---:|---:|---:|
| FordA | 0.074 / 0.114 FAIL | 0.102 / 0.019 FAIL | 0.173 / 0.000 PASS |
| ECG5000 | 0.072 / 0.003 FAIL | 0.079 / 0.006 PASS | 0.079 / 0.006 PASS |
| StarLightCurves | 0.050 / 0.002 PASS | 0.050 / 0.002 PASS | 0.050 / 0.002 PASS |
| Wafer | 0.003 / 0.004 PASS | 0.003 / 0.004 PASS | 0.003 / 0.004 PASS |
| ElectricDevices | 0.310 / 0.319 FAIL | 0.334 / 0.093 FAIL | 0.392 / 0.012 FAIL |
| Crop | 0.279 / 0.558 FAIL | 0.282 / 0.437 FAIL | 0.301 / 0.242 FAIL |
| GunPoint | 0.020 / 0.047 PASS | 0.200 / 0.067 FAIL | 0.200 / 0.067 PASS |
| Coffee | 0.000 / 0.179 PASS | 0.000 / 0.179 PASS | 0.000 / 0.179 PASS |

## Causal normalization

| Dataset (train/test x length; fit/val/cal) | Fixed | Probability | TEASER | alpha=.05 | alpha=.10 | alpha=.20 |
|---|---:|---:|---:|---:|---:|---:|
| FordA (3601/1320 x 500; 2301/400/900) | 0.798 / 0.799 / 0.798 / 0.200 / 0.799 | 0.844 / 0.844 / 0.844 / 0.228 / 0.807 | 0.918 / 0.918 / 0.918 / 0.396 / 0.728 | 0.926 / 0.925 / 0.926 / 0.463 / 0.680 | 0.893 / 0.894 / 0.893 / 0.309 / 0.779 | 0.798 / 0.798 / 0.798 / 0.201 / 0.798 |
| ECG5000 (500/4500 x 140; 201/99/200) | 0.901 / 0.386 / 0.380 / 0.200 / 0.847 | 0.901 / 0.386 / 0.380 / 0.200 / 0.847 | 0.931 / 0.453 / 0.481 / 0.359 / 0.760 | 0.925 / 0.429 / 0.446 / 0.258 / 0.824 | 0.902 / 0.389 / 0.386 / 0.204 / 0.846 | 0.902 / 0.389 / 0.386 / 0.204 / 0.846 |
| StarLightCurves (1000/8236 x 1024; 550/200/250) | 0.929 / 0.911 / 0.910 / 0.301 / 0.798 | 0.868 / 0.849 / 0.854 / 0.221 / 0.821 | 0.969 / 0.950 / 0.956 / 0.369 / 0.764 | 0.932 / 0.913 / 0.919 / 0.273 / 0.817 | 0.886 / 0.865 / 0.870 / 0.234 / 0.822 | 0.819 / 0.802 / 0.803 / 0.209 / 0.805 |
| Wafer (1000/6164 x 152; 550/200/250) | 0.994 / 0.977 / 0.983 / 0.204 / 0.884 | 0.994 / 0.977 / 0.983 / 0.204 / 0.884 | 0.999 / 0.996 / 0.996 / 0.353 / 0.785 | 0.997 / 0.990 / 0.992 / 0.213 / 0.880 | 0.997 / 0.990 / 0.992 / 0.213 / 0.880 | 0.997 / 0.990 / 0.992 / 0.213 / 0.880 |
| ElectricDevices (8926/7711 x 96; 4000/401/2000) | 0.574 / 0.471 / 0.459 / 0.406 / 0.584 | 0.589 / 0.539 / 0.554 / 0.377 / 0.606 | 0.524 / 0.389 / 0.356 / 0.346 / 0.581 | 0.661 / 0.590 / 0.604 / 0.661 / 0.448 | 0.650 / 0.591 / 0.605 / 0.553 / 0.529 | 0.612 / 0.564 / 0.577 / 0.423 / 0.594 |
| Crop (7200/16800 x 46; 4000/408/1800) | 0.560 / 0.560 / 0.556 / 0.413 / 0.573 | 0.581 / 0.581 / 0.572 / 0.408 / 0.586 | 0.634 / 0.634 / 0.629 / 0.512 / 0.551 | 0.678 / 0.678 / 0.675 / 0.871 / 0.217 | 0.677 / 0.677 / 0.673 / 0.799 / 0.310 | 0.665 / 0.665 / 0.660 / 0.699 / 0.414 |
| GunPoint (50/150 x 150; 30/10/10) | 0.680 / 0.681 / 0.679 / 0.200 / 0.735 | 0.767 / 0.768 / 0.766 / 0.223 / 0.772 | 0.893 / 0.893 / 0.893 / 0.455 / 0.677 | 0.913 / 0.913 / 0.913 / 0.489 / 0.655 | 0.913 / 0.913 / 0.913 / 0.489 / 0.655 | 0.840 / 0.840 / 0.840 / 0.276 / 0.778 |
| Coffee (28/28 x 286; 16/6/6) | 0.964 / 0.962 / 0.964 / 0.203 / 0.873 | 0.964 / 0.962 / 0.964 / 0.203 / 0.873 | 0.964 / 0.962 / 0.964 / 0.304 / 0.808 | 1.000 / 1.000 / 1.000 / 0.288 / 0.832 | 1.000 / 1.000 / 1.000 / 0.288 / 0.832 | 1.000 / 1.000 / 1.000 / 0.288 / 0.832 |
| **Mean** | 0.800 / 0.718 / 0.716 / 0.266 / 0.762 (n=8) | 0.813 / 0.738 / 0.739 / 0.258 / 0.774 (n=8) | 0.854 / 0.774 / 0.774 / 0.387 / 0.707 (n=8) | 0.879 / 0.805 / 0.809 / 0.440 / 0.669 (n=8) | 0.865 / 0.790 / 0.791 / 0.386 / 0.706 (n=8) | 0.829 / 0.756 / 0.757 / 0.314 / 0.743 (n=8) |

### Empirical reliability

| Dataset | alpha=.05 error / forced | alpha=.10 error / forced | alpha=.20 error / forced |
|---|---:|---:|---:|
| FordA | 0.074 / 0.146 FAIL | 0.107 / 0.018 FAIL | 0.202 / 0.002 FAIL |
| ECG5000 | 0.075 / 0.055 FAIL | 0.098 / 0.005 PASS | 0.098 / 0.005 PASS |
| StarLightCurves | 0.068 / 0.001 FAIL | 0.114 / 0.004 FAIL | 0.181 / 0.011 PASS |
| Wafer | 0.003 / 0.011 PASS | 0.003 / 0.011 PASS | 0.003 / 0.011 PASS |
| ElectricDevices | 0.339 / 0.354 FAIL | 0.350 / 0.174 FAIL | 0.388 / 0.021 FAIL |
| Crop | 0.322 / 0.695 FAIL | 0.323 / 0.579 FAIL | 0.335 / 0.431 FAIL |
| GunPoint | 0.087 / 0.207 FAIL | 0.087 / 0.207 PASS | 0.160 / 0.000 PASS |
| Coffee | 0.000 / 0.107 PASS | 0.000 / 0.107 PASS | 0.000 / 0.107 PASS |

## Causal minus standard: mean difference

| Method | Accuracy | Balanced accuracy | Macro-F1 | Earliness | Score |
|---|---:|---:|---:|---:|---:|
| Fixed | -0.045 | -0.055 | -0.054 | +0.038 | -0.036 |
| Probability | -0.047 | -0.054 | -0.053 | +0.019 | -0.031 |
| TEASER | -0.038 | -0.052 | -0.059 | -0.002 | -0.016 |
| alpha=.05 | -0.020 | -0.027 | -0.023 | +0.059 | -0.056 |
| alpha=.10 | -0.004 | -0.012 | -0.009 | +0.055 | -0.043 |
| alpha=.20 | -0.021 | -0.026 | -0.024 | +0.030 | -0.029 |

## Artifacts and issues

Final sequential-calibration results are in results_sequential.json and results.json. The original per-dataset records preserve the initial unconditioned-calibration diagnostic; final sequential records, probabilities, split indices, thresholds and supports, validation/test sweeps, and saved models are under runs/ects_v2/.
The server-side data download initially timed out and returned HTTP 429; verified files were relayed through Windows and their SHA-256 hashes checked against the dataset repository.
Archive: https://huggingface.co/datasets/ChengsenWang/UCR-2018 .
