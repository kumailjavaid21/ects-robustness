# ECTS scale benchmark

**Status:** complete 26-dataset main benchmark, five paired draws, seven conditions. Two small-calibration sets are separate and excluded from all main macro statistics. No target alpha is a finite-sample guarantee.

## Datasets and splits

Dataset files are from the pinned [UCR-2018 mirror](https://huggingface.co/datasets/ChengsenWang/UCR-2018), using its original train/test split. New files were SHA-256 verified against the repository LFS hashes. Domain labels are broad descriptors; related outline datasets may not be statistically independent. The calibration split follows v2 unchanged, including its class-reservation rule.

| Dataset | Cohort | Domain | Train | Test | Length | Classes | Fit | Validation | Calibration | Classes absent from calibration |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FordA | main | engine / sensor | 3601 | 1320 | 500 | 2 | 2301 | 400 | 900 | none |
| ECG5000 | main | ECG / health | 500 | 4500 | 140 | 5 | 201 | 99 | 200 | 4 |
| StarLightCurves | main | astronomy | 1000 | 8236 | 1024 | 3 | 550 | 200 | 250 | none |
| Wafer | main | semiconductor / sensor | 1000 | 6164 | 152 | 2 | 550 | 200 | 250 | none |
| ElectricDevices | main | device | 8926 | 7711 | 96 | 7 | 4000 | 401 | 2000 | none |
| Crop | main | satellite / spectral | 7200 | 16800 | 46 | 24 | 4000 | 408 | 1800 | none |
| FordB | main | engine / sensor | 3636 | 810 | 500 | 2 | 2327 | 400 | 909 | none |
| NonInvasiveFetalECGThorax1 | main | ECG / health | 1800 | 1965 | 750 | 42 | 994 | 357 | 449 | none |
| PhalangesOutlinesCorrect | main | medical image outline | 1800 | 858 | 80 | 2 | 990 | 360 | 450 | none |
| HandOutlines | main | image outline | 1000 | 370 | 2709 | 2 | 550 | 200 | 250 | none |
| UWaveGestureLibraryX | main | motion | 896 | 3582 | 315 | 8 | 494 | 178 | 224 | none |
| TwoPatterns | main | synthetic | 1000 | 4000 | 128 | 4 | 552 | 199 | 249 | none |
| Strawberry | main | spectroscopy | 613 | 370 | 235 | 2 | 290 | 123 | 200 | none |
| MiddlePhalanxOutlineCorrect | main | medical image outline | 600 | 291 | 80 | 2 | 280 | 120 | 200 | none |
| EthanolLevel | main | chemical / sensor | 504 | 500 | 1751 | 4 | 204 | 100 | 200 | none |
| MixedShapesRegularTrain | main | image shape | 500 | 2425 | 1024 | 5 | 200 | 100 | 200 | none |
| SwedishLeaf | main | image outline | 500 | 625 | 128 | 15 | 201 | 101 | 198 | none |
| FaceAll | main | image outline | 560 | 1690 | 131 | 14 | 252 | 112 | 196 | none |
| ShapesAll | main | image shape | 600 | 600 | 512 | 60 | 300 | 120 | 180 | none |
| SyntheticControl | main | synthetic | 300 | 300 | 60 | 6 | 180 | 60 | 60 | none |
| EOGHorizontalSignal | main | health / ocular | 362 | 362 | 1250 | 12 | 218 | 72 | 72 | none |
| ChlorineConcentration | main | chemical / sensor | 467 | 3840 | 166 | 3 | 281 | 93 | 93 | none |
| SemgHandMovementCh2 | main | health / muscle | 450 | 450 | 1500 | 6 | 270 | 90 | 90 | none |
| LargeKitchenAppliances | main | device | 375 | 375 | 720 | 3 | 225 | 75 | 75 | none |
| InsectWingbeatSound | main | acoustic / sensor | 220 | 1980 | 256 | 11 | 132 | 44 | 44 | none |
| CricketX | main | motion | 390 | 390 | 300 | 12 | 234 | 78 | 78 | none |
| TwoLeadECG | caveat | ECG / health | 23 | 1139 | 82 | 2 | 13 | 5 | 5 | none |
| ItalyPowerDemand | caveat | device / power | 67 | 1029 | 24 | 2 | 41 | 13 | 13 | none |

Three initially downloaded candidates (PLAID, MelbournePedestrian, AllGestureWiimoteX) contain non-finite data and were excluded to preserve the existing finite-series preprocessing. They were kept on disk. Some included datasets have substantially fewer than 200 calibration examples despite meeting the train-size criterion; interpret their reliability with care.

## Methods and analysis

Every stopping policy and threshold grid is fixed without using test labels or corruption identity. The model and stream generation follow v2: causal running normalization, 672-kernel MiniRocket + logistic regression per checkpoint, the same train/validation/calibration split and five deterministic paired draws. Alpha thresholds use one balanced mixed-corruption calibration set per draw and are reused unchanged across test conditions. Clean probability and clean TEASER use clean checkpoints; augmentation ablations use the same augmented checkpoints as ours. Augmented TEASER estimates safety out of fold while grouping each original with its augmented copy. Its safety estimator therefore requires separate fitting, even though its base classifiers are reused.

ECEC is an adaptation of the [authors? implementation](https://github.com/junweilvhfut/ECEC): per-checkpoint predicted-class precision is estimated on clean validation data, then confidence for the current class is one minus the product of its historical failure probabilities; a fixed threshold grid is swept. MiniRocket replaces its original classifier, and zero-prediction class/checkpoint cells use precision zero. It is not an exact reproduction of the original training protocol.

Results average datasets within each draw, then report the mean and sample SD across five draws. This SD reflects only the five augmentation/corruption draws: the UCR train/test and fit/calibration splits are fixed. The combined score is the harmonic mean of accuracy and one minus earliness. Forced final predictions are included in error. For probability/ECEC/ours, forced means no threshold was met even at full length; for TEASER, every full-length decision is counted as forced, consistent with v2. Gaussian noise uses fit-split signal power as in v2. Probability/TEASER/ECEC stopping settings are swept rather than tuned on test labels.

Matched-earliness comparisons below use the ?=0.10 operating point and linearly interpolate each baseline?s pre-specified empirical frontier at the same observed earliness, independently per dataset and draw. No extrapolation is used; ties in earliness use the first pre-specified setting rather than choosing by test error. Interpolation is a descriptive post-hoc comparison, not a deployable test-specific stopping policy. The sweep is not pruned into a test-error-selected lower envelope: doing so would select settings using test outcomes. Consequently the comparison does not prove mathematical dominance over every possible baseline mixture. Matched sample counts can differ by method and condition.

## Augmented + mixed calibration at ? = 0.10

Macro mean ? sample SD across the five draws. Error includes forced full-length decisions; 0.10 is a target, not a guarantee.

| Condition | Accuracy | Balanced accuracy | Macro-F1 | Error | Earliness | Forced fraction | Combined |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.7692 ? 0.0034 | 0.7394 ? 0.0032 | 0.7376 ? 0.0032 | 0.2308 ? 0.0034 | 0.6145 ? 0.0085 | 0.3346 ? 0.0129 | 0.4697 ? 0.0087 |
| noise_20db | 0.7296 ? 0.0017 | 0.6958 ? 0.0021 | 0.6914 ? 0.0019 | 0.2704 ? 0.0017 | 0.6629 ? 0.0074 | 0.3904 ? 0.0131 | 0.4192 ? 0.0081 |
| noise_10db | 0.6764 ? 0.0014 | 0.6361 ? 0.0016 | 0.6326 ? 0.0019 | 0.3236 ? 0.0014 | 0.7125 ? 0.0079 | 0.4609 ? 0.0105 | 0.3616 ? 0.0076 |
| noise_5db | 0.6170 ? 0.0018 | 0.5720 ? 0.0021 | 0.5661 ? 0.0018 | 0.3830 ? 0.0018 | 0.7394 ? 0.0076 | 0.5071 ? 0.0084 | 0.3242 ? 0.0065 |
| missing_10pct | 0.7556 ? 0.0025 | 0.7253 ? 0.0025 | 0.7233 ? 0.0026 | 0.2444 ? 0.0025 | 0.6255 ? 0.0075 | 0.3413 ? 0.0116 | 0.4584 ? 0.0080 |
| missing_30pct | 0.7295 ? 0.0042 | 0.6980 ? 0.0034 | 0.6955 ? 0.0037 | 0.2705 ? 0.0042 | 0.6415 ? 0.0071 | 0.3521 ? 0.0108 | 0.4409 ? 0.0077 |
| missing_50pct | 0.6883 ? 0.0057 | 0.6552 ? 0.0049 | 0.6515 ? 0.0058 | 0.3117 ? 0.0057 | 0.6533 ? 0.0065 | 0.3580 ? 0.0127 | 0.4254 ? 0.0066 |

## Fixed-length error gaps (clean minus augmented)

Positive values favor augmentation.

| Condition | 20% | 40% | 60% | 80% | 100% |
| --- | --- | --- | --- | --- | --- |
| clean | -0.0199 ? 0.0017 | -0.0175 ? 0.0023 | -0.0152 ? 0.0020 | -0.0153 ? 0.0017 | -0.0152 ? 0.0022 |
| noise_20db | 0.0869 ? 0.0028 | 0.1260 ? 0.0021 | 0.1629 ? 0.0020 | 0.1557 ? 0.0027 | 0.1543 ? 0.0038 |
| noise_10db | 0.0936 ? 0.0062 | 0.1627 ? 0.0021 | 0.2164 ? 0.0018 | 0.2560 ? 0.0021 | 0.2209 ? 0.0011 |
| noise_5db | 0.0764 ? 0.0048 | 0.1506 ? 0.0042 | 0.2066 ? 0.0030 | 0.2252 ? 0.0033 | 0.2284 ? 0.0024 |
| missing_10pct | 0.0015 ? 0.0044 | 0.0011 ? 0.0025 | -0.0037 ? 0.0036 | -0.0043 ? 0.0022 | -0.0028 ? 0.0030 |
| missing_30pct | 0.0403 ? 0.0013 | 0.0363 ? 0.0049 | 0.0305 ? 0.0034 | 0.0257 ? 0.0030 | 0.0302 ? 0.0033 |
| missing_50pct | 0.0672 ? 0.0035 | 0.0722 ? 0.0031 | 0.0597 ? 0.0035 | 0.0543 ? 0.0023 | 0.0592 ? 0.0032 |

### Does augmentation help at every evaluated prefix?

This is a macro-average comparison over all nine checkpoints, not a statement that every individual dataset benefits.

| Condition | Nonnegative gap at all 9 checkpoints | Minimum gap | Maximum gap | Checkpoints favoring clean |
| --- | --- | --- | --- | --- |
| clean | no | -0.0199 | -0.0119 | 9 |
| noise_20db | yes | 0.0869 | 0.1714 | 0 |
| noise_10db | yes | 0.0936 | 0.2560 | 0 |
| noise_5db | yes | 0.0764 | 0.2284 | 0 |
| missing_10pct | no | -0.0047 | 0.0039 | 6 |
| missing_30pct | yes | 0.0229 | 0.0403 | 0 |
| missing_50pct | yes | 0.0535 | 0.0722 | 0 |

## Matched-earliness results, paired tests and win/tie/loss

Margin is baseline error minus our error; positive favors ours. Wilcoxon tests use one mean paired margin per dataset (only datasets matched in all five draws). Holm adjustment covers the full family of valid condition ? baseline comparisons. W/T/L refers to those same datasets, with a numerical tolerance of 1e-12. A mean of matched draws can include a different subset of datasets per draw; the coverage column makes that explicit.

| Condition | Comparator | Matched error margin ? draw SD | Datasets/draw | Datasets in test | W/T/L | Wilcoxon p | Holm p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clean | Probability | -0.0053 ? 0.0026 | 20?21 | 19 | 7/0/12 | 0.1688 | 1 |
| clean | TEASER | 0.0065 ? 0.0042 | 10?12 | 10 | 4/0/6 | 0.625 | 1 |
| clean | ECEC adaptation | 0.0029 ? 0.0029 | 21?21 | 19 | 9/0/10 | 0.9217 | 1 |
| clean | Aug + probability | 0.0048 ? 0.0024 | 20?23 | 19 | 14/0/5 | 0.02893 | 0.5786 |
| clean | Aug + TEASER | 0.0199 ? 0.0030 | 10?13 | 10 | 8/0/2 | 0.01953 | 0.4297 |
| noise_20db | Probability | 0.1505 ? 0.0233 | 16?17 | 13 | 10/0/3 | 0.003418 | 0.08887 |
| noise_20db | TEASER | 0.2442 ? 0.0104 | 11?15 | 11 | 11/0/0 | 0.0009766 | 0.02734 |
| noise_20db | ECEC adaptation | 0.1446 ? 0.0033 | 20?20 | 19 | 17/0/2 | 3.815e-05 | 0.001297 |
| noise_20db | Aug + probability | 0.0074 ? 0.0022 | 20?23 | 20 | 13/0/7 | 0.06958 | 0.9023 |
| noise_20db | Aug + TEASER | 0.0172 ? 0.0021 | 11?15 | 11 | 7/0/4 | 0.04199 | 0.7559 |
| noise_10db | Probability | 0.2003 ? 0.0144 | 10?11 | 8 | 8/0/0 | 0.007812 | 0.1953 |
| noise_10db | TEASER | 0.2866 ? 0.0068 | 14?15 | 11 | 11/0/0 | 0.0009766 | 0.02734 |
| noise_10db | ECEC adaptation | 0.1993 ? 0.0049 | 16?18 | 16 | 15/0/1 | 0.0001526 | 0.004578 |
| noise_10db | Aug + probability | 0.0068 ? 0.0011 | 21?23 | 21 | 15/0/6 | 0.04599 | 0.7575 |
| noise_10db | Aug + TEASER | 0.0112 ? 0.0023 | 12?14 | 11 | 5/0/6 | 0.9658 | 1 |
| noise_5db | Probability | 0.1731 ? 0.0364 | 8?11 | 7 | 7/0/0 | 0.01562 | 0.3594 |
| noise_5db | TEASER | 0.2039 ? 0.0215 | 10?12 | 8 | 8/0/0 | 0.007812 | 0.1953 |
| noise_5db | ECEC adaptation | 0.1869 ? 0.0091 | 15?17 | 15 | 15/0/0 | 6.104e-05 | 0.001953 |
| noise_5db | Aug + probability | 0.0051 ? 0.0018 | 21?23 | 21 | 15/0/6 | 0.1281 | 1 |
| noise_5db | Aug + TEASER | 0.0051 ? 0.0029 | 12?14 | 8 | 4/0/4 | 0.6406 | 1 |
| missing_10pct | Probability | 0.0056 ? 0.0044 | 20?21 | 19 | 11/0/8 | 0.1819 | 1 |
| missing_10pct | TEASER | 0.0119 ? 0.0049 | 9?12 | 9 | 5/0/4 | 0.7344 | 1 |
| missing_10pct | ECEC adaptation | 0.0155 ? 0.0029 | 20?21 | 19 | 13/0/6 | 0.04456 | 0.7575 |
| missing_10pct | Aug + probability | 0.0038 ? 0.0031 | 20?23 | 19 | 12/0/7 | 0.09551 | 0.9551 |
| missing_10pct | Aug + TEASER | 0.0180 ? 0.0040 | 11?13 | 10 | 8/0/2 | 0.06445 | 0.9023 |
| missing_30pct | Probability | 0.0471 ? 0.0054 | 20?22 | 19 | 17/0/2 | 3.815e-05 | 0.001297 |
| missing_30pct | TEASER | 0.0351 ? 0.0088 | 9?13 | 8 | 6/0/2 | 0.07812 | 0.9023 |
| missing_30pct | ECEC adaptation | 0.0523 ? 0.0050 | 20?21 | 19 | 16/0/3 | 0.0004196 | 0.01217 |
| missing_30pct | Aug + probability | 0.0042 ? 0.0038 | 21?23 | 19 | 14/0/5 | 0.06629 | 0.9023 |
| missing_30pct | Aug + TEASER | 0.0186 ? 0.0040 | 11?13 | 10 | 8/0/2 | 0.01953 | 0.4297 |
| missing_50pct | Probability | 0.0935 ? 0.0034 | 18?19 | 18 | 17/0/1 | 1.526e-05 | 0.0005341 |
| missing_50pct | TEASER | 0.0746 ? 0.0084 | 10?16 | 8 | 7/0/1 | 0.03906 | 0.7422 |
| missing_50pct | ECEC adaptation | 0.0891 ? 0.0038 | 18?21 | 17 | 15/0/2 | 0.0001068 | 0.003311 |
| missing_50pct | Aug + probability | 0.0037 ? 0.0026 | 20?22 | 18 | 13/0/5 | 0.09874 | 0.9551 |
| missing_50pct | Aug + TEASER | 0.0132 ? 0.0052 | 12?15 | 10 | 7/0/3 | 0.04883 | 0.7575 |

## Descriptive matched frontier margins at common earliness levels

Each cell is the baseline-minus-ours error margin, mean ? sample SD across the five draw-specific dataset means, followed by the minimum and maximum number of matched datasets per draw. Values are descriptive interpolation without extrapolation; missing overlap is shown as n/a.

| Condition | Comparator | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
| --- | --- | --- | --- | --- | --- | --- |
| clean | Probability | -0.0096 ? 0.0019 (n=9?9) | -0.0106 ? 0.0051 (n=9?11) | -0.0132 ? 0.0059 (n=8?10) | -0.0075 ? 0.0030 (n=10?13) | -0.0072 ? 0.0046 (n=7?10) |
| clean | TEASER | n/a | -0.0072 ? 0.0027 (n=5?6) | -0.0035 ? 0.0042 (n=9?10) | 0.0081 ? 0.0035 (n=10?14) | 0.0128 ? 0.0030 (n=8?11) |
| clean | ECEC adaptation | 0.0105 ? 0.0019 (n=9?9) | 0.0005 ? 0.0040 (n=7?9) | 0.0027 ? 0.0045 (n=7?9) | 0.0166 ? 0.0032 (n=7?11) | 0.0142 ? 0.0069 (n=7?9) |
| clean | Aug + probability | 0.0015 ? 0.0032 (n=9?9) | -0.0005 ? 0.0027 (n=9?11) | 0.0020 ? 0.0043 (n=7?9) | 0.0033 ? 0.0026 (n=10?13) | 0.0021 ? 0.0054 (n=7?10) |
| clean | Aug + TEASER | n/a | 0.0054 ? 0.0044 (n=6?8) | 0.0098 ? 0.0026 (n=9?10) | 0.0202 ? 0.0060 (n=10?14) | 0.0234 ? 0.0048 (n=7?11) |
| noise_20db | Probability | 0.1099 ? 0.0066 (n=6?6) | 0.2086 ? 0.0198 (n=6?8) | 0.1388 ? 0.0133 (n=7?9) | 0.2082 ? 0.0127 (n=9?11) | 0.1725 ? 0.0116 (n=7?10) |
| noise_20db | TEASER | n/a | 0.3331 ? 0.0538 (n=3?4) | 0.2421 ? 0.0156 (n=8?11) | 0.2443 ? 0.0228 (n=12?15) | 0.2210 ? 0.0154 (n=12?15) |
| noise_20db | ECEC adaptation | 0.1848 ? 0.0097 (n=8?8) | 0.2241 ? 0.0181 (n=7?9) | 0.1844 ? 0.0085 (n=9?12) | 0.1967 ? 0.0222 (n=10?13) | 0.1716 ? 0.0228 (n=9?12) |
| noise_20db | Aug + probability | 0.0035 ? 0.0059 (n=8?8) | 0.0055 ? 0.0020 (n=9?10) | 0.0027 ? 0.0021 (n=8?12) | 0.0021 ? 0.0048 (n=11?14) | 0.0024 ? 0.0013 (n=12?14) |
| noise_20db | Aug + TEASER | n/a | 0.0075 ? 0.0040 (n=3?4) | 0.0098 ? 0.0037 (n=10?13) | 0.0124 ? 0.0055 (n=12?15) | 0.0112 ? 0.0021 (n=12?14) |
| noise_10db | Probability | 0.0985 ? 0.0077 (n=5?5) | 0.0365 ? 0.0100 (n=3?3) | 0.0671 ? 0.0388 (n=2?3) | 0.1792 ? 0.0324 (n=4?6) | 0.2191 ? 0.0227 (n=4?7) |
| noise_10db | TEASER | n/a | 0.3422 ? 0.0493 (n=3?4) | 0.2235 ? 0.0166 (n=9?11) | 0.2528 ? 0.0208 (n=11?13) | 0.2684 ? 0.0167 (n=9?14) |
| noise_10db | ECEC adaptation | 0.2065 ? 0.0063 (n=8?8) | 0.2924 ? 0.0088 (n=7?7) | 0.2486 ? 0.0211 (n=9?10) | 0.2463 ? 0.0166 (n=10?12) | 0.2672 ? 0.0301 (n=9?14) |
| noise_10db | Aug + probability | -0.0014 ? 0.0044 (n=8?8) | 0.0005 ? 0.0047 (n=7?8) | -0.0032 ? 0.0063 (n=9?10) | -0.0009 ? 0.0041 (n=10?12) | 0.0017 ? 0.0046 (n=11?14) |
| noise_10db | Aug + TEASER | n/a | 0.0045 ? 0.0017 (n=1?2) | 0.0018 ? 0.0090 (n=8?10) | 0.0081 ? 0.0049 (n=10?13) | 0.0029 ? 0.0053 (n=12?16) |
| noise_5db | Probability | 0.0007 ? 0.0076 (n=2?2) | 0.0550 ? 0.0107 (n=2?2) | 0.0561 ? 0.0482 (n=2?3) | 0.1178 ? 0.0325 (n=4?5) | 0.1829 ? 0.0352 (n=4?6) |
| noise_5db | TEASER | n/a | 0.2621 ? 0.0256 (n=4?5) | 0.2371 ? 0.0110 (n=8?9) | 0.2051 ? 0.0104 (n=12?15) | 0.1910 ? 0.0231 (n=10?11) |
| noise_5db | ECEC adaptation | 0.1987 ? 0.0096 (n=7?8) | 0.2333 ? 0.0117 (n=8?9) | 0.2336 ? 0.0169 (n=8?9) | 0.2114 ? 0.0051 (n=11?12) | 0.1654 ? 0.0113 (n=9?12) |
| noise_5db | Aug + probability | -0.0081 ? 0.0046 (n=7?8) | -0.0003 ? 0.0063 (n=8?9) | -0.0012 ? 0.0031 (n=9?10) | 0.0010 ? 0.0050 (n=10?14) | 0.0006 ? 0.0019 (n=10?11) |
| noise_5db | Aug + TEASER | n/a | n/a | -0.0036 ? 0.0079 (n=6?8) | 0.0040 ? 0.0059 (n=12?14) | -0.0009 ? 0.0037 (n=11?14) |
| missing_10pct | Probability | 0.0044 ? 0.0029 (n=9?9) | 0.0092 ? 0.0046 (n=8?11) | 0.0054 ? 0.0057 (n=9?11) | 0.0059 ? 0.0034 (n=10?13) | -0.0008 ? 0.0066 (n=7?10) |
| missing_10pct | TEASER | n/a | 0.0231 ? 0.0033 (n=3?4) | 0.0108 ? 0.0056 (n=10?12) | 0.0148 ? 0.0044 (n=10?14) | 0.0158 ? 0.0069 (n=8?12) |
| missing_10pct | ECEC adaptation | 0.0330 ? 0.0031 (n=9?9) | 0.0218 ? 0.0029 (n=6?9) | 0.0163 ? 0.0063 (n=8?10) | 0.0204 ? 0.0025 (n=7?11) | 0.0205 ? 0.0050 (n=7?9) |
| missing_10pct | Aug + probability | -0.0001 ? 0.0023 (n=9?9) | 0.0003 ? 0.0025 (n=8?10) | 0.0006 ? 0.0029 (n=8?11) | 0.0027 ? 0.0022 (n=10?13) | 0.0004 ? 0.0029 (n=7?11) |
| missing_10pct | Aug + TEASER | n/a | 0.0055 ? 0.0055 (n=5?7) | 0.0080 ? 0.0065 (n=10?12) | 0.0176 ? 0.0047 (n=10?14) | 0.0182 ? 0.0058 (n=8?12) |
| missing_30pct | Probability | 0.0770 ? 0.0058 (n=7?8) | 0.0708 ? 0.0048 (n=8?11) | 0.0643 ? 0.0055 (n=9?11) | 0.0461 ? 0.0118 (n=9?12) | 0.0306 ? 0.0054 (n=8?11) |
| missing_30pct | TEASER | n/a | 0.0927 ? 0.0167 (n=2?3) | 0.0548 ? 0.0060 (n=8?8) | 0.0490 ? 0.0098 (n=11?15) | 0.0481 ? 0.0074 (n=12?14) |
| missing_30pct | ECEC adaptation | 0.0968 ? 0.0107 (n=7?9) | 0.0659 ? 0.0042 (n=7?10) | 0.0674 ? 0.0047 (n=8?10) | 0.0377 ? 0.0138 (n=8?11) | 0.0467 ? 0.0112 (n=9?11) |
| missing_30pct | Aug + probability | -0.0008 ? 0.0032 (n=7?9) | 0.0000 ? 0.0016 (n=8?10) | 0.0017 ? 0.0027 (n=10?12) | 0.0023 ? 0.0032 (n=10?13) | 0.0008 ? 0.0029 (n=10?12) |
| missing_30pct | Aug + TEASER | n/a | 0.0051 ? 0.0029 (n=3?7) | 0.0112 ? 0.0033 (n=11?12) | 0.0144 ? 0.0049 (n=11?15) | 0.0151 ? 0.0030 (n=12?14) |
| missing_50pct | Probability | 0.1364 ? 0.0084 (n=7?8) | 0.1265 ? 0.0064 (n=7?9) | 0.0971 ? 0.0142 (n=7?9) | 0.0752 ? 0.0065 (n=8?10) | 0.0576 ? 0.0044 (n=10?11) |
| missing_50pct | TEASER | n/a | 0.1350 ? 0.0199 (n=2?4) | 0.1346 ? 0.0213 (n=5?8) | 0.1071 ? 0.0078 (n=13?16) | 0.0782 ? 0.0064 (n=13?16) |
| missing_50pct | ECEC adaptation | 0.1616 ? 0.0126 (n=7?8) | 0.1294 ? 0.0126 (n=7?9) | 0.1162 ? 0.0072 (n=8?11) | 0.0987 ? 0.0077 (n=9?12) | 0.0652 ? 0.0103 (n=10?12) |
| missing_50pct | Aug + probability | -0.0027 ? 0.0052 (n=7?8) | -0.0035 ? 0.0025 (n=8?9) | -0.0015 ? 0.0039 (n=10?13) | 0.0032 ? 0.0060 (n=12?14) | 0.0023 ? 0.0041 (n=11?14) |
| missing_50pct | Aug + TEASER | n/a | 0.0024 ? 0.0084 (n=2?5) | 0.0004 ? 0.0040 (n=9?10) | 0.0096 ? 0.0047 (n=13?16) | 0.0096 ? 0.0039 (n=13?16) |

## Nemenyi critical-difference diagrams

Each diagram ranks methods by matched error on datasets with all methods and all five draws matched; results use the Friedman omnibus test and the Nemenyi critical difference. Where fewer than three datasets overlap, the panel explicitly reports insufficient data. Common-overlap selection and post-hoc interpolation limit inference.

- [clean](cd_diagrams/clean.png): complete overlap N=5; Friedman p=0.1003.
- [noise_20db](cd_diagrams/noise_20db.png): complete overlap N=3; Friedman p=0.1485.
- [noise_10db](cd_diagrams/noise_10db.png): complete overlap N=2; Friedman p=n/a.
- [noise_5db](cd_diagrams/noise_5db.png): complete overlap N=2; Friedman p=n/a.
- [missing_10pct](cd_diagrams/missing_10pct.png): complete overlap N=5; Friedman p=0.5702.
- [missing_30pct](cd_diagrams/missing_30pct.png): complete overlap N=4; Friedman p=0.4159.
- [missing_50pct](cd_diagrams/missing_50pct.png): complete overlap N=2; Friedman p=n/a.

## Clean-data cost and series length

At each fixed checkpoint, positive error gap means augmentation helps; negative means a clean-model advantage. Spearman correlations below are exploratory across datasets, without multiplicity correction. They do not establish that series length causes the advantage.

| Condition | Prefix | Datasets | Mean error gap | Spearman ?(length,gap) | Unadjusted p |
| --- | --- | --- | --- | --- | --- |
| clean | 0.2 | 26 | -0.0199 | 0.1143 | 0.5783 |
| clean | 0.6 | 26 | -0.0152 | -0.2549 | 0.2089 |
| clean | 1.0 | 26 | -0.0152 | -0.2761 | 0.1722 |
| noise_20db | 0.2 | 26 | 0.0869 | 0.2583 | 0.2027 |
| noise_20db | 0.6 | 26 | 0.1629 | 0.3640 | 0.06753 |
| noise_20db | 1.0 | 26 | 0.1543 | 0.4957 | 0.01001 |
| noise_10db | 0.2 | 26 | 0.0936 | 0.1557 | 0.4477 |
| noise_10db | 0.6 | 26 | 0.2164 | 0.4889 | 0.01126 |
| noise_10db | 1.0 | 26 | 0.2209 | 0.4844 | 0.01214 |
| noise_5db | 0.2 | 26 | 0.0764 | 0.0927 | 0.6524 |
| noise_5db | 0.6 | 26 | 0.2066 | 0.5087 | 0.007957 |
| noise_5db | 1.0 | 26 | 0.2284 | 0.4622 | 0.01744 |
| missing_10pct | 0.2 | 26 | 0.0015 | -0.1522 | 0.4578 |
| missing_10pct | 0.6 | 26 | -0.0037 | -0.0031 | 0.9881 |
| missing_10pct | 1.0 | 26 | -0.0028 | 0.0044 | 0.9828 |
| missing_30pct | 0.2 | 26 | 0.0403 | -0.0263 | 0.8984 |
| missing_30pct | 0.6 | 26 | 0.0305 | 0.0794 | 0.6999 |
| missing_30pct | 1.0 | 26 | 0.0302 | 0.2104 | 0.3022 |
| missing_50pct | 0.2 | 26 | 0.0672 | 0.1748 | 0.393 |
| missing_50pct | 0.6 | 26 | 0.0597 | 0.2460 | 0.2258 |
| missing_50pct | 1.0 | 26 | 0.0592 | 0.2672 | 0.187 |

### Dataset-level augmentation effects at the 40% checkpoint

The three largest and smallest paired mean error gaps are shown for each condition; positive favors augmented training. These descriptive rankings use test outcomes and are not method-selection rules.

| Condition | Largest augmentation advantages | Smallest gaps / clean advantages |
| --- | --- | --- |
| clean | InsectWingbeatSound (+0.012), FordA (+0.010), FordB (+0.008) | MixedShapesRegularTrain (-0.056), Crop (-0.050), SwedishLeaf (-0.048) |
| noise_20db | NonInvasiveFetalECGThorax1 (+0.675), MixedShapesRegularTrain (+0.567), HandOutlines (+0.515) | FaceAll (-0.036), TwoPatterns (-0.006), SyntheticControl (-0.006) |
| noise_10db | MixedShapesRegularTrain (+0.549), NonInvasiveFetalECGThorax1 (+0.508), HandOutlines (+0.497) | FaceAll (-0.040), SyntheticControl (-0.025), TwoPatterns (-0.006) |
| noise_5db | MixedShapesRegularTrain (+0.511), HandOutlines (+0.438), InsectWingbeatSound (+0.324) | MiddlePhalanxOutlineCorrect (-0.020), FaceAll (-0.018), EthanolLevel (-0.014) |
| missing_10pct | MiddlePhalanxOutlineCorrect (+0.047), PhalangesOutlinesCorrect (+0.046), HandOutlines (+0.045) | MixedShapesRegularTrain (-0.049), SemgHandMovementCh2 (-0.045), EOGHorizontalSignal (-0.045) |
| missing_30pct | NonInvasiveFetalECGThorax1 (+0.182), HandOutlines (+0.161), FordA (+0.114) | EOGHorizontalSignal (-0.048), SemgHandMovementCh2 (-0.040), CricketX (-0.025) |
| missing_50pct | HandOutlines (+0.306), NonInvasiveFetalECGThorax1 (+0.291), MixedShapesRegularTrain (+0.270) | EOGHorizontalSignal (-0.044), ElectricDevices (-0.019), CricketX (-0.014) |

## Verdict and limitations

| Condition | Equal-earliness verdict |
| --- | --- |
| clean | no universal significant advantage established |
| noise_20db | no universal significant advantage established |
| noise_10db | no universal significant advantage established |
| noise_5db | no universal significant advantage established |
| missing_10pct | no universal significant advantage established |
| missing_30pct | no universal significant advantage established |
| missing_50pct | no universal significant advantage established |

These are exploratory cross-dataset comparisons of synthetic corruptions on a single archive with one fixed split per dataset. Dataset correlations, tuning and interval overlap limit a broad significance or external-validity claim. The empirical mixed-calibration rule is not a formal risk-control guarantee, especially with small calibration sets and forced full-length predictions. Causal normalization applies to the archived observations supplied to this pipeline; any upstream whole-series preprocessing performed before archival has not been audited or undone, and simulated corruption is applied to those archived values.

## Integrity and convergence checks

Prior-source snapshot comparison: 1673 files before; 0 changed, 0 removed and 0 added after. Details: [source_audit.json](source_audit.json).

Checkpoint model audit: 1506 models, 0 reached the configured logistic iteration limit; 0 audit read failures and 0 warning/error log lines. See [model_convergence_audit.csv](model_convergence_audit.csv) and [warnings_and_errors.json](logs/warnings_and_errors.json).

TEASER safety uses out-of-fold logistic predictions but, as in v2, the unsupervised MiniRocket feature transform is fitted once before those folds; this is not fully nested cross-validation. Augmented TEASER keeps each original and its augmented counterpart in the same safety fold.

Per-dataset and macro metrics, the matched pairs, test statistics, full draw-level records, code, models and figures are in this directory.
