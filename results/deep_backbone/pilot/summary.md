# Learned anytime ECTS: three-GPU methodological pilot

**Decision:** do not promote cross-checkpoint weight inheritance as the full method on this evidence. The validation winner is C_l0: independent heads without consistency. A learned TCN does show a real equal-prefix Gaussian-noise advantage over augmented MiniRocket (positive mean gains at all nine checkpoints), but missing-data gains are confined to part of the curve and there is a clean-data cost. At prefixes 0.5/0.6, all four corrupted conditions have positive mean gains; this does not imply dominance at every prefix or on every dataset. No ten-test Holm-corrected pilot comparison reaches 0.05; the smallest adjusted p-value is 0.078125 with only eight datasets.

All prior runs were read-only. Eight datasets, draws 0/1/2, five test conditions, nine checkpoints (20%-100%). Each reported mean first averages the eight datasets within a draw; +/- is sample SD over three paired draws. No test label or test condition selected a model, stopping setting, epoch, or corruption.

## Design and checks

**Strictly causal TCN:** five residual blocks, two kernel-3 convolutions/block, dilations 1/2/4/8/16, 64 channels, dropout 0.1, channel-only LayerNorm. The receptive field of local features is 125 samples; a causal prefix 1/2/4-bin temporal pyramid plus the current feature gives the classifier access to the entire observed prefix. No symmetric padding, time-axis normalization, or future pooling.

**Fair inheritance control:** each mode sees the same growing-checkpoint curriculum (two epochs/checkpoint), then at most 22 joint epochs. Loss is the mean CE over active checkpoints, plus optional lambda=0.1 adjacent Jensen-Shannon divergence. B copies the preceding already-trained head when introducing each new checkpoint; C introduces an independent random head. A uses one shared head. All modes have a shared TCN backbone; C means independent heads, not nine independent backbones. Same seeds, batches, augmentation copy, validation examples, optimizer, and early-stop criteria. Progressive initialization is not merely copying random weights. All eight inheritance events were checked exactly.

Adam lr=0.001, weight_decay=0.0001, gradient clip=1. Training uses AMP; inference is float32. The existing MiniRocket logistic regression uses no class weights, so the deep empirical-frequency CE also uses none. Balanced accuracy and macro-F1 are retained to expose majority-class failures. Early stopping restores the minimum all-checkpoint validation CE; all variants get the full curriculum plus at least six joint epochs.

Validation is a fixed 50:50 concatenation of clean validation and an independent six-corruption validation mixture. Training retains the exact prior clean-fit-plus-one-noise/missing-copy recipe. Calibration is the exact prior six-corruption mixture. The clean deep reference uses no training augmentation, but the same mixed validation protocol. Original fit caps and split indices are retained, including ECG5000 having only 201 fitting examples. Previously stored MiniRocket fixed-length/frontier metrics and calibration earliness are reused directly: no MiniRocket retraining. Raw corrupted test arrays were not persisted by scale_v1; they are deterministically reconstructed with its unchanged seeded generators. Per-stream SHA256 digests are recorded and checked equal across all deep variants.

**Validation-selected primary variant: C_l0 (Independent, lambda=0).** Selection is a SINGLE variant across the pilot, based only on mean validation accuracy across checkpoints/datasets/draws. All six augmented variants are reported, not just the winner.

| variant | validation_accuracy |
| --- | --- |
| C_l0 | 0.8021 |
| C_l01 | 0.8003 |
| B_l0 | 0.7998 |
| B_l01 | 0.7989 |
| A_l01 | 0.7869 |
| A_l0 | 0.7830 |

## Datasets and retained splits

| dataset | train_size | test_size | length | n_classes | fit_count | validation_count | calibration_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FordA | 3601 | 1320 | 500 | 2 | 2301 | 400 | 900 |
| ECG5000 | 500 | 4500 | 140 | 5 | 201 | 99 | 200 |
| StarLightCurves | 1000 | 8236 | 1024 | 3 | 550 | 200 | 250 |
| Wafer | 1000 | 6164 | 152 | 2 | 550 | 200 | 250 |
| ElectricDevices | 8926 | 7711 | 96 | 7 | 4000 | 401 | 2000 |
| Crop | 7200 | 16800 | 46 | 24 | 4000 | 408 | 1800 |
| NonInvasiveFetalECGThorax1 | 1800 | 1965 | 750 | 42 | 994 | 357 | 449 |
| HandOutlines | 1000 | 370 | 2709 | 2 | 550 | 200 | 250 |

## Q1: equal-earliness accuracy versus augmented MiniRocket

Positive gaps below mean the learned model reduces error (percentage points). Prefix lengths are exactly paired within each dataset; nominal fractions use the same rounded checkpoint lengths as all prior runs.

| condition | 0.2 | 0.4 | 0.5 | 0.6 | 0.8 | 1.0 | all 9 prefixes nonnegative |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clean | -2.44 +/- 0.47 | -0.33 +/- 0.39 | -0.13 +/- 0.51 | +0.12 +/- 0.32 | -2.41 +/- 0.85 | -2.17 +/- 0.47 | False |
| noise_10db | +3.65 +/- 0.19 | +4.28 +/- 0.55 | +4.19 +/- 0.24 | +4.64 +/- 0.22 | +2.22 +/- 0.32 | +2.01 +/- 0.62 | True |
| noise_5db | +3.47 +/- 1.69 | +4.87 +/- 0.70 | +4.82 +/- 0.55 | +5.48 +/- 0.54 | +3.50 +/- 0.51 | +3.61 +/- 1.01 | True |
| missing_30pct | -0.92 +/- 0.92 | +0.81 +/- 0.52 | +0.87 +/- 0.62 | +1.10 +/- 0.29 | -1.36 +/- 0.91 | -1.39 +/- 0.54 | False |
| missing_50pct | -1.46 +/- 1.42 | +0.22 +/- 0.32 | +0.51 +/- 0.38 | +0.42 +/- 0.43 | -1.66 +/- 0.85 | -1.31 +/- 0.44 | False |

Per-condition accuracy/error/earliness at the two primary fixed checkpoints:

| condition | method | prefix | accuracy % | error % | balanced accuracy % | earliness |
| --- | --- | --- | --- | --- | --- | --- |
| clean | A_clean | 0.5000 | 78.65 +/- 0.08 | 21.35 +/- 0.08 | 70.72 +/- 0.52 | 0.50 +/- 0.00 |
| clean | A_clean | 0.6000 | 77.51 +/- 0.60 | 22.49 +/- 0.60 | 69.71 +/- 0.23 | 0.60 +/- 0.00 |
| clean | C_l0 | 0.5000 | 83.24 +/- 0.55 | 16.76 +/- 0.55 | 75.35 +/- 0.68 | 0.50 +/- 0.00 |
| clean | C_l0 | 0.6000 | 83.26 +/- 0.37 | 16.74 +/- 0.37 | 75.74 +/- 0.29 | 0.60 +/- 0.00 |
| clean | MiniRocket_aug | 0.5000 | 83.37 +/- 0.04 | 16.63 +/- 0.04 | 75.84 +/- 0.16 | 0.50 +/- 0.00 |
| clean | MiniRocket_aug | 0.6000 | 83.14 +/- 0.07 | 16.86 +/- 0.07 | 75.90 +/- 0.09 | 0.60 +/- 0.00 |
| clean | MiniRocket_clean | 0.5000 | 84.81 +/- 0.00 | 15.19 +/- 0.00 | 76.91 +/- 0.00 | 0.50 +/- 0.00 |
| clean | MiniRocket_clean | 0.6000 | 84.77 +/- 0.00 | 15.23 +/- 0.00 | 77.25 +/- 0.00 | 0.60 +/- 0.00 |
| missing_30pct | A_clean | 0.5000 | 71.38 +/- 0.62 | 28.62 +/- 0.62 | 61.94 +/- 0.43 | 0.50 +/- 0.00 |
| missing_30pct | A_clean | 0.6000 | 71.02 +/- 0.83 | 28.98 +/- 0.83 | 61.56 +/- 0.64 | 0.60 +/- 0.00 |
| missing_30pct | C_l0 | 0.5000 | 79.43 +/- 0.64 | 20.57 +/- 0.64 | 71.47 +/- 0.57 | 0.50 +/- 0.00 |
| missing_30pct | C_l0 | 0.6000 | 79.90 +/- 0.19 | 20.10 +/- 0.19 | 72.26 +/- 0.25 | 0.60 +/- 0.00 |
| missing_30pct | MiniRocket_aug | 0.5000 | 78.55 +/- 0.27 | 21.45 +/- 0.27 | 70.87 +/- 0.44 | 0.50 +/- 0.00 |
| missing_30pct | MiniRocket_aug | 0.6000 | 78.80 +/- 0.24 | 21.20 +/- 0.24 | 71.28 +/- 0.40 | 0.60 +/- 0.00 |
| missing_30pct | MiniRocket_clean | 0.5000 | 71.82 +/- 0.13 | 28.18 +/- 0.13 | 63.37 +/- 0.18 | 0.50 +/- 0.00 |
| missing_30pct | MiniRocket_clean | 0.6000 | 73.48 +/- 0.05 | 26.52 +/- 0.05 | 64.55 +/- 0.05 | 0.60 +/- 0.00 |
| missing_50pct | A_clean | 0.5000 | 64.86 +/- 1.17 | 35.14 +/- 1.17 | 54.43 +/- 0.20 | 0.50 +/- 0.00 |
| missing_50pct | A_clean | 0.6000 | 65.12 +/- 1.25 | 34.88 +/- 1.25 | 54.64 +/- 0.75 | 0.60 +/- 0.00 |
| missing_50pct | C_l0 | 0.5000 | 75.36 +/- 0.35 | 24.64 +/- 0.35 | 67.21 +/- 0.22 | 0.50 +/- 0.00 |
| missing_50pct | C_l0 | 0.6000 | 75.81 +/- 0.24 | 24.19 +/- 0.24 | 67.85 +/- 0.35 | 0.60 +/- 0.00 |
| missing_50pct | MiniRocket_aug | 0.5000 | 74.85 +/- 0.35 | 25.15 +/- 0.35 | 66.89 +/- 0.49 | 0.50 +/- 0.00 |
| missing_50pct | MiniRocket_aug | 0.6000 | 75.39 +/- 0.31 | 24.61 +/- 0.31 | 67.49 +/- 0.31 | 0.60 +/- 0.00 |
| missing_50pct | MiniRocket_clean | 0.5000 | 62.99 +/- 0.21 | 37.01 +/- 0.21 | 54.15 +/- 0.25 | 0.50 +/- 0.00 |
| missing_50pct | MiniRocket_clean | 0.6000 | 65.03 +/- 0.18 | 34.97 +/- 0.18 | 55.52 +/- 0.12 | 0.60 +/- 0.00 |
| noise_10db | A_clean | 0.5000 | 60.85 +/- 2.54 | 39.15 +/- 2.54 | 48.13 +/- 2.12 | 0.50 +/- 0.00 |
| noise_10db | A_clean | 0.6000 | 60.52 +/- 2.12 | 39.48 +/- 2.12 | 47.75 +/- 1.62 | 0.60 +/- 0.00 |
| noise_10db | C_l0 | 0.5000 | 74.48 +/- 0.19 | 25.52 +/- 0.19 | 66.19 +/- 0.13 | 0.50 +/- 0.00 |
| noise_10db | C_l0 | 0.6000 | 75.28 +/- 0.21 | 24.72 +/- 0.21 | 67.21 +/- 0.16 | 0.60 +/- 0.00 |
| noise_10db | MiniRocket_aug | 0.5000 | 70.30 +/- 0.42 | 29.70 +/- 0.42 | 60.55 +/- 0.27 | 0.50 +/- 0.00 |
| noise_10db | MiniRocket_aug | 0.6000 | 70.64 +/- 0.42 | 29.36 +/- 0.42 | 61.29 +/- 0.45 | 0.60 +/- 0.00 |
| noise_10db | MiniRocket_clean | 0.5000 | 41.35 +/- 0.14 | 58.65 +/- 0.14 | 36.88 +/- 0.16 | 0.50 +/- 0.00 |
| noise_10db | MiniRocket_clean | 0.6000 | 48.13 +/- 0.04 | 51.87 +/- 0.04 | 34.75 +/- 0.07 | 0.60 +/- 0.00 |
| noise_5db | A_clean | 0.5000 | 54.94 +/- 1.91 | 45.06 +/- 1.91 | 42.48 +/- 1.44 | 0.50 +/- 0.00 |
| noise_5db | A_clean | 0.6000 | 54.56 +/- 1.17 | 45.44 +/- 1.17 | 42.18 +/- 0.83 | 0.60 +/- 0.00 |
| noise_5db | C_l0 | 0.5000 | 69.03 +/- 0.22 | 30.97 +/- 0.22 | 60.12 +/- 0.10 | 0.50 +/- 0.00 |
| noise_5db | C_l0 | 0.6000 | 69.82 +/- 0.17 | 30.18 +/- 0.17 | 61.25 +/- 0.16 | 0.60 +/- 0.00 |
| noise_5db | MiniRocket_aug | 0.5000 | 64.21 +/- 0.36 | 35.79 +/- 0.36 | 53.27 +/- 0.33 | 0.50 +/- 0.00 |
| noise_5db | MiniRocket_aug | 0.6000 | 64.34 +/- 0.38 | 35.66 +/- 0.38 | 53.63 +/- 0.94 | 0.60 +/- 0.00 |
| noise_5db | MiniRocket_clean | 0.5000 | 39.79 +/- 0.18 | 60.21 +/- 0.18 | 32.94 +/- 0.19 | 0.50 +/- 0.00 |
| noise_5db | MiniRocket_clean | 0.6000 | 44.37 +/- 0.05 | 55.63 +/- 0.05 | 30.59 +/- 0.02 | 0.60 +/- 0.00 |

Pilot paired Wilcoxon tests use each dataset's three-draw mean error difference; Holm across five conditions x two checkpoints (10 tests). With N=8 this is exploratory evidence, not the full 26-dataset benchmark.

| condition | nominal_fraction | comparison | N | margin | p_raw | p_holm |
| --- | --- | --- | --- | --- | --- | --- |
| clean | 0.5000 | C_l0 vs MiniRocket_aug | 8 | -0.0013 | 1.0000 | 1.0000 |
| clean | 0.6000 | C_l0 vs MiniRocket_aug | 8 | 0.0012 | 0.9453 | 1.0000 |
| missing_30pct | 0.5000 | C_l0 vs MiniRocket_aug | 8 | 0.0087 | 0.9453 | 1.0000 |
| missing_30pct | 0.6000 | C_l0 vs MiniRocket_aug | 8 | 0.0110 | 0.9453 | 1.0000 |
| missing_50pct | 0.5000 | C_l0 vs MiniRocket_aug | 8 | 0.0051 | 0.9453 | 1.0000 |
| missing_50pct | 0.6000 | C_l0 vs MiniRocket_aug | 8 | 0.0042 | 0.7422 | 1.0000 |
| noise_10db | 0.5000 | C_l0 vs MiniRocket_aug | 8 | 0.0419 | 0.0156 | 0.1406 |
| noise_10db | 0.6000 | C_l0 vs MiniRocket_aug | 8 | 0.0464 | 0.0078 | 0.0781 |
| noise_5db | 0.5000 | C_l0 vs MiniRocket_aug | 8 | 0.0482 | 0.0234 | 0.1875 |
| noise_5db | 0.6000 | C_l0 vs MiniRocket_aug | 8 | 0.0548 | 0.0234 | 0.1875 |

## Stopping at calibration targets 0.5 and 0.6

For each model/dataset/draw, choose among the SAME seven confidence thresholds (0.50,0.60,0.70,0.80,0.90,0.95,0.99) whose MIXED calibration mean earliness is closest to the target, using earliness only. Reuse that setting unchanged across all conditions. The cached mixed-calibrated MiniRocket baseline instead selects among its existing alpha sweep by the same criterion. The larger deep sweep is retained as calibration_target_dense in the CSV, not used for primary stopping comparisons. These are matched calibration targets, NOT guaranteed equal test earliness under distribution shift. No error-based or test-condition-specific interpolation/matching is presented as deployable evidence. The fixed-checkpoint comparison above is the decisive iso-earliness test.

| condition | target | deep_earliness | baseline_earliness | mean_absolute_dataset_earliness_gap | error_reduction_mean | error_reduction_sd | N_within_005_earliness |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.5000 | 0.4135 | 0.3858 | 0.0673 | -0.0124 | 0.0047 | 4 |
| clean | 0.6000 | 0.4858 | 0.4166 | 0.1052 | -0.0165 | 0.0044 | 2 |
| missing_30pct | 0.5000 | 0.4509 | 0.4346 | 0.0678 | -0.0016 | 0.0072 | 2 |
| missing_30pct | 0.6000 | 0.5307 | 0.4758 | 0.0935 | -0.0079 | 0.0086 | 2 |
| missing_50pct | 0.5000 | 0.4823 | 0.4695 | 0.0627 | -0.0028 | 0.0080 | 3 |
| missing_50pct | 0.6000 | 0.5676 | 0.5143 | 0.0854 | -0.0077 | 0.0072 | 1 |
| noise_10db | 0.5000 | 0.4969 | 0.5336 | 0.1309 | 0.0303 | 0.0021 | 1 |
| noise_10db | 0.6000 | 0.5786 | 0.5846 | 0.1187 | 0.0306 | 0.0022 | 3 |
| noise_5db | 0.5000 | 0.5380 | 0.5813 | 0.1344 | 0.0463 | 0.0012 | 2 |
| noise_5db | 0.6000 | 0.6263 | 0.6362 | 0.1057 | 0.0475 | 0.0024 | 3 |

| condition | method | target | accuracy % | error % | earliness |
| --- | --- | --- | --- | --- | --- |
| clean | C_l0 | 0.5000 | 83.16 +/- 0.24 | 16.84 +/- 0.24 | 0.41 +/- 0.01 |
| clean | C_l0 | 0.6000 | 83.71 +/- 0.44 | 16.29 +/- 0.44 | 0.49 +/- 0.01 |
| clean | MiniRocket_aug_mixed | 0.5000 | 86.08 +/- 0.20 | 13.92 +/- 0.20 | 0.44 +/- 0.02 |
| clean | MiniRocket_aug_mixed | 0.6000 | 86.41 +/- 0.31 | 13.59 +/- 0.31 | 0.48 +/- 0.00 |
| clean | MiniRocket_aug_probability | 0.5000 | 84.40 +/- 0.25 | 15.60 +/- 0.25 | 0.39 +/- 0.00 |
| clean | MiniRocket_aug_probability | 0.6000 | 85.36 +/- 0.01 | 14.64 +/- 0.01 | 0.42 +/- 0.00 |
| clean | MiniRocket_clean_probability | 0.5000 | 86.90 +/- 0.00 | 13.10 +/- 0.00 | 0.51 +/- 0.00 |
| clean | MiniRocket_clean_probability | 0.6000 | 86.96 +/- 0.00 | 13.04 +/- 0.00 | 0.54 +/- 0.00 |
| missing_30pct | C_l0 | 0.5000 | 80.32 +/- 0.36 | 19.68 +/- 0.36 | 0.45 +/- 0.00 |
| missing_30pct | C_l0 | 0.6000 | 81.14 +/- 0.60 | 18.86 +/- 0.60 | 0.53 +/- 0.02 |
| missing_30pct | MiniRocket_aug_mixed | 0.5000 | 82.52 +/- 0.20 | 17.48 +/- 0.20 | 0.49 +/- 0.02 |
| missing_30pct | MiniRocket_aug_mixed | 0.6000 | 83.06 +/- 0.24 | 16.94 +/- 0.24 | 0.54 +/- 0.00 |
| missing_30pct | MiniRocket_aug_probability | 0.5000 | 80.48 +/- 0.37 | 19.52 +/- 0.37 | 0.43 +/- 0.00 |
| missing_30pct | MiniRocket_aug_probability | 0.6000 | 81.93 +/- 0.31 | 18.07 +/- 0.31 | 0.48 +/- 0.01 |
| missing_30pct | MiniRocket_clean_probability | 0.5000 | 76.52 +/- 0.04 | 23.48 +/- 0.04 | 0.54 +/- 0.00 |
| missing_30pct | MiniRocket_clean_probability | 0.6000 | 77.04 +/- 0.03 | 22.96 +/- 0.03 | 0.57 +/- 0.00 |
| missing_50pct | C_l0 | 0.5000 | 76.80 +/- 0.41 | 23.20 +/- 0.41 | 0.48 +/- 0.01 |
| missing_50pct | C_l0 | 0.6000 | 77.78 +/- 0.43 | 22.22 +/- 0.43 | 0.57 +/- 0.02 |
| missing_50pct | MiniRocket_aug_mixed | 0.5000 | 78.75 +/- 0.19 | 21.25 +/- 0.19 | 0.52 +/- 0.02 |
| missing_50pct | MiniRocket_aug_mixed | 0.6000 | 79.36 +/- 0.33 | 20.64 +/- 0.33 | 0.57 +/- 0.01 |
| missing_50pct | MiniRocket_aug_probability | 0.5000 | 77.08 +/- 0.39 | 22.92 +/- 0.39 | 0.47 +/- 0.01 |
| missing_50pct | MiniRocket_aug_probability | 0.6000 | 78.55 +/- 0.30 | 21.45 +/- 0.30 | 0.51 +/- 0.01 |
| missing_50pct | MiniRocket_clean_probability | 0.5000 | 67.57 +/- 0.04 | 32.43 +/- 0.04 | 0.52 +/- 0.00 |
| missing_50pct | MiniRocket_clean_probability | 0.6000 | 68.38 +/- 0.03 | 31.62 +/- 0.03 | 0.55 +/- 0.00 |
| noise_10db | C_l0 | 0.5000 | 75.78 +/- 0.33 | 24.22 +/- 0.33 | 0.50 +/- 0.01 |
| noise_10db | C_l0 | 0.6000 | 76.99 +/- 0.32 | 23.01 +/- 0.32 | 0.58 +/- 0.01 |
| noise_10db | MiniRocket_aug_mixed | 0.5000 | 73.82 +/- 0.29 | 26.18 +/- 0.29 | 0.58 +/- 0.02 |
| noise_10db | MiniRocket_aug_mixed | 0.6000 | 74.01 +/- 0.27 | 25.99 +/- 0.27 | 0.63 +/- 0.01 |
| noise_10db | MiniRocket_aug_probability | 0.5000 | 72.75 +/- 0.16 | 27.25 +/- 0.16 | 0.53 +/- 0.02 |
| noise_10db | MiniRocket_aug_probability | 0.6000 | 73.94 +/- 0.10 | 26.06 +/- 0.10 | 0.58 +/- 0.01 |
| noise_10db | MiniRocket_clean_probability | 0.5000 | 47.64 +/- 0.05 | 52.36 +/- 0.05 | 0.33 +/- 0.00 |
| noise_10db | MiniRocket_clean_probability | 0.6000 | 48.16 +/- 0.09 | 51.84 +/- 0.09 | 0.34 +/- 0.00 |
| noise_5db | C_l0 | 0.5000 | 70.98 +/- 0.13 | 29.02 +/- 0.13 | 0.54 +/- 0.00 |
| noise_5db | C_l0 | 0.6000 | 72.18 +/- 0.23 | 27.82 +/- 0.23 | 0.63 +/- 0.01 |
| noise_5db | MiniRocket_aug_mixed | 0.5000 | 66.79 +/- 0.40 | 33.21 +/- 0.40 | 0.63 +/- 0.03 |
| noise_5db | MiniRocket_aug_mixed | 0.6000 | 67.10 +/- 0.27 | 32.90 +/- 0.27 | 0.68 +/- 0.01 |
| noise_5db | MiniRocket_aug_probability | 0.5000 | 66.35 +/- 0.14 | 33.65 +/- 0.14 | 0.58 +/- 0.02 |
| noise_5db | MiniRocket_aug_probability | 0.6000 | 67.42 +/- 0.01 | 32.58 +/- 0.01 | 0.64 +/- 0.00 |
| noise_5db | MiniRocket_clean_probability | 0.5000 | 45.52 +/- 0.04 | 54.48 +/- 0.04 | 0.33 +/- 0.00 |
| noise_5db | MiniRocket_clean_probability | 0.6000 | 46.07 +/- 0.02 | 53.93 +/- 0.02 | 0.36 +/- 0.00 |

## Q2: inheritance and consistency

Paired accuracy gains (percentage points), averaged over ALL nine matched checkpoints, eight datasets, then three draws. Positive values favor the first named variant. This isolates head initialization at the same curriculum; best-epoch selection can still produce different realized training time. Shared heads also change parameter count, so sharing vs independence is an architectural ablation, not solely initialization.

| comparison | condition | accuracy gain pp +/- SD |
| --- | --- | --- |
| shared minus independent, lambda=0 | clean | -1.82 +/- 1.05 |
| shared minus independent, lambda=0 | missing_30pct | -2.13 +/- 0.91 |
| shared minus independent, lambda=0 | missing_50pct | -2.60 +/- 0.86 |
| shared minus independent, lambda=0 | noise_10db | -2.10 +/- 0.61 |
| shared minus independent, lambda=0 | noise_5db | -2.65 +/- 1.07 |
| progressive minus independent, lambda=0 | clean | +0.15 +/- 0.05 |
| progressive minus independent, lambda=0 | missing_30pct | +0.03 +/- 0.26 |
| progressive minus independent, lambda=0 | missing_50pct | +0.08 +/- 0.48 |
| progressive minus independent, lambda=0 | noise_10db | -0.20 +/- 0.40 |
| progressive minus independent, lambda=0 | noise_5db | -0.09 +/- 0.24 |
| shared minus independent, lambda=0.1 | clean | -1.60 +/- 0.46 |
| shared minus independent, lambda=0.1 | missing_30pct | -1.72 +/- 0.47 |
| shared minus independent, lambda=0.1 | missing_50pct | -2.17 +/- 0.51 |
| shared minus independent, lambda=0.1 | noise_10db | -1.83 +/- 0.38 |
| shared minus independent, lambda=0.1 | noise_5db | -2.46 +/- 0.47 |
| progressive minus independent, lambda=0.1 | clean | -0.11 +/- 0.33 |
| progressive minus independent, lambda=0.1 | missing_30pct | -0.15 +/- 0.36 |
| progressive minus independent, lambda=0.1 | missing_50pct | -0.16 +/- 0.42 |
| progressive minus independent, lambda=0.1 | noise_10db | -0.05 +/- 0.61 |
| progressive minus independent, lambda=0.1 | noise_5db | -0.16 +/- 0.80 |
| consistency effect, shared | clean | +0.27 +/- 1.10 |
| consistency effect, shared | missing_30pct | +0.33 +/- 1.26 |
| consistency effect, shared | missing_50pct | +0.34 +/- 1.33 |
| consistency effect, shared | noise_10db | +0.12 +/- 0.84 |
| consistency effect, shared | noise_5db | +0.21 +/- 1.01 |
| consistency effect, progressive | clean | -0.20 +/- 0.43 |
| consistency effect, progressive | missing_30pct | -0.27 +/- 0.56 |
| consistency effect, progressive | missing_50pct | -0.32 +/- 0.89 |
| consistency effect, progressive | noise_10db | +0.01 +/- 0.41 |
| consistency effect, progressive | noise_5db | -0.05 +/- 0.45 |
| consistency effect, independent | clean | +0.05 +/- 0.27 |
| consistency effect, independent | missing_30pct | -0.08 +/- 0.11 |
| consistency effect, independent | missing_50pct | -0.08 +/- 0.15 |
| consistency effect, independent | noise_10db | -0.14 +/- 0.07 |
| consistency effect, independent | noise_5db | +0.02 +/- 0.27 |
| augmentation effect, shared lambda=0 | clean | +3.29 +/- 0.80 |
| augmentation effect, shared lambda=0 | missing_30pct | +6.20 +/- 0.84 |
| augmentation effect, shared lambda=0 | missing_50pct | +7.63 +/- 1.07 |
| augmentation effect, shared lambda=0 | noise_10db | +11.72 +/- 2.34 |
| augmentation effect, shared lambda=0 | noise_5db | +11.84 +/- 2.13 |

## Q3: clean-data cost

- C_l0 minus MiniRocket_aug clean error, averaged over checkpoints: 1.37 +/- 0.32 percentage points (positive = clean cost).
- C_l0 minus MiniRocket_clean clean error, averaged over checkpoints: 2.87 +/- 0.30 percentage points (positive = clean cost).

## Verdict

- clean: mean fixed-prefix error reduction -1.37 pp; wins at both 0.5/0.6 = False; nonnegative at every prefix = False.
- noise_10db: mean fixed-prefix error reduction +3.42 pp; wins at both 0.5/0.6 = True; nonnegative at every prefix = True.
- noise_5db: mean fixed-prefix error reduction +4.34 pp; wins at both 0.5/0.6 = True; nonnegative at every prefix = True.
- missing_30pct: mean fixed-prefix error reduction -0.29 pp; wins at both 0.5/0.6 = True; nonnegative at every prefix = False.
- missing_50pct: mean fixed-prefix error reduction -0.66 pp; wins at both 0.5/0.6 = True; nonnegative at every prefix = False.
- progressive minus independent, lambda=0: average corruption accuracy change -0.04 pp.
- progressive minus independent, lambda=0.1: average corruption accuracy change -0.13 pp.
- shared minus independent, lambda=0: average corruption accuracy change -2.37 pp.
- shared minus independent, lambda=0.1: average corruption accuracy change -2.05 pp.
- Promotion decision: do not promote as a general replacement: the equal-earliness advantage does not hold across all four corruptions in this pilot.

## Runtime, audits, and limitations

All three GPUs used: ['0', '1', '2']; hardware: Quadro RTX 8000. 168 trained networks; summed training + validation GPU occupancy = 1.945 GPU-hours. Inference and transfer time are additional. This is wall time while the training job occupies one GPU, not hardware-instruction utilization.

| variant | networks | mean_epochs | mean_best_epoch | train_gpu_hours |
| --- | --- | --- | --- | --- |
| A_clean | 24 | 30.0000 | 25.2917 | 0.1265 |
| A_l0 | 24 | 32.7917 | 27.4167 | 0.2927 |
| A_l01 | 24 | 33.6667 | 28.7500 | 0.3082 |
| B_l0 | 24 | 32.6667 | 27.4583 | 0.2970 |
| B_l01 | 24 | 32.8750 | 28.0000 | 0.3085 |
| C_l0 | 24 | 33.5417 | 28.8333 | 0.3087 |
| C_l01 | 24 | 32.9167 | 27.7917 | 0.3032 |

Causality audited on each of the 168 trained networks: suffix perturbation and truncated-prefix equivalence, plus zero gradient with respect to future input. No future observations enter normalization or prediction. Consistency couples training losses across prefixes, which does not grant an inference-time future input.

Pilot limitations: one prespecified TCN size/hyperparameter setting; curriculum and head choices can favor different data domains; fixed fit caps; three draws are insufficient for precise variance estimation; eight datasets are not a publishable-scale superiority claim. Input train/validation/calibration partitions are fixed as in prior runs; draws vary augmentation/corruption and paired network initialization.

All numeric tables, full curves, per-dataset metrics, checkpoints, predictions, training histories, and code are preserved. See source_audit.json for prior-input integrity and artifact_audit.json for completeness. Errors, if any, are retained verbatim in logs/.

Overall parallel wall time: 0.758 hours; worker failures: 0.

## Per-dataset review and final decision

Error reduction versus augmented MiniRocket, in percentage points, averaged across nine equal-prefix checkpoints and three draws. Positive = the learned validation winner is better. These are descriptive comparisons, not test-selected model choices.

| dataset | clean | noise_10db | noise_5db | missing_30pct | missing_50pct |
| --- | --- | --- | --- | --- | --- |
| Crop | 5.1537 | 7.6201 | 4.5924 | 8.6369 | 7.6631 |
| ECG5000 | -0.9070 | -0.3078 | -1.1613 | -0.9753 | -1.6889 |
| ElectricDevices | 3.3353 | 8.6428 | 7.7090 | 6.1254 | 5.4938 |
| FordA | 1.8070 | 2.4411 | 3.3109 | 2.4074 | 2.8143 |
| HandOutlines | -5.6557 | 0.2903 | 3.6236 | -4.9449 | -4.0941 |
| NonInvasiveFetalECGThorax1 | -14.5491 | 5.5339 | 12.8188 | -13.7480 | -15.8496 |
| StarLightCurves | -0.2999 | 2.1801 | 1.5429 | -0.1956 | -0.0279 |
| Wafer | 0.1845 | 0.9662 | 2.3163 | 0.4098 | 0.4032 |

The average noise gain is broad: seven of eight datasets improve when averaged over checkpoints under both noise levels. Missing-data effects are split: Crop, ElectricDevices, FordA and Wafer improve, while ECG5000, HandOutlines, NonInvasiveFetalECGThorax1 and StarLightCurves worsen. The fetal ECG dataset has a particularly large clean/missing-data cost; it also has the largest 5 dB gain. This is a condition-dependent trade-off, not general dominance.

**Inheritance:** progressive heads are approximately tied with independent heads and do not show a repeatable advantage. Full head sharing is worse. **Consistency:** lambda=0.1 gives small, mixed changes; the validation-selected model uses zero. The pilot therefore supports further investigation of a noise-robust learned backbone, but not an inheritance/consistency headline or wholesale replacement of augmented MiniRocket.

**Stopping interpretation:** common-grid probability stopping on the learned model also reduces noise error, but calibration-target matching leaves substantial per-dataset test-earliness gaps (only 1-4 of 8 are within 0.05 depending on condition/target). Those stopping comparisons are not promoted as exact equal-earliness evidence. The fixed-checkpoint comparisons are exact within each dataset and answer the backbone question directly.

## Resolved execution issues

Training worker failures: 0. No packages were installed or upgraded. The first report formatting pass encountered:

```text
ImportError: Missing optional dependency 'tabulate'.  Use pip or conda to install tabulate.
```

The report now uses a built-in Markdown formatter. All eight numeric CSV files produced before that formatting fix were SHA256-identical afterward (logs/numeric_hashes_after_report_fix.json). The original failure traceback is preserved.

One monitoring SSH connection also returned TimeoutError: timed out; reconnect succeeded and GPU jobs were unaffected.

Source integrity audit: all 5,419 protected files unchanged, with no additions or removals. All 168 model causality audits, 384 trained-head inheritance copies, and 24 augmentation-recipe comparisons passed.
