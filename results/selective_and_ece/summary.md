# ECTS selective decision pilot

## Scope and interpretation

All 26 main UCR datasets, five paired draws and seven conditions were replayed from the saved clean and augmented checkpoint models with causal running normalization. No model was retrained. Cached mixed-calibration indices and deterministic corruption streams were reused; all checkpoint accuracies were checked against the scale run. An initial fixed-point run stopped before writing results because two nearby ECEC thresholds were treated as equal; the match tolerance was corrected and the analysis rerun. The two small-calibration contrast datasets remain outside this main cohort.

A condition-agnostic method never receives test-condition identity or test labels. Two explicitly descriptive diagnostics below are not deployable policies: choosing a stopping setting separately using the observed earliness of each test condition, and selecting the highest-confidence fraction of a test batch to exactly match observed coverage. These diagnostics use no test labels for their selection, but they do use test-batch information and are not causal online procedures.

## Fixed operating points: all 26 datasets

For each dataset and method, the mean earliness for each grid setting over the five draws is compared with T in {0.5, 0.6}; the closest is selected, with lower setting breaking ties. The condition-posthoc version uses test stopping times for that condition and is descriptive only. The mixed-calibration-fixed version selects once using the five mixed-calibration draws and reuses the same setting under every condition. The latter satisfies the deployment constraint. Neither selection uses test error. Nevertheless, selecting on test stopping times is post-selection on the test sample and can affect error-dependent inference indirectly; post-hoc p-values are exploratory, whereas the mixed-calibration-fixed selection avoids that specific test-sample choice.

The selected settings may still fall far from T or from each other. Thus N=26 fixes the missing-overlap sample problem, but does not establish equal-earliness comparisons when the realized times differ. The table reports the actual mean earliness for both methods, the mean absolute gap across datasets, and the number of datasets with a gap greater than 0.05. Wilcoxon tests use one five-draw mean error per dataset; Holm correction covers 70 condition x target x baseline tests separately within each selection analysis. W/T/L is baseline error minus ours, with tolerance 1e-12.

| Selection | Condition | T | Comparator | Ours earliness | Comparator earliness | Mean absolute gap | Datasets gap > .05 | Error margin +/- draw SD | W/T/L | p raw | p Holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| condition_posthoc | clean | 0.5 | Probability | 0.582 | 0.485 | 0.128 | 17 | 0.0078 +/- 0.0035 | 13/0/13 | 0.7078 | 1 |
| condition_posthoc | clean | 0.5 | TEASER | 0.582 | 0.480 | 0.158 | 18 | 0.0288 +/- 0.0035 | 13/0/13 | 0.4374 | 1 |
| condition_posthoc | clean | 0.5 | ECEC adaptation | 0.582 | 0.480 | 0.129 | 18 | 0.0310 +/- 0.0035 | 17/0/9 | 0.02915 | 0.8454 |
| condition_posthoc | clean | 0.5 | Aug + probability | 0.582 | 0.472 | 0.116 | 14 | 0.0197 +/- 0.0030 | 21/0/5 | 0.0002175 | 0.008264 |
| condition_posthoc | clean | 0.5 | Aug + TEASER | 0.582 | 0.493 | 0.145 | 17 | 0.0395 +/- 0.0056 | 19/0/7 | 0.001298 | 0.04543 |
| condition_posthoc | clean | 0.6 | Probability | 0.619 | 0.569 | 0.111 | 13 | -0.0042 +/- 0.0018 | 9/0/17 | 0.227 | 1 |
| condition_posthoc | clean | 0.6 | TEASER | 0.619 | 0.612 | 0.120 | 17 | 0.0090 +/- 0.0018 | 12/0/14 | 0.9403 | 1 |
| condition_posthoc | clean | 0.6 | ECEC adaptation | 0.619 | 0.557 | 0.115 | 16 | 0.0222 +/- 0.0018 | 17/0/9 | 0.06303 | 1 |
| condition_posthoc | clean | 0.6 | Aug + probability | 0.619 | 0.565 | 0.120 | 15 | 0.0106 +/- 0.0019 | 17/1/8 | 0.09933 | 1 |
| condition_posthoc | clean | 0.6 | Aug + TEASER | 0.619 | 0.600 | 0.129 | 19 | 0.0262 +/- 0.0050 | 15/0/11 | 0.07515 | 1 |
| condition_posthoc | noise_20db | 0.5 | Probability | 0.596 | 0.458 | 0.167 | 18 | 0.1829 +/- 0.0018 | 25/0/1 | 4.172e-07 | 2.337e-05 |
| condition_posthoc | noise_20db | 0.5 | TEASER | 0.596 | 0.509 | 0.136 | 18 | 0.1862 +/- 0.0030 | 25/0/1 | 5.96e-08 | 4.053e-06 |
| condition_posthoc | noise_20db | 0.5 | ECEC adaptation | 0.596 | 0.472 | 0.140 | 18 | 0.1933 +/- 0.0031 | 26/0/0 | 2.98e-08 | 2.086e-06 |
| condition_posthoc | noise_20db | 0.5 | Aug + probability | 0.596 | 0.490 | 0.130 | 15 | 0.0200 +/- 0.0038 | 17/0/9 | 0.05585 | 1 |
| condition_posthoc | noise_20db | 0.5 | Aug + TEASER | 0.596 | 0.497 | 0.134 | 15 | 0.0353 +/- 0.0060 | 19/0/7 | 0.0002823 | 0.01044 |
| condition_posthoc | noise_20db | 0.6 | Probability | 0.629 | 0.518 | 0.160 | 17 | 0.1813 +/- 0.0015 | 22/0/4 | 5.037e-06 | 0.0002317 |
| condition_posthoc | noise_20db | 0.6 | TEASER | 0.629 | 0.610 | 0.110 | 17 | 0.1818 +/- 0.0025 | 23/0/3 | 4.083e-06 | 0.0001919 |
| condition_posthoc | noise_20db | 0.6 | ECEC adaptation | 0.629 | 0.557 | 0.119 | 15 | 0.1860 +/- 0.0013 | 25/0/1 | 8.941e-08 | 5.811e-06 |
| condition_posthoc | noise_20db | 0.6 | Aug + probability | 0.629 | 0.562 | 0.109 | 16 | 0.0118 +/- 0.0035 | 16/0/10 | 0.2173 | 1 |
| condition_posthoc | noise_20db | 0.6 | Aug + TEASER | 0.629 | 0.619 | 0.119 | 18 | 0.0115 +/- 0.0048 | 14/0/12 | 0.4678 | 1 |
| condition_posthoc | noise_10db | 0.5 | Probability | 0.595 | 0.399 | 0.220 | 23 | 0.2352 +/- 0.0043 | 23/0/3 | 1.281e-06 | 6.536e-05 |
| condition_posthoc | noise_10db | 0.5 | TEASER | 0.595 | 0.510 | 0.146 | 21 | 0.2336 +/- 0.0039 | 23/0/3 | 7.451e-07 | 4.023e-05 |
| condition_posthoc | noise_10db | 0.5 | ECEC adaptation | 0.595 | 0.471 | 0.155 | 17 | 0.2499 +/- 0.0039 | 25/0/1 | 7.451e-07 | 4.023e-05 |
| condition_posthoc | noise_10db | 0.5 | Aug + probability | 0.595 | 0.496 | 0.156 | 21 | 0.0190 +/- 0.0028 | 15/0/11 | 0.2687 | 1 |
| condition_posthoc | noise_10db | 0.5 | Aug + TEASER | 0.595 | 0.515 | 0.129 | 17 | 0.0175 +/- 0.0048 | 14/0/12 | 0.237 | 1 |
| condition_posthoc | noise_10db | 0.6 | Probability | 0.644 | 0.451 | 0.234 | 23 | 0.2349 +/- 0.0033 | 23/0/3 | 1.281e-06 | 6.536e-05 |
| condition_posthoc | noise_10db | 0.6 | TEASER | 0.644 | 0.591 | 0.129 | 21 | 0.2316 +/- 0.0029 | 23/0/3 | 7.451e-07 | 4.023e-05 |
| condition_posthoc | noise_10db | 0.6 | ECEC adaptation | 0.644 | 0.548 | 0.131 | 16 | 0.2397 +/- 0.0038 | 25/0/1 | 8.941e-08 | 5.811e-06 |
| condition_posthoc | noise_10db | 0.6 | Aug + probability | 0.644 | 0.586 | 0.115 | 16 | 0.0076 +/- 0.0024 | 14/0/12 | 0.7265 | 1 |
| condition_posthoc | noise_10db | 0.6 | Aug + TEASER | 0.644 | 0.621 | 0.128 | 21 | 0.0091 +/- 0.0050 | 11/0/15 | 0.8222 | 1 |
| condition_posthoc | noise_5db | 0.5 | Probability | 0.615 | 0.384 | 0.255 | 25 | 0.2207 +/- 0.0036 | 25/0/1 | 2.98e-07 | 1.729e-05 |
| condition_posthoc | noise_5db | 0.5 | TEASER | 0.615 | 0.510 | 0.156 | 21 | 0.2165 +/- 0.0027 | 25/0/1 | 5.96e-08 | 4.053e-06 |
| condition_posthoc | noise_5db | 0.5 | ECEC adaptation | 0.615 | 0.461 | 0.179 | 19 | 0.2251 +/- 0.0015 | 25/0/1 | 1.49e-07 | 9.239e-06 |
| condition_posthoc | noise_5db | 0.5 | Aug + probability | 0.615 | 0.528 | 0.152 | 19 | 0.0209 +/- 0.0034 | 17/1/8 | 0.05784 | 1 |
| condition_posthoc | noise_5db | 0.5 | Aug + TEASER | 0.615 | 0.544 | 0.136 | 16 | 0.0138 +/- 0.0050 | 13/0/13 | 0.6171 | 1 |
| condition_posthoc | noise_5db | 0.6 | Probability | 0.670 | 0.430 | 0.272 | 23 | 0.2179 +/- 0.0034 | 25/0/1 | 2.98e-07 | 1.729e-05 |
| condition_posthoc | noise_5db | 0.6 | TEASER | 0.670 | 0.599 | 0.143 | 19 | 0.2224 +/- 0.0037 | 26/0/0 | 2.98e-08 | 2.086e-06 |
| condition_posthoc | noise_5db | 0.6 | ECEC adaptation | 0.670 | 0.554 | 0.150 | 20 | 0.2253 +/- 0.0020 | 25/0/1 | 8.941e-08 | 5.811e-06 |
| condition_posthoc | noise_5db | 0.6 | Aug + probability | 0.670 | 0.604 | 0.121 | 17 | 0.0034 +/- 0.0033 | 14/0/12 | 0.9205 | 1 |
| condition_posthoc | noise_5db | 0.6 | Aug + TEASER | 0.670 | 0.626 | 0.124 | 19 | 0.0185 +/- 0.0046 | 13/0/13 | 0.4992 | 1 |
| condition_posthoc | missing_10pct | 0.5 | Probability | 0.583 | 0.470 | 0.131 | 18 | 0.0281 +/- 0.0037 | 20/0/6 | 0.0007454 | 0.02684 |
| condition_posthoc | missing_10pct | 0.5 | TEASER | 0.583 | 0.474 | 0.152 | 19 | 0.0397 +/- 0.0051 | 17/0/9 | 0.01761 | 0.546 |
| condition_posthoc | missing_10pct | 0.5 | ECEC adaptation | 0.583 | 0.474 | 0.126 | 17 | 0.0489 +/- 0.0032 | 23/0/3 | 9.149e-06 | 0.0003934 |
| condition_posthoc | missing_10pct | 0.5 | Aug + probability | 0.583 | 0.480 | 0.115 | 17 | 0.0186 +/- 0.0038 | 18/0/8 | 0.006131 | 0.2023 |
| condition_posthoc | missing_10pct | 0.5 | Aug + TEASER | 0.583 | 0.484 | 0.139 | 17 | 0.0349 +/- 0.0033 | 15/0/11 | 0.01761 | 0.546 |
| condition_posthoc | missing_10pct | 0.6 | Probability | 0.621 | 0.569 | 0.096 | 11 | 0.0090 +/- 0.0029 | 18/0/8 | 0.1427 | 1 |
| condition_posthoc | missing_10pct | 0.6 | TEASER | 0.621 | 0.622 | 0.116 | 19 | 0.0151 +/- 0.0019 | 15/0/11 | 0.3403 | 1 |
| condition_posthoc | missing_10pct | 0.6 | ECEC adaptation | 0.621 | 0.550 | 0.110 | 14 | 0.0360 +/- 0.0016 | 21/0/5 | 6.032e-05 | 0.002473 |
| condition_posthoc | missing_10pct | 0.6 | Aug + probability | 0.621 | 0.585 | 0.106 | 14 | 0.0044 +/- 0.0021 | 12/0/14 | 0.9007 | 1 |
| condition_posthoc | missing_10pct | 0.6 | Aug + TEASER | 0.621 | 0.607 | 0.114 | 17 | 0.0185 +/- 0.0024 | 12/1/13 | 0.237 | 1 |
| condition_posthoc | missing_30pct | 0.5 | Probability | 0.581 | 0.469 | 0.137 | 17 | 0.0687 +/- 0.0029 | 24/0/2 | 5.662e-07 | 3.114e-05 |
| condition_posthoc | missing_30pct | 0.5 | TEASER | 0.581 | 0.507 | 0.132 | 18 | 0.0744 +/- 0.0041 | 23/0/3 | 1.106e-05 | 0.0004644 |
| condition_posthoc | missing_30pct | 0.5 | ECEC adaptation | 0.581 | 0.465 | 0.136 | 17 | 0.0990 +/- 0.0042 | 25/0/1 | 1.49e-07 | 9.239e-06 |
| condition_posthoc | missing_30pct | 0.5 | Aug + probability | 0.581 | 0.481 | 0.117 | 15 | 0.0207 +/- 0.0026 | 19/0/7 | 0.006689 | 0.2141 |
| condition_posthoc | missing_30pct | 0.5 | Aug + TEASER | 0.581 | 0.498 | 0.134 | 17 | 0.0329 +/- 0.0050 | 18/1/7 | 0.002661 | 0.09046 |
| condition_posthoc | missing_30pct | 0.6 | Probability | 0.621 | 0.562 | 0.100 | 13 | 0.0516 +/- 0.0044 | 20/0/6 | 0.0001092 | 0.004368 |
| condition_posthoc | missing_30pct | 0.6 | TEASER | 0.621 | 0.619 | 0.124 | 18 | 0.0548 +/- 0.0027 | 21/0/5 | 0.0001092 | 0.004368 |
| condition_posthoc | missing_30pct | 0.6 | ECEC adaptation | 0.621 | 0.550 | 0.111 | 16 | 0.0816 +/- 0.0032 | 25/0/1 | 2.086e-07 | 1.231e-05 |
| condition_posthoc | missing_30pct | 0.6 | Aug + probability | 0.621 | 0.565 | 0.106 | 14 | 0.0077 +/- 0.0028 | 16/0/10 | 0.7644 | 1 |
| condition_posthoc | missing_30pct | 0.6 | Aug + TEASER | 0.621 | 0.613 | 0.111 | 19 | 0.0150 +/- 0.0037 | 13/0/13 | 0.3533 | 1 |
| condition_posthoc | missing_50pct | 0.5 | Probability | 0.570 | 0.484 | 0.144 | 17 | 0.0969 +/- 0.0061 | 22/0/4 | 6.169e-06 | 0.0002714 |
| condition_posthoc | missing_50pct | 0.5 | TEASER | 0.570 | 0.511 | 0.143 | 19 | 0.0999 +/- 0.0044 | 21/0/5 | 5.037e-06 | 0.0002317 |
| condition_posthoc | missing_50pct | 0.5 | ECEC adaptation | 0.570 | 0.470 | 0.124 | 14 | 0.1319 +/- 0.0048 | 25/0/1 | 5.96e-08 | 4.053e-06 |
| condition_posthoc | missing_50pct | 0.5 | Aug + probability | 0.570 | 0.484 | 0.117 | 17 | 0.0184 +/- 0.0063 | 18/0/8 | 0.02915 | 0.8454 |
| condition_posthoc | missing_50pct | 0.5 | Aug + TEASER | 0.570 | 0.501 | 0.128 | 17 | 0.0263 +/- 0.0067 | 16/0/10 | 0.03817 | 1 |
| condition_posthoc | missing_50pct | 0.6 | Probability | 0.629 | 0.526 | 0.133 | 17 | 0.0977 +/- 0.0049 | 23/0/3 | 1.639e-06 | 8.032e-05 |
| condition_posthoc | missing_50pct | 0.6 | TEASER | 0.629 | 0.593 | 0.097 | 15 | 0.1048 +/- 0.0043 | 23/0/3 | 2.086e-06 | 0.0001001 |
| condition_posthoc | missing_50pct | 0.6 | ECEC adaptation | 0.629 | 0.557 | 0.110 | 17 | 0.1232 +/- 0.0039 | 24/0/2 | 1.49e-07 | 9.239e-06 |
| condition_posthoc | missing_50pct | 0.6 | Aug + probability | 0.629 | 0.581 | 0.106 | 16 | 0.0088 +/- 0.0072 | 14/0/12 | 0.5317 | 1 |
| condition_posthoc | missing_50pct | 0.6 | Aug + TEASER | 0.629 | 0.614 | 0.100 | 17 | 0.0176 +/- 0.0059 | 15/0/11 | 0.2687 | 1 |
| mixed_calibration_fixed | clean | 0.5 | Probability | 0.558 | 0.594 | 0.124 | 20 | -0.0193 +/- 0.0032 | 7/0/19 | 0.01633 | 0.4573 |
| mixed_calibration_fixed | clean | 0.5 | TEASER | 0.558 | 0.481 | 0.157 | 21 | 0.0183 +/- 0.0032 | 10/0/16 | 0.9403 | 1 |
| mixed_calibration_fixed | clean | 0.5 | ECEC adaptation | 0.558 | 0.472 | 0.150 | 17 | 0.0235 +/- 0.0032 | 15/0/11 | 0.237 | 1 |
| mixed_calibration_fixed | clean | 0.5 | Aug + probability | 0.558 | 0.464 | 0.110 | 14 | 0.0300 +/- 0.0037 | 20/1/5 | 0.0008354 | 0.03509 |
| mixed_calibration_fixed | clean | 0.5 | Aug + TEASER | 0.558 | 0.454 | 0.161 | 21 | 0.0397 +/- 0.0062 | 16/0/10 | 0.01295 | 0.4015 |
| mixed_calibration_fixed | clean | 0.6 | Probability | 0.586 | 0.644 | 0.137 | 19 | -0.0184 +/- 0.0021 | 7/0/19 | 0.006131 | 0.233 |
| mixed_calibration_fixed | clean | 0.6 | TEASER | 0.586 | 0.586 | 0.132 | 20 | 0.0059 +/- 0.0021 | 10/0/16 | 0.3802 | 1 |
| mixed_calibration_fixed | clean | 0.6 | ECEC adaptation | 0.586 | 0.553 | 0.126 | 17 | 0.0175 +/- 0.0021 | 14/0/12 | 0.3666 | 1 |
| mixed_calibration_fixed | clean | 0.6 | Aug + probability | 0.586 | 0.515 | 0.097 | 15 | 0.0202 +/- 0.0026 | 19/1/6 | 0.00178 | 0.07118 |
| mixed_calibration_fixed | clean | 0.6 | Aug + TEASER | 0.586 | 0.550 | 0.136 | 19 | 0.0327 +/- 0.0052 | 16/0/10 | 0.03122 | 0.7181 |
| mixed_calibration_fixed | noise_20db | 0.5 | Probability | 0.598 | 0.464 | 0.177 | 22 | 0.1847 +/- 0.0028 | 24/0/2 | 4.172e-07 | 2.253e-05 |
| mixed_calibration_fixed | noise_20db | 0.5 | TEASER | 0.598 | 0.510 | 0.147 | 20 | 0.1920 +/- 0.0036 | 24/0/2 | 1.49e-07 | 9.09e-06 |
| mixed_calibration_fixed | noise_20db | 0.5 | ECEC adaptation | 0.598 | 0.481 | 0.152 | 18 | 0.1924 +/- 0.0034 | 25/0/1 | 8.941e-08 | 5.633e-06 |
| mixed_calibration_fixed | noise_20db | 0.5 | Aug + probability | 0.598 | 0.504 | 0.122 | 15 | 0.0244 +/- 0.0047 | 17/0/9 | 0.07092 | 1 |
| mixed_calibration_fixed | noise_20db | 0.5 | Aug + TEASER | 0.598 | 0.490 | 0.143 | 18 | 0.0329 +/- 0.0053 | 19/0/7 | 0.0005907 | 0.0254 |
| mixed_calibration_fixed | noise_20db | 0.6 | Probability | 0.635 | 0.527 | 0.154 | 19 | 0.1836 +/- 0.0019 | 23/0/3 | 3.278e-06 | 0.0001639 |
| mixed_calibration_fixed | noise_20db | 0.6 | TEASER | 0.635 | 0.638 | 0.123 | 17 | 0.1721 +/- 0.0029 | 24/0/2 | 4.083e-06 | 0.0002001 |
| mixed_calibration_fixed | noise_20db | 0.6 | ECEC adaptation | 0.635 | 0.556 | 0.126 | 16 | 0.1912 +/- 0.0019 | 25/0/1 | 1.49e-07 | 9.09e-06 |
| mixed_calibration_fixed | noise_20db | 0.6 | Aug + probability | 0.635 | 0.567 | 0.107 | 15 | 0.0158 +/- 0.0044 | 16/0/10 | 0.09407 | 1 |
| mixed_calibration_fixed | noise_20db | 0.6 | Aug + TEASER | 0.635 | 0.598 | 0.123 | 18 | 0.0254 +/- 0.0060 | 16/0/10 | 0.06688 | 1 |
| mixed_calibration_fixed | noise_10db | 0.5 | Probability | 0.653 | 0.375 | 0.291 | 24 | 0.2469 +/- 0.0037 | 24/0/2 | 1.49e-07 | 9.09e-06 |
| mixed_calibration_fixed | noise_10db | 0.5 | TEASER | 0.653 | 0.482 | 0.216 | 22 | 0.2477 +/- 0.0027 | 24/0/2 | 1.49e-07 | 9.09e-06 |
| mixed_calibration_fixed | noise_10db | 0.5 | ECEC adaptation | 0.653 | 0.477 | 0.181 | 19 | 0.2517 +/- 0.0036 | 26/0/0 | 2.98e-08 | 2.086e-06 |
| mixed_calibration_fixed | noise_10db | 0.5 | Aug + probability | 0.653 | 0.559 | 0.137 | 18 | 0.0148 +/- 0.0030 | 13/1/12 | 0.5483 | 1 |
| mixed_calibration_fixed | noise_10db | 0.5 | Aug + TEASER | 0.653 | 0.533 | 0.147 | 22 | 0.0272 +/- 0.0045 | 16/0/10 | 0.0272 | 0.6527 |
| mixed_calibration_fixed | noise_10db | 0.6 | Probability | 0.695 | 0.413 | 0.293 | 22 | 0.2514 +/- 0.0029 | 24/0/2 | 4.172e-07 | 2.253e-05 |
| mixed_calibration_fixed | noise_10db | 0.6 | TEASER | 0.695 | 0.609 | 0.181 | 23 | 0.2353 +/- 0.0023 | 24/0/2 | 1.49e-07 | 9.09e-06 |
| mixed_calibration_fixed | noise_10db | 0.6 | ECEC adaptation | 0.695 | 0.550 | 0.171 | 20 | 0.2559 +/- 0.0029 | 26/0/0 | 2.98e-08 | 2.086e-06 |
| mixed_calibration_fixed | noise_10db | 0.6 | Aug + probability | 0.695 | 0.625 | 0.114 | 15 | 0.0080 +/- 0.0034 | 14/0/12 | 0.5651 | 1 |
| mixed_calibration_fixed | noise_10db | 0.6 | Aug + TEASER | 0.695 | 0.644 | 0.132 | 21 | 0.0227 +/- 0.0048 | 15/0/11 | 0.1227 | 1 |
| mixed_calibration_fixed | noise_5db | 0.5 | Probability | 0.692 | 0.358 | 0.347 | 26 | 0.2261 +/- 0.0032 | 26/0/0 | 2.98e-08 | 2.086e-06 |
| mixed_calibration_fixed | noise_5db | 0.5 | TEASER | 0.692 | 0.484 | 0.249 | 24 | 0.2286 +/- 0.0028 | 26/0/0 | 2.98e-08 | 2.086e-06 |
| mixed_calibration_fixed | noise_5db | 0.5 | ECEC adaptation | 0.692 | 0.465 | 0.231 | 22 | 0.2333 +/- 0.0020 | 25/0/1 | 8.941e-08 | 5.633e-06 |
| mixed_calibration_fixed | noise_5db | 0.5 | Aug + probability | 0.692 | 0.602 | 0.144 | 18 | 0.0127 +/- 0.0028 | 14/0/12 | 0.3276 | 1 |
| mixed_calibration_fixed | noise_5db | 0.5 | Aug + TEASER | 0.692 | 0.571 | 0.150 | 19 | 0.0231 +/- 0.0045 | 16/0/10 | 0.1227 | 1 |
| mixed_calibration_fixed | noise_5db | 0.6 | Probability | 0.737 | 0.397 | 0.348 | 23 | 0.2290 +/- 0.0020 | 26/0/0 | 2.98e-08 | 2.086e-06 |
| mixed_calibration_fixed | noise_5db | 0.6 | TEASER | 0.737 | 0.612 | 0.217 | 23 | 0.2212 +/- 0.0037 | 26/0/0 | 2.98e-08 | 2.086e-06 |
| mixed_calibration_fixed | noise_5db | 0.6 | ECEC adaptation | 0.737 | 0.536 | 0.231 | 23 | 0.2290 +/- 0.0025 | 26/0/0 | 2.98e-08 | 2.086e-06 |
| mixed_calibration_fixed | noise_5db | 0.6 | Aug + probability | 0.737 | 0.671 | 0.117 | 15 | 0.0047 +/- 0.0033 | 13/0/13 | 0.6893 | 1 |
| mixed_calibration_fixed | noise_5db | 0.6 | Aug + TEASER | 0.737 | 0.686 | 0.152 | 23 | 0.0164 +/- 0.0045 | 14/1/11 | 0.4273 | 1 |
| mixed_calibration_fixed | missing_10pct | 0.5 | Probability | 0.563 | 0.587 | 0.137 | 20 | -0.0012 +/- 0.0040 | 14/0/12 | 0.8809 | 1 |
| mixed_calibration_fixed | missing_10pct | 0.5 | TEASER | 0.563 | 0.498 | 0.147 | 21 | 0.0282 +/- 0.0044 | 16/0/10 | 0.1227 | 1 |
| mixed_calibration_fixed | missing_10pct | 0.5 | ECEC adaptation | 0.563 | 0.477 | 0.139 | 18 | 0.0385 +/- 0.0032 | 20/0/6 | 0.001604 | 0.06577 |
| mixed_calibration_fixed | missing_10pct | 0.5 | Aug + probability | 0.563 | 0.460 | 0.118 | 15 | 0.0292 +/- 0.0046 | 17/0/9 | 0.01401 | 0.4062 |
| mixed_calibration_fixed | missing_10pct | 0.5 | Aug + TEASER | 0.563 | 0.460 | 0.156 | 21 | 0.0369 +/- 0.0040 | 16/0/10 | 0.01761 | 0.4756 |
| mixed_calibration_fixed | missing_10pct | 0.6 | Probability | 0.592 | 0.637 | 0.148 | 18 | 0.0006 +/- 0.0027 | 17/0/9 | 0.5995 | 1 |
| mixed_calibration_fixed | missing_10pct | 0.6 | TEASER | 0.592 | 0.606 | 0.132 | 21 | 0.0172 +/- 0.0031 | 14/0/12 | 0.5317 | 1 |
| mixed_calibration_fixed | missing_10pct | 0.6 | ECEC adaptation | 0.592 | 0.559 | 0.107 | 15 | 0.0314 +/- 0.0029 | 19/0/7 | 0.006131 | 0.233 |
| mixed_calibration_fixed | missing_10pct | 0.6 | Aug + probability | 0.592 | 0.515 | 0.102 | 15 | 0.0201 +/- 0.0026 | 16/0/10 | 0.01761 | 0.4756 |
| mixed_calibration_fixed | missing_10pct | 0.6 | Aug + TEASER | 0.592 | 0.559 | 0.128 | 18 | 0.0290 +/- 0.0020 | 14/0/12 | 0.04347 | 0.9563 |
| mixed_calibration_fixed | missing_30pct | 0.5 | Probability | 0.573 | 0.561 | 0.169 | 23 | 0.0410 +/- 0.0037 | 19/0/7 | 0.008635 | 0.2849 |
| mixed_calibration_fixed | missing_30pct | 0.5 | TEASER | 0.573 | 0.520 | 0.147 | 19 | 0.0646 +/- 0.0035 | 21/0/5 | 5.165e-05 | 0.002427 |
| mixed_calibration_fixed | missing_30pct | 0.5 | ECEC adaptation | 0.573 | 0.481 | 0.131 | 19 | 0.0817 +/- 0.0036 | 23/0/3 | 2.623e-06 | 0.0001338 |
| mixed_calibration_fixed | missing_30pct | 0.5 | Aug + probability | 0.573 | 0.465 | 0.126 | 15 | 0.0302 +/- 0.0023 | 18/0/8 | 0.01105 | 0.3534 |
| mixed_calibration_fixed | missing_30pct | 0.5 | Aug + TEASER | 0.573 | 0.473 | 0.148 | 19 | 0.0354 +/- 0.0045 | 18/0/8 | 0.007291 | 0.2479 |
| mixed_calibration_fixed | missing_30pct | 0.6 | Probability | 0.606 | 0.611 | 0.178 | 21 | 0.0425 +/- 0.0044 | 19/0/7 | 0.006689 | 0.2341 |
| mixed_calibration_fixed | missing_30pct | 0.6 | TEASER | 0.606 | 0.632 | 0.137 | 19 | 0.0500 +/- 0.0033 | 18/0/8 | 0.006131 | 0.233 |
| mixed_calibration_fixed | missing_30pct | 0.6 | ECEC adaptation | 0.606 | 0.560 | 0.103 | 15 | 0.0761 +/- 0.0042 | 23/0/3 | 6.169e-06 | 0.0002961 |
| mixed_calibration_fixed | missing_30pct | 0.6 | Aug + probability | 0.606 | 0.523 | 0.108 | 15 | 0.0196 +/- 0.0027 | 17/0/9 | 0.04633 | 0.973 |
| mixed_calibration_fixed | missing_30pct | 0.6 | Aug + TEASER | 0.606 | 0.577 | 0.122 | 18 | 0.0270 +/- 0.0037 | 16/1/9 | 0.05585 | 1 |
| mixed_calibration_fixed | missing_50pct | 0.5 | Probability | 0.586 | 0.524 | 0.199 | 23 | 0.0817 +/- 0.0055 | 22/0/4 | 0.0001448 | 0.006516 |
| mixed_calibration_fixed | missing_50pct | 0.5 | TEASER | 0.586 | 0.527 | 0.145 | 18 | 0.0972 +/- 0.0039 | 23/0/3 | 2.086e-06 | 0.0001085 |
| mixed_calibration_fixed | missing_50pct | 0.5 | ECEC adaptation | 0.586 | 0.482 | 0.129 | 16 | 0.1209 +/- 0.0037 | 24/0/2 | 1.49e-07 | 9.09e-06 |
| mixed_calibration_fixed | missing_50pct | 0.5 | Aug + probability | 0.586 | 0.474 | 0.132 | 15 | 0.0305 +/- 0.0055 | 18/0/8 | 0.005613 | 0.2189 |
| mixed_calibration_fixed | missing_50pct | 0.5 | Aug + TEASER | 0.586 | 0.488 | 0.141 | 20 | 0.0322 +/- 0.0054 | 16/0/10 | 0.01295 | 0.4015 |
| mixed_calibration_fixed | missing_50pct | 0.6 | Probability | 0.621 | 0.572 | 0.204 | 19 | 0.0809 +/- 0.0055 | 22/0/4 | 0.0001259 | 0.005791 |
| mixed_calibration_fixed | missing_50pct | 0.6 | TEASER | 0.621 | 0.643 | 0.145 | 20 | 0.0818 +/- 0.0043 | 21/0/5 | 0.000248 | 0.01091 |
| mixed_calibration_fixed | missing_50pct | 0.6 | ECEC adaptation | 0.621 | 0.558 | 0.114 | 18 | 0.1153 +/- 0.0040 | 24/0/2 | 2.98e-07 | 1.639e-05 |
| mixed_calibration_fixed | missing_50pct | 0.6 | Aug + probability | 0.621 | 0.533 | 0.113 | 15 | 0.0201 +/- 0.0064 | 17/0/9 | 0.01761 | 0.4756 |
| mixed_calibration_fixed | missing_50pct | 0.6 | Aug + TEASER | 0.621 | 0.596 | 0.117 | 18 | 0.0231 +/- 0.0062 | 16/0/10 | 0.1165 | 1 |

Friedman and Nemenyi diagrams use all 26 datasets at T=0.5. Horizontal groups are shown only when the Friedman omnibus test rejects; within a group, rank differences do not exceed the Nemenyi critical difference, which is not evidence of equivalence. The post-hoc diagrams have condition-aware setting selection and must not be interpreted as deployable results; the mixed-calibration-fixed diagrams use a single setting across conditions.

- [clean, condition_posthoc](cd_diagrams/clean_condition_posthoc.png): N=26; Friedman p=0.0006097; Nemenyi CD=1.479.
- [noise_20db, condition_posthoc](cd_diagrams/noise_20db_condition_posthoc.png): N=26; Friedman p=8.327e-17; Nemenyi CD=1.479.
- [noise_10db, condition_posthoc](cd_diagrams/noise_10db_condition_posthoc.png): N=26; Friedman p=2.41e-14; Nemenyi CD=1.479.
- [noise_5db, condition_posthoc](cd_diagrams/noise_5db_condition_posthoc.png): N=26; Friedman p=2.633e-16; Nemenyi CD=1.479.
- [missing_10pct, condition_posthoc](cd_diagrams/missing_10pct_condition_posthoc.png): N=26; Friedman p=0.0002662; Nemenyi CD=1.479.
- [missing_30pct, condition_posthoc](cd_diagrams/missing_30pct_condition_posthoc.png): N=26; Friedman p=3.178e-11; Nemenyi CD=1.479.
- [missing_50pct, condition_posthoc](cd_diagrams/missing_50pct_condition_posthoc.png): N=26; Friedman p=2.115e-11; Nemenyi CD=1.479.
- [clean, mixed_calibration_fixed](cd_diagrams/clean_mixed_calibration_fixed.png): N=26; Friedman p=5.922e-05; Nemenyi CD=1.479.
- [noise_20db, mixed_calibration_fixed](cd_diagrams/noise_20db_mixed_calibration_fixed.png): N=26; Friedman p=6.126e-15; Nemenyi CD=1.479.
- [noise_10db, mixed_calibration_fixed](cd_diagrams/noise_10db_mixed_calibration_fixed.png): N=26; Friedman p=4.981e-15; Nemenyi CD=1.479.
- [noise_5db, mixed_calibration_fixed](cd_diagrams/noise_5db_mixed_calibration_fixed.png): N=26; Friedman p=1.03e-16; Nemenyi CD=1.479.
- [missing_10pct, mixed_calibration_fixed](cd_diagrams/missing_10pct_mixed_calibration_fixed.png): N=26; Friedman p=0.025; Nemenyi CD=1.479.
- [missing_30pct, mixed_calibration_fixed](cd_diagrams/missing_30pct_mixed_calibration_fixed.png): N=26; Friedman p=0.0001485; Nemenyi CD=1.479.
- [missing_50pct, mixed_calibration_fixed](cd_diagrams/missing_50pct_mixed_calibration_fixed.png): N=26; Friedman p=2.461e-07; Nemenyi CD=1.479.

## Selective stopping rule

The augmented model may DECIDE at any checkpoint when confidence exceeds its fixed time-dependent threshold, WAIT otherwise, and ABSTAIN if no checkpoint passes. The candidate family is prespecified: 9 base confidence levels x 2 time slopes = 18 rules. For each candidate, its committed calibration errors receive a one-sided Clopper-Pearson upper bound with Bonferroni delta/18, where alpha=0.10 and delta=0.10. Among certified candidates, select maximal mixed-calibration coverage and break ties by earlier decisions. If none is certified, abstain on every test case. No test condition or label enters this policy.

This is a Learn-then-Test-style finite-candidate procedure, not a universal error guarantee: the exact bound assumes an appropriate independent/exchangeable calibration-to-test sampling model and a conditional binomial model. The original mixed calibration uses balanced fixed corruption quotas, and separate test corruptions are shifted from that mixture, so those assumptions and per-condition risk control are not assured. The bound is per dataset/draw, not simultaneous across all datasets and draws. See [Learn then Test](https://arxiv.org/abs/2110.01052).

Calibration certified a rule for 59 of 130 dataset-draw pairs; the rest abstain completely. A conditional error is undefined when there are no commitments and is never replaced with zero.

| Condition | Coverage, all 26 | Committed error | Earliness on commitments | Datasets with commitments/draw | Pooled committed error |
| --- | --- | --- | --- | --- | --- |
| clean | 0.3534 +/- 0.0176 | 0.0636 +/- 0.0068 | 0.5838 +/- 0.0465 | 11-13 | 0.0601 +/- 0.0073 |
| noise_20db | 0.3157 +/- 0.0097 | 0.0689 +/- 0.0067 | 0.6140 +/- 0.0396 | 11-13 | 0.0643 +/- 0.0037 |
| noise_10db | 0.2764 +/- 0.0068 | 0.0799 +/- 0.0025 | 0.6329 +/- 0.0393 | 11-13 | 0.0718 +/- 0.0023 |
| noise_5db | 0.2430 +/- 0.0105 | 0.1185 +/- 0.0170 | 0.6618 +/- 0.0330 | 11-13 | 0.0879 +/- 0.0032 |
| missing_10pct | 0.3478 +/- 0.0165 | 0.0669 +/- 0.0071 | 0.5920 +/- 0.0455 | 11-13 | 0.0617 +/- 0.0072 |
| missing_30pct | 0.3342 +/- 0.0156 | 0.0753 +/- 0.0070 | 0.6096 +/- 0.0440 | 11-13 | 0.0651 +/- 0.0070 |
| missing_50pct | 0.3150 +/- 0.0146 | 0.0973 +/- 0.0098 | 0.6247 +/- 0.0441 | 11-13 | 0.0776 +/- 0.0087 |

Coverage averages all 26 datasets within each draw. Committed-error and committed-earliness averages exclude dataset-draws with no decisions; their denominator can vary. Pooled error weights datasets by their number of committed test cases. The +/- values are sample SDs of the five draw-level aggregates, not dataset-sampling confidence intervals.

### Comparisons and significance

The deployable augmented-probability baseline selects its stopping threshold and abstention floor on mixed calibration to approximate our calibration coverage and committed earliness; the final-only max-probability baseline matches calibration coverage, while a separate naive rule uses a fixed 0.90 final confidence threshold. Realized test coverage and earliness can differ. The exact test-coverage baseline is OFFLINE and transductive: it ranks the entire unlabeled test batch to match our realized count, so it cannot be deployed as an online condition-agnostic policy.

The strict matching subset requires absolute coverage and earliness gaps each <= 0.05 in every included draw. Selection uses no errors, but it is post-hoc and test-feature dependent, so p-values on this selected subset are exploratory. Wilcoxon uses only datasets with decisions for both methods in all five included draws; no error is invented for zero-coverage cases. Holm correction covers all valid condition x baseline x matching-subset comparisons.

| Condition | Comparator | Subset | Datasets | Error margin +/- SD | Coverage gap | Earliness gap | W/T/L | p Holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | Aug + probability, calibration matched | all_evaluable | 11 | 0.0148 +/- 0.0031 | -0.016 | -0.146 | 8/0/3 | 0.9766 |
| clean | Aug + probability, calibration matched | within_0.05_coverage_and_earliness | 3 | 0.0096 +/- 0.0078 | -0.000 | -0.025 | 1/0/2 | 1 |
| clean | Final max-probability, calibration matched | all_evaluable | 11 | 0.0003 +/- 0.0035 | -0.017 | +0.416 | 2/1/8 | 1 |
| clean | Final max-probability, calibration matched | within_0.05_coverage_and_earliness | 0 | 0.0001 +/- 0.0009 | -0.001 | +0.028 | 0/0/0 | n/a |
| clean | Final max-probability, threshold .90 | all_evaluable | 11 | -0.0057 +/- 0.0065 | -0.011 | +0.416 | 3/0/8 | 1 |
| clean | Final max-probability, threshold .90 | within_0.05_coverage_and_earliness | 0 | n/a | n/a | n/a | 0/0/0 | n/a |
| clean | Aug + probability, exact test-coverage match (OFFLINE) | all_evaluable | 11 | 0.0149 +/- 0.0030 | +0.000 | -0.144 | 8/1/2 | 0.8203 |
| clean | Aug + probability, exact test-coverage match (OFFLINE) | within_0.05_coverage_and_earliness | 3 | 0.0098 +/- 0.0081 | +0.000 | -0.027 | 1/1/1 | 1 |
| noise_20db | Aug + probability, calibration matched | all_evaluable | 11 | 0.0158 +/- 0.0017 | -0.008 | -0.122 | 7/0/4 | 1 |
| noise_20db | Aug + probability, calibration matched | within_0.05_coverage_and_earliness | 5 | 0.0053 +/- 0.0068 | -0.003 | -0.023 | 1/0/4 | 1 |
| noise_20db | Final max-probability, calibration matched | all_evaluable | 11 | -0.0010 +/- 0.0049 | -0.002 | +0.385 | 2/0/9 | 1 |
| noise_20db | Final max-probability, calibration matched | within_0.05_coverage_and_earliness | 0 | 0.0039 +/- 0.0032 | -0.001 | +0.028 | 0/0/0 | n/a |
| noise_20db | Final max-probability, threshold .90 | all_evaluable | 11 | -0.0027 +/- 0.0096 | +0.001 | +0.385 | 3/0/8 | 1 |
| noise_20db | Final max-probability, threshold .90 | within_0.05_coverage_and_earliness | 0 | n/a | n/a | n/a | 0/0/0 | n/a |
| noise_20db | Aug + probability, exact test-coverage match (OFFLINE) | all_evaluable | 11 | 0.0168 +/- 0.0025 | +0.000 | -0.119 | 8/1/2 | 1 |
| noise_20db | Aug + probability, exact test-coverage match (OFFLINE) | within_0.05_coverage_and_earliness | 5 | 0.0080 +/- 0.0062 | +0.000 | -0.023 | 2/1/2 | 1 |
| noise_10db | Aug + probability, calibration matched | all_evaluable | 11 | 0.0313 +/- 0.0087 | +0.006 | -0.116 | 7/0/4 | 1 |
| noise_10db | Aug + probability, calibration matched | within_0.05_coverage_and_earliness | 3 | 0.0049 +/- 0.0081 | +0.002 | -0.012 | 0/0/3 | 1 |
| noise_10db | Final max-probability, calibration matched | all_evaluable | 11 | -0.0025 +/- 0.0023 | -0.000 | +0.366 | 3/0/8 | 1 |
| noise_10db | Final max-probability, calibration matched | within_0.05_coverage_and_earliness | 0 | 0.0125 +/- 0.0337 | -0.002 | +0.038 | 0/0/0 | n/a |
| noise_10db | Final max-probability, threshold .90 | all_evaluable | 11 | 0.0008 +/- 0.0130 | -0.017 | +0.366 | 3/0/8 | 1 |
| noise_10db | Final max-probability, threshold .90 | within_0.05_coverage_and_earliness | 0 | n/a | n/a | n/a | 0/0/0 | n/a |
| noise_10db | Aug + probability, exact test-coverage match (OFFLINE) | all_evaluable | 11 | 0.0288 +/- 0.0050 | +0.000 | -0.117 | 7/1/3 | 1 |
| noise_10db | Aug + probability, exact test-coverage match (OFFLINE) | within_0.05_coverage_and_earliness | 3 | 0.0085 +/- 0.0079 | +0.000 | -0.015 | 0/1/2 | 1 |
| noise_5db | Aug + probability, calibration matched | all_evaluable | 11 | 0.0359 +/- 0.0142 | +0.023 | -0.113 | 9/0/2 | 1 |
| noise_5db | Aug + probability, calibration matched | within_0.05_coverage_and_earliness | 3 | 0.0020 +/- 0.0064 | +0.003 | -0.010 | 1/0/2 | 1 |
| noise_5db | Final max-probability, calibration matched | all_evaluable | 11 | -0.0121 +/- 0.0081 | -0.003 | +0.338 | 3/0/8 | 1 |
| noise_5db | Final max-probability, calibration matched | within_0.05_coverage_and_earliness | 0 | 0.0061 +/- 0.0075 | -0.004 | +0.032 | 0/0/0 | n/a |
| noise_5db | Final max-probability, threshold .90 | all_evaluable | 11 | -0.0119 +/- 0.0168 | -0.033 | +0.338 | 2/0/9 | 1 |
| noise_5db | Final max-probability, threshold .90 | within_0.05_coverage_and_earliness | 0 | 0.2473 +/- 0.0997 | +0.038 | +0.020 | 0/0/0 | n/a |
| noise_5db | Aug + probability, exact test-coverage match (OFFLINE) | all_evaluable | 11 | 0.0334 +/- 0.0078 | +0.000 | -0.119 | 9/1/1 | 0.8203 |
| noise_5db | Aug + probability, exact test-coverage match (OFFLINE) | within_0.05_coverage_and_earliness | 3 | -0.0005 +/- 0.0050 | +0.000 | -0.008 | 1/1/1 | 1 |
| missing_10pct | Aug + probability, calibration matched | all_evaluable | 11 | 0.0142 +/- 0.0035 | -0.017 | -0.137 | 8/0/3 | 1 |
| missing_10pct | Aug + probability, calibration matched | within_0.05_coverage_and_earliness | 4 | 0.0079 +/- 0.0058 | -0.002 | -0.025 | 1/0/3 | 1 |
| missing_10pct | Final max-probability, calibration matched | all_evaluable | 11 | -0.0017 +/- 0.0052 | -0.015 | +0.407 | 3/0/8 | 1 |
| missing_10pct | Final max-probability, calibration matched | within_0.05_coverage_and_earliness | 0 | 0.0020 +/- 0.0008 | -0.001 | +0.028 | 0/0/0 | n/a |
| missing_10pct | Final max-probability, threshold .90 | all_evaluable | 11 | -0.0064 +/- 0.0084 | -0.010 | +0.407 | 3/0/8 | 1 |
| missing_10pct | Final max-probability, threshold .90 | within_0.05_coverage_and_earliness | 0 | n/a | n/a | n/a | 0/0/0 | n/a |
| missing_10pct | Aug + probability, exact test-coverage match (OFFLINE) | all_evaluable | 11 | 0.0147 +/- 0.0035 | +0.000 | -0.135 | 8/1/2 | 1 |
| missing_10pct | Aug + probability, exact test-coverage match (OFFLINE) | within_0.05_coverage_and_earliness | 4 | 0.0079 +/- 0.0067 | +0.000 | -0.024 | 1/1/2 | 1 |
| missing_30pct | Aug + probability, calibration matched | all_evaluable | 11 | 0.0165 +/- 0.0035 | -0.013 | -0.127 | 7/0/4 | 1 |
| missing_30pct | Aug + probability, calibration matched | within_0.05_coverage_and_earliness | 4 | 0.0001 +/- 0.0023 | -0.002 | -0.014 | 1/0/3 | 1 |
| missing_30pct | Final max-probability, calibration matched | all_evaluable | 11 | -0.0066 +/- 0.0056 | -0.012 | +0.390 | 3/0/8 | 1 |
| missing_30pct | Final max-probability, calibration matched | within_0.05_coverage_and_earliness | 0 | 0.0039 +/- 0.0026 | -0.001 | +0.027 | 0/0/0 | n/a |
| missing_30pct | Final max-probability, threshold .90 | all_evaluable | 11 | -0.0091 +/- 0.0088 | -0.014 | +0.390 | 3/0/8 | 1 |
| missing_30pct | Final max-probability, threshold .90 | within_0.05_coverage_and_earliness | 0 | n/a | n/a | n/a | 0/0/0 | n/a |
| missing_30pct | Aug + probability, exact test-coverage match (OFFLINE) | all_evaluable | 11 | 0.0173 +/- 0.0036 | +0.000 | -0.125 | 7/1/3 | 1 |
| missing_30pct | Aug + probability, exact test-coverage match (OFFLINE) | within_0.05_coverage_and_earliness | 4 | 0.0010 +/- 0.0033 | +0.000 | -0.013 | 1/1/2 | 1 |
| missing_50pct | Aug + probability, calibration matched | all_evaluable | 11 | 0.0251 +/- 0.0068 | -0.003 | -0.120 | 8/0/3 | 1 |
| missing_50pct | Aug + probability, calibration matched | within_0.05_coverage_and_earliness | 3 | 0.0022 +/- 0.0029 | -0.001 | -0.007 | 0/0/3 | 1 |
| missing_50pct | Final max-probability, calibration matched | all_evaluable | 11 | -0.0119 +/- 0.0084 | -0.013 | +0.374 | 3/0/8 | 1 |
| missing_50pct | Final max-probability, calibration matched | within_0.05_coverage_and_earliness | 0 | -0.0017 +/- 0.0030 | -0.001 | +0.025 | 0/0/0 | n/a |
| missing_50pct | Final max-probability, threshold .90 | all_evaluable | 11 | -0.0133 +/- 0.0145 | -0.026 | +0.374 | 3/0/8 | 1 |
| missing_50pct | Final max-probability, threshold .90 | within_0.05_coverage_and_earliness | 0 | n/a | n/a | n/a | 0/0/0 | n/a |
| missing_50pct | Aug + probability, exact test-coverage match (OFFLINE) | all_evaluable | 11 | 0.0253 +/- 0.0071 | +0.000 | -0.120 | 8/1/2 | 1 |
| missing_50pct | Aug + probability, exact test-coverage match (OFFLINE) | within_0.05_coverage_and_earliness | 3 | 0.0023 +/- 0.0029 | +0.000 | -0.006 | 0/1/2 | 1 |

## Calibration mechanism: top-label ECE

ECE uses 15 equal-width confidence bins at the stated checkpoint. Signed confidence minus accuracy > 0 indicates aggregate overconfidence; a small ECE does not establish conditional or per-class calibration. All values below are 26-dataset means +/- sample SD across draws.

| Condition | Prefix | Clean ECE | Augmented ECE | Clean signed gap | Augmented signed gap |
| --- | --- | --- | --- | --- | --- |
| clean | 0.4 | 0.1205 +/- 0.0000 | 0.1266 +/- 0.0012 | +0.0851 | +0.1050 |
| clean | 0.6 | 0.1037 +/- 0.0000 | 0.1081 +/- 0.0016 | +0.0653 | +0.0869 |
| clean | 1.0 | 0.0721 +/- 0.0000 | 0.0733 +/- 0.0015 | +0.0291 | +0.0500 |
| noise_20db | 0.4 | 0.3086 +/- 0.0023 | 0.1460 +/- 0.0027 | +0.2973 | +0.1227 |
| noise_20db | 0.6 | 0.3105 +/- 0.0010 | 0.1229 +/- 0.0020 | +0.3032 | +0.0943 |
| noise_20db | 1.0 | 0.2572 +/- 0.0013 | 0.0899 +/- 0.0021 | +0.2386 | +0.0556 |
| noise_10db | 0.4 | 0.4291 +/- 0.0019 | 0.1585 +/- 0.0016 | +0.4178 | +0.1317 |
| noise_10db | 0.6 | 0.4475 +/- 0.0013 | 0.1402 +/- 0.0029 | +0.4342 | +0.0993 |
| noise_10db | 1.0 | 0.3935 +/- 0.0010 | 0.1180 +/- 0.0038 | +0.3743 | +0.0653 |
| noise_5db | 0.4 | 0.4787 +/- 0.0028 | 0.1681 +/- 0.0030 | +0.4671 | +0.1514 |
| noise_5db | 0.6 | 0.5115 +/- 0.0008 | 0.1485 +/- 0.0030 | +0.4979 | +0.1190 |
| noise_5db | 1.0 | 0.4535 +/- 0.0010 | 0.1270 +/- 0.0024 | +0.4372 | +0.0808 |
| missing_10pct | 0.4 | 0.1482 +/- 0.0033 | 0.1369 +/- 0.0027 | +0.1149 | +0.1158 |
| missing_10pct | 0.6 | 0.1220 +/- 0.0011 | 0.1190 +/- 0.0031 | +0.0841 | +0.0985 |
| missing_10pct | 1.0 | 0.0920 +/- 0.0018 | 0.0823 +/- 0.0009 | +0.0471 | +0.0589 |
| missing_30pct | 0.4 | 0.1978 +/- 0.0023 | 0.1511 +/- 0.0043 | +0.1753 | +0.1348 |
| missing_30pct | 0.6 | 0.1618 +/- 0.0016 | 0.1310 +/- 0.0026 | +0.1348 | +0.1106 |
| missing_30pct | 1.0 | 0.1359 +/- 0.0013 | 0.0959 +/- 0.0020 | +0.0961 | +0.0726 |
| missing_50pct | 0.4 | 0.2615 +/- 0.0023 | 0.1724 +/- 0.0048 | +0.2502 | +0.1600 |
| missing_50pct | 0.6 | 0.2149 +/- 0.0026 | 0.1501 +/- 0.0023 | +0.1981 | +0.1352 |
| missing_50pct | 1.0 | 0.1844 +/- 0.0011 | 0.1153 +/- 0.0030 | +0.1518 | +0.0935 |

## Verdict

- clean: high-coverage, <= 0.10 committed error across all 26 datasets is not established.
- noise_20db: high-coverage, <= 0.10 committed error across all 26 datasets is not established.
- noise_10db: high-coverage, <= 0.10 committed error across all 26 datasets is not established.
- noise_5db: high-coverage, <= 0.10 committed error across all 26 datasets is not established.
- missing_10pct: high-coverage, <= 0.10 committed error across all 26 datasets is not established.
- missing_30pct: high-coverage, <= 0.10 committed error across all 26 datasets is not established.
- missing_50pct: high-coverage, <= 0.10 committed error across all 26 datasets is not established.

A result is not evidence of a stopping-layer improvement at matched coverage and earliness unless the reported strict-matching subset is sufficiently large and its corrected paired comparison supports it. A zero-coverage policy may have no measured error; that is not successful high-coverage risk control.

The UCR train/test and calibration splits are fixed across draws. Dataset correlations, small calibration sets, test-condition-specific diagnostics, and possible upstream archive preprocessing limit external validity and causal claims. Detailed per-dataset results, certificates, code, fixed-point settings, and uncertainty summaries are included in the CSV files.

Prior-source audit: 4170 files before and 4170 after; 0 changed, 0 removed, 0 added.
Replay audit: 910 condition records checked against saved checkpoint accuracies; 780 prior mixed-calibration threshold vectors rederived, with 0 mismatches. No models were retrained.
