# ECTS fixed-length and corruption-agnostic stopping frontiers

**Verdict:** At equal fixed prefixes, augmentation is no worse on the macro-mean curve at every available checkpoint for: Noise 20 dB, Noise 10 dB, Noise 5 dB, Missing 30%, Missing 50%. At matched mean earliness, the augmented stopping envelope is below both baselines throughout shared support for: Noise 20 dB, Noise 10 dB, Noise 5 dB, Missing 30%, Missing 50%. These are aggregate, post-hoc comparisons; they do not establish per-dataset dominance, statistical significance, or a deployable test-conditioned choice.

Eight datasets, five paired augmentation/test-corruption draws, and seven conditions replay the saved v2 checkpoint classifiers. All cells are means +/- sample SD over five draws: first average datasets within each draw. The earlier robustness run was read-only.

The old run did not store corrupted test streams or full test probabilities, so they were regenerated from its deterministic seeds and saved classifiers. TEASER safety estimators were absent and were rebuilt once per dataset with the exact v2 settings; all regenerated probability, alpha=.10, and default-TEASER per-case decisions were checked against v2.

ItalyPowerDemand has no 20% or 30% checkpoint: nine observations are the MiniRocket minimum, whereas those nominal prefixes contain five and eight observations. Fixed-length macro curves at 20% and 30% therefore use seven datasets; all other fixed-length points and all stopping results use eight. Within each dataset the models see exactly the same observed prefix.

## A. Fixed-length error gaps

Cells are clean-model error minus augmented-model error; positive means augmentation improves accuracy. The last columns ask whether the augmented mean error is no greater at every available nominal checkpoint (20%, 30%, ..., 100%), across the macro curve and within each dataset.

| Condition | 20% (n=7) | 40% | 60% | 80% | 100% | Macro all checkpoints? | Dataset count all checkpoints |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clean | -0.017 +/- 0.006 | -0.009 +/- 0.005 | -0.008 +/- 0.008 | -0.011 +/- 0.004 | -0.009 +/- 0.005 | NO | 0/8 |
| Noise 20 dB | +0.079 +/- 0.006 | +0.072 +/- 0.006 | +0.049 +/- 0.006 | +0.045 +/- 0.003 | +0.042 +/- 0.003 | YES | 3/8 |
| Noise 10 dB | +0.095 +/- 0.004 | +0.111 +/- 0.003 | +0.103 +/- 0.005 | +0.174 +/- 0.005 | +0.081 +/- 0.007 | YES | 6/8 |
| Noise 5 dB | +0.087 +/- 0.004 | +0.110 +/- 0.004 | +0.112 +/- 0.006 | +0.156 +/- 0.005 | +0.156 +/- 0.008 | YES | 5/8 |
| Missing 10% | -0.005 +/- 0.006 | -0.002 +/- 0.005 | +0.002 +/- 0.009 | -0.006 +/- 0.007 | -0.001 +/- 0.005 | NO | 1/8 |
| Missing 30% | +0.023 +/- 0.002 | +0.021 +/- 0.008 | +0.026 +/- 0.013 | +0.017 +/- 0.010 | +0.031 +/- 0.006 | YES | 3/8 |
| Missing 50% | +0.044 +/- 0.003 | +0.041 +/- 0.008 | +0.047 +/- 0.013 | +0.041 +/- 0.011 | +0.057 +/- 0.009 | YES | 4/8 |

## B. Matched stopping earliness

Sweeps use probability thresholds 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99; mixed-calibration alpha 0.02, 0.05, 0.10, 0.15, 0.20, 0.30; and TEASER consecutive-safe counts 2, 3, 4, 5. Every setting is fixed without using the test condition. The per-condition curves and comparisons are post-hoc descriptions, not a rule for selecting a setting after observing test conditions.
At each shared earliness, the lower convex envelope linearly interpolates between swept settings; this represents a randomized mixture of fixed policies, not an independently measured operating point. We do not extrapolate. Margins are baseline error minus augmented error, so positive means the augmented policy has lower error. Targets use the range jointly supported by all three methods in all five draws.

| Condition | Matched earliness | Probability minus augmented | TEASER minus augmented | Better-baseline minus augmented | Aug lower than both at this point? |
|---|---:|---:|---:|---:|---|
| Clean | 0.4465 | -0.026 +/- 0.009 | -0.005 +/- 0.009 | -0.026 +/- 0.009 | NO |
| Clean | 0.4798 | -0.023 +/- 0.008 | -0.001 +/- 0.008 | -0.023 +/- 0.008 | NO |
| Clean | 0.5130 | -0.019 +/- 0.007 | -0.002 +/- 0.007 | -0.019 +/- 0.007 | NO |
| Noise 20 dB | 0.4762 | +0.096 +/- 0.010 | +0.082 +/- 0.011 | +0.082 +/- 0.011 | YES |
| Noise 20 dB | 0.5104 | +0.101 +/- 0.008 | +0.084 +/- 0.009 | +0.084 +/- 0.009 | YES |
| Noise 20 dB | 0.5446 | +0.103 +/- 0.006 | +0.082 +/- 0.007 | +0.082 +/- 0.007 | YES |
| Noise 10 dB | 0.4978 | +0.137 +/- 0.011 | +0.117 +/- 0.011 | +0.117 +/- 0.011 | YES |
| Noise 10 dB | 0.5348 | +0.143 +/- 0.008 | +0.121 +/- 0.009 | +0.121 +/- 0.009 | YES |
| Noise 10 dB | 0.5719 | +0.145 +/- 0.009 | +0.122 +/- 0.009 | +0.122 +/- 0.009 | YES |
| Noise 5 dB | 0.5006 | +0.130 +/- 0.013 | +0.103 +/- 0.014 | +0.103 +/- 0.014 | YES |
| Noise 5 dB | 0.5346 | +0.141 +/- 0.009 | +0.110 +/- 0.010 | +0.110 +/- 0.010 | YES |
| Noise 5 dB | 0.5686 | +0.146 +/- 0.007 | +0.113 +/- 0.009 | +0.113 +/- 0.009 | YES |
| Missing 10% | 0.4588 | -0.017 +/- 0.011 | +0.005 +/- 0.010 | -0.017 +/- 0.011 | NO |
| Missing 10% | 0.4948 | -0.014 +/- 0.010 | +0.009 +/- 0.009 | -0.014 +/- 0.010 | NO |
| Missing 10% | 0.5309 | -0.011 +/- 0.009 | +0.008 +/- 0.009 | -0.011 +/- 0.009 | NO |
| Missing 30% | 0.4783 | +0.022 +/- 0.014 | +0.040 +/- 0.015 | +0.022 +/- 0.014 | YES |
| Missing 30% | 0.5166 | +0.026 +/- 0.013 | +0.045 +/- 0.014 | +0.026 +/- 0.013 | YES |
| Missing 30% | 0.5548 | +0.029 +/- 0.012 | +0.046 +/- 0.013 | +0.029 +/- 0.012 | YES |
| Missing 50% | 0.4888 | +0.047 +/- 0.015 | +0.058 +/- 0.015 | +0.047 +/- 0.015 | YES |
| Missing 50% | 0.5279 | +0.055 +/- 0.014 | +0.066 +/- 0.015 | +0.055 +/- 0.014 | YES |
| Missing 50% | 0.5671 | +0.059 +/- 0.013 | +0.069 +/- 0.013 | +0.059 +/- 0.013 | YES |

### Equal-earliness verdict by condition

YES requires lower augmented error than both comparison envelopes at every breakpoint in their common mean-curve support. NO is a failure at at least one breakpoint; no common support is reported as unavailable. These are descriptive means, not statistical significance claims.

| Condition | Augmented strictly dominates both on shared mean support? | Common earliness range |
|---|---|---:|
| Clean | NO | 0.3942 - 0.5734 |
| Noise 20 dB | YES | 0.4409 - 0.6140 |
| Noise 10 dB | YES | 0.4596 - 0.6108 |
| Noise 5 dB | YES | 0.4653 - 0.6041 |
| Missing 10% | NO | 0.4105 - 0.5927 |
| Missing 30% | YES | 0.4305 - 0.6187 |
| Missing 50% | YES | 0.4366 - 0.6344 |

## Artifacts and scope

fixed_length_curves.csv and frontier_points.csv contain per-dataset and macro mean/SD values; fixed_length_curves_draws.csv and frontier_points_draws.csv contain individual-draw points. The ItalyPowerDemand 9/24 checkpoint is included as an actual-only per-dataset point and in its dataset-level dominance check. accuracy_vs_prefix.png and earliness_vs_error_frontier.png show the curves and SD bars.
The five draws vary augmentation and corruption while holding each dataset split fixed. No finite-sample risk guarantees or claims about raw physical SNR follow. Original data were previously normalized per series, and synthetic noise is applied in archive units.
