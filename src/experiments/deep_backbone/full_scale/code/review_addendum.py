from pathlib import Path
import json
p=Path('/home/ubuntu/host_share/kumail_mad/runs/ects_deep_scale_v1')
s=p/'summary.md'
original=s.read_text()
backup=p/'logs/summary_before_review.md'
with backup.open('x') as f:f.write(original)
answers="""## Final interpretation

**Verdict: the learned backbone has a limited, statistically supported noise advantage. The mask does not repair the missing-data weakness, and neither learned configuration dominates augmented MiniRocket across conditions.**

1. **Noise: partially confirmed at equal earliness.** At the exact halfway checkpoint, TCN_aug reduces error versus augmented MiniRocket by **3.09 +/- 0.30 percentage points at 10 dB** (Wilcoxon-Holm p=0.005328, W/T/L=22/0/4) and **3.89 +/- 0.52 points at 5 dB** (p=0.007622, 21/0/5). This supports the lower part of the pilot's 3-5 point claim at that checkpoint on N=26. The mask variant retains those gains: 3.03 +/- 0.40 and 3.80 +/- 0.65 points, p=0.014769 and 0.010702. At checkpoint 0.6, both variants still have positive mean gains, but neither survives the declared 28-test Holm family (plain p=0.224071 at 10 dB, 0.061197 at 5 dB; mask p=0.231951 and 0.057858). Mild noise (20 dB) is not significant. Averaging all nine fixed checkpoints, only 5 dB is significant: plain +2.96 +/- 0.26 points, p=0.023134; mask +2.99 +/- 0.51, p=0.020215 in the separate 14-test curve-average family. Only 5 dB has nonnegative mean gains at every checkpoint.

2. **Missingness mask: no demonstrated recovery.** At missing 30%/50%, TCN_aug_mask gains just 0.29/0.32 points over augmented MiniRocket at checkpoint 0.5 and -0.34/+0.01 points at checkpoint 0.6; all adjusted p=1. Its incremental gains over the plain TCN are only 0.03/0.13 points at 0.5 and 0.09/0.15 at 0.6, also not significant (all adjusted p=1). Across all checkpoints it remains worse than augmented MiniRocket by 1.44 points at missing 30% and 1.11 at missing 50%. Noise gains are preserved, but the requested missing-data fix is not supported.

3. **One configuration for both corruptions: no.** No model demonstrates significant improvement over augmented MiniRocket for both noise and missingness. The clean-data accuracy cost averaged across the nine checkpoints is **3.06 +/- 0.35 points for TCN_aug** and **3.04 +/- 0.29 for TCN_aug_mask**; at checkpoints 0.5/0.6 the costs are approximately 1.1-1.4 points. These are observed macro costs, not claims of significance at each fixed checkpoint.

**Deployable stopping is a separate negative result.** At mixed-calibration-selected targets 0.5 and 0.6, neither TCN has a Holm-significant positive error margin against augmented MiniRocket in any condition. The clean-data stopping cost is significant. Realized test earliness differs: for example, at 5 dB and calibration target 0.5, plain TCN decides at mean fraction 0.634 versus MiniRocket's 0.692, with a nonsignificant 1.34-point lower error. Therefore these stopping results cannot establish superiority at exactly equal test earliness; the fixed-prefix comparisons above provide the exact matched-time evidence. All setting choices were made from mixed calibration alone and held across test conditions.

**Recommendation:** retain the TCN as a noise-specific benchmark contribution, not a general replacement for augmented MiniRocket. Do not promote the mask variant as a successful missingness method. The full N=26 is the requested benchmark, but eight datasets informed the earlier model selection; the new-18 descriptive analysis below is an additional sensitivity check, not an independent preregistered replication.

**Execution and checks.** Used GPUs 0, 1 and 2 (Quadro RTX 8000). The 260 artifacts comprise 236 new fits and 24 compatible pilot checkpoints reused unchanged. New training plus validation consumed 1.396 GPU-hours; summed GPU-assigned job time including preparation/inference was 1.572 hours, and parallel worker wall time was 33.60 minutes. All 260 models passed value/mask causality checks (zero future gradients; maximum prefix-logit numerical difference 3.82e-6). All 7,025 protected prior files were unchanged; there was no baseline retraining or package installation.

**Mask-use audit.** All 130 trained mask models have nonzero learned mask-channel weights. For FordA draw 0, missing 50%, 32 held-out samples, replacing genuine flags by all-observed flags changes mean absolute class probability by 0.00323 (maximum 0.03792) and changes 2 of 288 checkpoint decisions. The original inference reproduces cached probabilities. This is a post-training implementation check, not an additional accuracy evaluation or tuning step; details are in mask_usage_audit.json.

"""
start=original.index('## Key answers')
end=original.index('## Data, splits and method')
s.write_text(original[:start]+answers+original[end:])
print('SUMMARY_REVIEWED',len(s.read_text().splitlines()),'lines')
for fn in ['artifact_audit.json','source_audit.json','mask_usage_audit.json']:
 a=json.loads((p/fn).read_text())
 print(fn, a if fn!='source_audit.json' else {k:v for k,v in a.items() if not isinstance(v,dict)})
