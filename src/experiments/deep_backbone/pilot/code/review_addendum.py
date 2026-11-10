"""Post-run review of the prespecified validation winner; no test-based selection."""
import json
import pandas as pd
from settings import *
from report import table
d=pd.read_csv(OUT/'deep_pilot_metrics.csv')
best=json.loads((OUT/'validation_selection.json').read_text())['primary_variant']
a=d[(d.kind=='fixed')&(d.method==best)];b=d[(d.kind=='fixed')&(d.method=='MiniRocket_aug')]
x=a.merge(b,on=['dataset','draw','condition','setting'],suffixes=('_deep','_rocket'))
x['error_reduction']=x.error_rocket-x.error_deep
x[['dataset','draw','condition','setting','error_reduction']].to_csv(OUT/'per_dataset_fixed_gaps.csv',index=False)
g=x.groupby(['dataset','condition']).error_reduction.mean().unstack().reindex(columns=CONDITIONS)*100
summary=(OUT/'summary.md').read_text()
with (OUT/'logs/summary_before_review.md').open('x') as f:f.write(summary)
lead=(
'**Decision:** do not promote cross-checkpoint weight inheritance as the full method on this evidence. '
'The validation winner is C_l0: independent heads without consistency. A learned TCN does show a real '
'equal-prefix Gaussian-noise advantage over augmented MiniRocket (positive mean gains at all nine checkpoints), '
'but missing-data gains are confined to part of the curve and there is a clean-data cost. '
'At prefixes 0.5/0.6, all four corrupted conditions have positive mean gains; this does not imply dominance '
'at every prefix or on every dataset. No ten-test Holm-corrected pilot comparison reaches 0.05; '
'the smallest adjusted p-value is 0.078125 with only eight datasets.\n\n')
title,rest=summary.split('\n',1);summary=title+'\n\n'+lead+rest.lstrip('\n')
extra=['','## Per-dataset review and final decision','',
'Error reduction versus augmented MiniRocket, in percentage points, averaged across nine equal-prefix checkpoints and '
'three draws. Positive = the learned validation winner is better. These are descriptive comparisons, not '
'test-selected model choices.','',table(g.reset_index()),'',
'The average noise gain is broad: seven of eight datasets improve when averaged over checkpoints under both noise levels. '
'Missing-data effects are split: Crop, ElectricDevices, FordA and Wafer improve, while ECG5000, HandOutlines, '
'NonInvasiveFetalECGThorax1 and StarLightCurves worsen. The fetal ECG dataset has a particularly large clean/missing-data cost; '
'it also has the largest 5 dB gain. This is a condition-dependent trade-off, not general dominance.','',
'**Inheritance:** progressive heads are approximately tied with independent heads and do not show a repeatable advantage. '
'Full head sharing is worse. **Consistency:** lambda=0.1 gives small, mixed changes; the validation-selected model uses zero. '
'The pilot therefore supports further investigation of a noise-robust learned backbone, but not an inheritance/consistency '
'headline or wholesale replacement of augmented MiniRocket.','',
'**Stopping interpretation:** common-grid probability stopping on the learned model also reduces noise error, but calibration-target '
'matching leaves substantial per-dataset test-earliness gaps (only 1-4 of 8 are within 0.05 depending on condition/target). '
'Those stopping comparisons are not promoted as exact equal-earliness evidence. The fixed-checkpoint comparisons are exact '
'within each dataset and answer the backbone question directly.','',
'## Resolved execution issues','',
'Training worker failures: 0. No packages were installed or upgraded. The first report formatting pass encountered:',
'',chr(96)*3+'text']
log=(OUT/'logs/finalizer.log').read_text()
extra.append(next(z for z in log.splitlines() if z.startswith('ImportError:')))
extra += [chr(96)*3,'',
'The report now uses a built-in Markdown formatter. All eight numeric CSV files produced before that formatting fix '
'were SHA256-identical afterward (logs/numeric_hashes_after_report_fix.json). The original failure traceback is preserved.',
'',
'One monitoring SSH connection also returned TimeoutError: timed out; reconnect succeeded and GPU jobs were unaffected.','',
'Source integrity audit: all 5,419 protected files unchanged, with no additions or removals. All 168 model causality audits, '
'384 trained-head inheritance copies, and 24 augmentation-recipe comparisons passed.','']
(OUT/'summary.md').write_text(summary+'\n'.join(extra))
print('REVIEW_ADDENDUM_DONE',best,flush=True)
