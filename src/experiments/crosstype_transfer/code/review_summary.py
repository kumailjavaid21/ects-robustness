from pathlib import Path
import json
import numpy as np,pandas as pd
p=Path('/home/ubuntu/host_share/kumail_mad/runs/ects_crosstype_v1')
s=p/'summary.md';original=s.read_text()
with (p/'logs/summary_before_review.md').open('x') as f:f.write(original)
fs=pd.read_csv(p/'fixed_length_stats.csv');tr=pd.read_csv(p/'transfer_gap_draws.csv');tm=pd.read_csv(p/'crosstype_transfer_gap.csv')
def pm(a,b,scale=100):return f'{a*scale:+.2f} +/- {b*scale:.2f}'
def md(df):
 return '\n'.join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']+
  ['| '+' | '.join(str(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)])
text=['## Reviewed conclusion','',
 '**The representational-robustness thesis is supported for specific unseen corruptions, not universally. '
 'Noise+random-missing augmentation transfers to spikes and coarse quantization. It does not consistently transfer '
 'to causal time warping, baseline drift, or the tested burst missingness schedules. The single mixed stopping '
 'calibration does not transfer uniformly to these unseen shifts.**','',
 'The table averages error reductions over all nine MATCHED checkpoints (then datasets/draws). Positive numbers favor '
 'the augmented model. +/- is sample SD of five paired macro draw means; p_Holm uses the declared separate 30-test '
 'curve-average family, not the 60-test primary fixed-checkpoint family.','']
rows=[]
for cond in ['spike_005','spike_015','warp_moderate','warp_strong','quantization_8','quantization_4','drift_small','drift_large','burst_3x10','burst_1x30']:
 a=fs[(fs.condition==cond)&(fs.family=='curve_average_30')&(fs.method=='MiniRocket_aug')&(fs.baseline=='MiniRocket_clean')].iloc[0]
 b=fs[(fs.condition==cond)&(fs.family=='curve_average_30')&(fs.method=='TCN_aug')&(fs.baseline=='MiniRocket_aug')].iloc[0]
 rows.append({'condition':cond,'MR augmentation vs clean, pp +/- SD':pm(a.error_reduction_mean,a.error_reduction_sd),
  'p_Holm':f'{a.p_holm:.4g}','TCN vs augmented MR, pp +/- SD':pm(b.error_reduction_mean,b.error_reduction_sd),'TCN p_Holm':f'{b.p_holm:.4g}'})
text+=[md(pd.DataFrame(rows)),'',
 '- **Spikes: positive transfer.** Augmented MiniRocket has lower mean error than the clean backbone at every one '
 'of the nine matched prefixes for both severities. Both curve-average gains survive Holm. In the stricter primary '
 'checkpoint family, the 15% spike gains at .5/.6 are +6.34/+4.70 pp (p=.0407/.0209); the 5% spike gains '
 '+7.31/+7.17 pp do not survive correction (p=.2546/.0820). Thus the result is not significant at every time/severity.',
 '- **Quantization: positive transfer.** Augmented MiniRocket is better in mean at all nine matched prefixes for both '
 '8 and 4 levels; both curve-average gains are significant. Three of the four primary .5/.6 comparisons are significant. '
 'The fourth, 4 levels at .6, has p=.0534.',
 '- **Warp and drift: no positive transfer.** Augmented MiniRocket is worse in mean at every matched prefix for '
 'both severities of each type. All four curve-average losses survive the separate Holm correction. '
 'At primary checkpoints, moderate warp at .6 and small drift at .6 are significant losses.',
 '- **Burst missingness: no clear MR augmentation benefit.** Curve-average effects are +0.63 pp for three shorter '
 'bursts and -0.30 pp for one long burst, neither significant; none of the four primary comparisons is a significant gain.','',
 '**The learned spike advantage extends.** TCN_aug improves on augmented MiniRocket at every matched prefix for '
 'both spike severities. Curve-average gains are +4.59 +/- 0.70 pp (5% spikes, p=.0064) and +3.97 +/- 0.47 pp '
 '(15%, p=.0142). At exact prefix .6, gains are +5.35 +/- 1.57 pp (p=.0286) and +4.30 +/- 1.53 pp (p=.0136); '
 'the .5 gains do not survive the primary correction. TCN also wins several burst comparisons: at .6, +5.35/+7.56 pp '
 'against augmented MiniRocket (p=.0131/.0097). These are conditional results, not a universal learned-model win: '
 'under large drift the TCN loses 5.44 pp on the curve average versus augmented MiniRocket (p=.0112).','',
 '**Deployable stopping does not establish an additional TCN win.** No positive TCN-versus-augmented-MiniRocket '
 'stopping comparison survives the 140-test Holm family at either calibration target. The TCN has significant stopping '
 'losses under both drift severities. Augmented MiniRocket has significant stopping wins over clean TEASER/ECEC '
 'under spikes and several wins under quantization, but realized waiting times differ; the exact-prefix comparisons '
 'are the evidence for representational transfer at equal earliness.','',
 '**Mixed calibration: no uniform small-gap transfer.** For the primary MiniRocket_aug recipe, same-alpha '
 'mixed-minus-oracle gaps below average both severities and both calibration targets. They are descriptive macro '
 'comparisons, not tests at equal realized time. The added absolute per-dataset gaps expose cancellation.','']
a=tr[(tr.method=='MiniRocket_aug')&(tr.diagnostic=='oracle_same_alpha')]
draw=a.groupby(['type','draw'])[['error_gap','earliness_gap']].mean()
agg=draw.groupby('type').agg(['mean','std'])
rows=[]
for typ in ['spike','warp','quantization','drift','burst']:
 r=agg.loc[typ];g=tm[(tm.type==typ)&(tm.method=='MiniRocket_aug')&(tm.diagnostic=='oracle_same_alpha')]
 rows.append({'type':typ,'mixed - oracle error pp +/- SD':pm(r[('error_gap','mean')],r[('error_gap','std')]),
  'mixed - oracle decision fraction +/- SD':pm(r[('earliness_gap','mean')],r[('earliness_gap','std')],1),
  'mean absolute dataset error gap, pp':f'{100*g.mean_absolute_dataset_error_gap.mean():.2f}',
  'mean absolute dataset time gap':f'{g.mean_absolute_dataset_earliness_gap.mean():.3f}'})
text += [md(pd.DataFrame(rows)),'',
 'For example, under 5% spikes at calibration target .5, mixed versus oracle error is 50.75% versus 48.30%, '
 'but decision fraction is .564 versus .833. Under 4-level quantization it is 56.04% versus 53.53%, with times '
 '.479 versus .838. These are substantial differences in waiting, even where error gaps look modest. '
 'Drift has small macro same-alpha gaps, but its mean absolute per-dataset time gap is about .145, so even this is '
 'not uniform transfer. The type-matched oracle alpha grid also fails to reach the desired .5/.6 calibration '
 'earliness on some datasets; choosing its closest setting does not make the oracle exactly time-matched. '
 'A matched oracle can score worse, particularly for warp, and is not a proven upper bound.','',
 '**A small oracle gap is not an error guarantee.** At common alpha=.10, mixed-calibrated MiniRocket_aug macro '
 'test error ranges from 28.24% to 62.66% across these shifts. The condition-matched oracle also remains well above '
 '10%, because these forced-decision models cannot in general meet that target. This experiment supports some '
 'cross-type representation transfer, not reliable error control under arbitrary unseen corruptions.','',
 '**Burst-schedule scope.** Observed gaps are sampled by a multinomial allocation of the nonmissing positions, '
 'so burst locations concentrate near evenly spaced positions (the single long burst is near the middle in expectation). '
 'They are not uniform-start bursts. This explains the non-monotone prefix curves, and burst findings apply to '
 'this structured placement scheme. Both versions still remove exactly round(.30*L) samples, use causal forward '
 'fill, and never inspect future values. No corruption setting was changed after inspecting these results.','',
 '**Execution:** zero training or installations; 33.71 minutes inference-worker wall time across the three GPUs. '
 'All 1,300 condition records are present. The audit reproduces 10,400 selected decisions and 15,600 oracle '
 'threshold vectors; all 10,103 protected files are unchanged. One preflight import-path issue was fixed before '
 'evaluation and is preserved verbatim below.','']
insert=original.index('## Explicit per-type answers')
s.write_text(original[:insert]+'\n'.join(text)+'\n'+original[insert:])
dom=fs[fs.family.isin(['primary_fixed_60','other_prefixes_210'])].groupby(['method','baseline','condition']).agg(
 minimum_mean_gain=('error_reduction_mean','min'),maximum_mean_gain=('error_reduction_mean','max')).reset_index()
dom['mean_better_all_nine']=dom.minimum_mean_gain>=0
dom.to_csv(p/'dominance_by_condition.csv',index=False)
with (p/'review_complete.json').open('x') as f:json.dump({'figure_ticks_reviewed':True,'curves_and_CD_inspected':True,
 'corruptions_or_models_changed_after_inspection':False,'protected_files_unchanged':10103,'training_performed':False},f,indent=2)
print('SUMMARY_REVIEW_COMPLETE',len(s.read_text().splitlines()),'lines')
