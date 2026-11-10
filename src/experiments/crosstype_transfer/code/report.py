import json
import numpy as np,pandas as pd
from settings import *
def table(df):
 def fmt(x):
  if isinstance(x,(float,np.floating)):return f'{x:.4f}'
  return str(x).replace('|','/').replace('\n',' ')
 return '\n'.join(['| '+' | '.join(map(str,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']+
  ['| '+' | '.join(fmt(x) for x in r)+' |' for r in df.itertuples(index=False,name=None)])
def pm(a,b,scale=100):return f'{a*scale:.2f} +/- {b*scale:.2f}'
def gains(df,stopping=False):
 rows=[]
 for r in df.itertuples(index=False):
  q={'condition':r.condition,'model':r.method,'baseline':r.baseline,'time/target':r.setting,
   'gain pp +/- SD':pm(r.error_reduction_mean,r.error_reduction_sd),'W/T/L':f'{r.wins}/{r.ties}/{r.losses}',
   'p_raw':r.p_raw,'p_holm':r.p_holm,'rank_biserial':r.rank_biserial}
  if stopping:q.update(model_time=r.model_earliness,baseline_time=r.baseline_earliness,
    mean_absolute_dataset_time_gap=r.mean_absolute_dataset_earliness_gap,N_gap_within_005=r.N_gap_within_005)
  rows.append(q)
 return table(pd.DataFrame(rows))
def main():
 fs=pd.read_csv(OUT/'fixed_length_stats.csv');ss=pd.read_csv(OUT/'crosstype_fixed_op_stats.csv')
 sm=pd.read_csv(OUT/'fixed_op_macro.csv');tm=pd.read_csv(OUT/'crosstype_transfer_gap.csv')
 rank=pd.read_csv(OUT/'friedman_nemenyi.csv');primary=fs[fs.family=='primary_fixed_60']
 status=json.loads((OUT/'orchestration_status.json').read_text());verdict=[]
 text=['# Frozen ECTS models: unseen-corruption transfer','',
 '**Scope:** 26 main datasets x five paired draws x ten unseen settings x nine causal checkpoints. No retraining. '
 'New corruptions and analysis families were fixed before inspecting their results. All previous runs were read-only.','',
 'Macro means first average all 26 datasets within each draw; +/- denotes sample SD (ddof=1) over five draw means. '
 'Significance uses 26 paired dataset-level five-draw means, two-sided Wilcoxon, with globally Holm-corrected families. '
 'A significant gain requires positive mean gain AND positive signed-rank direction. Rank-biserial effects and nonzero-pair counts are retained.','',
 '## Explicit per-type answers','']
 for typ in ('spike','warp','quantization','drift','burst'):
  aug=primary[(primary.type==typ)&(primary.method=='MiniRocket_aug')&(primary.baseline=='MiniRocket_clean')]
  deep=primary[(primary.type==typ)&(primary.method=='TCN_aug')&(primary.baseline=='MiniRocket_aug')]
  nw=int(((aug.error_reduction_mean>0)&(aug.rank_biserial>0)&(aug.p_holm<.05)).sum());nd=int(((deep.error_reduction_mean>0)&(deep.rank_biserial>0)&(deep.p_holm<.05)).sum())
  tr=tm[(tm.type==typ)&(tm.method=='MiniRocket_aug')&(tm.diagnostic=='oracle_same_alpha')]
  trtime=tm[(tm.type==typ)&(tm.method=='MiniRocket_aug')&(tm.diagnostic=='oracle_calibration_time_matched')]
  joint=int(tr.small_joint_macro_gap.sum());jointtime=int(trtime.small_joint_macro_gap.sum())
  verdict.append({'type':typ,'aug_significant_wins_of_4':nw,'TCN_vs_aug_significant_wins_of_4':nd,
   'same_alpha_small_joint_gaps_of_4':joint,'oracle_cal_time_small_joint_gaps_of_4':jointtime,
   'mean_aug_error_reduction':float(aug.error_reduction_mean.mean()),
   'mean_same_alpha_error_gap':float(tr.error_gap_mean.mean()),'mean_same_alpha_time_gap':float(tr.earliness_gap_mean.mean())})
  text.append(f'- **{typ}:** augmented MiniRocket beats the clean backbone significantly at **{nw}/4** primary exact-time '
   f'comparisons (two severities x checkpoints .5/.6); mean error reduction {aug.error_reduction_mean.mean()*100:+.2f} pp. '
   f'TCN beats augmented MiniRocket significantly at {nd}/4. The mixed/oracle same-alpha gap is jointly small at {joint}/4 '
   f'operating comparisons; calibration-time-matched oracle gives {jointtime}/4. Mean same-alpha error gap '
   f'{tr.error_gap_mean.mean()*100:+.2f} pp and decision-fraction gap {tr.earliness_gap_mean.mean():+.3f}.')
 n=int(((primary.method=='MiniRocket_aug')&(primary.baseline=='MiniRocket_clean')&(primary.error_reduction_mean>0)&(primary.rank_biserial>0)&(primary.p_holm<.05)).sum())
 positive=int((primary[(primary.method=='MiniRocket_aug')&(primary.baseline=='MiniRocket_clean')].error_reduction_mean>0).sum())
 text+=['',f'**Overall:** augmentation has lower mean error at {positive}/20 primary matched checkpoints, with '
 f'{n}/20 significant wins. Generalization is condition-specific; universal superiority and universal calibration transfer '
 'are not implied. The per-type counts and the signed tables below determine the honest conclusion.','',
 'Clean Probability/TEASER/ECEC use the same frozen clean backbone. At forced checkpoints, that common backbone is the '
 'equal-earliness reference; actual baseline stopping policies are evaluated separately.','',
 '## Corruption specification and causal checks','',
 '| Type | Frozen settings in archived normalized units |','| --- | --- |',
 '| Spike | Bernoulli fractions .05/.15, independent random-sign additive amplitude 5 |',
 '| Warp | One-sided monotone causal clock, speed amplitude .4/.8, independent phase/frequency; source index <= current time |',
 '| Quantization | 8/4 equally spaced levels on fixed [-3,3], with clipping; no per-series range estimation |',
 '| Drift | Signed linear ramp from zero to .5/1.5 units over declared horizon |',
 '| Burst | Exact round(.30*L) missing: three shorter runs vs one long run; forward fill, initial holes=0 |','',
 'Both burst settings remove 30%; severity here changes contiguity. Burst tests an unseen temporal missingness pattern, '
 'rather than a wholly new measurement mechanism. Each random mechanism uses per-series, '
 'dataset/draw/partition/type seeds, shared across severity. Calibration and test seeds are independent. '
 'Quantization is deterministic across draws; clean-model outputs are therefore identical across those draws.','',
 'Corrupt raw archive values, then apply the unchanged prefix-only running mean/std normalization. The pre-existing '
 'archive normalization is retained. The known horizon sets schedules but no future signal values are read. '
 'Warp interpolation reads only acquired samples; this one-sided warp differs from an offline spline warp. '
 'Future-suffix perturbation, truncated-prefix reproduction and normalization audits pass for all ten settings. '
 'Every dataset receives an additional TCN inference-prefix equivalence check; the prior audit already covers all 130 frozen TCN checkpoints.','',
 '## Exact-time error comparisons (N=26)','',
 'Positive gain = baseline error minus model error, in percentage points. Primary Holm family: 60 tests '
 '(three pairings x ten conditions x checkpoints .5/.6). Other checkpoints: separate 210-test exploratory family. '
 'All-nine-prefix averages: separate 30-test family.','']
 for frac in (.5,.6):
  text += [f'Checkpoint {frac}:','',gains(primary[np.isclose(pd.to_numeric(primary.setting),frac)]),'']
 text+=['All-nine-prefix averages:','',gains(fs[fs.family=='curve_average_30']),'',
 '![Accuracy curves](accuracy_vs_prefix_crosstype.png)','',
 '## Deployable operating points fixed from original mixed calibration','',
 'Every per-dataset setting is copied exactly from deep_scale_v1: five-draw mean MIXED calibration earliness '
 'closest to .5/.6, held across every unseen condition. Saved per-draw thresholds, clean TEASER safety models and '
 'ECEC rates are reused. No new type, test confidence, test label or test condition enters deployed policy selection. '
 'MiniRocket_aug and TCN_aug retain the six-alpha sweep and force a decision at full length. TEASER is the saved aeon '
 'rule on clean MiniRocket probabilities. ECEC is the prior in-pipeline adaptation, not a new official reproduction.','',
 '**Matching limit:** equal calibration targets do not imply equal realized test earliness. These N=26 stopping '
 'comparisons retain every dataset, report actual waiting differences, and do not replace exact-prefix evidence. '
 'Holm correction spans all 140 stopping tests (two augmented models vs three clean baselines plus TCN vs augmented '
 'MiniRocket, x ten conditions x two targets).','',gains(ss,True),'',
 'Macro stopping metrics:','']
 rows=[]
 for r in sm.itertuples(index=False):
  rows.append({'condition':r.condition,'target':r.target,'method':r.method,
   'accuracy % +/- SD':pm(r.accuracy_mean,r.accuracy_std),'balanced acc % +/- SD':pm(r.balanced_accuracy_mean,r.balanced_accuracy_std),
   'error % +/- SD':pm(r.error_mean,r.error_std),'time +/- SD':pm(r.earliness_mean,r.earliness_std,1),
   'forced % +/- SD':pm(r.forced_fraction_mean,r.forced_fraction_std)})
 text += [table(pd.DataFrame(rows)),'','![Operating points](fixedop_crosstype.png)','',
 '## Oracle transfer gaps: diagnostic only','',
 'Type-and-severity-matched oracle thresholds use only the original calibration labels, with independently corrupted calibration '
 'streams. They are never used by a deployable method. A matched oracle is a reference, not necessarily an upper bound. '
 'We report same-alpha oracle; oracle alpha selected by five-draw mean type-matched CALIBRATION earliness; '
 'and common alpha=.10. Error gap = mixed minus oracle; time gap = mixed minus oracle. A lower error can reflect '
 'longer waiting. A small gap does not mean either rule attains alpha=.10: actual errors and forced fractions are reported. '
 'The predeclared descriptive small-gap criterion requires BOTH absolute error gap <=2 pp and '
 'absolute decision-fraction gap <=.05. These are MACRO gaps; the additional mean absolute per-dataset gaps show '
 'heterogeneity without cancellation. A macro-small gap does not imply each dataset transfers well. Shifted reliability '
 'is not formally guaranteed.','']
 for diag in ('oracle_same_alpha','oracle_calibration_time_matched','fixed_alpha_0.10'):
  rows=[]
  for r in tm[tm.diagnostic==diag].itertuples(index=False):
   rows.append({'condition':r.condition,'model':r.method,'target':r.target if r.target>0 else 'alpha=.10',
    'mixed error % +/- SD':pm(r.mixed_error_mean,r.mixed_error_std),'oracle error % +/- SD':pm(r.oracle_error_mean,r.oracle_error_std),
    'gap pp +/- SD':pm(r.error_gap_mean,r.error_gap_std),'mixed time':r.mixed_earliness_mean,'oracle time':r.oracle_earliness_mean,
    'time gap +/- SD':pm(r.earliness_gap_mean,r.earliness_gap_std,1),
    'mixed forced %':r.mixed_forced_fraction_mean*100,'oracle forced %':r.oracle_forced_fraction_mean*100,
    'mean absolute dataset error gap pp':r.mean_absolute_dataset_error_gap*100,
    'mean absolute dataset time gap':r.mean_absolute_dataset_earliness_gap,'small joint macro gap':r.small_joint_macro_gap})
  text += [diag,'',table(pd.DataFrame(rows)),'']
 text += ['## N=26 Nemenyi diagrams','',
 'Five stopping methods, all 26 datasets, both calibration targets. Nemenyi alpha=.05 controls within each panel; '
 'rank comparisons inherit realized waiting-time differences.','',table(rank),'']
 for c in CONDITIONS:
  for target in TARGETS:text.append(f'![{c}, target {target}](cd_diagrams/{c}_target{target}.png)')
 text += ['','## Datasets and per-dataset effects','',table(pd.read_csv(DEEP/'dataset_sizes.csv')),'',
 'Same splits and fit caps as before; small calibration counts in some main datasets remain a limitation. '
 'TwoLeadECG/ItalyPowerDemand are excluded from the main 26.','']
 per=pd.read_csv(OUT/'fixed_length_per_dataset.csv')
 for model,baseline in [('MiniRocket_aug','MiniRocket_clean'),('TCN_aug','MiniRocket_aug')]:
  p=per[per.method.isin([model,baseline])].groupby(['dataset','condition','method']).error_mean.mean().unstack()
  view=((p[baseline]-p[model])*100).unstack().reindex(index=DATASETS,columns=CONDITIONS)
  text += [f'{model} vs {baseline}, error reduction pp averaged over nine checkpoints:','',table(view.reset_index()),'']
 text += ['## Runtime and audit','',
 f"Parallel worker wall time {status['wall_seconds']/60:.2f} minutes. Frozen TCN inference on three RTX 8000 GPUs; "
 'MiniRocket transforms on CPU. No fitting or package installation. Normalized stream hashes, compact test predictions, '
 'full oracle calibration probabilities, policy settings and all draw metrics are retained. Original files are '
 'checked by a before/after SHA-256 audit.','']
 if (OUT/'logs/setup_issues.txt').exists():text += ['Preflight issue corrected before evaluation:','',chr(96)*3+'text',
  (OUT/'logs/setup_issues.txt').read_text().strip(),chr(96)*3,'']
 text += ['Inference worker errors: '+json.dumps(status['errors']), '',
 'See fixed_length_stats.csv for exact-time tests; crosstype_fixed_op_stats.csv for deployable stopping; '
 'crosstype_transfer_gap.csv for macro transfer gaps; transfer_gap_draws.csv for every dataset/draw diagnostic.','']
 with (OUT/'summary.md').open('x') as f:f.write('\n'.join(text)+'\n')
 pd.DataFrame(verdict).to_csv(OUT/'type_verdicts.csv',index=False)
 print('REPORT_COMPLETE',flush=True)
if __name__=='__main__':main()
