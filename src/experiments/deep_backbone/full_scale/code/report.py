"""Generate the full-scale report without optional Markdown packages."""
import sys,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from settings import *
from analysis import *
def savej(path,x):
 with path.open('x') as f:json.dump(x,f,indent=2)
def run():
 fixed,front,cal,tr,meta=collect()
 fixed.to_csv(OUT/'deep_scale_fixedlength.csv',index=False)
 front.to_csv(OUT/'stopping_frontier_draws.csv',index=False);cal.to_csv(OUT/'calibration_earliness.csv',index=False)
 tr.to_csv(OUT/'training_times.csv',index=False)
 dataset_table=pd.DataFrame([{k:m[k] for k in ['dataset','train_size','test_size','length','n_classes','fit_count','validation_count','calibration_count']} for m in meta])
 dataset_table.to_csv(OUT/'dataset_sizes.csv',index=False)
 fd,fm=aggregate(fixed,['method','condition','checkpoint','nominal_fraction'])
 fm.to_csv(OUT/'fixedlength_macro.csv',index=False)
 per=fixed.groupby(['dataset','method','condition','checkpoint'])[METRICS].agg(['mean','std'])
 per.columns=['_'.join(x) for x in per.columns];per.reset_index().to_csv(OUT/'fixedlength_per_dataset.csv',index=False)
 fs=fixed_statistics(fixed);fs.to_csv(OUT/'deep_scale_fixedlength_stats.csv',index=False)
 choices=choose_calibration(cal);choices.to_csv(OUT/'fixedop_calibration_choices.csv',index=False)
 sel=select_frontiers(front,choices);sel.to_csv(OUT/'fixedop_selected_draws.csv',index=False)
 ss=stopping_statistics(sel);ss.to_csv(OUT/'deep_scale_fixedop_stats.csv',index=False)
 sd,sm=aggregate(sel,['method','condition','target']);sm.to_csv(OUT/'fixedop_macro.csv',index=False)
 ranks=diagrams(sel);ranks.to_csv(OUT/'cd_diagrams/friedman_nemenyi.csv',index=False)
 # Fixed-length figure: exact same observed prefixes within each dataset and no stop-policy selection.
 fig,axes=plt.subplots(2,4,figsize=(21,10),sharex=True,sharey=True)
 colors={'TCN_aug':'#d55e00','TCN_aug_mask':'#0072b2','MiniRocket_aug':'#111111','MiniRocket_clean':'#777777'}
 for ax,cond in zip(axes.flat,CONDITIONS):
  for model in FIXED_METHODS:
   q=fm[(fm.condition==cond)&(fm.method==model)].sort_values('checkpoint')
   x=q.nominal_fraction.to_numpy();a=q.accuracy_mean.to_numpy()*100;s=q.accuracy_std.to_numpy()*100
   name='MiniRocket augmented' if model=='MiniRocket_aug' else NAMES[model]
   ax.plot(x,a,label=name,color=colors[model],ls='--' if model=='MiniRocket_clean' else '-',lw=2)
   if model!='MiniRocket_clean':ax.fill_between(x,a-s,a+s,color=colors[model],alpha=.11)
  ax.set_title(cond);ax.set_xlabel('Observed prefix fraction');ax.set_ylabel('Accuracy (%)');ax.grid(alpha=.2);ax.tick_params(labelbottom=True)
 handles,labels=axes[0,0].get_legend_handles_labels();axes.flat[-1].axis('off');axes.flat[-1].legend(handles,labels,loc='center')
 fig.suptitle('Causal TCNs vs cached MiniRocket: 26 datasets, 5 paired draws; bands = sample SD')
 fig.tight_layout();fig.savefig(OUT/'accuracy_vs_prefix_deep_26.png',dpi=160);plt.close(fig)
 # Realized stopping earliness, with every condition-agnostic setting evaluated.
 _,fr=aggregate(front,['method','condition','setting'])
 fig,axes=plt.subplots(2,4,figsize=(21,10))
 for ax,cond in zip(axes.flat,CONDITIONS):
  for model in STOP_METHODS:
   q=fr[(fr.condition==cond)&(fr.method==model)].sort_values('earliness_mean')
   ax.plot(q.earliness_mean,q.error_mean,label=NAMES[model],marker='.',lw=1.8)
  ax.set_title(cond);ax.set_xlabel('Realized mean earliness');ax.set_ylabel('Error');ax.grid(alpha=.2)
 handles,labels=axes[0,0].get_legend_handles_labels();axes.flat[-1].axis('off');axes.flat[-1].legend(handles,labels,loc='center',fontsize=9)
 fig.suptitle('Corruption-agnostic stopping sweeps (no test-condition setting selection)')
 fig.tight_layout();fig.savefig(OUT/'stopping_frontiers_deep_26.png',dpi=160);plt.close(fig)
 # Dataset-level effects and pilot/new-dataset sensitivity, without changing the main N=26.
 diffs=[];cohort=[];clean_cost=[]
 for model in VARIANTS:
  a=fixed[fixed.method==model];b=fixed[fixed.method=='MiniRocket_aug']
  x=a.merge(b,on=['dataset','draw','condition','checkpoint'],suffixes=('_deep','_rocket'))
  x['error_reduction']=x.error_rocket-x.error_deep
  x['model']=model;diffs.append(x[['dataset','draw','condition','checkpoint','model','error_reduction']])
  for label,names in [('pilot_8',PILOT_DATASETS),('new_18',tuple(n for n in DATASETS if n not in PILOT_DATASETS))]:
   for cond in CONDITIONS:
    z=x[(x.dataset.isin(names))&(x.condition==cond)]
    for cp in [3,4,'all_prefixes']:
     zz=z if cp=='all_prefixes' else z[z.checkpoint==cp]
     dr=zz.groupby('draw').error_reduction.mean()
     cohort.append({'model':model,'cohort':label,'N':len(names),'condition':cond,
        'prefix':cp if isinstance(cp,str) else (cp+2)/10,'error_reduction_mean':dr.mean(),'error_reduction_sd':dr.std(ddof=1)})
  for cp in [3,4,'all_prefixes']:
   z=x[x.condition=='clean']
   if cp!='all_prefixes':z=z[z.checkpoint==cp]
   dr=-z.groupby('draw').error_reduction.mean()
   clean_cost.append({'model':model,'prefix':cp if isinstance(cp,str) else (cp+2)/10,
                      'error_increase_mean':dr.mean(),'error_increase_sd':dr.std(ddof=1)})
 effects=pd.concat(diffs,ignore_index=True);effects.to_csv(OUT/'paired_fixedlength_effects.csv',index=False)
 pd.DataFrame(cohort).to_csv(OUT/'pilot_new_dataset_sensitivity.csv',index=False)
 cc=pd.DataFrame(clean_cost);cc.to_csv(OUT/'clean_data_cost.csv',index=False)
 # Explicit condition verdicts at the prespecified primary checkpoints, and across the complete curve.
 decisions=[]
 for model in VARIANTS:
  for cond in CONDITIONS:
   q=fs[(fs.model==model)&(fs.baseline=='MiniRocket_aug')&(fs.condition==cond)]
   primary=q[q.family=='primary_fixed_28'];curve=q[q.family=='curve_average_14'].iloc[0]
   prefix=q[q.family.isin(['primary_fixed_28','exploratory_other_prefixes_98'])]
   decisions.append({'model':model,'condition':cond,
      'wins_both_primary_prefixes':bool((primary.error_reduction_mean>0).all()),
      'significant_wins_both_primary':bool(((primary.error_reduction_mean>0)&(primary.p_holm<.05)).all()),
      'mean_nonnegative_all_nine_prefixes':bool((prefix.error_reduction_mean>=0).all()),
      'curve_average_error_reduction':float(curve.error_reduction_mean),'curve_average_p_holm':float(curve.p_holm)})
 decision=pd.DataFrame(decisions);decision.to_csv(OUT/'condition_verdicts.csv',index=False)
 savej(OUT/'decision_summary.json',{'by_condition':decisions,'new_fits':int(tr.reused_from.isna().sum()),
       'pilot_reused':int(tr.reused_from.notna().sum()),'train_gpu_hours':float(tr.train_seconds.sum()/3600)})
 # Markdown report, focusing on the 3 requested questions and uncertainty.
 text=['# Full-scale learned ECTS backbone benchmark','',
 '**Scope:** 26 main datasets x 5 paired draws x 7 conditions x 9 checkpoints. Independent heads, lambda=0, '
 'with and without an observed/filled mask. Model architectures and training settings were frozen before testing. '
 'All previous runs were read-only; no MiniRocket, TEASER or ECEC model was retrained.','',
 'Each macro mean gives all 26 datasets equal weight within each draw, then averages the five draw means. '
 '+/- denotes sample SD (ddof=1) across those five paired draw means. Dataset-level significance uses each dataset\'s '
 'five-draw mean and ALWAYS N=26; no overlap filtering or test-error-based setting selection.','',
 '## Key answers','']
 for model in VARIANTS:
  q=decision[decision.model==model]
  noise=q[q.condition.str.startswith('noise_')];missing=q[q.condition.isin(['missing_30pct','missing_50pct'])]
  allcor=q[q.condition!='clean']
  text.append(f"- **{model}:** significant gains at BOTH 0.5/0.6 for noise 20/10/5 dB = "
      +str(dict(zip(noise.condition,noise.significant_wins_both_primary)))
      +"; missing 30/50% = "+str(dict(zip(missing.condition,missing.significant_wins_both_primary)))+".")
  text.append(f"  Mean error no higher at ALL nine checkpoints across ALL six corruptions: {bool(allcor.mean_nonnegative_all_nine_prefixes.all())}.")
 text+=['',
 'The fixed-checkpoint tests are the decisive equal-earliness evidence. A positive average effect, a statistically '
 'significant effect, and dominance at every checkpoint are distinct claims. The tables below report all three.','',
 '## Data, splits and method','',table(dataset_table),'',
 'The exact prior fit/validation/calibration indices and fit caps are retained. Some calibration sets remain smaller '
 'than 100; these data are kept in the main 26-dataset benchmark as requested. TwoLeadECG and ItalyPowerDemand are excluded.','',
 'Architecture: the pilot causal TCN, five residual blocks with two causal kernel-3 convolutions each, dilations '
 '1/2/4/8/16, 64 channels, dropout 0.1, per-timestep channel-only LayerNorm; causal 1/2/4-bin prefix pooling plus the '
 'last feature; nine independent linear heads. The local receptive field is 125 samples, while prefix pooling uses '
 'all observations up to the checkpoint. No future samples, time-axis normalization, or noncausal padding are used.','',
 'Training: exact pilot curriculum (two epochs per newly introduced checkpoint, then at most 22 joint epochs), '
 'Adam lr=0.001, weight decay=0.0001, gradient clip=1, mean checkpoint CE, no consistency, no class reweighting '
 '(matching the previous unweighted logistic and pilot settings). Early stopping on unchanged clean+mixed validation '
 'CE, with at least six joint epochs; fit retains the original clean examples plus one fixed augmented copy.','',
 'The value channel uses the unchanged causal running normalization, including forward-filled positions as before. '
 'The binary mask is NOT normalized: 1 for acquired observations (including noisy observations), 0 for missing/imputed '
 'positions. It does not identify noise levels or class labels. Masks are reconstructed from the exact seeded corruption '
 'generator, rather than guessed from equal neighboring values. Both variants see identical signal arrays. '
 'The mask model adds 64 zero-initialized first-layer weights; other initial parameters and RNG states are paired. '
 'Only the mask variant receives acquisition flags; any mask gain is therefore an information/representation gain, '
 'not evidence that the same signal-only input suffices.','',
 f"Training artifacts: {int(tr.reused_from.notna().sum())} compatible pilot models reused and {int(tr.reused_from.isna().sum())} new fits. "
 'The reused 24 models have identical split, augmentation, validation and calibration hashes; five previous test conditions '
 'reuse their saved predictions. The two newly requested conditions are evaluated using those frozen checkpoints. '
 'All other corrupted streams use the unchanged deterministic generators; raw corrupted arrays were not stored by scale_v1.','',
 '## Fixed-prefix gains and Wilcoxon-Holm (N=26)','',
 'Positive margins mean lower TCN error than augmented MiniRocket, in percentage points. The primary Holm family '
 'contains 28 tests: two TCN variants x seven conditions x prefixes 0.5/0.6. '
 'The other 98 checkpoint tests, 14 all-prefix-average tests, and 14 mask-versus-plain tests each have separate declared families.','']
 rows=[]
 for r in fs[fs.family=='primary_fixed_28'].itertuples(index=False):
  rows.append({'model':r.model,'condition':r.condition,'prefix':r.setting,
    'error reduction pp +/- SD':pm(r.error_reduction_mean,r.error_reduction_sd),
    'W/T/L':f'{r.wins}/{r.ties}/{r.losses}','p_raw':r.p_raw,'p_holm':r.p_holm})
 text += [table(pd.DataFrame(rows)),'','All nine-prefix and curve-average results:','',table(decision),'',
 '## Accuracy and error at the two exact fixed checkpoints','']
 rows=[]
 for r in fm[fm.checkpoint.isin([3,4])].itertuples(index=False):
  rows.append({'method':r.method,'condition':r.condition,'prefix':r.nominal_fraction,
               'accuracy % +/- SD':pm(r.accuracy_mean,r.accuracy_std),
               'error % +/- SD':pm(r.error_mean,r.error_std),'earliness':r.earliness_mean,
               'balanced accuracy % +/- SD':pm(r.balanced_accuracy_mean,r.balanced_accuracy_std)})
 text += [table(pd.DataFrame(rows)),'','## Does adding the mask help?','']
 rows=[]
 for r in fs[fs.family=='mask_incremental_fixed_14'].itertuples(index=False):
  rows.append({'condition':r.condition,'prefix':r.setting,'mask error reduction pp +/- SD':pm(r.error_reduction_mean,r.error_reduction_sd),
               'W/T/L':f'{r.wins}/{r.ties}/{r.losses}','p_holm':r.p_holm})
 text += [table(pd.DataFrame(rows)),'',
 'This direct paired mask-versus-plain comparison measures the incremental flag benefit. Beating the unmasked TCN '
 'does not by itself show that the mask TCN beats augmented MiniRocket; use the primary table for that second question.','',
 '## Deployable mixed-calibration-fixed stopping','',
 'For each dataset/method, choose one swept setting whose FIVE-DRAW MEAN mixed-calibration earliness is closest to '
 '0.5 or 0.6, using earliness only. That setting is then fixed across draws and every test condition; its per-draw '
 'thresholds are calibrated on that draw\'s mixed calibration set. Both TCNs and augmented MiniRocket use the same '
 'sequential empirical calibration algorithm and alpha grid {0.02,0.05,0.10,0.15,0.20,0.30}, forcing a final decision. '
 'The clean baselines retain their cached probability/TEASER/ECEC sweeps: aeon TEASER uses the existing MiniRocket checkpoint estimator, and ECEC is the prior in-pipeline adaptation rather than a new official-code reproduction. This is NOT a formal error guarantee under '
 'distribution shift. No condition-matched oracle calibration is used.','',
 '**Matching limit:** realized TEST earliness can differ even when calibration targets match. These fixed-operating-point '
 'tests use all 26 datasets but are not substituted for exact fixed-prefix comparisons. The error margin may partly '
 'reflect unequal waiting; per-dataset absolute gaps and realized mean earliness are reported below.','']
 rows=[]
 for r in ss[ss.family=='fixedop_112'].itertuples(index=False):
  rows.append({'model':r.model,'baseline':r.baseline,'condition':r.condition,'target':r.setting,
     'error reduction pp +/- SD':pm(r.error_reduction_mean,r.error_reduction_sd),
     'TCN earliness':r.model_earliness,'baseline earliness':r.baseline_earliness,
     'mean |dataset gap|':r.mean_absolute_dataset_earliness_gap,'N gap <=.05':r.N_gap_within_005,
     'W/T/L':f'{r.wins}/{r.ties}/{r.losses}','p_raw':r.p_raw,'p_holm':r.p_holm})
 text += [table(pd.DataFrame(rows)),'',
 'Holm correction for this table spans all 112 comparisons (2 TCNs x 4 baselines x 7 conditions x 2 targets), '
 'matching the global-family convention of select_v1. The separate mask-versus-plain stopping family has 14 tests.',
 '', 'Realized mean accuracy/error/earliness for all six methods:','']
 rows=[]
 for r in sm.itertuples(index=False):
  rows.append({'method':r.method,'condition':r.condition,'target':r.target,'accuracy % +/- SD':pm(r.accuracy_mean,r.accuracy_std),
               'error % +/- SD':pm(r.error_mean,r.error_std),'earliness +/- SD':pm(r.earliness_mean,r.earliness_std,1)})
 text += [table(pd.DataFrame(rows)),'','## Nemenyi diagrams','',
 'Each diagram uses all 26 datasets and all six stopping methods at one mixed-calibration target. '
 'The Nemenyi CD controls pairwise rank comparisons within that diagram at 0.05; omnibus Friedman p-values and '
 'mean ranks are below. These ranks inherit the realized-earliness matching limitation.','',
 table(ranks),'']
 for cond in CONDITIONS:
  for target in TARGETS:text.append(f'![{cond}, target {target}](cd_diagrams/{cond}_target{target}.png)')
 text += ['','## Clean-data cost','',
 'Positive values are increased clean error versus augmented MiniRocket, in percentage points.','']
 view=cc.copy();view['clean error cost pp +/- SD']=[pm(a,b) for a,b in zip(view.error_increase_mean,view.error_increase_sd)]
 text += [table(view[['model','prefix','clean error cost pp +/- SD']]),'',
 '## Per-dataset heterogeneity and confirmation cohort','',
 'The main aggregate includes the eight pilot datasets and eighteen additional datasets. It is not a completely new '
 'test-dataset collection; model architecture selection used the pilot validation results. The new-18 sensitivity table '
 'below shows whether the average effect persists outside the pilot datasets, without replacing the requested N=26 tests.','']
 cv=pd.DataFrame(cohort);cv=cv[(cv.cohort=='new_18')&(cv.prefix=='all_prefixes')].copy()
 cv['error reduction pp +/- SD']=[pm(a,b) for a,b in zip(cv.error_reduction_mean,cv.error_reduction_sd)]
 text += [table(cv[['model','condition','N','error reduction pp +/- SD']]),'',
 'Per-dataset mean error reduction (percentage points), averaged across all checkpoints/draws:','']
 for model in VARIANTS:
  ds=effects[effects.model==model].groupby(['dataset','condition']).error_reduction.mean().unstack().reindex(columns=CONDITIONS)*100
  text += ['',model,'',table(ds.reset_index())]
 status=loadj(OUT/'orchestration_status.json')
 text += ['','## Runtime and audits','',
 f"All three RTX 8000 GPUs used. Current-run training + validation occupancy: {tr.train_seconds.sum()/3600:.3f} GPU-hours. "
 f"Parallel worker wall time: {status['wall_seconds']/3600:.3f} hours. This excludes subsequent reporting/transfer. "
 'Reused pilot training time is not charged again.','',
 'All original sources/data/runs are hash-audited in source_audit.json. Artifact completeness, mask pairing, zero future-input '
 'gradients, suffix-perturbation equivalence, and pilot reuse are checked in artifact_audit.json. '
 'No datasets were removed and no prior result was overwritten. Error tracebacks, if any, are preserved in logs/.','',
 'Files: deep_scale_fixedlength.csv contains all dataset/draw/checkpoint records; fixedlength_macro.csv and '
 'fixedlength_per_dataset.csv contain means/SDs. deep_scale_fixedlength_stats.csv contains exact-prefix tests; '
 'deep_scale_fixedop_stats.csv contains the deployable fixed-setting tests. All trained models, compact predictions, '
 'calibration probabilities, splits/provenance, code and logs are retained.','']
 with (OUT/'summary.md').open('x') as f:f.write('\n'.join(text)+'\n')
 print('REPORT_DONE','fixed_rows',len(fixed),'fixedop_tests',len(ss),'GPU_hours',tr.train_seconds.sum()/3600,flush=True)
if __name__=='__main__':run()
