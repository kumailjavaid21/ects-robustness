"""Decision-focused pilot report. All variant/stop selection uses validation/calibration only."""
import sys,json,math,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from settings import *
METRICS=('accuracy','error','balanced_accuracy','macro_f1','earliness','combined')
NAMES={'A_l0':'Shared, lambda=0','B_l0':'Progressive, lambda=0','C_l0':'Independent, lambda=0',
 'A_l01':'Shared, lambda=0.1','B_l01':'Progressive, lambda=0.1','C_l01':'Independent, lambda=0.1',
 'A_clean':'Shared, clean training','MiniRocket_aug':'MiniRocket augmented',
 'MiniRocket_clean':'MiniRocket clean','MiniRocket_aug_probability':'MiniRocket aug + probability',
 'MiniRocket_aug_mixed':'MiniRocket aug + mixed-cal','MiniRocket_clean_probability':'MiniRocket clean + probability'}
def write(path,s):
 if path.exists() and path.read_text()==s:return
 with path.open('x') as f:f.write(s)
def jread(p):return json.loads(p.read_text())
def sd(x):return float(np.std(x,ddof=1))
def mpm(x,percent=True):
 x=np.asarray(x,float);factor=100 if percent else 1
 return f'{x.mean()*factor:.2f} +/- {sd(x)*factor:.2f}'
def table(frame):
 def value(x):
  if isinstance(x,(float,np.floating)):return f'{x:.4f}'
  return str(x).replace('|','\\|').replace('\\n',' ')
 lines=['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
 lines += ['| '+' | '.join(value(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None)]
 return '\n'.join(lines)
def holm(p):
 p=np.asarray(p);order=np.argsort(p);out=np.empty(len(p));v=0.
 for i,j in enumerate(order):v=max(v,(len(p)-i)*p[j]);out[j]=min(1.,v)
 return out

def collect():
 rows=[];training=[];meta=[]
 for name in DATASETS:
  meta.append(jread(SCALE/'draw_meta'/f'{name}_base.json'))
  for draw in DRAWS:
   for tag,mode,lam,aug in VARIANTS:
    rec=jread(OUT/'records'/f'{name}_draw{draw}_{tag}.json')
    train={k:v for k,v in rec['training'].items() if k not in ('history','inheritance_events','causality_audit')}
    train['causality_passed']=rec['training']['causality_audit']['passed']
    train['inheritance_events']=len(rec['training']['inheritance_events'])
    training.append(train)
    for row in rec['rows']:
     rr=dict(row)
     if rr['kind']=='calibration_target':rr['kind']='calibration_target_dense'
     rows.append({'dataset':name,'draw':draw,'method':tag,**rr})
    # Primary matched-calibration comparison uses exactly the cached baseline's seven taus.
    shared_taus=(.5,.6,.7,.8,.9,.95,.99)
    eligible=[r for r in rec['calibration_frontier'] if r['tau'] in shared_taus]
    for target in TARGETS:
     ch=min(eligible,key=lambda a:(abs(a['earliness']-target),a['tau']))
     for cond in CONDITIONS:
      rr=next(r for r in rec['rows'] if r['kind']=='frontier' and r['condition']==cond and abs(r['setting']-ch['tau'])<1e-12)
      rows.append({'dataset':name,'draw':draw,'method':tag,**rr,'kind':'calibration_target',
         'setting':target,'selected_tau':ch['tau'],'calibration_earliness':ch['earliness']})
   cm=jread(ROOT/'runs/ects_select_v1/draw_meta'/f'{name}_draw{draw}.json')['calibration_earliness']
   for cond in CONDITIONS:
    rec=jread(SCALE/'records'/f'{name}_draw{draw}_{cond}.json')
    for j,r in enumerate(rec['fixed']):
     for label,key in [('MiniRocket_aug','augmented'),('MiniRocket_clean','clean')]:
      rows.append({'dataset':name,'draw':draw,'condition':cond,'method':label,'kind':'fixed',
                   'setting':j,'nominal_fraction':(j+2)/10,**r[key]})
    for label,key in [('MiniRocket_clean_probability','probability'),('MiniRocket_aug_probability','aug_probability'),
                      ('MiniRocket_aug_mixed','ours')]:
     for r in rec['frontiers'][key]:
      rows.append({'dataset':name,'draw':draw,'condition':cond,'method':label,'kind':'frontier',**r})
     for target in TARGETS:
      chosen=min(cm[key],key=lambda a:(abs(a['earliness']-target),a['setting']))
      rr=next(r for r in rec['frontiers'][key] if abs(r['setting']-chosen['setting'])<1e-12)
      rows.append({'dataset':name,'draw':draw,'condition':cond,'method':label,'kind':'calibration_target',
         **rr,'setting':target,'selected_tau':chosen['setting'],'calibration_earliness':chosen['earliness']})
 return pd.DataFrame(rows),pd.DataFrame(training),meta

def aggregate(df):
 keys=['method','kind','condition','setting']
 draw=df.groupby(keys+['draw'],dropna=False)[list(METRICS)].mean().reset_index()
 agg=draw.groupby(keys,dropna=False)[list(METRICS)].agg(['mean','std'])
 agg.columns=['_'.join(x) for x in agg.columns]
 return draw,agg.reset_index()

def paired(df,one,two,kind,setting=None):
 x=df[(df.method==one)&(df.kind==kind)]
 y=df[(df.method==two)&(df.kind==kind)]
 if setting is not None:x=x[np.isclose(x.setting,setting)];y=y[np.isclose(y.setting,setting)]
 key=['dataset','draw','condition','setting']
 return x.merge(y,on=key,suffixes=('_one','_two'))

def run():
 df,tr,meta=collect();tr.to_csv(OUT/'training_times.csv',index=False)
 # This selection uses ONLY held-out validation accuracies. No condition/test statistic is consulted.
 v=tr[tr.augmentation].groupby('variant').best_val_accuracy.mean().sort_values(ascending=False,kind='stable')
 best=v.index[0];write(OUT/'validation_selection.json',json.dumps({'primary_variant':best,'criterion':'mean held-out clean+mixed validation accuracy across checkpoints, datasets and draws',
       'ranking':v.to_dict(),'test_labels_used':False},indent=2))
 df.to_csv(OUT/'deep_pilot_metrics.csv',index=False)
 draw,macro=aggregate(df);macro.to_csv(OUT/'macro_metrics.csv',index=False)
 per=df.groupby(['dataset','method','kind','condition','setting'])[list(METRICS)].agg(['mean','std'])
 per.columns=['_'.join(x) for x in per.columns];per.reset_index().to_csv(OUT/'per_dataset_metrics.csv',index=False)
 # Fixed checkpoint gaps: positive means deep has lower error.
 pairs=paired(df,best,'MiniRocket_aug','fixed')
 gaps=[];tests=[]
 for (cond,j),g in pairs.groupby(['condition','setting']):
  d=(g.assign(margin=g.error_two-g.error_one).groupby('draw').margin.mean()).reindex(DRAWS)
  gd=g.assign(margin=g.error_two-g.error_one).groupby('dataset').margin.mean()
  gaps.append({'condition':cond,'nominal_fraction':(j+2)/10,'error_reduction_mean':d.mean(),
               'error_reduction_sd':sd(d),'wins':int((gd>1e-12).sum()),'ties':int((abs(gd)<=1e-12).sum()),
               'losses':int((gd < -1e-12).sum()),'datasets':len(gd)})
  if int(j) in (3,4):
   p=float(wilcoxon(gd,alternative='two-sided',zero_method='wilcox',method='auto').pvalue) if np.any(abs(gd)>1e-12) else 1.
   tests.append({'condition':cond,'nominal_fraction':(j+2)/10,'comparison':best+' vs MiniRocket_aug',
                 'N':len(gd),'margin':gd.mean(),'p_raw':p})
 gaps=pd.DataFrame(gaps);gaps.to_csv(OUT/'fixed_length_gaps.csv',index=False)
 tests=pd.DataFrame(tests);tests['p_holm']=holm(tests.p_raw);tests.to_csv(OUT/'pilot_significance.csv',index=False)
 # Honest stopping comparison: match calibration earliness, retain actual test mismatch.
 stops=[];matched=paired(df,best,'MiniRocket_aug_probability','calibration_target')
 for (cond,target),g in matched.groupby(['condition','setting']):
  m=(g.error_two-g.error_one).groupby(g.draw).mean();e1=g.groupby('draw').earliness_one.mean();e2=g.groupby('draw').earliness_two.mean()
  dsg=g.groupby('dataset')[['earliness_one','earliness_two']].mean()
  stops.append({'condition':cond,'target':target,'deep_earliness':e1.mean(),'baseline_earliness':e2.mean(),
     'mean_absolute_dataset_earliness_gap':float(abs(dsg.earliness_one-dsg.earliness_two).mean()),
     'error_reduction_mean':m.mean(),'error_reduction_sd':sd(m),
     'N_within_005_earliness':int((abs(dsg.earliness_one-dsg.earliness_two)<=.05).sum())})
 stops=pd.DataFrame(stops);stops.to_csv(OUT/'stopping_target_margins.csv',index=False)
 # Ablations, average all nine matched checkpoints within each dataset.
 ab=[]
 for one,two,label in [('A_l0','C_l0','shared minus independent, lambda=0'),
  ('B_l0','C_l0','progressive minus independent, lambda=0'),
  ('A_l01','C_l01','shared minus independent, lambda=0.1'),
  ('B_l01','C_l01','progressive minus independent, lambda=0.1'),
  ('A_l01','A_l0','consistency effect, shared'),
  ('B_l01','B_l0','consistency effect, progressive'),
  ('C_l01','C_l0','consistency effect, independent'),
  ('A_l0','A_clean','augmentation effect, shared lambda=0')]:
  pair=paired(df,one,two,'fixed')
  for cond,g in pair.groupby('condition'):
   margins=(g.accuracy_one-g.accuracy_two).groupby(g.draw).mean()
   ab.append({'comparison':label,'condition':cond,'accuracy_gain_mean':margins.mean(),'accuracy_gain_sd':sd(margins)})
 ab=pd.DataFrame(ab);ab.to_csv(OUT/'ablation_effects.csv',index=False)
 # Plots: seven conditions are NOT silently pooled; every displayed difference is at the same prefix.
 fig,axes=plt.subplots(2,3,figsize=(18,10),sharex=True,sharey=True)
 palette={'A_l0':'#006d77','B_l0':'#e76f51','C_l0':'#7b2cbf',
          'A_l01':'#006d77','B_l01':'#e76f51','C_l01':'#7b2cbf',
          'A_clean':'#777777','MiniRocket_aug':'#111111','MiniRocket_clean':'#3a86ff'}
 for ax,cond in zip(axes.flat,CONDITIONS):
  for method in [x[0] for x in VARIANTS]+['MiniRocket_aug','MiniRocket_clean']:
   q=macro[(macro.kind=='fixed')&(macro.condition==cond)&(macro.method==method)].sort_values('setting')
   xs=(q.setting+2)/10;ys=q.accuracy_mean*100
   ax.plot(xs,ys,label=NAMES[method]+(' [validation winner]' if method==best else ''),
       color=palette[method],ls='--' if method.endswith('l01') else ':'
       if method=='A_clean' else '-',lw=2.5 if method in (best,'MiniRocket_aug') else 1.3)
   if method in (best,'MiniRocket_aug'):
    ax.fill_between(xs,ys-100*q.accuracy_std,ys+100*q.accuracy_std,color=palette[method],alpha=.13)
  ax.set_title(cond);ax.grid(alpha=.2);ax.set_xlabel('Observed prefix fraction');ax.set_ylabel('Accuracy (%)')
 handles,labels=axes[0,0].get_legend_handles_labels();axes.flat[-1].axis('off')
 axes.flat[-1].legend(handles,labels,loc='center',fontsize=10)
 fig.suptitle('Causal TCN vs cached MiniRocket: 8 datasets, 3 paired draws; shaded bands = sample SD')
 fig.tight_layout();fig.savefig(OUT/'accuracy_vs_prefix_deep.png',dpi=160);plt.close(fig)
 # All stop sweeps are condition-agnostic. The plot is evaluation, not condition-based selection.
 fig,axes=plt.subplots(2,3,figsize=(17,10))
 for ax,cond in zip(axes.flat,CONDITIONS):
  for meth in [best,'MiniRocket_aug_probability','MiniRocket_aug_mixed','MiniRocket_clean_probability']:
   q=macro[(macro.kind=='frontier')&(macro.condition==cond)&(macro.method==meth)].sort_values('earliness_mean')
   ax.plot(q.earliness_mean,q.error_mean,marker='.',label=NAMES[meth])
  ax.set_title(cond);ax.set_xlabel('Mean fraction observed');ax.set_ylabel('Error');ax.grid(alpha=.2)
 handles,labels=axes[0,0].get_legend_handles_labels();axes.flat[-1].axis('off');axes.flat[-1].legend(handles,labels,loc='center')
 fig.suptitle('Stopping sweeps (deep grid is denser); no test-condition tuning');fig.tight_layout()
 fig.savefig(OUT/'earliness_vs_error_deep.png',dpi=160);plt.close(fig)
 text=['# Learned anytime ECTS: three-GPU methodological pilot','',
 'All prior runs were read-only. Eight datasets, draws 0/1/2, five test conditions, nine checkpoints (20%-100%). '
 'Each reported mean first averages the eight datasets within a draw; +/- is sample SD over three paired draws. '
 'No test label or test condition selected a model, stopping setting, epoch, or corruption.',
 '', '## Design and checks','',
 '**Strictly causal TCN:** five residual blocks, two kernel-3 convolutions/block, dilations 1/2/4/8/16, 64 channels, '
 'dropout 0.1, channel-only LayerNorm. The receptive field of local features is 125 samples; a causal prefix '
 '1/2/4-bin temporal pyramid plus the current feature gives the classifier access to the entire observed prefix. '
 'No symmetric padding, time-axis normalization, or future pooling.',
 '',
 '**Fair inheritance control:** each mode sees the same growing-checkpoint curriculum (two epochs/checkpoint), then '
 'at most 22 joint epochs. Loss is the mean CE over active checkpoints, plus optional lambda=0.1 adjacent Jensen-Shannon divergence. '
 'B copies the preceding already-trained head when introducing each new checkpoint; C introduces an independent random head. '
 'A uses one shared head. All modes have a shared TCN backbone; C means independent heads, not nine independent backbones. '
 'Same seeds, batches, augmentation copy, validation examples, optimizer, and early-stop criteria. '
 'Progressive initialization is not merely copying random weights. All eight inheritance events were checked exactly.',
 '',
 'Adam lr=0.001, weight_decay=0.0001, gradient clip=1. Training uses AMP; inference is float32. '
 'The existing MiniRocket logistic regression uses no class weights, so the deep empirical-frequency CE also uses none. '
 'Balanced accuracy and macro-F1 are retained to expose majority-class failures. '
 'Early stopping restores the minimum all-checkpoint validation CE; all variants get the full curriculum plus at least six joint epochs.',
 '',
 'Validation is a fixed 50:50 concatenation of clean validation and an independent six-corruption validation mixture. '
 'Training retains the exact prior clean-fit-plus-one-noise/missing-copy recipe. Calibration is the exact prior six-corruption mixture. '
 'The clean deep reference uses no training augmentation, but the same mixed validation protocol. '
 'Original fit caps and split indices are retained, including ECG5000 having only 201 fitting examples. '
 'Previously stored MiniRocket fixed-length/frontier metrics and calibration earliness are reused directly: no MiniRocket retraining. '
 'Raw corrupted test arrays were not persisted by scale_v1; they are deterministically reconstructed with its unchanged seeded generators. '
 'Per-stream SHA256 digests are recorded and checked equal across all deep variants.',
 '',
 f'**Validation-selected primary variant: {best} ({NAMES[best]}).** Selection is a SINGLE variant across the pilot, '
 'based only on mean validation accuracy across checkpoints/datasets/draws. All six augmented variants are reported, not just the winner.',
 '', table(v.rename('validation_accuracy').reset_index()),'',
 '## Datasets and retained splits','',
 table(pd.DataFrame([{k:m[k] for k in ('dataset','train_size','test_size','length','n_classes','fit_count','validation_count','calibration_count')} for m in meta])),
 '', '## Q1: equal-earliness accuracy versus augmented MiniRocket','',
 'Positive gaps below mean the learned model reduces error (percentage points). Prefix lengths are exactly paired within '
 'each dataset; nominal fractions use the same rounded checkpoint lengths as all prior runs.',
 '']
 gt=[]
 for cond in CONDITIONS:
  g=gaps[gaps.condition==cond]
  row={'condition':cond}
  for frac in [.2,.4,.5,.6,.8,1.]:
   q=g[np.isclose(g.nominal_fraction,frac)].iloc[0]
   row[f'{frac:.1f}']=f'{100*q.error_reduction_mean:+.2f} +/- {100*q.error_reduction_sd:.2f}'
  row['all 9 prefixes nonnegative']=bool((g.error_reduction_mean>=0).all());gt.append(row)
 text+=[table(pd.DataFrame(gt)),'','Per-condition accuracy/error/earliness at the two primary fixed checkpoints:','']
 short=df[(df.kind=='fixed')&(df.method.isin([best,'A_clean','MiniRocket_aug','MiniRocket_clean']))&df.setting.isin([3,4])]
 tmp=[]
 for (cond,meth,j),g in short.groupby(['condition','method','setting']):
  z=g.groupby('draw')[list(METRICS)].mean()
  tmp.append({'condition':cond,'method':meth,'prefix':(j+2)/10,'accuracy %':mpm(z.accuracy),
      'error %':mpm(z.error),'balanced accuracy %':mpm(z.balanced_accuracy),'earliness':mpm(z.earliness,False)})
 text+=[table(pd.DataFrame(tmp)),'',
 'Pilot paired Wilcoxon tests use each dataset\'s three-draw mean error difference; Holm across five conditions x two checkpoints '
 '(10 tests). With N=8 this is exploratory evidence, not the full 26-dataset benchmark.', '',
 table(tests),'','## Stopping at calibration targets 0.5 and 0.6','',
 'For each model/dataset/draw, choose among the SAME seven confidence thresholds (0.50,0.60,0.70,0.80,0.90,0.95,0.99) whose MIXED calibration mean earliness is closest to '
 'the target, using earliness only. Reuse that setting unchanged across all conditions. The cached mixed-calibrated '
 'MiniRocket baseline instead selects among its existing alpha sweep by the same criterion. '
 'The larger deep sweep is retained as calibration_target_dense in the CSV, not used for primary stopping comparisons. These are matched calibration targets, NOT guaranteed equal test earliness under distribution shift. '
 'No error-based or test-condition-specific interpolation/matching is presented as deployable evidence. '
 'The fixed-checkpoint comparison above is the decisive iso-earliness test.','',
 table(stops),'']
 st=df[(df.kind=='calibration_target')&df.method.isin([best,'MiniRocket_aug_probability','MiniRocket_aug_mixed','MiniRocket_clean_probability'])]
 rows=[]
 for (c,m,t),g in st.groupby(['condition','method','setting']):
  z=g.groupby('draw')[list(METRICS)].mean()
  rows.append({'condition':c,'method':m,'target':t,'accuracy %':mpm(z.accuracy),
               'error %':mpm(z.error),'earliness':mpm(z.earliness,False)})
 text+=[table(pd.DataFrame(rows)),'','## Q2: inheritance and consistency','',
 'Paired accuracy gains (percentage points), averaged over ALL nine matched checkpoints, eight datasets, then three draws. '
 'Positive values favor the first named variant. This isolates head initialization at the same curriculum; best-epoch selection '
 'can still produce different realized training time. Shared heads also change parameter count, so sharing vs independence is '
 'an architectural ablation, not solely initialization.','']
 aview=ab.copy();aview['accuracy gain pp +/- SD']=[f'{100*a:+.2f} +/- {100*b:.2f}' for a,b in zip(aview.accuracy_gain_mean,aview.accuracy_gain_sd)]
 text+=[table(aview[['comparison','condition','accuracy gain pp +/- SD']]),'',
 '## Q3: clean-data cost','']
 for baseline in ['MiniRocket_aug','MiniRocket_clean']:
  p=paired(df,best,baseline,'fixed');p=p[p.condition=='clean']
  x=(p.error_one-p.error_two).groupby(p.draw).mean()
  text.append(f'- {best} minus {baseline} clean error, averaged over checkpoints: {mpm(x)} percentage points (positive = clean cost).')
 text+=['','## Verdict','']
 for cond in CONDITIONS:
  gg=gaps[gaps.condition==cond];q=gg[np.isclose(gg.nominal_fraction,.5)|np.isclose(gg.nominal_fraction,.6)]
  mean=gg.error_reduction_mean.mean();all9=bool((gg.error_reduction_mean>=0).all())
  text.append(f'- {cond}: mean fixed-prefix error reduction {100*mean:+.2f} pp; wins at both 0.5/0.6 = {bool((q.error_reduction_mean>0).all())}; nonnegative at every prefix = {all9}.')
 corrupted=gaps[gaps.condition!='clean']
 robust_all=all(corrupted.groupby('condition').error_reduction_mean.mean()>0)
 for label in ['progressive minus independent, lambda=0','progressive minus independent, lambda=0.1',
               'shared minus independent, lambda=0','shared minus independent, lambda=0.1']:
  value=ab[(ab.comparison==label)&(ab.condition!='clean')].accuracy_gain_mean.mean()
  text.append(f'- {label}: average corruption accuracy change {100*value:+.2f} pp.')
 text.append('- Promotion decision: '+('promising for a larger controlled experiment, but not a confirmed full-method improvement; only eight datasets and three draws were tested.'
     if robust_all else 'do not promote as a general replacement: the equal-earliness advantage does not hold across all four corruptions in this pilot.'))
 text+=['','## Runtime, audits, and limitations','',
 f"All three GPUs used: {sorted(tr.gpu.astype(str).unique())}; hardware: {tr.gpu_name.iloc[0]}. "
 f"{len(tr)} trained networks; summed training + validation GPU occupancy = {tr.train_seconds.sum()/3600:.3f} GPU-hours. "
 'Inference and transfer time are additional. This is wall time while the training job occupies one GPU, not hardware-instruction utilization.',
 '',table(tr.groupby(['variant']).agg(networks=('variant','size'),mean_epochs=('epochs','mean'),
       mean_best_epoch=('best_epoch','mean'),train_gpu_hours=('gpu_hours','sum')).reset_index()),'',
 'Causality audited on each of the 168 trained networks: suffix perturbation and truncated-prefix equivalence, '
 'plus zero gradient with respect to future input. No future observations enter normalization or prediction. '
 'Consistency couples training losses across prefixes, which does not grant an inference-time future input.',
 '',
 'Pilot limitations: one prespecified TCN size/hyperparameter setting; curriculum and head choices can favor different data domains; '
 'fixed fit caps; three draws are insufficient for precise variance estimation; eight datasets are not a publishable-scale superiority claim. '
 'Input train/validation/calibration partitions are fixed as in prior runs; draws vary augmentation/corruption and paired network initialization.',
 '',
 'All numeric tables, full curves, per-dataset metrics, checkpoints, predictions, training histories, and code are preserved. '
 'See source_audit.json for prior-input integrity and artifact_audit.json for completeness. Errors, if any, are retained verbatim in logs/.',
 '']
 status=OUT/'orchestration_status.json'
 if status.exists():
  stat=jread(status);text.append(f"Overall parallel wall time: {stat['wall_seconds']/3600:.3f} hours; worker failures: {len(stat['failed'])}.")
 write(OUT/'summary.md','\n'.join(text)+'\n')
 print('REPORT_DONE',best,'records',len(df),'train_gpu_hours',tr.train_seconds.sum()/3600,flush=True)

if __name__=='__main__':run()
