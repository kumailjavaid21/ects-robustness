"""N=26 paired dataset tests. All deployable choices are copied from previous mixed calibration."""
import json
import numpy as np,pandas as pd
from scipy.stats import wilcoxon,rankdata,studentized_range,friedmanchisquare
from settings import *
PAIRS_FIXED=(('MiniRocket_aug','MiniRocket_clean'),('TCN_aug','MiniRocket_clean'),('TCN_aug','MiniRocket_aug'))
PAIRS_STOP=tuple((m,b) for m in AUG_METHODS for b in ('Probability','TEASER','ECEC'))+(('TCN_aug','MiniRocket_aug'),)
NAMES={'MiniRocket_clean':'MiniRocket clean','MiniRocket_aug':'MiniRocket augmented + mixed-cal','TCN_aug':'TCN augmented',
 'Probability':'Clean + probability','TEASER':'TEASER (clean)','ECEC':'ECEC adaptation (clean)'}
def holm(values):
 p=np.asarray(values,float);out=np.empty(len(p));order=np.argsort(p);running=0.
 for i,j in enumerate(order):running=max(running,min(1.,(len(p)-i)*p[j]));out[j]=running
 return out
def correct(df):
 df=df.copy();df['p_holm']=np.nan
 for f,g in df.groupby('family'):df.loc[g.index,'p_holm']=holm(g.p_raw)
 return df
def pair(piv,early,model,baseline,cond,setting,family):
 diff=piv[baseline]-piv[model];v=diff.groupby('dataset').mean();draw=diff.groupby('draw').mean()
 assert len(v)==26 and len(draw)==5 and np.isfinite(v).all()
 if np.all(abs(v)<=1e-14):w,p=0.,1.
 else:
  r=wilcoxon(v.to_numpy(),alternative='two-sided',zero_method='wilcox',method='auto');w,p=float(r.statistic),float(r.pvalue)
 nonzero=v.to_numpy()[v.to_numpy()!=0]
 ranks=rankdata(abs(nonzero)) if len(nonzero) else np.array([])
 signed=float(np.sum(ranks*np.sign(nonzero))) if len(nonzero) else 0.
 biserial=signed/float(ranks.sum()) if len(nonzero) else 0.
 out={'rank_biserial':biserial,'effective_nonzero_pairs':len(nonzero),'method':model,'baseline':baseline,'condition':cond,'type':TYPES[cond],'setting':setting,'family':family,'N':26,
  'error_reduction_mean':float(v.mean()),'error_reduction_sd':float(draw.std(ddof=1)),
  'model_error':float(piv[model].mean()),'baseline_error':float(piv[baseline].mean()),
  'wins':int((v>1e-12).sum()),'ties':int((abs(v)<=1e-12).sum()),'losses':int((v< -1e-12).sum()),'W':w,'p_raw':p}
 if early is not None:
  e=early.groupby('dataset').mean();gap=abs(e[model]-e[baseline])
  out.update(model_earliness=float(e[model].mean()),baseline_earliness=float(e[baseline].mean()),
   mean_absolute_dataset_earliness_gap=float(gap.mean()),N_gap_within_005=int((gap<=.05).sum()))
 return out
def aggregate(df,keys,metrics=METRICS):
 per=df.groupby(keys+['dataset'])[list(metrics)].agg(['mean','std'])
 per.columns=['_'.join(x) for x in per.columns]
 draws=df.groupby(keys+['draw'])[list(metrics)].mean().reset_index()
 macro=draws.groupby(keys)[list(metrics)].agg(['mean','std']);macro.columns=['_'.join(x) for x in macro.columns]
 return per.reset_index(),draws,macro.reset_index()
def collect():
 fixed=[];selected=[];front=[];cal=[];run=[]
 for name in DATASETS:
  for d in DRAWS:
   for c in CONDITIONS:
    r=json.loads((OUT/'records'/f'{name}_draw{d}_{c}.json').read_text());key={'dataset':name,'draw':d,'condition':c,'type':TYPES[c]}
    fixed.extend({**key,**q} for q in r['fixed'])
    selected.extend({**key,**q} for q in r['selected'])
    front.extend({**key,**q} for q in r['frontiers'])
    cal.extend({**key,**{k:v for k,v in q.items() if k in ('method','setting','earliness')}} for q in r['oracle_calibration'])
    run.append({**key,'seconds':r['seconds'],'test_missing_fraction':r['test_missing_fraction']})
 return tuple(pd.DataFrame(a) for a in (fixed,selected,front,cal,run))
def fixed_stats(f):
 rows=[]
 for cond in CONDITIONS:
  for j in range(9):
   x=f[(f.condition==cond)&(f.checkpoint==j)]
   p=x.pivot(index=['dataset','draw'],columns='method',values='error')
   e=x.pivot(index=['dataset','draw'],columns='method',values='earliness')
   family='primary_fixed_60' if j in (3,4) else 'other_prefixes_210'
   for model,base in PAIRS_FIXED:rows.append(pair(p,e,model,base,cond,(j+2)/10,family))
  x=f[f.condition==cond];p=x.groupby(['dataset','draw','method']).error.mean().unstack()
  e=x.groupby(['dataset','draw','method']).earliness.mean().unstack()
  for model,base in PAIRS_FIXED:rows.append(pair(p,e,model,base,cond,'all_prefixes','curve_average_30'))
 return correct(pd.DataFrame(rows))
def stop_stats(s):
 rows=[]
 for cond in CONDITIONS:
  for target in TARGETS:
   x=s[(s.condition==cond)&(s.target==target)]
   p=x.pivot(index=['dataset','draw'],columns='method',values='error')
   e=x.pivot(index=['dataset','draw'],columns='method',values='earliness')
   for model,base in PAIRS_STOP:rows.append(pair(p,e,model,base,cond,target,'fixedop_140'))
 return correct(pd.DataFrame(rows))
def transfer(front,cal):
 # Oracle selection uses only calibration earliness, never test labels/confidence/error.
 mixed_choices=pd.read_csv(DEEP/'fixedop_calibration_choices.csv')
 cm={(r.dataset,r.method,float(r.target)):float(r.setting) for r in mixed_choices.itertuples() if r.method in AUG_METHODS}
 rows=[];choices=[]
 cg=cal.groupby(['dataset','condition','method','setting']).earliness.mean().reset_index()
 for (name,c,method),g in cg.groupby(['dataset','condition','method']):
  for target in TARGETS:
   q=g.assign(distance=abs(g.earliness-target)).sort_values(['distance','setting'],kind='stable').iloc[0]
   choices.append({'dataset':name,'condition':c,'method':method,'target':target,'setting':float(q.setting),'calibration_earliness':float(q.earliness)})
 oc={(q['dataset'],q['condition'],q['method'],q['target']):q['setting'] for q in choices}
 index={(r.dataset,r.draw,r.condition,r.method,r.regime,float(r.setting)):r for r in front.itertuples()}
 for name in DATASETS:
  for d in DRAWS:
   for c in CONDITIONS:
    for method in AUG_METHODS:
     for target in TARGETS:
      alpha=cm[(name,method,target)];mixed=index[(name,d,c,method,'mixed',alpha)]
      for diagnostic,oa in [('oracle_same_alpha',alpha),('oracle_calibration_time_matched',oc[(name,c,method,target)])]:
       oracle=index[(name,d,c,method,'oracle',oa)]
       rows.append({'dataset':name,'draw':d,'condition':c,'type':TYPES[c],'method':method,'target':target,
        'diagnostic':diagnostic,'mixed_setting':alpha,'oracle_setting':oa,
        'mixed_error':mixed.error,'oracle_error':oracle.error,'error_gap':mixed.error-oracle.error,
        'mixed_earliness':mixed.earliness,'oracle_earliness':oracle.earliness,'earliness_gap':mixed.earliness-oracle.earliness,
        'mixed_forced_fraction':mixed.forced_fraction,'oracle_forced_fraction':oracle.forced_fraction})
     mixed=index[(name,d,c,method,'mixed',.1)];oracle=index[(name,d,c,method,'oracle',.1)]
     rows.append({'dataset':name,'draw':d,'condition':c,'type':TYPES[c],'method':method,'target':-1.,
      'diagnostic':'fixed_alpha_0.10','mixed_setting':.1,'oracle_setting':.1,
      'mixed_error':mixed.error,'oracle_error':oracle.error,'error_gap':mixed.error-oracle.error,
      'mixed_earliness':mixed.earliness,'oracle_earliness':oracle.earliness,'earliness_gap':mixed.earliness-oracle.earliness,
      'mixed_forced_fraction':mixed.forced_fraction,'oracle_forced_fraction':oracle.forced_fraction})
 df=pd.DataFrame(rows);keys=['condition','type','method','target','diagnostic']
 vals=['mixed_error','oracle_error','error_gap','mixed_earliness','oracle_earliness','earliness_gap','mixed_forced_fraction','oracle_forced_fraction']
 draws=df.groupby(keys+['draw'])[vals].mean().reset_index()
 macro=draws.groupby(keys)[vals].agg(['mean','std']);macro.columns=['_'.join(x) for x in macro.columns];macro=macro.reset_index()
 dataset_gaps=df.groupby(keys+['dataset'])[['error_gap','earliness_gap']].mean().abs()
 absolute=dataset_gaps.groupby(keys).mean().rename(columns={'error_gap':'mean_absolute_dataset_error_gap','earliness_gap':'mean_absolute_dataset_earliness_gap'}).reset_index()
 macro=macro.merge(absolute,on=keys,validate='one_to_one')
 macro['small_joint_macro_gap']=(abs(macro.error_gap_mean)<=.02)&(abs(macro.earliness_gap_mean)<=.05)
 return df,macro,pd.DataFrame(choices)
def diagrams(sel):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 rows=[]
 for c in CONDITIONS:
  for target in TARGETS:
   a=sel[(sel.condition==c)&(sel.target==target)].groupby(['dataset','method']).error.mean().unstack().reindex(index=DATASETS,columns=METHODS)
   assert a.shape==(26,5) and np.isfinite(a.to_numpy()).all()
   m=a.to_numpy();ranks=np.array([rankdata(v) for v in m]).mean(0);k=5
   cd=float(studentized_range.ppf(.95,k,np.inf)/np.sqrt(2)*np.sqrt(k*(k+1)/(6*26)))
   f,p=friedmanchisquare(*[m[:,j] for j in range(k)])
   ordered=sorted(zip(METHODS,ranks),key=lambda x:x[1]);values=[r for _,r in ordered]
   fig,ax=plt.subplots(figsize=(11,5))
   for j,(method,rank) in enumerate(ordered):ax.scatter(rank,j,c='#176e93');ax.text(rank+.06,j,NAMES[method],va='center')
   ax.plot([1,1+cd],[k+.1]*2,c='black',lw=3);ax.text(1,k+.3,f'Nemenyi CD={cd:.3f} (alpha=.05 within panel)')
   intervals=[]
   if p<.05:
    for i in range(k):
     for j in range(i+1,k):
      if values[j]-values[i]<=cd:intervals.append((i,j))
    intervals=[(i,j) for i,j in intervals if not any(a<=i and b>=j and (a,b)!=(i,j) for a,b in intervals)]
    for line,(i,j) in enumerate(intervals):ax.plot([values[i],values[j]],[k+.65+.2*line]*2,c='#555',lw=2)
   ax.set(xlim=(.8,7.2),ylim=(-.7,k+1+.2*len(intervals)),yticks=[],xticks=range(1,6),
    xlabel='Mean error rank (1 = best); realized test earliness can differ',title=f'{c}, calibration target {target}: N=26, Friedman p={p:.4g}')
   fig.tight_layout();fig.savefig(OUT/'cd_diagrams'/f'{c}_target{target}.png',dpi=160);plt.close(fig)
   rows.append({'condition':c,'target':target,'N':26,'CD':cd,'friedman_stat':float(f),'friedman_p':float(p),**dict(zip(METHODS,ranks))})
 return pd.DataFrame(rows)
def plots(fm,sm):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 colors={'MiniRocket_clean':'#777','MiniRocket_aug':'black','TCN_aug':'#1676b0'}
 fig,axes=plt.subplots(3,4,figsize=(21,13),sharex=True,sharey=True)
 for ax,c in zip(axes.flat,CONDITIONS):
  for method in FIXED_METHODS:
   g=fm[(fm.condition==c)&(fm.method==method)].sort_values('checkpoint')
   x=g.nominal_fraction.to_numpy();m=g.accuracy_mean.to_numpy()*100;s=g.accuracy_std.to_numpy()*100
   ax.plot(x,m,c=colors[method],label=NAMES[method].replace(' + mixed-cal',''));ax.fill_between(x,m-s,m+s,color=colors[method],alpha=.13)
  ax.set_title(c);ax.set_xlabel('Observed prefix fraction');ax.set_ylabel('Accuracy (%)');ax.grid(alpha=.2)
  ax.tick_params(axis='x',labelbottom=True);ax.set_xticks([.2,.4,.6,.8,1.])
 for ax in axes.flat[len(CONDITIONS):]:ax.axis('off')
 h,l=axes.flat[0].get_legend_handles_labels();fig.legend(h,l,loc='lower right',bbox_to_anchor=(.97,.12))
 fig.suptitle('Unseen corruption transfer: frozen models; 26 datasets, 5 draws; bands = sample SD',fontsize=18)
 fig.tight_layout(rect=(0,0,1,.97));fig.savefig(OUT/'accuracy_vs_prefix_crosstype.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(3,4,figsize=(21,13))
 colors2={'MiniRocket_aug':'black','TCN_aug':'#1676b0','Probability':'#c44','TEASER':'#9867b8','ECEC':'#d78b24'}
 for ax,c in zip(axes.flat,CONDITIONS):
  for method in METHODS:
   g=sm[(sm.condition==c)&(sm.method==method)].sort_values('target')
   ax.errorbar(g.earliness_mean,g.error_mean*100,xerr=g.earliness_std,yerr=g.error_std*100,label=NAMES[method],c=colors2[method],marker='o',capsize=2)
  ax.set_title(c);ax.set(xlabel='Realized decision fraction',ylabel='Error (%)');ax.grid(alpha=.2)
 for ax in axes.flat[len(CONDITIONS):]:ax.axis('off')
 h,l=axes.flat[0].get_legend_handles_labels();fig.legend(h,l,loc='lower right',bbox_to_anchor=(.99,.10))
 fig.suptitle('Previous mixed-calibration operating settings, fixed across unseen test types',fontsize=18)
 fig.tight_layout(rect=(0,0,1,.97));fig.savefig(OUT/'fixedop_crosstype.png',dpi=160);plt.close(fig)
def main():
 f,s,front,cal,runtime=collect()
 assert len(f)==35100 and len(s)==13000 and len(front)==31200 and len(cal)==15600
 f.to_csv(OUT/'crosstype_fixed_length.csv',index=False);s.to_csv(OUT/'fixed_op_draws.csv',index=False)
 front.to_csv(OUT/'frontier_draws.csv',index=False);cal.to_csv(OUT/'oracle_calibration_earliness.csv',index=False);runtime.to_csv(OUT/'runtime.csv',index=False)
 fp,fd,fm=aggregate(f,['condition','type','method','checkpoint','nominal_fraction'])
 sp,sd,sm=aggregate(s,['condition','type','method','target'])
 fp.to_csv(OUT/'fixed_length_per_dataset.csv',index=False);fm.to_csv(OUT/'fixed_length_macro.csv',index=False)
 sp.to_csv(OUT/'fixed_op_per_dataset.csv',index=False);sm.to_csv(OUT/'fixed_op_macro.csv',index=False)
 fs=fixed_stats(f);ss=stop_stats(s)
 assert len(fs[fs.family=='primary_fixed_60'])==60 and len(ss)==140 and (ss.N==26).all()
 fs.to_csv(OUT/'fixed_length_stats.csv',index=False);ss.to_csv(OUT/'crosstype_fixed_op_stats.csv',index=False)
 t,tm,oc=transfer(front,cal);t.to_csv(OUT/'transfer_gap_draws.csv',index=False);tm.to_csv(OUT/'crosstype_transfer_gap.csv',index=False)
 oc.to_csv(OUT/'oracle_only_calibration_choices.csv',index=False)
 ranks=diagrams(s);ranks.to_csv(OUT/'friedman_nemenyi.csv',index=False);plots(fm,sm)
 # Copy previous deployed settings as provenance, without modifying their source.
 with (OUT/'mixed_fixed_choices.csv').open('x') as out:
  out.write((DEEP/'fixedop_calibration_choices.csv').read_text())
 print('ANALYSIS_COMPLETE','fixed',len(f),'selected',len(s),'N=26',flush=True)
if __name__=='__main__':main()
