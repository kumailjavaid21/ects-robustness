"""Full N=26 analysis; settings selected ONLY from five-draw mixed calibration earliness."""
import json,math
import numpy as np,pandas as pd
from scipy.stats import wilcoxon,rankdata,studentized_range,friedmanchisquare
from settings import *
METRICS=['accuracy','error','balanced_accuracy','macro_f1','earliness','combined']
BASE_MAP={'MiniRocket_aug':'ours','Probability':'probability','TEASER':'TEASER','ECEC':'ECEC'}
STOP_METHODS=list(VARIANTS)+list(BASE_MAP)
FIXED_METHODS=list(VARIANTS)+['MiniRocket_aug','MiniRocket_clean']
NAMES={'TCN_aug':'TCN augmented','TCN_aug_mask':'TCN augmented + mask','MiniRocket_aug':'MiniRocket augmented + mixed-cal',
       'MiniRocket_clean':'MiniRocket clean','Probability':'MiniRocket clean + probability','TEASER':'TEASER (clean)','ECEC':'ECEC adaptation (clean)'}
PILOT_DATASETS=('FordA','ECG5000','StarLightCurves','Wafer','ElectricDevices','Crop','NonInvasiveFetalECGThorax1','HandOutlines')
def loadj(p):return json.loads(p.read_text())
def table(df):
 def fmt(x):
  if isinstance(x,(float,np.floating)):return f'{x:.4f}'
  return str(x).replace('|','\\|').replace('\n',' ')
 out=['| '+' | '.join(map(str,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']
 out+=['| '+' | '.join(fmt(x) for x in r)+' |' for r in df.itertuples(index=False,name=None)]
 return '\n'.join(out)
def pm(mean,std,scale=100):return f'{scale*mean:.2f} +/- {scale*std:.2f}'
def holm(values):
 p=np.asarray(values,float);order=np.argsort(p);out=np.empty(len(p));v=0.
 for i,j in enumerate(order):v=max(v,min(1.,(len(p)-i)*p[j]));out[j]=v
 return out
def test_pair(per,draw_diffs,model,baseline,condition,setting,family):
 v=per.to_numpy(dtype=float)
 assert len(v)==26 and np.isfinite(v).all()
 if np.all(abs(v)<=1e-14):w,p=0.,1.
 else:
  z=wilcoxon(v,alternative='two-sided',zero_method='wilcox',method='auto');w,p=float(z.statistic),float(z.pvalue)
 return {'model':model,'baseline':baseline,'condition':condition,'setting':setting,'family':family,'N':26,
   'error_reduction_mean':float(v.mean()),'error_reduction_sd':float(draw_diffs.std(ddof=1)),
   'wins':int((v>1e-12).sum()),'ties':int((abs(v)<=1e-12).sum()),'losses':int((v< -1e-12).sum()),
   'wilcoxon_W':w,'p_raw':p}
def correct(df):
 df=df.copy();df['p_holm']=np.nan
 for family,g in df.groupby('family'):df.loc[g.index,'p_holm']=holm(g.p_raw)
 return df
def aggregate(df,keys):
 # Macro each dataset equally within a draw, then mean/sample SD of five macro draw means.
 draws=df.groupby(keys+['draw'])[METRICS].mean().reset_index()
 macro=draws.groupby(keys)[METRICS].agg(['mean','std']);macro.columns=['_'.join(x) for x in macro.columns]
 return draws,macro.reset_index()

def collect():
 fixed=[];front=[];cal=[];training=[];datasets=[]
 for name in DATASETS:
  datasets.append(loadj(SCALE/'draw_meta'/f'{name}_base.json'))
  for draw in DRAWS:
   for tag in VARIANTS:
    r=loadj(OUT/'records'/f'{name}_draw{draw}_{tag}.json')
    tr={k:v for k,v in r['training'].items() if k not in ('history','causality_audit')}
    tr['causality_passed']=r['training']['causality_audit']['passed'];training.append(tr)
    for q in r['calibration_frontiers']:
     if q['kind']=='mixed':cal.append({'dataset':name,'draw':draw,'method':tag,'setting':q['setting'],'earliness':q['earliness']})
    for q in r['rows']:
     if q['kind']=='fixed':
      fixed.append({'dataset':name,'draw':draw,'method':tag,'condition':q['condition'],'checkpoint':int(q['setting']),
                    'nominal_fraction':q['nominal_fraction'],**{k:q[k] for k in METRICS}})
     elif q['kind']=='mixed':
      front.append({'dataset':name,'draw':draw,'method':tag,'condition':q['condition'],'setting':q['setting'],
                    **{k:q[k] for k in METRICS},'forced_fraction':q['forced_fraction']})
   oldcal=loadj(SELECT/'draw_meta'/f'{name}_draw{draw}.json')['calibration_earliness']
   for tag,source in BASE_MAP.items():
    for q in oldcal[source]:cal.append({'dataset':name,'draw':draw,'method':tag,'setting':q['setting'],'earliness':q['earliness']})
   for cond in CONDITIONS:
    r=loadj(SCALE/'records'/f'{name}_draw{draw}_{cond}.json')
    for j,q in enumerate(r['fixed']):
     for tag,source in [('MiniRocket_aug','augmented'),('MiniRocket_clean','clean')]:
      fixed.append({'dataset':name,'draw':draw,'method':tag,'condition':cond,'checkpoint':j,
                    'nominal_fraction':(j+2)/10,**{k:q[source][k] for k in METRICS}})
    for tag,source in BASE_MAP.items():
     for q in r['frontiers'][source]:
      front.append({'dataset':name,'draw':draw,'method':tag,'condition':cond,'setting':q['setting'],
                    **{k:q[k] for k in METRICS},'forced_fraction':q['forced_fraction']})
 return pd.DataFrame(fixed),pd.DataFrame(front),pd.DataFrame(cal),pd.DataFrame(training),datasets

def fixed_statistics(df):
 rows=[]
 for cond in CONDITIONS:
  for j in range(9):
   a=df[(df.condition==cond)&(df.checkpoint==j)]
   piv=a.pivot(index=['dataset','draw'],columns='method',values='error')
   for model in VARIANTS:
    diff=piv['MiniRocket_aug']-piv[model];per=diff.groupby('dataset').mean();d=diff.groupby('draw').mean()
    family='primary_fixed_28' if j in (3,4) else 'exploratory_other_prefixes_98'
    row=test_pair(per,d,model,'MiniRocket_aug',cond,(j+2)/10,family)
    row.update(model_error=float(piv[model].mean()),baseline_error=float(piv['MiniRocket_aug'].mean()))
    rows.append(row)
   if j in (3,4):
    diff=piv['TCN_aug']-piv['TCN_aug_mask']
    rows.append(test_pair(diff.groupby('dataset').mean(),diff.groupby('draw').mean(),'TCN_aug_mask','TCN_aug',cond,(j+2)/10,'mask_incremental_fixed_14'))
  a=df[df.condition==cond]
  piv=a.groupby(['dataset','draw','method']).error.mean().unstack('method')
  for model in VARIANTS:
   diff=piv['MiniRocket_aug']-piv[model]
   rows.append(test_pair(diff.groupby('dataset').mean(),diff.groupby('draw').mean(),model,'MiniRocket_aug',cond,'all_prefixes','curve_average_14'))
 return correct(pd.DataFrame(rows))

def choose_calibration(cal):
 e=cal.groupby(['dataset','method','setting']).agg(earliness=('earliness','mean'),n_draws=('draw','nunique')).reset_index()
 assert (e.n_draws==5).all()
 choices=[]
 for (name,method),g in e.groupby(['dataset','method'],sort=False):
  for target in TARGETS:
   q=g.assign(distance=abs(g.earliness-target)).sort_values(['distance','setting'],kind='stable').iloc[0]
   choices.append({'dataset':name,'method':method,'target':target,'setting':float(q.setting),'calibration_earliness':float(q.earliness)})
 return pd.DataFrame(choices)
def select_frontiers(front,choices):
 rows=[]
 for q in choices.itertuples(index=False):
  g=front[(front.dataset==q.dataset)&(front.method==q.method)&np.isclose(front.setting,q.setting,rtol=0,atol=1e-12)]
  assert len(g)==35 and g.draw.nunique()==5 and g.condition.nunique()==7
  for x in g.to_dict('records'):rows.append({**x,'target':q.target,'calibration_earliness':q.calibration_earliness})
 return pd.DataFrame(rows)
def stopping_statistics(sel):
 rows=[]
 for cond in CONDITIONS:
  for target in TARGETS:
   a=sel[(sel.condition==cond)&(sel.target==target)]
   piv=a.pivot(index=['dataset','draw'],columns='method',values='error')
   e=a.groupby(['dataset','method']).earliness.mean().unstack()
   for model in VARIANTS:
    for baseline in BASE_MAP:
     diff=piv[baseline]-piv[model]
     row=test_pair(diff.groupby('dataset').mean(),diff.groupby('draw').mean(),model,baseline,cond,target,'fixedop_112')
     eg=abs(e[model]-e[baseline])
     row.update(model_error=float(piv[model].mean()),baseline_error=float(piv[baseline].mean()),
       model_earliness=float(e[model].mean()),baseline_earliness=float(e[baseline].mean()),
       mean_absolute_dataset_earliness_gap=float(eg.mean()),N_gap_within_005=int((eg<=.05).sum()))
     rows.append(row)
   diff=piv['TCN_aug']-piv['TCN_aug_mask'];eg=abs(e['TCN_aug']-e['TCN_aug_mask'])
   row=test_pair(diff.groupby('dataset').mean(),diff.groupby('draw').mean(),'TCN_aug_mask','TCN_aug',cond,target,'fixedop_mask_14')
   row.update(model_earliness=e['TCN_aug_mask'].mean(),baseline_earliness=e['TCN_aug'].mean(),mean_absolute_dataset_earliness_gap=eg.mean(),N_gap_within_005=int((eg<=.05).sum()))
   rows.append(row)
 return correct(pd.DataFrame(rows))

def diagrams(sel):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 rows=[]
 for cond in CONDITIONS:
  for target in TARGETS:
   a=sel[(sel.condition==cond)&(sel.target==target)]
   matrix=a.groupby(['dataset','method']).error.mean().unstack().reindex(index=DATASETS,columns=STOP_METHODS)
   assert matrix.shape==(26,6) and np.isfinite(matrix.to_numpy()).all()
   m=matrix.to_numpy();avg=np.array([rankdata(r) for r in m]).mean(0);k=6;n=26
   cd=float(studentized_range.ppf(.95,k,np.inf)/np.sqrt(2)*np.sqrt(k*(k+1)/(6*n)))
   stat,p=friedmanchisquare(*[m[:,i] for i in range(k)])
   ordered=sorted(zip(STOP_METHODS,avg),key=lambda x:x[1]);values=[x[1] for x in ordered]
   fig,ax=plt.subplots(figsize=(11,5))
   for j,(name,rank) in enumerate(ordered):
    ax.scatter(rank,j,c='#186e94');ax.text(rank+.07,j,NAMES[name],va='center')
   ax.plot([1,1+cd],[k+.15]*2,c='black',lw=3);ax.text(1,k+.35,f'Nemenyi CD={cd:.3f} (alpha=0.05 within this diagram)')
   intervals=[]
   if p<.05:
    for i in range(k):
     for j in range(i+1,k):
      if values[j]-values[i]<=cd+1e-12:intervals.append((i,j))
    intervals=[(i,j) for i,j in intervals if not any(a<=i and b>=j and (a,b)!=(i,j) for a,b in intervals)]
    for line,(i,j) in enumerate(intervals):ax.plot([values[i],values[j]],[k+.7+.22*line]*2,lw=2,c='#555')
   ax.set_xlim(.8,8);ax.set_ylim(-.7,k+1.1+.22*len(intervals));ax.set_yticks([]);ax.set_xticks(range(1,7))
   ax.set_xlabel('Mean error rank (1 = best); realized test earliness can differ')
   ax.set_title(f'{cond}, calibration target {target}: N=26, Friedman p={p:.4g}')
   fig.tight_layout();fig.savefig(OUT/'cd_diagrams'/f'{cond}_target{target}.png',dpi=160);plt.close(fig)
   rows.append({'condition':cond,'target':target,'N':26,'k':6,'CD':cd,'friedman_stat':float(stat),'friedman_p':float(p),**dict(zip(STOP_METHODS,avg))})
 return pd.DataFrame(rows)
