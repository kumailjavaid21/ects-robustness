"""Full-coverage fixed-operating-point analysis, with no error-based tuning."""
import sys,json,math
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon,rankdata,studentized_range,friedmanchisquare
ROOT=Path('/home/ubuntu/host_share/kumail_mad');HERE=ROOT/'runs/ects_select_v1';SCALE=ROOT/'runs/ects_scale_v1'
sys.path.insert(0,str(SCALE));from scale_worker import MAIN
sys.path.insert(0,str(ROOT/'src'));from ects.robustness import CONDITIONS
METHODS=['probability','TEASER','ECEC','aug_probability','aug_TEASER','ours']
NAMES={'probability':'Probability','TEASER':'TEASER','ECEC':'ECEC adaptation','aug_probability':'Aug + probability','aug_TEASER':'Aug + TEASER','ours':'Ours'}
TARGETS=(.5,.6)
def load():
    d=pd.read_csv(SCALE/'per_dataset_draws.csv')
    d=d[(d.cohort=='main')&(d.kind=='frontier')&d.dataset.isin(MAIN)].copy()
    if set(d.dataset)!=set(MAIN):raise ValueError('Main dataset set differs')
    return d
def choose_posthoc(df):
    # Test labels/errors are not read or used to choose a setting.
    e=df.groupby(['dataset','condition','method','setting'],as_index=False).agg(earliness=('earliness','mean'),n_draws=('draw','nunique'))
    if not (e.n_draws==5).all():raise ValueError('Five draws required')
    records=[]
    for (name,cond,method),a in e.groupby(['dataset','condition','method'],sort=False):
        for target in TARGETS:
            q=a.assign(distance=(a.earliness-target).abs()).sort_values(['distance','setting'],kind='stable').iloc[0]
            records.append({'dataset':name,'condition':cond,'method':method,'target':target,'setting':q.setting,'selection_earliness':q.earliness,'selection':'condition_posthoc'})
    return pd.DataFrame(records)
def selected(df,choices):
    result=[]
    for q in choices.itertuples(index=False):
        a=df[(df.dataset==q.dataset)&(df.condition==q.condition)&(df.method==q.method)&np.isclose(df.setting,q.setting,rtol=0,atol=1e-12)]
        if a.draw.nunique()!=5 or len(a)!=5:raise ValueError('Selected draw mismatch '+str(q))
        for x in a.itertuples(index=False):
            result.append({'dataset':q.dataset,'condition':q.condition,'target':q.target,'method':q.method,'setting':q.setting,'draw':x.draw,
                'error':x.error,'earliness':x.earliness,'accuracy':x.accuracy,'balanced_accuracy':x.balanced_accuracy})
    return pd.DataFrame(result)
def statistics(sel,tag):
    out=[]
    for cond in CONDITIONS:
        for target in TARGETS:
            a=sel[(sel.condition==cond)&np.isclose(sel.target,target)]
            if a.dataset.nunique()!=len(MAIN):raise ValueError('Incomplete dataset comparison')
            per=a.groupby(['dataset','method'],as_index=False).agg(error=('error','mean'),earliness=('earliness','mean'))
            pivot=per.pivot(index='dataset',columns='method',values='error')
            e=per.groupby('method').earliness.mean()
            e_pivot=per.pivot(index='dataset',columns='method',values='earliness')
            for method in METHODS[:-1]:
                d=pivot[method]-pivot['ours'];v=d.to_numpy()
                if len(v)!=26 or not np.isfinite(v).all():raise ValueError('Expected 26 finite differences')
                if np.all(np.abs(v)<=1e-14):stat,pvalue=0.,1.
                else:
                    z=wilcoxon(v,zero_method='wilcox',alternative='two-sided',method='auto');stat,pvalue=float(z.statistic),float(z.pvalue)
                draws=a.pivot_table(index=['dataset','draw'],columns='method',values='error')
                dg=(draws[method]-draws['ours']).groupby('draw').mean()
                out.append({'selection':tag,'condition':cond,'target':target,'baseline':method,'n_datasets':len(v),
                   'baseline_earliness':float(e[method]),'ours_earliness':float(e['ours']),
                   'mean_absolute_earliness_gap':float((e_pivot[method]-e_pivot['ours']).abs().mean()),
                   'datasets_with_earliness_gap_over_005':int(((e_pivot[method]-e_pivot['ours']).abs()>.05).sum()),
                   'baseline_error':float(per[per.method==method].error.mean()),'ours_error':float(per[per.method=='ours'].error.mean()),
                   'margin_baseline_minus_ours':float(d.mean()),'margin_sd_across_draw_means':float(dg.std(ddof=1)),
                   'wins':int(np.sum(v>1e-12)),'ties':int(np.sum(np.abs(v)<=1e-12)),'losses':int(np.sum(v< -1e-12)),
                   'wilcoxon_stat':stat,'p_raw':pvalue})
    result=pd.DataFrame(out);result['p_holm']=np.nan
    inds=result.index[result.p_raw.notna()].tolist();inds.sort(key=lambda i:result.loc[i,'p_raw']);running=0.
    for k,i in enumerate(inds):
        running=max(running,min(1.,(len(inds)-k)*result.loc[i,'p_raw']));result.loc[i,'p_holm']=running
    return result
def diagrams(sel,tag):
    rows=[]
    for cond in CONDITIONS:
        a=sel[(sel.condition==cond)&np.isclose(sel.target,.5)]
        z=a.groupby(['dataset','method']).error.mean().unstack()
        z=z.reindex(index=list(MAIN),columns=METHODS)
        if z.shape!=(26,6) or z.isna().any().any():raise ValueError('Incomplete CD rows')
        matrix=z.to_numpy();ranks=np.array([rankdata(x) for x in matrix]);avg=ranks.mean(0);n=26;k=len(METHODS)
        cd=float(studentized_range.ppf(.95,k,np.inf)/np.sqrt(2)*np.sqrt(k*(k+1)/(6*n)))
        try:fstat,fp=friedmanchisquare(*[matrix[:,i] for i in range(k)])
        except ValueError:fstat,fp=np.nan,np.nan
        fig,ax=plt.subplots(figsize=(10,4))
        ordered=sorted(zip(METHODS,avg),key=lambda z:z[1])
        for i,(name,r) in enumerate(ordered):
            ax.scatter(r,i,c='#2878b5');ax.text(r+.05,i,NAMES[name],va='center')
        ax.plot([1,1+cd],[k+.1,k+.1],color='black',lw=3);ax.text(1,k+.25,f'Nemenyi CD={cd:.3f}')
        intervals=[]
        if np.isfinite(fp) and fp<.05:
            values=[r for _,r in ordered]
            for i in range(k):
                for j in range(i+1,k):
                    if values[j]-values[i]<=cd+1e-12:intervals.append((i,j))
            intervals=[(i,j) for i,j in intervals if not any((a<=i and b>=j and (a,b)!=(i,j)) for a,b in intervals)]
            for line,(i,j) in enumerate(intervals):
                ax.plot([values[i],values[j]],[k+.55+.22*line]*2,color='#333333',lw=2)
        top=k+.9+.22*len(intervals)
        ax.set_xlim(.8,k+2);ax.set_ylim(-1,top);ax.set_yticks([]);ax.set_xticks(range(1,k+1));ax.set_xlabel('Mean rank (1 = lower error)')
        ax.set_title(f'{cond}: N=26, Friedman p={fp:.4g} [{tag}]');fig.tight_layout()
        fig.savefig(HERE/'cd_diagrams'/f'{cond}_{tag}.png',dpi=160);plt.close(fig)
        rows.append({'selection':tag,'condition':cond,'n':n,'cd':cd,'friedman_stat':fstat,'friedman_p':fp,**{m:float(v) for m,v in zip(METHODS,avg)}})
    return pd.DataFrame(rows)
if __name__=='__main__':
    df=load();c=choose_posthoc(df);s=selected(df,c);c.to_csv(HERE/'posthoc_choices.csv',index=False)
    s.to_csv(HERE/'posthoc_selected_draws.csv',index=False);statistics(s,'condition_posthoc').to_csv(HERE/'posthoc_stats.csv',index=False)
    diagrams(s,'condition_posthoc').to_csv(HERE/'cd_diagrams/posthoc_ranks.csv',index=False)
    print('POSTHOC_FIXED_OP_COMPLETE',len(c),'choices, all tests N=26',flush=True)
