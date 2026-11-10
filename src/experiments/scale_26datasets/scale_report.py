"""Aggregate five paired draws and perform dataset-level matched-earliness tests."""
import json,sys,math,argparse
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon,rankdata,studentized_range,friedmanchisquare,spearmanr
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from scale_worker import MAIN,CAVEAT
from ects.robustness import CONDITIONS
METHODS=['probability','TEASER','ECEC','aug_probability','aug_TEASER','ours']
METRICS=['accuracy','balanced_accuracy','macro_f1','error','earliness','combined','forced_fraction']
NAMES={'probability':'Probability','TEASER':'TEASER','ECEC':'ECEC adaptation','aug_probability':'Aug + probability','aug_TEASER':'Aug + TEASER','ours':'Aug + mixed calibration'}
DOMAINS={'FordA':'engine / sensor','FordB':'engine / sensor','ECG5000':'ECG / health','StarLightCurves':'astronomy','Wafer':'semiconductor / sensor','ElectricDevices':'device','Crop':'satellite / spectral','NonInvasiveFetalECGThorax1':'ECG / health','PhalangesOutlinesCorrect':'medical image outline','HandOutlines':'image outline','UWaveGestureLibraryX':'motion','TwoPatterns':'synthetic','Strawberry':'spectroscopy','MiddlePhalanxOutlineCorrect':'medical image outline','EthanolLevel':'chemical / sensor','MixedShapesRegularTrain':'image shape','SwedishLeaf':'image outline','FaceAll':'image outline','ShapesAll':'image shape','SyntheticControl':'synthetic','EOGHorizontalSignal':'health / ocular','CricketX':'motion','ChlorineConcentration':'chemical / sensor','SemgHandMovementCh2':'health / muscle','LargeKitchenAppliances':'device','InsectWingbeatSound':'acoustic / sensor','TwoLeadECG':'ECG / health','ItalyPowerDemand':'device / power'}
def table(headers,rows):
    def clean(x):return str(x).replace('|','\\|').replace('\n',' ')
    return '| '+' | '.join(map(clean,headers))+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(clean,r))+' |\n' for r in rows)
def pm(mean,sd,digits=4):
    return f'{mean:.{digits}f} ? {sd:.{digits}f}' if np.isfinite(mean) and np.isfinite(sd) else 'n/a'
def aggframe(records):
    rows=[]
    for rec in records:
        name=rec['dataset'];cohort='main' if name in MAIN else 'small-calibration'
        base={'dataset':name,'cohort':cohort,'draw':rec['draw'],'condition':rec['condition'],'test_size':rec['test_size'],'series_length':rec['series_length']}
        for j,q in enumerate(rec['fixed']):
            for m in ('clean','augmented'):
                rows.append({**base,'kind':'fixed','method':m,'setting':round((j+2)/10,2),'actual_checkpoint_fraction':q['actual_fraction'],**{k:q[m][k] for k in METRICS}})
        for m,results in rec['frontiers'].items():
            for q in results:
                rows.append({**base,'kind':'frontier','method':m,'setting':q['setting'],'actual_checkpoint_fraction':np.nan,**{k:q[k] for k in METRICS}})
    return pd.DataFrame(rows)
def complete_records():
    records=[];missing=[]
    for name in MAIN+CAVEAT:
        for draw in range(5):
            for cond in CONDITIONS:
                p=HERE/'records'/f'{name}_draw{draw}_{cond}.json'
                if p.exists():
                    q=json.loads(p.read_text())
                    if q['dataset']!=name or q['draw']!=draw or q['condition']!=cond:raise ValueError('Record mismatch '+str(p))
                    records.append(q)
                else:missing.append((name,draw,cond))
    return records,missing
def interpolate(df,target):
    # Duplicate time points use the first prespecified sweep setting, never lowest test error.
    arr=df.sort_values(['earliness','setting']).drop_duplicates('earliness',keep='first')
    x=arr['earliness'].to_numpy();y=arr['error'].to_numpy()
    if len(x)==0 or target<x[0]-1e-10 or target>x[-1]+1e-10:return np.nan
    return float(np.interp(target,x,y))
def matches(df):
    rows=[]
    for (name,draw,cond),g in df[(df.cohort=='main')&(df.kind=='frontier')].groupby(['dataset','draw','condition'],sort=False):
        ours=g[(g.method=='ours')&np.isclose(g.setting,.1)]
        if len(ours)!=1:continue
        e=float(ours.earliness.iloc[0]);err=float(ours.error.iloc[0])
        for m in METHODS[:-1]:
            ref=interpolate(g[g.method==m],e)
            rows.append({'dataset':name,'draw':draw,'condition':cond,'baseline':m,'target_earliness':e,
                         'ours_error':err,'baseline_error_interpolated':ref,'margin_baseline_minus_ours':ref-err})
    return pd.DataFrame(rows)
def grid_matches(df):
    rows=[]
    for (name,draw,cond),g in df[(df.cohort=='main')&(df.kind=='frontier')].groupby(['dataset','draw','condition'],sort=False):
        for target in (.3,.4,.5,.6,.7):
            our_error=interpolate(g[g.method=='ours'],target)
            for m in METHODS[:-1]:
                ref=interpolate(g[g.method==m],target)
                rows.append({'dataset':name,'draw':draw,'condition':cond,'baseline':m,'target_earliness':target,
                  'ours_error_interpolated':our_error,'baseline_error_interpolated':ref,'margin_baseline_minus_ours':ref-our_error})
    return pd.DataFrame(rows)
def stats(matches_df):
    results=[]
    for cond in CONDITIONS:
        for m in METHODS[:-1]:
            a=matches_df[(matches_df.condition==cond)&(matches_df.baseline==m)].dropna(subset=['margin_baseline_minus_ours'])
            per=a.groupby('dataset').margin_baseline_minus_ours.agg(['mean','count'])
            per=per[per['count']==5];d=per['mean'].to_numpy();n=len(d)
            p=np.nan;w=np.nan
            if n>=2:
                if np.all(np.abs(d)<1e-14):p=1.;w=0.
                else:
                    r=wilcoxon(d,zero_method='wilcox',alternative='two-sided',method='auto');p=float(r.pvalue);w=float(r.statistic)
            drawmean=a.groupby('draw').margin_baseline_minus_ours.mean()
            perdraw=a.groupby('draw').dataset.nunique()
            results.append({'condition':cond,'baseline':m,'n_datasets_all_5_draws':n,'n_min_matched_per_draw':int(perdraw.min()) if len(perdraw) else 0,
                'n_max_matched_per_draw':int(perdraw.max()) if len(perdraw) else 0,'n_draws':len(drawmean),
                'margin_mean_over_draws':float(drawmean.mean()) if len(drawmean) else np.nan,
                'margin_sample_sd_over_draws':float(drawmean.std(ddof=1)) if len(drawmean)>1 else np.nan,
                'wins':int(np.sum(d>1e-12)),'ties':int(np.sum(np.abs(d)<=1e-12)),'losses':int(np.sum(d< -1e-12)),
                'wilcoxon_stat':w,'p_raw':p})
    s=pd.DataFrame(results);s['p_holm']=np.nan
    indices=s.index[s.p_raw.notna()].tolist();indices.sort(key=lambda i:s.loc[i,'p_raw']);running=0.
    for rank,i in enumerate(indices):
        running=max(running,min(1.,(len(indices)-rank)*s.loc[i,'p_raw']));s.loc[i,'p_holm']=running
    return s

def plot_prefix(macro):
    fig,axes=plt.subplots(2,4,figsize=(18,8),sharex=True,sharey=True);colors={'clean':'#2878b5','augmented':'#d95f02'}
    for ax,cond in zip(axes.flat,CONDITIONS):
        for m in ('clean','augmented'):
            a=macro[(macro.cohort=='main')&(macro.condition==cond)&(macro.kind=='fixed')&(macro.method==m)].sort_values('setting')
            if len(a):ax.errorbar(a.earliness_mean,a.accuracy_mean,yerr=a.accuracy_sd,marker='o',ms=3,label=m,color=colors[m],capsize=2)
        ax.set_title(cond);ax.grid(alpha=.2);ax.set_xlabel('Observed fraction');ax.set_ylabel('Accuracy')
    axes.flat[-1].axis('off');axes.flat[0].legend();fig.suptitle('Fixed checkpoint: 26-dataset mean; bars = sample SD across five draws');fig.tight_layout()
    fig.savefig(HERE/'accuracy_vs_prefix.png',dpi=180);plt.close(fig)

def plot_frontier(macro):
    fig,axes=plt.subplots(2,4,figsize=(19,9),sharex=True,sharey=True)
    colors=['#2878b5','#763b9f','#999999','#7ba23f','#d6a20e','#d95f02']
    for ax,cond in zip(axes.flat,CONDITIONS):
        for m,col in zip(METHODS,colors):
            a=macro[(macro.cohort=='main')&(macro.condition==cond)&(macro.kind=='frontier')&(macro.method==m)].sort_values('earliness_mean')
            if len(a):ax.errorbar(a.earliness_mean,a.error_mean,xerr=a.earliness_sd,yerr=a.error_sd,marker='o',ms=3,label=NAMES[m],color=col,capsize=1.5,lw=1)
        ax.set_title(cond);ax.set_xlabel('Earliness');ax.set_ylabel('Error');ax.grid(alpha=.2)
    axes.flat[-1].axis('off');h,l=axes.flat[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3);fig.suptitle('Corruption-agnostic stopping sweeps; 26-dataset mean ? draw SD');fig.tight_layout(rect=(0,.08,1,.97))
    fig.savefig(HERE/'earliness_vs_error_frontier.png',dpi=180);plt.close(fig)

def cd_diagrams(matches_df):
    result=[]
    for cond in CONDITIONS:
        a=matches_df[(matches_df.condition==cond)].dropna(subset=['margin_baseline_minus_ours'])
        counts=a.groupby(['dataset','baseline']).draw.nunique();eligible=[n for n in MAIN if all(counts.get((n,m),0)==5 for m in METHODS[:-1])]
        matrix=[]
        for n in eligible:
            q=a[a.dataset==n];ours=q.groupby('draw').ours_error.first().mean()
            matrix.append([q[q.baseline==m].baseline_error_interpolated.mean() for m in METHODS[:-1]]+[ours])
        M=np.array(matrix,dtype=float).reshape(-1,len(METHODS));N=len(M);fig,ax=plt.subplots(figsize=(10,4))
        if N>=3:
            ranks=np.array([rankdata(row,method='average') for row in M]);avg=ranks.mean(0)
            cd=float(studentized_range.ppf(.95,len(METHODS),np.inf)/np.sqrt(2)*np.sqrt(len(METHODS)*(len(METHODS)+1)/(6*N)))
            try:fp=float(friedmanchisquare(*[M[:,i] for i in range(len(METHODS))]).pvalue)
            except ValueError:fp=np.nan
            for i,(m,r) in enumerate(sorted(zip(METHODS,avg),key=lambda z:z[1])):
                ax.scatter(r,i,color='#2878b5');ax.text(r+.05,i,NAMES[m],va='center',fontsize=9)
            ax.plot([1,1+cd],[len(METHODS)+.1,len(METHODS)+.1],color='black',lw=3);ax.text(1,len(METHODS)+.3,f'Nemenyi CD={cd:.3f}',fontsize=9)
            ax.set_xlim(.8,max(len(METHODS)+1.9,1+cd));ax.set_ylim(-1,len(METHODS)+.7);ax.set_yticks([]);ax.set_xticks(range(1,len(METHODS)+1));ax.set_xlabel('Mean rank (1 = lowest matched error)')
            ax.set_title(f'{cond}: N={N} complete matched datasets; Friedman p={fp:.4g}')
            row={'condition':cond,'N':N,'critical_difference':cd,'friedman_p':fp,**{m:float(r) for m,r in zip(METHODS,avg)}}
        else:
            ax.text(.5,.5,f'Insufficient complete matched overlap: N={N}; no Nemenyi test',ha='center',va='center',transform=ax.transAxes);ax.set_axis_off()
            row={'condition':cond,'N':N,'critical_difference':np.nan,'friedman_p':np.nan}
        fig.tight_layout();fig.savefig(HERE/'cd_diagrams'/(cond+'.png'),dpi=160);plt.close(fig);result.append(row)
    return pd.DataFrame(result)
def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args()
    records,missing=complete_records()
    print('RECORDS',len(records),'MISSING',len(missing),'datasets represented',len(set(x['dataset'] for x in records)),flush=True)
    if args.check:
        if records:
            d=aggframe(records);print('ROWS',len(d),'matched',len(matches(d)),flush=True)
        return
    if missing:raise RuntimeError('Incomplete requested benchmark: '+repr(missing[:15]))
    df=aggframe(records);df.to_csv(HERE/'per_dataset_draws.csv',index=False)
    keys=['cohort','condition','kind','method','setting']
    per=df.groupby(keys+['dataset'],dropna=False).agg(**{k+'_mean':(k,'mean') for k in METRICS},**{k+'_sd':(k,'std') for k in METRICS},n_draws=('draw','nunique'),test_size=('test_size','first'),series_length=('series_length','first')).reset_index()
    per.to_csv(HERE/'per_dataset_metrics.csv',index=False)
    draws=df.groupby(keys+['draw'],dropna=False).agg(**{k:(k,'mean') for k in METRICS},n_datasets=('dataset','nunique')).reset_index()
    macro=draws.groupby(keys,dropna=False).agg(**{k+'_mean':(k,'mean') for k in METRICS},**{k+'_sd':(k,'std') for k in METRICS},n_draws=('draw','nunique'),n_datasets_min=('n_datasets','min'),n_datasets_max=('n_datasets','max')).reset_index()
    macro.to_csv(HERE/'macro_metrics.csv',index=False)
    matching=matches(df);matching.to_csv(HERE/'matched_earliness_draws.csv',index=False)
    stat=stats(matching);stat.to_csv(HERE/'significance_tests.csv',index=False)
    grid=grid_matches(df);grid.to_csv(HERE/'matched_frontier_grid.csv',index=False)
    plot_prefix(macro);plot_frontier(macro);cd=cd_diagrams(matching);cd.to_csv(HERE/'cd_diagrams/mean_ranks.csv',index=False)
    metas={name:json.loads((HERE/'draw_meta'/f'{name}_base.json').read_text()) for name in MAIN+CAVEAT}
    paragraphs=[]
    paragraphs.append('# ECTS scale benchmark\n\n')
    paragraphs.append('**Status:** complete 26-dataset main benchmark, five paired draws, seven conditions. Two small-calibration sets are separate and excluded from all main macro statistics. No target alpha is a finite-sample guarantee.\n\n')
    paragraphs.append('## Datasets and splits\n\nDataset files are from the pinned [UCR-2018 mirror](https://huggingface.co/datasets/ChengsenWang/UCR-2018), using its original train/test split. New files were SHA-256 verified against the repository LFS hashes. Domain labels are broad descriptors; related outline datasets may not be statistically independent. The calibration split follows v2 unchanged, including its class-reservation rule.\n\n')
    rows=[]
    for name in MAIN+CAVEAT:
        x=metas[name];rows.append([name,'main' if name in MAIN else 'caveat',DOMAINS.get(name,'other'),x['train_size'],x['test_size'],x['length'],x['n_classes'],x['fit_count'],x['validation_count'],x['calibration_count'],','.join(map(str,x['missing_cal_classes'])) or 'none'])
    paragraphs.append(table(['Dataset','Cohort','Domain','Train','Test','Length','Classes','Fit','Validation','Calibration','Classes absent from calibration'],rows)+'\n')
    paragraphs.append('Three initially downloaded candidates (PLAID, MelbournePedestrian, AllGestureWiimoteX) contain non-finite data and were excluded to preserve the existing finite-series preprocessing. They were kept on disk. Some included datasets have substantially fewer than 200 calibration examples despite meeting the train-size criterion; interpret their reliability with care.\n\n')
    paragraphs.append('## Methods and analysis\n\nEvery stopping policy and threshold grid is fixed without using test labels or corruption identity. The model and stream generation follow v2: causal running normalization, 672-kernel MiniRocket + logistic regression per checkpoint, the same train/validation/calibration split and five deterministic paired draws. Alpha thresholds use one balanced mixed-corruption calibration set per draw and are reused unchanged across test conditions. Clean probability and clean TEASER use clean checkpoints; augmentation ablations use the same augmented checkpoints as ours. Augmented TEASER estimates safety out of fold while grouping each original with its augmented copy. Its safety estimator therefore requires separate fitting, even though its base classifiers are reused.\n\n')
    paragraphs.append('ECEC is an adaptation of the [authors? implementation](https://github.com/junweilvhfut/ECEC): per-checkpoint predicted-class precision is estimated on clean validation data, then confidence for the current class is one minus the product of its historical failure probabilities; a fixed threshold grid is swept. MiniRocket replaces its original classifier, and zero-prediction class/checkpoint cells use precision zero. It is not an exact reproduction of the original training protocol.\n\n')
    paragraphs.append('Results average datasets within each draw, then report the mean and sample SD across five draws. This SD reflects only the five augmentation/corruption draws: the UCR train/test and fit/calibration splits are fixed. The combined score is the harmonic mean of accuracy and one minus earliness. Forced final predictions are included in error. For probability/ECEC/ours, forced means no threshold was met even at full length; for TEASER, every full-length decision is counted as forced, consistent with v2. Gaussian noise uses fit-split signal power as in v2. Probability/TEASER/ECEC stopping settings are swept rather than tuned on test labels.\n\n')
    paragraphs.append('Matched-earliness comparisons below use the ?=0.10 operating point and linearly interpolate each baseline?s pre-specified empirical frontier at the same observed earliness, independently per dataset and draw. No extrapolation is used; ties in earliness use the first pre-specified setting rather than choosing by test error. Interpolation is a descriptive post-hoc comparison, not a deployable test-specific stopping policy. The sweep is not pruned into a test-error-selected lower envelope: doing so would select settings using test outcomes. Consequently the comparison does not prove mathematical dominance over every possible baseline mixture. Matched sample counts can differ by method and condition.\n\n')
    paragraphs.append('## Augmented + mixed calibration at ? = 0.10\n\nMacro mean ? sample SD across the five draws. Error includes forced full-length decisions; 0.10 is a target, not a guarantee.\n\n')
    op=[]
    for cond in CONDITIONS:
        a=macro[(macro.cohort=='main')&(macro.kind=='frontier')&(macro.condition==cond)&(macro.method=='ours')&np.isclose(macro.setting,.1)]
        if len(a)!=1:raise ValueError('Missing unique alpha=.10 macro result for '+cond)
        z=a.iloc[0]
        op.append([cond,pm(z.accuracy_mean,z.accuracy_sd),pm(z.balanced_accuracy_mean,z.balanced_accuracy_sd),pm(z.macro_f1_mean,z.macro_f1_sd),pm(z.error_mean,z.error_sd),pm(z.earliness_mean,z.earliness_sd),pm(z.forced_fraction_mean,z.forced_fraction_sd),pm(z.combined_mean,z.combined_sd)])
    paragraphs.append(table(['Condition','Accuracy','Balanced accuracy','Macro-F1','Error','Earliness','Forced fraction','Combined'],op)+'\n')
    paragraphs.append('## Fixed-length error gaps (clean minus augmented)\n\nPositive values favor augmentation.\n\n')
    fixedrows=[]
    for cond in CONDITIONS:
        cells=[]
        for frac in (.2,.4,.6,.8,1.):
            a=df[(df.cohort=='main')&(df.condition==cond)&(df.kind=='fixed')&np.isclose(df.setting,frac)]
            pivot=a.pivot_table(index=['dataset','draw'],columns='method',values='error')
            gap=(pivot['clean']-pivot['augmented']).groupby('draw').mean()
            cells.append(pm(gap.mean(),gap.std(ddof=1)))
        fixedrows.append([cond]+cells)
    paragraphs.append(table(['Condition','20%','40%','60%','80%','100%'],fixedrows)+'\n')
    paragraphs.append('### Does augmentation help at every evaluated prefix?\n\nThis is a macro-average comparison over all nine checkpoints, not a statement that every individual dataset benefits.\n\n')
    qrows=[]
    for cond in CONDITIONS:
        a=df[(df.cohort=='main')&(df.condition==cond)&(df.kind=='fixed')]
        pivot=a.pivot_table(index=['dataset','draw','setting'],columns='method',values='error')
        gaps=(pivot['clean']-pivot['augmented']).groupby(['draw','setting']).mean().groupby('setting').mean()
        qrows.append([cond,'yes' if (gaps>=-1e-12).all() else 'no',f'{gaps.min():.4f}',f'{gaps.max():.4f}',int((gaps< -1e-12).sum())])
    paragraphs.append(table(['Condition','Nonnegative gap at all 9 checkpoints','Minimum gap','Maximum gap','Checkpoints favoring clean'],qrows)+'\n')
    paragraphs.append('## Matched-earliness results, paired tests and win/tie/loss\n\nMargin is baseline error minus our error; positive favors ours. Wilcoxon tests use one mean paired margin per dataset (only datasets matched in all five draws). Holm adjustment covers the full family of valid condition ? baseline comparisons. W/T/L refers to those same datasets, with a numerical tolerance of 1e-12. A mean of matched draws can include a different subset of datasets per draw; the coverage column makes that explicit.\n\n')
    sr=[]
    for _,r in stat.iterrows():
        sr.append([r.condition,NAMES[r.baseline],pm(r.margin_mean_over_draws,r.margin_sample_sd_over_draws),f"{int(r.n_min_matched_per_draw)}?{int(r.n_max_matched_per_draw)}",int(r.n_datasets_all_5_draws),f"{int(r.wins)}/{int(r.ties)}/{int(r.losses)}",f"{r.p_raw:.4g}" if np.isfinite(r.p_raw) else 'n/a',f"{r.p_holm:.4g}" if np.isfinite(r.p_holm) else 'n/a'])
    paragraphs.append(table(['Condition','Comparator','Matched error margin ? draw SD','Datasets/draw','Datasets in test','W/T/L','Wilcoxon p','Holm p'],sr)+'\n')
    paragraphs.append('## Descriptive matched frontier margins at common earliness levels\n\nEach cell is the baseline-minus-ours error margin, mean ? sample SD across the five draw-specific dataset means, followed by the minimum and maximum number of matched datasets per draw. Values are descriptive interpolation without extrapolation; missing overlap is shown as n/a.\n\n')
    gr=[]
    for cond in CONDITIONS:
        for method in METHODS[:-1]:
            cells=[]
            for target in (.3,.4,.5,.6,.7):
                a=grid[(grid.condition==cond)&(grid.baseline==method)&np.isclose(grid.target_earliness,target)].dropna(subset=['margin_baseline_minus_ours'])
                d=a.groupby('draw').margin_baseline_minus_ours.mean();counts=a.groupby('draw').dataset.nunique()
                cells.append((pm(d.mean(),d.std(ddof=1))+' (n='+str(int(counts.min()))+'?'+str(int(counts.max()))+')') if len(d)==5 else 'n/a')
            gr.append([cond,NAMES[method]]+cells)
    paragraphs.append(table(['Condition','Comparator','0.3','0.4','0.5','0.6','0.7'],gr)+'\n')
    paragraphs.append('## Nemenyi critical-difference diagrams\n\nEach diagram ranks methods by matched error on datasets with all methods and all five draws matched; results use the Friedman omnibus test and the Nemenyi critical difference. Where fewer than three datasets overlap, the panel explicitly reports insufficient data. Common-overlap selection and post-hoc interpolation limit inference.\n\n')
    for _,r in cd.iterrows():
        fp=f"{r.friedman_p:.4g}" if np.isfinite(r.friedman_p) else 'n/a'
        paragraphs.append(f"- [{r.condition}](cd_diagrams/{r.condition}.png): complete overlap N={int(r.N)}; Friedman p={fp}.\n")
    paragraphs.append('\n## Clean-data cost and series length\n\nAt each fixed checkpoint, positive error gap means augmentation helps; negative means a clean-model advantage. Spearman correlations below are exploratory across datasets, without multiplicity correction. They do not establish that series length causes the advantage.\n\n')
    rr=[]
    for cond in CONDITIONS:
        for frac in (.2,.6,1.):
            a=df[(df.cohort=='main')&(df.condition==cond)&(df.kind=='fixed')&np.isclose(df.setting,frac)]
            p=a.pivot_table(index=['dataset','draw','series_length'],columns='method',values='error').reset_index();p['gap']=p.clean-p.augmented
            d=p.groupby(['dataset','series_length']).gap.mean().reset_index();rho,pv=spearmanr(d.series_length,d.gap)
            rr.append([cond,frac,len(d),f'{d.gap.mean():.4f}',f'{rho:.4f}',f'{pv:.4g}'])
    paragraphs.append(table(['Condition','Prefix','Datasets','Mean error gap','Spearman ?(length,gap)','Unadjusted p'],rr)+'\n')
    paragraphs.append('### Dataset-level augmentation effects at the 40% checkpoint\n\nThe three largest and smallest paired mean error gaps are shown for each condition; positive favors augmented training. These descriptive rankings use test outcomes and are not method-selection rules.\n\n')
    dr=[]
    for cond in CONDITIONS:
        a=df[(df.cohort=='main')&(df.condition==cond)&(df.kind=='fixed')&np.isclose(df.setting,.4)]
        t=a.pivot_table(index=['dataset','draw'],columns='method',values='error');q=(t['clean']-t['augmented']).groupby('dataset').mean().sort_values()
        fmt=lambda ss:', '.join(f'{n} ({v:+.3f})' for n,v in ss.items())
        dr.append([cond,fmt(q.tail(3).iloc[::-1]),fmt(q.head(3))])
    paragraphs.append(table(['Condition','Largest augmentation advantages','Smallest gaps / clean advantages'],dr)+'\n')
    paragraphs.append('## Verdict and limitations\n\n')
    winners=[]
    for cond in CONDITIONS:
        a=stat[stat.condition==cond];allcovered=bool((a.n_datasets_all_5_draws==len(MAIN)).all())
        sig=bool(((a.p_holm<.05)&(a.margin_mean_over_draws>0)).all())
        if allcovered and sig:verdict='yes, for the prespecified ?=0.10 comparison and all five comparators'
        elif sig:verdict='positive and Holm-significant on matched subsets; incomplete overlap prevents an all-dataset claim'
        else:verdict='no universal significant advantage established'
        winners.append([cond,verdict])
    paragraphs.append(table(['Condition','Equal-earliness verdict'],winners)+'\n')
    paragraphs.append('These are exploratory cross-dataset comparisons of synthetic corruptions on a single archive with one fixed split per dataset. Dataset correlations, tuning and interval overlap limit a broad significance or external-validity claim. The empirical mixed-calibration rule is not a formal risk-control guarantee, especially with small calibration sets and forced full-length predictions. Causal normalization applies to the archived observations supplied to this pipeline; any upstream whole-series preprocessing performed before archival has not been audited or undone, and simulated corruption is applied to those archived values.\n\n')
    paragraphs.append('## Integrity and convergence checks\n\n')
    audit=json.loads((HERE/'source_audit.json').read_text())
    changed=len(audit['changed']);removed=len(audit['removed']);added=len(audit['added'])
    paragraphs.append(f"Prior-source snapshot comparison: {audit['files_hashed_before']} files before; {changed} changed, {removed} removed and {added} added after. Details: [source_audit.json](source_audit.json).\n\n")
    convergence=pd.read_csv(HERE/'model_convergence_audit.csv');hit=int(convergence.hit_iteration_limit.sum())
    notices=json.loads((HERE/'logs/warnings_and_errors.json').read_text())
    paragraphs.append(f"Checkpoint model audit: {len(convergence)} models, {hit} reached the configured logistic iteration limit; {len(notices['audit_failures'])} audit read failures and {len(notices['notices'])} warning/error log lines. See [model_convergence_audit.csv](model_convergence_audit.csv) and [warnings_and_errors.json](logs/warnings_and_errors.json).\n\n")
    paragraphs.append('TEASER safety uses out-of-fold logistic predictions but, as in v2, the unsupervised MiniRocket feature transform is fitted once before those folds; this is not fully nested cross-validation. Augmented TEASER keeps each original and its augmented counterpart in the same safety fold.\n\n')
    paragraphs.append('Per-dataset and macro metrics, the matched pairs, test statistics, full draw-level records, code, models and figures are in this directory.\n')
    (HERE/'summary.md').write_text(''.join(paragraphs),encoding='utf-8')
    print('AGGREGATION_COMPLETE',len(records),len(df),flush=True)
if __name__=='__main__':main()
