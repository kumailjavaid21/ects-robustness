"""Summarize five-draw robustness results and produce error-bar figures."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .run_robust_v2 import ROOT,OUT,DATASETS,METHODS,ALPHA
from .robustness import CONDITIONS

METHOD_NAMES={'probability':'Probability','TEASER':'TEASER','realistic':'Augmented + mixed cal','oracle':'Augmented + matched cal'}
LABELS={'clean':'Clean','noise_20db':'Noise 20 dB','noise_10db':'Noise 10 dB','noise_5db':'Noise 5 dB',
 'missing_10pct':'Missing 10%','missing_30pct':'Missing 30%','missing_50pct':'Missing 50%'}
KEYS=('accuracy','balanced_accuracy','macro_f1','earliness','combined','test_error','forced_fraction')
KEY_NAMES={'accuracy':'Accuracy','balanced_accuracy':'Balanced accuracy','macro_f1':'Macro-F1',
 'earliness':'Earliness','combined':'Combined score','test_error':'Test error at stop','forced_fraction':'Forced fraction'}
LARGE=('FordA','ECG5000','StarLightCurves','Wafer','ElectricDevices','Crop')
def collect():
    result={}
    for dataset in DATASETS:
        result[dataset]=[]
        for draw in range(5):
            path=OUT/'records'/f'{dataset}_draw{draw}.json'
            if not path.exists():raise FileNotFoundError(f'Missing completed draw: {path}')
            r=json.loads(path.read_text())
            if r['dataset']!=dataset or r['draw']!=draw or set(r['conditions'])!=set(CONDITIONS):
                raise ValueError('Incomplete or mismatched draw '+str(path))
            result[dataset].append(r)
    return result
def values(rows,names,condition,method,key):
    return np.array([np.mean([rows[name][draw]['conditions'][condition]['results'][method][key]
                              for name in names]) for draw in range(5)],float)
def fmt(a):return f'{float(np.mean(a)):.3f} +/- {float(np.std(a,ddof=1)):.3f}'
def figure(rows):
    fig,axes=plt.subplots(1,2,figsize=(13.5,6),sharey=True,constrained_layout=True)
    colors={'probability':'#415a77','TEASER':'#6a4c93','realistic':'#b51f3c','oracle':'#188a89'}
    panels=[(['clean','noise_20db','noise_10db','noise_5db'],['Clean','20 dB','10 dB','5 dB'],'Gaussian noise'),
            (['clean','missing_10pct','missing_30pct','missing_50pct'],['Clean','10%','30%','50%'],'Missing samples')]
    for ax,(conds,ticks,title) in zip(axes,panels):
        for method in METHODS:
            arrays=[values(rows,DATASETS,c,method,'test_error') for c in conds]
            means=[np.mean(x) for x in arrays];sds=[np.std(x,ddof=1) for x in arrays]
            ax.errorbar(range(4),means,yerr=sds,marker='o',linewidth=2,capsize=3,
                        color=colors[method],label=METHOD_NAMES[method])
        ax.axhline(ALPHA,color='#444',linestyle='--',linewidth=1.1,label='0.10 reference')
        ax.set_xticks(range(4),ticks);ax.set_title(title);ax.set_xlabel('Corruption level')
        ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.22)
    axes[0].set_ylabel('Error at stopping (unweighted dataset mean)')
    handles,labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncols=3,frameon=False)
    fig.suptitle('ECTS robustness: mean +/- SD over five paired draws',fontsize=14)
    with (OUT/'reliability_vs_corruption_v2.png').open('xb') as f:fig.savefig(f,format='png',dpi=170)
    plt.close(fig)
def write_csv(rows):
    with (OUT/'metrics_draws.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['dataset','draw','condition','method',*KEYS])
        for name in DATASETS:
            for draw in range(5):
                for cond in CONDITIONS:
                    for method in METHODS:
                        v=rows[name][draw]['conditions'][cond]['results'][method]
                        w.writerow([name,draw,cond,method,*[v[k] for k in KEYS]])
    with (OUT/'metrics_mean_std.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['dataset','condition','method','metric','mean','sample_std','n_draws'])
        for name in (*DATASETS,'macro_all_8','macro_large_6'):
            names=DATASETS if name=='macro_all_8' else LARGE if name=='macro_large_6' else (name,)
            for cond in CONDITIONS:
                for method in METHODS:
                    for key in KEYS:
                        a=values(rows,names,cond,method,key)
                        w.writerow([name,cond,method,key,float(a.mean()),float(a.std(ddof=1)),len(a)])
def summary(rows):
    lines=['# ECTS robustness: mixed calibration versus matched oracle','',
      'Eight UCR datasets; each cell is mean +/- sample standard deviation over five independent train-augmentation and test-corruption draws (ddof=1). Macro rows average the eight datasets within each draw before taking the five-draw mean and SD; they are not confidence intervals.',
      '','Each draw trains one augmented 672-feature MiniRocket/logistic model per checkpoint. Training retains clean fit examples and adds one noise/missing-augmented copy. The fit subset is capped at 4,000 examples, calibration at 2,000, and validation at roughly 400; unused training rows remain untouched. All models use causal running normalization.',
      '', '**Realistic:** one fixed set of sequential alpha=.10 thresholds per draw is calibrated on a balanced mixture of six corruption types and levels; it is reused unchanged for clean and every corrupted test condition. It does not use test condition or test labels.',
      '**Oracle reference:** the same augmented model is calibrated separately on held-out data matched to each test condition. It uses knowledge of the condition and is not guaranteed to outperform mixed calibration in a finite sample.',
      '','Probability and aeon TEASER use fixed clean-trained v2-style models and clean validation/training; TEASER uses three-fold logistic cross-validation on shared unsupervised MiniRocket features (two folds if necessary) and a linear SGD one-class safety classifier; their clean results are identical across draws. They were not designed to guarantee the .10 error target. Separate calibrations and test corruptions use independent seeds; severity levels within a draw share noise/masks for paired comparison.',
      '', 'Noise is specified in archived normalized units using fit-set signal power. Missing data use causal forward fill with zero before the first observation. Initial archive preprocessing may contain unknown non-affine effects. One fixed data split is used per dataset; five draws do not capture split or dataset sampling uncertainty.',
      '', 'The per-checkpoint accepted calibration groups have empirical error <= .10, but the first-stop test error is not guaranteed: thresholds are data-dependent and any remaining cases are forced at the final checkpoint.',
      '', '## Dataset sizes','',
      '| Dataset | Train / test x length | Fit / validation / calibration | Missing calibration classes |',
      '|---|---:|---:|---|']
    beats=sum(np.mean(values(rows,DATASETS,c,'realistic','test_error')) <
              min(np.mean(values(rows,DATASETS,c,m,'test_error')) for m in ('probability','TEASER')) for c in CONDITIONS)
    v5=values(rows,DATASETS,'noise_5db','realistic','test_error')
    vm=values(rows,DATASETS,'missing_50pct','realistic','test_error')
    c5=sum(np.mean(values(rows,(n,),'noise_5db','realistic','test_error'))<=ALPHA+1e-12 for n in DATASETS)
    cm=sum(np.mean(values(rows,(n,),'missing_50pct','realistic','test_error'))<=ALPHA+1e-12 for n in DATASETS)
    lines[2:2]=[f'**Verdict:** Mixed calibration has lower eight-dataset mean error than both baselines in {beats}/7 conditions, but does not reliably meet the 10% target: at 5 dB mean error is {fmt(v5)} with {c5}/8 datasets meeting the target; at 50% missingness it is {fmt(vm)} with {cm}/8 meeting it. The mixed rule generally waits longer and does not dominate on the combined score.','']
    for name in DATASETS:
        b=rows[name][0]['base'];tr,L=b['train_shape'];te,_=b['test_shape']
        lines.append(f"| {name} | {tr} / {te} x {L} | {b['fit_count']} / {b['validation_count']} / {b['calibration_count']} | {b['missing_cal_classes']} |")
    lines+=['','ECG5000 has only two training examples of its rarest class, both reserved for fitting. TwoLeadECG and ItalyPowerDemand have very small calibration splits; a balanced six-type mixture is impossible when fewer than six calibration examples are available. ItalyPowerDemand checkpoints start at 9/24 observations because MiniRocket needs at least nine samples.']
    for key in KEYS:
        lines+=['',f'## Eight-dataset mean: {KEY_NAMES[key]}','',
                '| Condition | '+' | '.join(METHOD_NAMES[x] for x in METHODS)+' |',
                '|---|---:|---:|---:|---:|']
        for cond in CONDITIONS:
            lines.append('| '+LABELS[cond]+' | '+' | '.join(fmt(values(rows,DATASETS,cond,m,key)) for m in METHODS)+' |')
    lines+=['','## Realistic versus oracle: mean test-error difference','',
        'Positive numbers mean mixed calibration has higher error; negative numbers mean it has lower error.','',
        '| Condition | Mixed minus oracle, mean +/- SD | Datasets where mixed mean error <= .10 |',
        '|---|---:|---:|']
    for cond in CONDITIONS:
        diff=values(rows,DATASETS,cond,'realistic','test_error')-values(rows,DATASETS,cond,'oracle','test_error')
        count=sum(np.mean(values(rows,(name,),cond,'realistic','test_error'))<=ALPHA+1e-12 for name in DATASETS)
        lines.append(f'| {LABELS[cond]} | {fmt(diff)} | {count}/8 |')
    lines+=['','## Six larger datasets: mean test error','',
      '| Condition | '+' | '.join(METHOD_NAMES[x] for x in METHODS)+' |',
      '|---|---:|---:|---:|---:|']
    for cond in CONDITIONS:
        lines.append('| '+LABELS[cond]+' | '+' | '.join(fmt(values(rows,LARGE,cond,m,'test_error')) for m in METHODS)+' |')
    lines+=['','## Per-dataset metrics','',
        'Cells: accuracy / balanced accuracy / macro-F1 / earliness / combined / test error / forced fraction; every value is mean +/- sample SD over the five draws.']
    for name in DATASETS:
        lines+=['',f'### {name}','',
            '| Condition | '+' | '.join(METHOD_NAMES[x] for x in METHODS)+' |',
            '|---|---:|---:|---:|---:|']
        for cond in CONDITIONS:
            cells=[]
            for method in METHODS:
                cells.append(' / '.join(fmt(values(rows,(name,),cond,method,k)) for k in KEYS))
            lines.append('| '+LABELS[cond]+' | '+' | '.join(cells)+' |')
    lines+=['','## Artifacts','',
      'The error-bar plot, all individual-draw predictions, mixed and oracle calibration thresholds, split indices, checkpoint models, metrics_draws.csv and metrics_mean_std.csv are saved in runs/ects_robust_v2/. The earlier project data and models were only read. The first pilot reported a syntax error before fitting; it was corrected, and all dataset workers finished without recorded experiment errors or optimizer convergence warnings.']
    return '\n'.join(lines)+'\n'
def main():
    rows=collect();figure(rows);write_csv(rows)
    with (OUT/'summary.md').open('x') as f:f.write(summary(rows))
    print('ROBUST_V2_REPORT_DONE',flush=True)
if __name__=='__main__':main()
