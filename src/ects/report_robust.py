"""Create robustness summary and plot from finished condition records."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .run_robust import ROOT,OUT,DATASETS,METHODS,ALPHA
from .robustness import CONDITIONS

NAMES={'probability':'Probability','TEASER':'TEASER','plain_clean_cal':'Plain, clean cal',
       'plain_matched_cal':'Plain, matched cal','aug_matched_cal':'Augmented, matched cal'}
LABELS={'clean':'Clean','noise_20db':'Noise 20 dB','noise_10db':'Noise 10 dB','noise_5db':'Noise 5 dB',
        'missing_10pct':'Missing 10%','missing_30pct':'Missing 30%','missing_50pct':'Missing 50%'}
def collect():
    rows=[]
    for dataset in DATASETS:
        path=OUT/'records'/f'{dataset}_complete.json'
        if not path.exists():raise FileNotFoundError(f'Missing completed dataset: {dataset}')
        rows.append(json.loads(path.read_text()))
    return rows
def average(rows,condition,method,key):
    return float(np.mean([r['conditions'][condition]['results'][method][key] for r in rows]))
def table_cell(v):
    return ' / '.join(f"{v[k]:.3f}" for k in ('accuracy','balanced_accuracy','macro_f1','earliness','combined','test_error','forced_fraction'))
def make_plot(rows):
    fig,axes=plt.subplots(1,2,figsize=(12.5,5.5),sharey=True,constrained_layout=True)
    colors={'probability':'#415a77','TEASER':'#6a4c93','plain_clean_cal':'#dc7b21',
            'plain_matched_cal':'#188a89','aug_matched_cal':'#b51f3c'}
    panels=[(['clean','noise_20db','noise_10db','noise_5db'],['Clean','20 dB','10 dB','5 dB'],'Gaussian noise (larger corruption to the right)'),
            (['clean','missing_10pct','missing_30pct','missing_50pct'],['Clean','10%','30%','50%'],'Missing samples')]
    for ax,(conds,ticks,title) in zip(axes,panels):
        for method in METHODS:
            vals=[average(rows,c,method,'test_error') for c in conds]
            ax.plot(range(4),vals,marker='o',linewidth=2,markersize=5,color=colors[method],label=NAMES[method])
        ax.axhline(ALPHA,color='#444444',linewidth=1.2,linestyle='--',label='0.10 reference')
        ax.set_xticks(range(4),ticks);ax.set_title(title);ax.set_xlabel('Corruption level')
        ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.22)
    axes[0].set_ylabel('Test error at stopping (unweighted dataset mean)')
    handles,labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncols=3,frameon=False)
    fig.suptitle('ECTS robustness: causal inputs, one paired corruption draw',fontsize=14)
    with (OUT/'reliability_vs_corruption.png').open('xb') as f:fig.savefig(f,format='png',dpi=170)
    plt.close(fig)
def summary(rows,status):
    lines=['# ECTS robustness: causal checkpoint MiniRocket','',
      'Four UCR datasets: FordA, ECG5000, StarLightCurves, and Wafer. The original v2 fit/calibration/test splits and 20%-100% checkpoint locations are retained. All models use causal running normalization and one CPU thread.',
      '', 'Each test corruption is a single deterministic, paired draw: Gaussian noise uses a fixed scale derived from fit-set signal power in the archived normalized units; the same noise draw is scaled across SNRs. Missingness uses a shared uniform draw for nested masks, causal forward fill, and zero before the first observation. The mask is not a separate classifier channel.',
      '', 'Augmented training retains every clean fit example and adds one augmented copy. For each copy, noise is independently selected from none, 20, 10, and 5 dB, and missingness from none, 10%, 30%, and 50%; the two can occur together. A separate MiniRocket+logistic model is trained per checkpoint.',
      '', '**Calibration access:** plain clean-cal uses the unmodified v2 alpha=.10 thresholds. Plain matched-cal and augmented matched-cal each calibrate on a separately corrupted copy of the held-out calibration split with the same corruption type and level as the test condition. This assumes the test condition is known, but uses no test labels. The matched plain ablation separates recalibration from model augmentation.',
      '', 'Probability thresholds were selected on clean validation data. Aeon TEASER was refitted on clean fit data using the same v2 checkpoint models and its linear one-class safety classifier; its forced fraction counts cases reaching the final checkpoint. These baselines do not claim an error target.',
      '', 'All error-at-stop values are empirical. Sequential calibration controls its selected calibration groups, not necessarily the unseen stopping error; repeated selection, small groups, distribution shift and forced final decisions provide no finite-sample guarantee. ECG5000 has an unrepresented rare class in its calibration split.',
      '', 'Cells: accuracy / balanced accuracy / macro-F1 / earliness / combined score / test error / forced fraction. Combined is the harmonic mean of accuracy and (1-earliness). Means weight the four datasets equally.']
    lines+=['','## Mean across datasets','',
            '| Condition | '+' | '.join(NAMES[m] for m in METHODS)+' |',
            '|---|---:|---:|---:|---:|---:|']
    for cond in CONDITIONS:
        vals=[]
        for method in METHODS:
            keys=('accuracy','balanced_accuracy','macro_f1','earliness','combined','test_error','forced_fraction')
            vals.append(table_cell({k:average(rows,cond,method,k) for k in keys}))
        lines.append('| '+LABELS[cond]+' | '+' | '.join(vals)+' |')
    for r in rows:
        n,L=r['train_shape'];test,_=r['test_shape']
        lines+=['',f"## {r['dataset']} (train {n}, test {test}, fit {r['fit_count']}, calibration {r['calibration_count']})",'',
          '| Condition | '+' | '.join(NAMES[m] for m in METHODS)+' |',
          '|---|---:|---:|---:|---:|---:|']
        for cond in CONDITIONS:
            v=r['conditions'][cond]['results']
            lines.append('| '+LABELS[cond]+' | '+' | '.join(table_cell(v[m]) for m in METHODS)+' |')
    lines+=['','## Is reliability retained?','']
    for cond in ('clean','noise_5db','missing_50pct'):
        vals={m:average(rows,cond,m,'test_error') for m in METHODS}
        errors=[r['conditions'][cond]['results']['aug_matched_cal']['test_error'] for r in rows]
        passed=sum(x<=ALPHA+1e-12 for x in errors)
        lines.append(f"- {LABELS[cond]}: probability {vals['probability']:.3f}, TEASER {vals['TEASER']:.3f}, plain clean-cal {vals['plain_clean_cal']:.3f}, plain matched-cal {vals['plain_matched_cal']:.3f}, augmented matched-cal {vals['aug_matched_cal']:.3f}. Augmented meets .10 in {passed}/{len(rows)} datasets.")
    lines+=['','Augmented-versus-plain differences combine the effects of training and condition-matched calibration. Compare the two matched-cal rows to isolate training augmentation. These are four dataset averages from one seeded corruption draw, not uncertainty bounds.',
      '', 'Saved artifacts: per-condition predictions and calibration thresholds, fit augmentation counts, and augmented checkpoint models are under runs/ects_robust/. Original v2 models and data were read without modification.']
    if status.get('errors'):lines+=['','Errors:']+['- '+e.replace('\n',' ') for e in status['errors']]
    if any(r.get('warnings') for r in rows):lines+=['',f"Optimizer convergence warnings occurred at {sum(len(r['warnings']) for r in rows)} checkpoints; details are in records."]
    return '\n'.join(lines)+'\n'
def main():
    rows=collect();status=json.loads((OUT/'run_status.json').read_text())
    make_plot(rows)
    with (OUT/'summary.md').open('x') as f:f.write(summary(rows,status))
    print('ROBUST_REPORT_DONE',flush=True)
if __name__=='__main__':main()
