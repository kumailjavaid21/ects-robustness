"""Aggregate paired checkpoint replays and describe interpolated operating frontiers."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from frontier_worker import ROOT,OUT,DATASETS,TAUS,ALPHAS,GS
from ects.robustness import CONDITIONS

METHODS=('probability','TEASER','augmented')
METHOD_NAMES={'probability':'Probability','TEASER':'TEASER','augmented':'Augmented + mixed'}
FRACTIONS=np.arange(2,11,dtype=float)/10
SELECT=(.2,.4,.6,.8,1.)
CONDITION_NAMES={'clean':'Clean','noise_20db':'Noise 20 dB','noise_10db':'Noise 10 dB','noise_5db':'Noise 5 dB',
                 'missing_10pct':'Missing 10%','missing_30pct':'Missing 30%','missing_50pct':'Missing 50%'}
KEYS=('accuracy','balanced_accuracy','macro_f1','error','earliness')
SETTINGS={'probability':TAUS,'TEASER':GS,'augmented':ALPHAS}

def fmt(x):
    x=np.asarray(x,dtype=float);return f'{x.mean():+.3f} +/- {x.std(ddof=1):.3f}'
def plain(x):
    x=np.asarray(x,dtype=float);return f'{x.mean():.3f} +/- {x.std(ddof=1):.3f}'
def collect():
    result={}
    for n in DATASETS:
        result[n]=[]
        for d in range(5):
            c={}
            for cond in CONDITIONS:
                path=OUT/'records'/f'{n}_draw{d}_{cond}.json'
                if not path.exists():raise FileNotFoundError(f'Missing replay record {path}')
                r=json.loads(path.read_text())
                if (r['dataset'],r['draw'],r['condition'])!=(n,d,cond):raise ValueError('Record metadata mismatch')
                c[cond]=r
            result[n].append(c)
    return result

def checkpoint(record,fraction):
    L=record['series_length'];j=round(fraction*10)
    # Match the actual ceil-based nominal checkpoint, not a later substitute.
    t=int(np.ceil(j*L/10))
    return next((row for row in record['fixed'] if row['timepoint']==t),None)

def eligible(rows,cond,fraction):
    return [n for n in DATASETS if checkpoint(rows[n][0][cond],fraction) is not None]

def fixed_values(rows,names,cond,fraction,model,key):
    out=[]
    for d in range(5):
        vals=[]
        for name in names:
            x=checkpoint(rows[name][d][cond],fraction)
            if x is not None:vals.append(x[model][key])
        if not vals:raise ValueError('No available checkpoints')
        out.append(np.mean(vals))
    return np.asarray(out,float)

def setting_row(record,method,setting):
    return next(r for r in record['frontiers'][method] if abs(r['setting']-setting)<1e-12)

def front_values(rows,names,cond,method,setting,key):
    return np.array([np.mean([setting_row(rows[n][d][cond],method,setting)[key] for n in names]) for d in range(5)])

def points_for_draw(rows,cond,method,draw):
    return [(float(np.mean([setting_row(rows[n][draw][cond],method,s)['earliness'] for n in DATASETS])),
             float(np.mean([setting_row(rows[n][draw][cond],method,s)['error'] for n in DATASETS])))
             for s in SETTINGS[method]]

def points_for_means(rows,cond,method):
    return [(float(front_values(rows,DATASETS,cond,method,s,'earliness').mean()),
             float(front_values(rows,DATASETS,cond,method,s,'error').mean())) for s in SETTINGS[method]]

def lower_hull(points):
    # Convex lower envelope of fixed settings, allowing randomized mixtures.
    grouped={}
    for x,y in points:grouped[x]=min(grouped.get(x,float('inf')),y)
    pts=sorted(grouped.items());h=[]
    for p in pts:
        while len(h)>=2:
            a,b=h[-2],h[-1]
            cross=(b[0]-a[0])*(p[1]-b[1])-(b[1]-a[1])*(p[0]-b[0])
            if cross<=1e-14:h.pop()
            else:break
        h.append(p)
    return h

def interp(h,x):
    if len(h)==1:
        if abs(x-h[0][0])>1e-10:raise ValueError('Single-point hull')
        return h[0][1]
    if x<h[0][0]-1e-10 or x>h[-1][0]+1e-10:raise ValueError('Interpolation would extrapolate')
    return float(np.interp(x,[p[0] for p in h],[p[1] for p in h]))

def compare_condition(rows,cond):
    # Use the common support of all methods and all draws to report mean +/- SD.
    bydraw=[{m:lower_hull(points_for_draw(rows,cond,m,d)) for m in METHODS} for d in range(5)]
    lo=max(h[m][0][0] for h in bydraw for m in METHODS)
    hi=min(h[m][-1][0] for h in bydraw for m in METHODS)
    targets=[]
    if hi-lo>1e-8:
        for f in (.25,.50,.75):
            x=lo+f*(hi-lo);vals=[]
            for h in bydraw:
                p=interp(h['probability'],x);t=interp(h['TEASER'],x);a=interp(h['augmented'],x)
                vals.append({'probability_margin':p-a,'teaser_margin':t-a,'both_margin':min(p,t)-a,
                             'aug_error':a,'probability_error':p,'teaser_error':t})
            targets.append({'earliness':x,'draws':vals})
    meanh={m:lower_hull(points_for_means(rows,cond,m)) for m in METHODS}
    mlo=max(meanh[m][0][0] for m in METHODS);mhi=min(meanh[m][-1][0] for m in METHODS)
    if mhi<mlo-1e-10:
        verdict='NO SHARED SUPPORT';breaks=[]
    else:
        breaks=sorted(set([mlo,mhi]+[x for m in METHODS for x,y in meanh[m] if mlo<x<mhi]))
        wins=[interp(meanh['probability'],x)-interp(meanh['augmented'],x)>1e-12 and
              interp(meanh['TEASER'],x)-interp(meanh['augmented'],x)>1e-12 for x in breaks]
        verdict='YES' if all(wins) else 'NO'
    return {'strict_5_draw_support':[lo,hi] if hi>=lo else None,'targets':targets,
            'mean_support':[mlo,mhi] if mhi>=mlo else None,'dominates_on_mean_envelopes':verdict,
            'mean_breakpoints_checked':breaks}

def write_csv(rows):
    fixed_fields=['scope','dataset','condition','model','checkpoint_kind','timepoint','nominal_fraction','n_datasets','n_draws']
    fixed_fields += [f'{key}_{suffix}' for key in ('actual_fraction','accuracy','balanced_accuracy','macro_f1','error') for suffix in ('mean','sd')]
    with (OUT/'fixed_length_curves.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fixed_fields);w.writeheader()
        for name in (*DATASETS,'macro'):
            for cond in CONDITIONS:
                for fraction in FRACTIONS:
                    names=eligible(rows,cond,fraction) if name=='macro' else [name]
                    if name!='macro' and name not in eligible(rows,cond,fraction):continue
                    for model in ('clean','augmented'):
                        obj={'scope':'macro' if name=='macro' else 'dataset','dataset':name,'condition':cond,
                             'model':model,'checkpoint_kind':'nominal','timepoint':'' if name=='macro' else checkpoint(rows[name][0][cond],fraction)['timepoint'],
                             'nominal_fraction':float(fraction),'n_datasets':len(names),'n_draws':5}
                        for key in ('actual_fraction','accuracy','balanced_accuracy','macro_f1','error'):
                            if key=='actual_fraction':
                                v=np.array([np.mean([checkpoint(rows[n][d][cond],fraction)['actual_fraction'] for n in names]) for d in range(5)])
                            else:v=fixed_values(rows,names,cond,fraction,model,key)
                            obj[key+'_mean']=float(v.mean());obj[key+'_sd']=float(v.std(ddof=1))
                        w.writerow(obj)
        # ItalyPowerDemand's v2 minimum-length checkpoint (9/24) is real but has
        # no nominal 20%-100% decile label. Preserve it rather than dropping it.
        name='ItalyPowerDemand'
        for cond in CONDITIONS:
            for model in ('clean','augmented'):
                entries=[next(x for x in rows[name][d][cond]['fixed'] if x['timepoint']==9) for d in range(5)]
                obj={'scope':'dataset','dataset':name,'condition':cond,'model':model,
                     'checkpoint_kind':'actual_only','timepoint':9,'nominal_fraction':'','n_datasets':1,'n_draws':5}
                for key in ('actual_fraction','accuracy','balanced_accuracy','macro_f1','error'):
                    v=np.asarray([x['actual_fraction'] if key=='actual_fraction' else x[model][key] for x in entries])
                    obj[key+'_mean']=float(v.mean());obj[key+'_sd']=float(v.std(ddof=1))
                w.writerow(obj)
    with (OUT/'fixed_length_curves_draws.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['dataset','draw','condition','model','timepoint','actual_fraction','nominal_fraction','accuracy','balanced_accuracy','macro_f1','error'])
        for name in DATASETS:
            for d in range(5):
                for cond in CONDITIONS:
                    record=rows[name][d][cond];L=record['series_length']
                    nominal={int(np.ceil(j*L/10)):j/10 for j in range(2,11)}
                    for entry in record['fixed']:
                        for model in ('clean','augmented'):
                            v=entry[model]
                            w.writerow([name,d,cond,model,entry['timepoint'],entry['actual_fraction'],
                              nominal.get(entry['timepoint'],''),v['accuracy'],v['balanced_accuracy'],v['macro_f1'],v['error']])
    front_fields=['scope','dataset','condition','method','setting','n_datasets','n_draws']
    front_fields += [f'{key}_{suffix}' for key in KEYS for suffix in ('mean','sd')]
    with (OUT/'frontier_points.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=front_fields);w.writeheader()
        for name in (*DATASETS,'macro'):
            names=DATASETS if name=='macro' else (name,)
            for cond in CONDITIONS:
                for method in METHODS:
                    for setting in SETTINGS[method]:
                        obj={'scope':'macro' if name=='macro' else 'dataset','dataset':name,'condition':cond,
                             'method':method,'setting':setting,'n_datasets':len(names),'n_draws':5}
                        for key in KEYS:
                            v=front_values(rows,names,cond,method,setting,key)
                            obj[key+'_mean']=float(v.mean());obj[key+'_sd']=float(v.std(ddof=1))
                        w.writerow(obj)
    with (OUT/'frontier_points_draws.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['dataset','draw','condition','method','setting',*KEYS])
        for name in DATASETS:
            for d in range(5):
                for cond in CONDITIONS:
                    for method in METHODS:
                        for setting in SETTINGS[method]:
                            m=setting_row(rows[name][d][cond],method,setting)
                            w.writerow([name,d,cond,method,setting,*[m[k] for k in KEYS]])

def draw_figures(rows):
    fig,axes=plt.subplots(2,4,figsize=(17,8),sharex=True,sharey=True,constrained_layout=True)
    for i,cond in enumerate(CONDITIONS):
        ax=axes.flat[i]
        for model,color in (('clean','#305875'),('augmented','#b32240')):
            xs=[];ys=[];sd=[]
            for f in FRACTIONS:
                names=eligible(rows,cond,f);v=fixed_values(rows,names,cond,f,model,'accuracy')
                xs.append(f);ys.append(v.mean());sd.append(v.std(ddof=1))
            ax.errorbar(xs,ys,yerr=sd,marker='o',markersize=3,linewidth=1.5,capsize=2,color=color,label=model.title())
        ax.set_title(CONDITION_NAMES[cond]);ax.set_xticks([.2,.4,.6,.8,1.]);ax.grid(alpha=.2);ax.set_ylim(0,1.02)
        if i%4==0:ax.set_ylabel('Accuracy')
        if i>=4:ax.set_xlabel('Nominal prefix fraction')
    axes.flat[7].axis('off')
    fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='outside lower center',ncols=2,frameon=False)
    fig.suptitle('Fixed-checkpoint accuracy: mean +/- SD over five paired draws; 20%-30% use seven datasets')
    with (OUT/'accuracy_vs_prefix.png').open('xb') as f:fig.savefig(f,format='png',dpi=150)
    plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(17,8),sharex=False,sharey=True,constrained_layout=True)
    colors={'probability':'#305875','TEASER':'#704a91','augmented':'#b32240'}
    for i,cond in enumerate(CONDITIONS):
        ax=axes.flat[i]
        for method in METHODS:
            data=[]
            for setting in SETTINGS[method]:
                x=front_values(rows,DATASETS,cond,method,setting,'earliness')
                y=front_values(rows,DATASETS,cond,method,setting,'error')
                data.append((x.mean(),y.mean(),x.std(ddof=1),y.std(ddof=1)))
            data=sorted(data)
            ax.errorbar([v[0] for v in data],[v[1] for v in data],
                        xerr=[v[2] for v in data],yerr=[v[3] for v in data],
                        marker='o',markersize=3,linewidth=1.2,capsize=2,color=colors[method],label=METHOD_NAMES[method])
        ax.set_title(CONDITION_NAMES[cond]);ax.grid(alpha=.2)
        if i%4==0:ax.set_ylabel('Error at stopping')
        if i>=4:ax.set_xlabel('Earliness')
    axes.flat[7].axis('off')
    fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='outside lower center',ncols=3,frameon=False)
    fig.suptitle('Corruption-agnostic stopping sweeps: mean +/- SD over five paired draws')
    with (OUT/'earliness_vs_error_frontier.png').open('xb') as f:fig.savefig(f,format='png',dpi=150)
    plt.close(fig)

def summary(rows,comparisons):
    lines=['# ECTS fixed-length and corruption-agnostic stopping frontiers','',
      'Eight datasets, five paired augmentation/test-corruption draws, and seven conditions replay the saved v2 checkpoint classifiers. All cells are means +/- sample SD over five draws: first average datasets within each draw. The earlier robustness run was read-only.',
      '', 'The old run did not store corrupted test streams or full test probabilities, so they were regenerated from its deterministic seeds and saved classifiers. TEASER safety estimators were absent and were rebuilt once per dataset with the exact v2 settings; all regenerated probability, alpha=.10, and default-TEASER per-case decisions were checked against v2.',
      '', 'ItalyPowerDemand has no 20% or 30% checkpoint: nine observations are the MiniRocket minimum, whereas those nominal prefixes contain five and eight observations. Fixed-length macro curves at 20% and 30% therefore use seven datasets; all other fixed-length points and all stopping results use eight. Within each dataset the models see exactly the same observed prefix.',
      '', '## A. Fixed-length error gaps','',
      'Cells are clean-model error minus augmented-model error; positive means augmentation improves accuracy. The last columns ask whether the augmented mean error is no greater at every available nominal checkpoint (20%, 30%, ..., 100%), across the macro curve and within each dataset.','',
      '| Condition | 20% (n=7) | 40% | 60% | 80% | 100% | Macro all checkpoints? | Dataset count all checkpoints |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    fixed_wins=[]
    for cond in CONDITIONS:
        gaps=[]
        for f in FRACTIONS:
            names=eligible(rows,cond,f)
            gaps.append(float((fixed_values(rows,names,cond,f,'clean','error')-
                               fixed_values(rows,names,cond,f,'augmented','error')).mean()))
        if all(x>=-1e-12 for x in gaps):fixed_wins.append(CONDITION_NAMES[cond])
    frontier_wins=[CONDITION_NAMES[c] for c in CONDITIONS if comparisons[c]['dominates_on_mean_envelopes']=='YES']
    verdict=('**Verdict:** At equal fixed prefixes, augmentation is no worse on the macro-mean curve at every available checkpoint for: '
             +', '.join(fixed_wins)+'. At matched mean earliness, the augmented stopping envelope is below both baselines throughout shared support for: '
             +', '.join(frontier_wins)+'. These are aggregate, post-hoc comparisons; they do not establish per-dataset dominance, statistical significance, or a deployable test-conditioned choice.')
    lines[2:2]=[verdict,'']
    for cond in CONDITIONS:
        vals=[]
        for f in SELECT:
            names=eligible(rows,cond,f)
            vals.append(fmt(fixed_values(rows,names,cond,f,'clean','error')-fixed_values(rows,names,cond,f,'augmented','error')))
        checks=[]
        for f in FRACTIONS:
            names=eligible(rows,cond,f)
            gap=fixed_values(rows,names,cond,f,'clean','error')-fixed_values(rows,names,cond,f,'augmented','error')
            checks.append(float(gap.mean())>=-1e-12)
        dataset_count=0
        for n in DATASETS:
            gaps=[]
            for t in rows[n][0][cond]['times']:
                gap=[]
                for d in range(5):
                    entry=next(x for x in rows[n][d][cond]['fixed'] if x['timepoint']==t)
                    gap.append(entry['clean']['error']-entry['augmented']['error'])
                gaps.append(float(np.mean(gap)))
            dataset_count+=int(all(g>=-1e-12 for g in gaps))
        lines.append('| '+CONDITION_NAMES[cond]+' | '+' | '.join(vals)+' | '+('YES' if all(checks) else 'NO')+f' | {dataset_count}/8 |')
    lines+=['','## B. Matched stopping earliness','',
      'Sweeps use probability thresholds 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99; mixed-calibration alpha 0.02, 0.05, 0.10, 0.15, 0.20, 0.30; and TEASER consecutive-safe counts 2, 3, 4, 5. Every setting is fixed without using the test condition. The per-condition curves and comparisons are post-hoc descriptions, not a rule for selecting a setting after observing test conditions.',
      'At each shared earliness, the lower convex envelope linearly interpolates between swept settings; this represents a randomized mixture of fixed policies, not an independently measured operating point. We do not extrapolate. Margins are baseline error minus augmented error, so positive means the augmented policy has lower error. Targets use the range jointly supported by all three methods in all five draws.',
      '', '| Condition | Matched earliness | Probability minus augmented | TEASER minus augmented | Better-baseline minus augmented | Aug lower than both at this point? |',
      '|---|---:|---:|---:|---:|---|']
    for cond in CONDITIONS:
        comp=comparisons[cond]
        if not comp['targets']:
            lines.append('| '+CONDITION_NAMES[cond]+' | N/A | N/A | N/A | N/A | No common five-draw support |')
        for q in comp['targets']:
            ds=q['draws'];p=np.array([x['probability_margin'] for x in ds]);t=np.array([x['teaser_margin'] for x in ds]);b=np.array([x['both_margin'] for x in ds])
            lines.append(f"| {CONDITION_NAMES[cond]} | {q['earliness']:.4f} | {fmt(p)} | {fmt(t)} | {fmt(b)} | "+('YES' if p.mean()>0 and t.mean()>0 else 'NO')+' |')
    lines+=['','### Equal-earliness verdict by condition','',
      'YES requires lower augmented error than both comparison envelopes at every breakpoint in their common mean-curve support. NO is a failure at at least one breakpoint; no common support is reported as unavailable. These are descriptive means, not statistical significance claims.','',
      '| Condition | Augmented strictly dominates both on shared mean support? | Common earliness range |',
      '|---|---|---:|']
    for cond in CONDITIONS:
        c=comparisons[cond];scope=c['mean_support'];text=f'{scope[0]:.4f} - {scope[1]:.4f}' if scope else 'Unavailable'
        lines.append(f"| {CONDITION_NAMES[cond]} | {c['dominates_on_mean_envelopes']} | {text} |")
    lines+=['','## Artifacts and scope','',
      'fixed_length_curves.csv and frontier_points.csv contain per-dataset and macro mean/SD values; fixed_length_curves_draws.csv and frontier_points_draws.csv contain individual-draw points. The ItalyPowerDemand 9/24 checkpoint is included as an actual-only per-dataset point and in its dataset-level dominance check. accuracy_vs_prefix.png and earliness_vs_error_frontier.png show the curves and SD bars.',
      'The five draws vary augmentation and corruption while holding each dataset split fixed. No finite-sample risk guarantees or claims about raw physical SNR follow. Original data were previously normalized per series, and synthetic noise is applied in archive units.','']
    return '\n'.join(lines)

def main():
    rows=collect();comp={c:compare_condition(rows,c) for c in CONDITIONS}
    with (OUT/'comparison_details.json').open('x') as f:json.dump(comp,f,indent=2,allow_nan=False)
    write_csv(rows);draw_figures(rows)
    with (OUT/'summary.md').open('x') as f:f.write(summary(rows,comp))
    print('FRONTIER_REPORT_DONE',flush=True)
if __name__=='__main__':main()
