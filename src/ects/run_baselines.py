"""Validation-selected fixed-prefix and confidence-threshold baselines on UCR."""
import csv,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .download import NAMES
from .metrics import evaluate
from .prefix import PrefixClassifier,prefix_times,probabilities_over_time,fixed_predictions,threshold_predictions
ROOT=Path(__file__).resolve().parents[2];DATA=ROOT/'data/ucr';OUT=ROOT/'runs/ects'
FRACTIONS=np.arange(1,11)/10
THRESHOLDS=np.array([.50,.55,.60,.65,.70,.75,.80,.85,.90,.95,.99,1.0])

def load_split(name,split):
    df=pd.read_parquet(DATA/name/(split+'.parquet'))
    X=[]
    for r in df['data']:
        a=np.stack(r)
        if a.ndim!=2 or a.shape[1]!=1:raise ValueError('Expected univariate time series')
        X.append(a[:,0])
    X=np.stack(X).astype(np.float64);y=df['label'].to_numpy()
    if not np.isfinite(X).all():raise ValueError(f'Nonfinite input: {name}/{split}')
    return X,y

def sweep(probs,classes,times,y):
    records=[]
    for i,fraction in enumerate(FRACTIONS):
        pred,t=fixed_predictions(probs,classes,times,i)
        records.append(dict(baseline='fixed',setting=float(fraction),**evaluate(y,pred,t,int(times[-1]))))
    for threshold in THRESHOLDS:
        pred,t=threshold_predictions(probs,classes,times,float(threshold))
        records.append(dict(baseline='threshold',setting=float(threshold),**evaluate(y,pred,t,int(times[-1]))))
    return records

def select(rows,method):
    candidates=[x for x in rows if x['baseline']==method]
    return max(candidates,key=lambda x:(x['combined'],x['accuracy'],-x['earliness'],-x['setting']))

def run_one(name):
    started=time.perf_counter()
    X,y=load_split(name,'train');T,z=load_split(name,'test')
    if X.shape[1]!=T.shape[1] or not set(np.unique(z))<=set(np.unique(y)):raise ValueError('Train/test incompatible')
    idx=np.arange(len(y))
    fit_idx,val_idx=train_test_split(idx,test_size=.25,random_state=42,stratify=y)
    length=X.shape[1];times=prefix_times(length,FRACTIONS)
    model=PrefixClassifier().fit(X[fit_idx],y[fit_idx])
    val_probs=probabilities_over_time(model,X[val_idx],times)
    val_rows=sweep(val_probs,model.classes_,times,y[val_idx])
    selected={method:select(val_rows,method) for method in ('fixed','threshold')}
    model=PrefixClassifier().fit(X,y)
    test_probs=probabilities_over_time(model,T,times)
    test_rows=sweep(test_probs,model.classes_,times,z)
    chosen={}
    for method in selected:
        setting=selected[method]['setting']
        chosen[method]=next(r for r in test_rows if r['baseline']==method and r['setting']==setting)
    return {'dataset':name,'train_shape':list(X.shape),'test_shape':list(T.shape),
        'classes':np.unique(y).tolist(),'fit_count':len(fit_idx),'validation_count':len(val_idx),
        'fit_indices':fit_idx.tolist(),'validation_indices':val_idx.tolist(),'prefix_times':times.tolist(),
        'selection':selected,'test_selected':chosen,'validation_sweep':val_rows,'test_sweep':test_rows,
        'seconds':time.perf_counter()-started}

def means(records,method):
    return {k:float(np.mean([r['test_selected'][method][k] for r in records])) for k in ('accuracy','earliness','combined')}

def summary_md(results):
    lines=['# Reliable Early Classification of Time Series: simple baselines','',
      'Six UCR 2018 train/test splits were obtained as Parquet from the Hugging Face mirror of ChengsenWang/UCR-2018; SHA-256 hashes match the repository LFS metadata.',
      '', 'Classifier: StandardScaler plus logistic regression fitted on complete training series. At a prefix, unseen samples are replaced with per-time-point training means. No values from the unobserved test suffix are read.',
      '','Fractions: 10%, 20%, ..., 100% of samples, rounded up. Confidence rule: stop at the first fraction where maximum predicted probability reaches the chosen threshold; force a decision at 100%. These scores are not calibrated probability guarantees.',
      '', 'Each setting is chosen by the accuracy/speed harmonic mean on a stratified 25% training validation split (seed 42); ties prefer accuracy, then earlier decisions. The classifier is refitted on all official training data before the selected setting is evaluated on the untouched official test split.',
      '', 'Earliness = mean fraction observed (lower is earlier). Combined = harmonic mean of accuracy and 1 - earliness; macro means weight each dataset equally. These are generic baselines, not reproductions of a named published ECTS method.',
      '', '**Limitation:** the UCR series in this mirror are already normalized using full-series statistics. Even though the harness hides future values, prefix values can encode information from future measurements through upstream normalization. This is not a leakage-free raw-stream experiment.',
      '', '| Dataset | Train shape | Test shape | Fixed fraction | Fixed acc. | Fixed earliness | Fixed score | Threshold | Threshold acc. | Threshold earliness | Threshold score |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        a=r['test_selected']['fixed'];b=r['test_selected']['threshold']
        sh=lambda x:f"{x[0]} x {x[1]}"
        lines.append(f"| {r['dataset']} | {sh(r['train_shape'])} | {sh(r['test_shape'])} | {a['setting']:.2f} | {a['accuracy']:.4f} | {a['earliness']:.4f} | {a['combined']:.4f} | {b['setting']:.2f} | {b['accuracy']:.4f} | {b['earliness']:.4f} | {b['combined']:.4f} |")
    a=means(results,'fixed');b=means(results,'threshold')
    lines.append(f"| **Mean** | - | - | - | **{a['accuracy']:.4f}** | **{a['earliness']:.4f}** | **{a['combined']:.4f}** | - | **{b['accuracy']:.4f}** | **{b['earliness']:.4f}** | **{b['combined']:.4f}** |")
    lines+=['','Coffee and TwoLeadECG have 28 and 23 training examples respectively, leaving small validation subsets; threshold selection is consequently noisy. Wafer is class-imbalanced, so accuracy alone can conceal minority-class errors.','',
      'Source and manifest: https://huggingface.co/datasets/ChengsenWang/UCR-2018 . Data and hashes: data/ucr/manifest.json. Full validation and test sweeps, split indices and timings: runs/ects/results.json and runs/ects/sweeps.csv.','']
    return '\n'.join(lines)

def main():
    t=time.perf_counter();results=[]
    for name in NAMES:
        r=run_one(name);results.append(r)
        print('ECTS_DONE',name,'shape',r['train_shape'],r['test_shape'],'fixed',r['test_selected']['fixed'],'threshold',r['test_selected']['threshold'],'seconds',round(r['seconds'],2),flush=True)
    with (OUT/'results.json').open('x') as f:json.dump({'results':results,'mean_fixed':means(results,'fixed'),'mean_threshold':means(results,'threshold'),'total_seconds':time.perf_counter()-t},f,indent=2)
    with (OUT/'sweeps.csv').open('x',newline='') as f:
        fields=['dataset','split','baseline','setting','accuracy','earliness','combined']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in results:
            for split in ('validation','test'):
                for row in r[split+'_sweep']:writer.writerow({'dataset':r['dataset'],'split':split,**row})
    with (OUT/'baselines_summary.md').open('x') as f:f.write(summary_md(results))
    print('ECTS_ALL_DONE',flush=True)
if __name__=='__main__':main()
