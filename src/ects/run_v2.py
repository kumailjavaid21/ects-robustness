"""ECTS v2: independent selective calibration of checkpoint MiniRocket classifiers."""
import json,time,traceback,warnings
from pathlib import Path
import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from .download_v2 import NAMES
from .rocket_v2 import load_arrow,split_indices,checkpoints,fit_checkpoint,calibrate_sequential,decisions,metrics
from .anytime import normalize
from .teaser_rocket import evaluate_teaser

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/ects_v2'
MODES=('standard','causal');ALPHAS=(.05,.1,.2)
THRESHOLDS=(.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99,1.)
KERNELS=672
METHODS=('fixed','threshold','TEASER','alpha_0.05','alpha_0.10','alpha_0.20')
LABELS=('Fixed','Probability','TEASER','alpha=.05','alpha=.10','alpha=.20')

def sweep(p,classes,y,times):
    rows=[];n=len(y);L=int(times[-1])
    for j,t in enumerate(times):
        pred=classes[p[:,j].argmax(1)]
        rows.append(dict(baseline='fixed',setting=float(t/L),**metrics(y,pred,np.full(n,t,dtype=int),L)))
    for threshold in THRESHOLDS:
        pred,t,forced=decisions(p,classes,times,np.full(len(times),threshold))
        rows.append(dict(baseline='threshold',setting=float(threshold),forced_full_fraction=float(forced.mean()),**metrics(y,pred,t,L)))
    return rows

def select(rows,kind):
    return max((r for r in rows if r['baseline']==kind),
               key=lambda r:(r['combined'],r['accuracy'],-r['earliness'],-r['setting']))

def run_one(name,mode,X,y,T,z,indices):
    fit,val,cal,unused=indices;started=time.perf_counter()
    norm=normalize(X,mode);test=normalize(T,mode)
    xf=norm[fit];xv=norm[val];xc=norm[cal];yf=y[fit];yv=y[val];yc=y[cal]
    times=checkpoints(X.shape[1]);classes=None
    pval=[];pcal=[];ptest=[];registry={};training=[];solver_warnings=[]
    for t in times:
        step=time.perf_counter()
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('always',ConvergenceWarning)
            transform,classifier,features,probs=fit_checkpoint(xf,yf,xv,xc,test,int(t),n_kernels=KERNELS)
        solver_warnings.extend(str(w.message) for w in seen if issubclass(w.category,ConvergenceWarning))
        newclasses=classifier.classes_
        if classes is None:classes=newclasses
        elif not np.array_equal(classes,newclasses):raise ValueError('Checkpoint class order changed')
        pval.append(probs[0]);pcal.append(probs[1]);ptest.append(probs[2])
        registry[int(t)]={'fit_input':xf[:,:t], 'fit_y':yf,'fit_features':features,
             'test_input':test[:,:t],'test_probs':probs[2],'transform':transform,'classifier':classifier}
        checkpoint=OUT/'models'/f'{name}_{mode}_{int(t):04d}.joblib'
        with checkpoint.open('xb') as f:joblib.dump({'transform':transform,'classifier':classifier,'mode':mode,'length':int(t)},f,compress=3)
        training.append({'timepoint':int(t),'seconds':time.perf_counter()-step,'iterations':classifier[-1].n_iter_.tolist()})
        print('CHECKPOINT',name,mode,int(t),'/',X.shape[1],'seconds',round(time.perf_counter()-step,2),flush=True)
    pval=np.stack(pval,axis=1);pcal=np.stack(pcal,axis=1);ptest=np.stack(ptest,axis=1)
    val_sweep=sweep(pval,classes,yv,times);test_sweep=sweep(ptest,classes,z,times)
    results={};chosen={}
    for method in ('fixed','threshold'):
        chosen[method]=select(val_sweep,method)
        results[method]=next(x for x in test_sweep if x['baseline']==method and x['setting']==chosen[method]['setting'])
    try:results['TEASER']=evaluate_teaser(xf,yf,test,z,times,registry)
    except Exception:
        err=traceback.format_exc();print('TEASER_ERROR',name,mode,err,flush=True)
        results['TEASER']={'baseline':'TEASER','error':err}
    calibration={}
    for alpha in ALPHAS:
        lambdas,support,calerr,at_risk=calibrate_sequential(pcal,yc,classes,alpha)
        pred,t,forced=decisions(ptest,classes,times,lambdas);r=metrics(z,pred,t,int(times[-1]))
        err=1-r['accuracy'];r.update({'baseline':'reliability','alpha':alpha,'test_error':err,
                 'meets_target':err<=alpha+1e-12,'forced_fraction':float(forced.mean()),'forced_count':int(forced.sum())})
        results[f'alpha_{alpha:.2f}']=r
        cpred,ct,cf=decisions(pcal,classes,times,lambdas)
        calibration[f'{alpha:.2f}']={'thresholds':[float(x) if np.isfinite(x) else None for x in lambdas],
              'support':support.tolist(),'at_risk':at_risk.tolist(),'per_checkpoint_error':calerr.tolist(),
              'calibration_error_at_stop':float(np.mean(cpred!=yc)),'calibration_forced_fraction':float(cf.mean())}
    predictionfile=OUT/'predictions'/f'{name}_{mode}.npz'
    with predictionfile.open('xb') as f:
        np.savez_compressed(f,classes=classes,times=times,val_prob=pval,cal_prob=pcal,test_prob=ptest,
                            val_y=yv,cal_y=yc,test_y=z)
    row={'dataset':name,'mode':mode,'train_shape':list(X.shape),'test_shape':list(T.shape),
        'sizes':{'fit':len(fit),'validation':len(val),'calibration':len(cal),'unused':len(unused)},
        'indices':{'fit':fit.tolist(),'validation':val.tolist(),'calibration':cal.tolist(),'unused':unused.tolist()},
        'missing_cal_classes':sorted(set(map(int,np.unique(y)))-set(map(int,np.unique(yc)))),
        'checkpoint_lengths':times.tolist(),'kernels':KERNELS,'training':training,'warnings':solver_warnings,
        'selected':chosen,'validation_sweep':val_sweep,'test_sweep':test_sweep,'calibration':calibration,
        'results':results,'seconds':time.perf_counter()-started}
    with (OUT/'records'/f'{name}_{mode}.json').open('x') as f:json.dump(row,f,indent=2,allow_nan=False)
    return row

def summarize(records,errors,seconds):
    lines=['# ECTS v2: checkpoint MiniRocket and selective stopping','',
      f'Summed dataset-mode processing time: {seconds:.1f} seconds; CPU-only, one worker/thread. Official train/test splits were maintained.',
      'Each checkpoint at 20%, 30%, ..., 100% has its own aeon MiniRocket (672 features) and scaled logistic regression (C=0.1, tol=0.001, max_iter=400). C was selected in a small Crop validation pilot and then fixed across all datasets. Fit size is capped at 4,000, validation at about 400, calibration at about 2,000; unused training rows are recorded.',
      'Standard uses the archived whole-series-normalized series. Causal normalizes each observation with running prefix statistics, cancelling prior per-series positive affine normalization up to numerical precision, not unknown non-affine preprocessing.',
      '','Fixed and probability settings are chosen on validation; probabilities use thresholds .05, .10, .20, ..., .90, .95, .99, 1.00. Aeon TEASER uses the same checkpoint MiniRocket models, 3-fold out-of-fold logistic predictions on shared unsupervised features (2-fold if needed), and a linear SGD one-class safety classifier. Unsupervised MiniRocket features are fitted on all fit examples before the TEASER folds. TEASER uses the same nine checkpoints.',
      '','Our thresholds are selected sequentially on held-out calibration cases still undecided at each checkpoint: use the largest accepted set with empirical error <= alpha. Tied confidence values are handled together. If none qualifies, the checkpoint is disabled. Unaccepted cases are forced to decide at 100%.',
      '**Important:** these empirical checkpoint conditions do not control error at the first crossing. The data-dependent threshold selection, repeated checks, and forced decisions provide no finite-sample or anytime guarantee. Calibration error at the actual stopping point and test forced fractions are saved in JSON.',
      '','Small-data contrast: GunPoint and Coffee have only 10 and 6 calibration examples. ECG5000 has 200 calibration examples but its rarest class (only two training examples) is entirely assigned to fit, so calibration does not cover that class.',
      '', 'Cells: accuracy / balanced accuracy / macro-F1 / earliness / combined score. Combined is the harmonic mean of accuracy and (1 - earliness); dataset means are unweighted.']
    def fmt(r):
        if 'error' in r:return 'ERROR'
        return ' / '.join(f"{r[k]:.3f}" for k in ('accuracy','balanced_accuracy','macro_f1','earliness','combined'))
    for mode in MODES:
        data=[x for x in records if x['mode']==mode]
        lines+=['',f'## {mode.title()} normalization','',
          '| Dataset (train/test x length; fit/val/cal) | '+' | '.join(LABELS)+' |',
          '|---|---:|---:|---:|---:|---:|---:|']
        for r in data:
            n,m=r['train_shape'];q,_=r['test_shape'];sz=r['sizes']
            cells=[fmt(r['results'][method]) for method in METHODS]
            lines.append(f"| {r['dataset']} ({n}/{q} x {m}; {sz['fit']}/{sz['validation']}/{sz['calibration']}) | "+' | '.join(cells)+' |')
        means=[]
        for method in METHODS:
            valid=[r['results'][method] for r in data if 'error' not in r['results'][method]]
            if not valid:means.append('N/A');continue
            avg={k:float(np.mean([x[k] for x in valid])) for k in ('accuracy','balanced_accuracy','macro_f1','earliness','combined')}
            means.append(fmt(avg)+f' (n={len(valid)})')
        lines.append('| **Mean** | '+' | '.join(means)+' |')
        lines+=['','### Empirical reliability','',
            '| Dataset | alpha=.05 error / forced | alpha=.10 error / forced | alpha=.20 error / forced |',
            '|---|---:|---:|---:|']
        for r in data:
            cells=[]
            for alpha in ALPHAS:
                x=r['results'][f'alpha_{alpha:.2f}']
                cells.append(f"{x['test_error']:.3f} / {x['forced_fraction']:.3f} "+('PASS' if x['meets_target'] else 'FAIL'))
            lines.append('| '+r['dataset']+' | '+' | '.join(cells)+' |')
    lines+=['','## Causal minus standard: mean difference','',
       '| Method | Accuracy | Balanced accuracy | Macro-F1 | Earliness | Score |',
       '|---|---:|---:|---:|---:|---:|']
    for method,label in zip(METHODS,LABELS):
        a=[r['results'][method] for r in records if r['mode']=='standard' and 'error' not in r['results'][method]]
        b=[r['results'][method] for r in records if r['mode']=='causal' and 'error' not in r['results'][method]]
        if not a or not b:continue
        ks=('accuracy','balanced_accuracy','macro_f1','earliness','combined')
        delta=[np.mean([x[k] for x in b])-np.mean([x[k] for x in a]) for k in ks]
        lines.append('| '+label+' | '+' | '.join(f'{x:+.3f}' for x in delta)+' |')
    lines+=['','## Artifacts and issues','',
      'Final sequential-calibration results are in results_sequential.json and results.json. The original per-dataset records preserve the initial unconditioned-calibration diagnostic; final sequential records, probabilities, split indices, thresholds and supports, validation/test sweeps, and saved models are under runs/ects_v2/.',
      'The server-side data download initially timed out and returned HTTP 429; verified files were relayed through Windows and their SHA-256 hashes checked against the dataset repository.',
      'Archive: https://huggingface.co/datasets/ChengsenWang/UCR-2018 .']
    if errors:lines+=['', 'Errors:']+['- '+e.replace('\n',' ') for e in errors]
    if any(r['warnings'] for r in records):
        lines+=['',f"Optimizer convergence warnings occurred in {sum(len(r['warnings']) for r in records)} checkpoints; details are in the records and run log."]
    return '\n'.join(lines)+'\n'

def main():
    for sub in ('models','records','predictions'): (OUT/sub).mkdir(exist_ok=True)
    start=time.perf_counter();records=[];errors=[]
    for name in NAMES:
        X,y=load_arrow(name,'train');T,z=load_arrow(name,'test');indices=split_indices(y)
        print('DATASET_START',name,list(X.shape),list(T.shape),'sizes',[len(x) for x in indices],flush=True)
        for mode in MODES:
            try:
                r=run_one(name,mode,X,y,T,z,indices);records.append(r)
                print('DATASET_MODE_DONE',name,mode,'seconds',round(r['seconds'],2),'results',json.dumps(r['results']),flush=True)
            except Exception:
                error=f'{name} {mode}: '+traceback.format_exc();errors.append(error);print('DATASET_MODE_ERROR',error,flush=True)
    seconds=time.perf_counter()-start
    with (OUT/'results.json').open('x') as f:json.dump({'records':records,'errors':errors,'seconds':seconds},f,indent=2,allow_nan=False)
    with (OUT/'summary.md').open('x') as f:f.write(summarize(records,errors,seconds))
    print('ECTS_V2_DONE','records',len(records),'errors',len(errors),'seconds',round(seconds,2),flush=True)
if __name__=='__main__':main()
