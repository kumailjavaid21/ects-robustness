"""Compare a causal anytime classifier, TEASER, and empirical-risk stopping rules."""
import csv,json,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from .download import NAMES
from .run_baselines import load_split,select,FRACTIONS,THRESHOLDS
from .metrics import evaluate
from .prefix import prefix_times
from .anytime import normalize,train,probabilities
from .calibrated_stop import calibrate,predict

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/ects_v1'
ALPHAS=(.05,.10,.20)
MODES=('standard','causal')
BASELINES=('fixed','threshold','TEASER')
def split(y):
    idx=np.arange(len(y))
    train_idx,hold_idx=train_test_split(idx,test_size=.4,random_state=42,stratify=y)
    val_idx,cal_idx=train_test_split(hold_idx,test_size=.5,random_state=43,stratify=y[hold_idx])
    if set(train_idx)&set(val_idx) or set(train_idx)&set(cal_idx) or set(val_idx)&set(cal_idx):
        raise AssertionError('Split overlap')
    return train_idx,val_idx,cal_idx

def gpu_choice():
    if not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable')
    try:
        out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        candidates=[tuple(int(v.strip()) for v in line.split(',')) for line in out.splitlines()]
        idx=min(candidates,key=lambda x:(x[2]>=10,x[1],x[2]))[0]
        return torch.device('cuda:'+str(idx))
    except Exception as e:
        print('GPU_SELECTION_FALLBACK',repr(e),flush=True)
        return torch.device('cuda:0')

def teaser(x,y,test,labels,times):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import OneClassSVM
    from aeon.classification.early_classification import TEASER
    from .teaser_prefix import PrefixCVClassifier
    base=PrefixCVClassifier(make_pipeline(StandardScaler(),LogisticRegression(max_iter=600,random_state=42)),random_state=42)
    points=sorted(set(max(3,int(t)) for t in times))
    est=TEASER(estimator=base,one_class_classifier=OneClassSVM(),
         one_class_param_grid={'gamma':['scale']},classification_points=points,n_jobs=1,random_state=42)
    started=time.perf_counter();est.fit(x[:,None,:],y)
    pred,decisions=est.predict(test[:,None,:])
    if not np.all(decisions):raise ValueError('TEASER did not decide all samples at full input')
    t=np.asarray(est.state_info[:,3],dtype=int)
    result=evaluate(labels,pred,t,test.shape[1])
    result.update({'baseline':'TEASER','setting':'native','seconds':time.perf_counter()-started})
    return result

def sweep_full(probs,classes,y):
    # All baselines may decide at every observed timestep.
    from .prefix import threshold_predictions
    n,length,_=probs.shape
    times=np.arange(1,length+1);grid=np.transpose(probs,(1,0,2));rows=[]
    for i,t in enumerate(times):
        pred=classes[grid[i].argmax(axis=1)]
        rows.append(dict(baseline='fixed',setting=float(t/length),
                         **evaluate(y,pred,np.full(n,t,dtype=int),length)))
    for threshold in THRESHOLDS:
        pred,t=threshold_predictions(grid,classes,times,float(threshold))
        rows.append(dict(baseline='threshold',setting=float(threshold),
                         **evaluate(y,pred,t,length)))
    return rows

def run(name,mode,device,epochs=60):
    X,y=load_split(name,'train');T,z=load_split(name,'test')
    train_idx,val_idx,cal_idx=split(y)
    norm=normalize(X,mode);test=normalize(T,mode)
    started=time.perf_counter();model,classes,losses=train(norm[train_idx],y[train_idx],device,epochs=epochs)
    train_seconds=time.perf_counter()-started
    valp=probabilities(model,norm[val_idx],device)
    calp=probabilities(model,norm[cal_idx],device)
    testp=probabilities(model,test,device)
    times=prefix_times(X.shape[1],FRACTIONS)
    val_rows=sweep_full(valp,classes,y[val_idx])
    test_rows=sweep_full(testp,classes,z)
    results={};selected={}
    for method in ('fixed','threshold'):
        selected[method]=select(val_rows,method)
        results[method]=next(r for r in test_rows if r['baseline']==method and r['setting']==selected[method]['setting'])
    try:
        results['TEASER']=teaser(norm[train_idx],y[train_idx],test,z,times)
    except Exception:
        err=traceback.format_exc();print('TEASER_ERROR',name,mode,err,flush=True)
        results['TEASER']={'baseline':'TEASER','error':err}
    calibration={}
    for alpha in ALPHAS:
        lambdas,support,cal_error=calibrate(calp,y[cal_idx],classes,alpha)
        pred,t,forced=predict(testp,classes,lambdas)
        metric=evaluate(z,pred,t,T.shape[1]);metric.update({'baseline':'reliability','alpha':alpha,
            'test_error':1-metric['accuracy'],'meets_target':(1-metric['accuracy'])<=alpha+1e-12,
            'forced_full_count':int(forced.sum()),'forced_full_fraction':float(forced.mean())})
        results[f'alpha_{alpha:.2f}']=metric
        calibration[f'{alpha:.2f}']={'thresholds':[float(v) if np.isfinite(v) else None for v in lambdas],
            'n_accepted':support.tolist(),'empirical_errors':cal_error.tolist()}
    namefile=OUT/f'{name}_{mode}_model.pt'
    with namefile.open('xb') as f:
        torch.save({'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},'classes':classes.tolist(),
                    'mode':mode,'epochs':epochs,'hidden':32,'seed':42},f)
    return {'dataset':name,'mode':mode,'train_shape':list(X.shape),'test_shape':list(T.shape),
       'split':{'fit_indices':train_idx.tolist(),'validation_indices':val_idx.tolist(),'calibration_indices':cal_idx.tolist()},
       'split_sizes':[len(train_idx),len(val_idx),len(cal_idx)],'train_seconds':train_seconds,
       'losses':losses,'selected':selected,'validation_sweep':val_rows,'test_sweep':test_rows,
       'calibration':calibration,'results':results,'model_path':str(namefile)}

def fmt(m):
    return f"{m['accuracy']:.3f} / {m['earliness']:.3f} / {m['combined']:.3f}"

def make_summary(records,device,seconds):
    methods=['fixed','threshold','TEASER','alpha_0.05','alpha_0.10','alpha_0.20']
    method_names=['Fixed fraction','Probability threshold','TEASER','Risk alpha 0.05','Risk alpha 0.10','Risk alpha 0.20']
    lines=['# ECTS v1: causal normalization and empirical reliability stopping','',
       f'Device: {device} ({torch.cuda.get_device_name(device)}). Total runtime: {seconds:.1f} seconds.',
       'All models use the official UCR train/test splits. The official training set is stratified into 60% fit / 20% validation / 20% calibration (approximately; seeds 42 and 43).',
       '','Standard uses the archived whole-series-normalized values. Causal uses, at each timestep, only that series prefix running mean and standard deviation. Running z-scores cancel prior per-series positive affine standardization except numerical rounding, but cannot undo other unknown preprocessing.',
       '','The anytime model is a unidirectional 32-hidden-unit GRU with a class head at every timestep. It uses random-length prefix loss and full-length loss, trained for 60 epochs on GPU. The fixed and probability-threshold methods use the same GRU, can stop at every timestep, and select on validation. TEASER uses its own prefix classifiers (aeon TEASER with scaled logistic regression, 10 points, OneClassSVM gamma=scale); its safety-model cross-validation is restricted to each fitted prefix; no model is refitted on validation or calibration.',
       '', 'At every timestep, reliability thresholds select the largest calibration confidence set with empirical error at most alpha. If no nonempty set qualifies, its threshold is disabled. Test cases stop at the first passing timestep, or are forced to decide at full length. Calibration uses only 5-200 examples depending on the dataset.',
       '', '**No finite-sample or anytime risk guarantee:** these are empirical per-time conditions. Selection at the first qualifying time and forced final predictions can change the risk at stopping, and small calibration samples make thresholds unstable. Confidence is not calibrated probability.',
       '', 'Cells show test accuracy / mean earliness / harmonic mean of accuracy and (1 - earliness). Arithmetic means weight datasets equally.']
    for mode in MODES:
        lines+=['',f'## {mode.title()} normalization','',
          '| Dataset | Fixed | Probability | TEASER | alpha=0.05 | alpha=0.10 | alpha=0.20 |',
          '|---|---:|---:|---:|---:|---:|---:|']
        rr=[r for r in records if r['mode']==mode]
        for r in rr:
            vals=[fmt(r['results'][m]) if 'error' not in r['results'][m] else 'ERROR' for m in methods]
            lines.append('| '+r['dataset']+' | '+' | '.join(vals)+' |')
        cells=[]
        for method in methods:
            valid=[r['results'][method] for r in rr if 'error' not in r['results'][method]]
            if valid:
                avg={k:float(np.mean([v[k] for v in valid])) for k in ('accuracy','earliness','combined')}
                cells.append(fmt(avg)+f' (n={len(valid)})')
            else:cells.append('N/A')
        lines.append('| **Mean** | '+' | '.join(cells)+' |')
        lines+=['','### Reliability at stopping','',
            '| Dataset | alpha=0.05 error / meets? / forced | alpha=0.10 error / meets? / forced | alpha=0.20 error / meets? / forced |',
            '|---|---:|---:|---:|']
        for r in rr:
            vals=[]
            for alpha in ALPHAS:
                v=r['results'][f'alpha_{alpha:.2f}']
                vals.append(f"{v['test_error']:.3f} / {'yes' if v['meets_target'] else 'NO'} / {v['forced_full_fraction']:.3f}")
            lines.append('| '+r['dataset']+' | '+' | '.join(vals)+' |')
    lines+=['','## Causal minus standard: mean differences','',
            '| Method | Accuracy difference | Earliness difference | Score difference |',
            '|---|---:|---:|---:|']
    for method,label in zip(methods,method_names):
        means={}
        for mode in MODES:
            vals=[r['results'][method] for r in records if r['mode']==mode and 'error' not in r['results'][method]]
            means[mode]={k:float(np.mean([v[k] for v in vals])) for k in ('accuracy','earliness','combined')}
        dif=[means['causal'][k]-means['standard'][k] for k in ('accuracy','earliness','combined')]
        lines.append('| '+label+' | '+' | '.join(f'{x:+.3f}' for x in dif)+' |')
    lines+=['','Forced = fraction that did not cross any calibrated threshold and was required to decide at the full length.',
       'Because the threshold is selected independently for each time and the earliest crossing is used, the calibration-set guarantee at each time does not imply reliability at the random stopping time.',
       '','Coffee and TwoLeadECG have particularly few calibration examples; selecting among many stopping times can also overfit the small validation sets. Wafer is imbalanced: causal alpha=0.20 predicts class 1 for all 6,164 test samples, including all 665 minority-class examples incorrectly.',
       '','Data, model checkpoints, split indices, per-timestep thresholds, calibration support, loss curves, sweeps and test metrics are saved in runs/ects_v1/ and data/ucr/. The corrected every-timestep comparison with prefix-restricted TEASER CV is in results_teaser_prefix.json; earlier diagnostics are preserved.','']
    return '\n'.join(lines)

def main():
    device=gpu_choice();print('DEVICE',device,torch.cuda.get_device_name(device),flush=True)
    t=time.perf_counter();records=[]
    for mode in MODES:
        for name in NAMES:
            r=run(name,mode,device);records.append(r)
            print('ECTS_V1_DONE',mode,name,'split',r['split_sizes'],'train_seconds',round(r['train_seconds'],2),
                  'loss',round(r['losses'][-1],4),'results',json.dumps(r['results']),flush=True)
            with (OUT/f'{name}_{mode}_results.json').open('x') as f:json.dump(r,f,indent=2,allow_nan=False)
    seconds=time.perf_counter()-t
    with (OUT/'results.json').open('x') as f:json.dump({'device':str(device),'device_name':torch.cuda.get_device_name(device),'seconds':seconds,'records':records},f,indent=2,allow_nan=False)
    with (OUT/'summary.md').open('x') as f:f.write(make_summary(records,device,seconds))
    print('ECTS_V1_ALL_DONE','seconds',round(seconds,2),flush=True)
if __name__=='__main__':main()
