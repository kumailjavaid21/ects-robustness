"""Replay immutable v2 checkpoints on deterministic test streams; no checkpoint fitting."""
import argparse,json,time,traceback
from pathlib import Path
import joblib,numpy as np
from ects.rocket_v2 import load_arrow,calibrate_sequential,decisions,metrics
from ects.anytime import normalize
from ects.robustness import CONDITIONS
from ects.robustness_v2 import corrupt
from ects.run_robust_v2 import proba,prepare_teaser,teaser_decide
from ects.teaser_rocket import REGISTRY
from fast_teaser import all_decisions

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'runs/ects_robust_v2';OUT=ROOT/'runs/ects_frontier_v1'
DATASETS=('FordA','ECG5000','StarLightCurves','Wafer','ElectricDevices','Crop','TwoLeadECG','ItalyPowerDemand')
TAUS=(.50,.60,.70,.80,.90,.95,.99);ALPHAS=(.02,.05,.10,.15,.20,.30);GS=(2,3,4,5)
def save(path,obj):
    with path.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)
def unique(path):
    if not path.exists():return path
    i=1
    while path.with_name(path.stem+f'_retry{i}'+path.suffix).exists():i+=1
    return path.with_name(path.stem+f'_retry{i}'+path.suffix)
def metric(y,p,t,forced,L):
    m=metrics(y,p,t,L);m.update({'error':1-m['accuracy'],'forced_fraction':float(np.mean(forced))})
    return m
def base_models(name,times):
    source=ROOT/'runs/ects_v2/models' if name not in ('TwoLeadECG','ItalyPowerDemand') else SRC/'base_models'
    models={'transforms':[],'classifiers':[]}
    for t in times:
        filename=f'{name}_causal_{int(t):04d}.joblib' if name not in ('TwoLeadECG','ItalyPowerDemand') else f'{name}_base_{int(t):04d}.joblib'
        obj=joblib.load(source/filename);models['transforms'].append(obj['transform']);models['classifiers'].append(obj['classifier'])
    return models
def aug_models(name,draw,times):
    models={'transforms':[],'classifiers':[]}
    for t in times:
        path=SRC/'models'/f'{name}_aug_draw{draw}_{int(t):04d}.joblib'
        obj=joblib.load(path)
        if obj['dataset']!=name or obj['length']!=int(t):raise ValueError('Mismatched augmentation checkpoint '+str(path))
        models['transforms'].append(obj['transform']);models['classifiers'].append(obj['classifier'])
    return models
def get_teaser(name,base,xfit,yfit,test_clean,cleanp,times):
    path=OUT/'teaser_safety'/f'{name}.joblib'
    if path.exists():
        model=joblib.load(path)
        REGISTRY.clear()
        for t,est in zip(times,model._estimators):REGISTRY[int(t)]=est._entry
        print('TEASER_SAFETY_REUSED',name,'g',model._consecutive_predictions,flush=True)
        return model
    # V2 stored no TEASER safety estimator. Rebuild exactly its v2 OOF safety fitting,
    # using the saved clean MiniRocket transforms and classifiers.
    model=prepare_teaser(base,xfit,yfit,test_clean,cleanp,times)
    for estimator in model._estimators:
        # Training features and test arrays are not needed after safety fitting.
        for key in ('fit_input','fit_y','fit_features','test_input','test_probs'):estimator._entry[key]=None
    with path.open('xb') as f:joblib.dump(model,f,compress=3)
    print('TEASER_SAFETY_REBUILT',name,'g',model._consecutive_predictions,flush=True)
    return model
def run_dataset(name,draws,conditions):
    complete=lambda d,c:(OUT/'records'/f'{name}_draw{d}_{c}.json').exists()
    if all(complete(d,c) for d in draws for c in conditions):
        print('REUSE_COMPLETE_DATASET',name,flush=True);return
    reference=json.loads((SRC/'records'/f'{name}_draw0.json').read_text());meta=reference['base']
    X,y=load_arrow(name,'train');T,z=load_arrow(name,'test')
    fit=np.array(meta['fit_indices']);times=np.array(meta['checkpoint_lengths'],int);L=T.shape[1]
    classes=np.array(meta['classes']);power=meta['fit_signal_power'];tau_original=meta['base_probability_threshold']
    xfit=normalize(X[fit],'causal');test_clean=normalize(T,'causal')
    base=base_models(name,times);cleanp=proba(base,test_clean,times)
    teaser=get_teaser(name,base,xfit,y[fit],test_clean,cleanp,times)
    default_g=int(teaser._consecutive_predictions)
    # Reconnect a cached TEASER's fitted estimators to the active module registry.
    REGISTRY.clear()
    for t,est in zip(times,teaser._estimators):REGISTRY[int(t)]=est._entry
    print('DATASET_READY',name,'n_test',len(z),'times',times.tolist(),'default_g',default_g,flush=True)
    for draw in draws:
        if all(complete(draw,c) for c in conditions):
            print('REUSE_DRAW',name,draw,flush=True);continue
        reference=json.loads((SRC/'records'/f'{name}_draw{draw}.json').read_text())
        if reference['base']['calibration_indices']!=meta['calibration_indices']:raise ValueError('Calibration split changed')
        with np.load(reference['calibration_artifact']) as d:
            cp=d['mixed_prob'];cy=d['cal_y'];cclasses=d['classes'];ct=d['times']
            if not np.array_equal(classes,cclasses) or not np.array_equal(times,ct):raise ValueError('Calibration metadata mismatch')
            thresholds={}
            for alpha in ALPHAS:
                lam,support,err,risk=calibrate_sequential(cp,cy,classes,alpha)
                thresholds[alpha]=lam
                if alpha==.10:
                    restored=[float(a) if np.isfinite(a) else None for a in lam]
                    if restored!=reference['realistic_calibration']['thresholds']:raise ValueError('Original mixed-calibration thresholds do not reproduce')
        meta_path=OUT/'draw_meta'/f'{name}_draw{draw}.json'
        if not meta_path.exists():
            save(meta_path,{'dataset':name,'draw':draw,'times':times.tolist(),'classes':classes.tolist(),
              'test_shape':list(T.shape),'mixed_thresholds':{str(a):[float(x) if np.isfinite(x) else None for x in v] for a,v in thresholds.items()},
              'teaser_default_g':default_g,'n_calibration':len(cy)})
        augmented=aug_models(name,draw,times)
        for condition in conditions:
            path=OUT/'records'/f'{name}_draw{draw}_{condition}.json'
            if path.exists():print('REUSE_CONDITION',name,draw,condition,flush=True);continue
            started=time.perf_counter()
            test=normalize(corrupt(T,name,draw,'test',condition,power),'causal')
            pb=cleanp if condition=='clean' else proba(base,test,times)
            pa=proba(augmented,test,times)
            original=reference['conditions'][condition]
            with np.load(original['artifact']) as old:
                if not np.array_equal(old['y'],z) or not np.array_equal(old['times'],times):raise ValueError('Cached target/times mismatch')
                pp,tt,ff=decisions(pb,classes,times,np.full(len(times),tau_original))
                if not (np.array_equal(pp,old['probability_pred']) and np.array_equal(tt,old['probability_time'])):
                    raise ValueError('Regenerated clean model probability decisions differ from v2')
                pp,tt,ff=decisions(pa,classes,times,thresholds[.10])
                if not (np.array_equal(pp,old['realistic_pred']) and np.array_equal(tt,old['realistic_time'])):
                    raise ValueError('Regenerated augmented decisions differ from v2')
                fast=all_decisions(teaser,pb,times,tuple(sorted(set(GS+(default_g,)))))
                pp,tt,ff=fast[default_g]
                if not (np.array_equal(pp,old['TEASER_pred']) and np.array_equal(tt,old['TEASER_time'])):
                    raise ValueError('Reconstructed TEASER decisions differ from v2')
            fixed=[]
            for j,t in enumerate(times):
                a=classes[pa[:,j,:].argmax(1)];b=classes[pb[:,j,:].argmax(1)]
                time_arr=np.full(len(z),int(t));not_forced=np.zeros(len(z),dtype=bool)
                fixed.append({'timepoint':int(t),'actual_fraction':float(t/L),
                    'clean':metric(z,b,time_arr,not_forced,L),'augmented':metric(z,a,time_arr,not_forced,L)})
            swept={'probability':[],'TEASER':[],'augmented':[]}
            for tau in TAUS:
                p,t,f=decisions(pb,classes,times,np.full(len(times),tau))
                swept['probability'].append({'setting':float(tau),**metric(z,p,t,f,L)})
            for alpha in ALPHAS:
                p,t,f=decisions(pa,classes,times,thresholds[alpha])
                swept['augmented'].append({'setting':float(alpha),**metric(z,p,t,f,L)})
            for g in GS:
                p,t,f=fast[g]
                swept['TEASER'].append({'setting':int(g),**metric(z,p,t,f,L)})
            record={'dataset':name,'draw':draw,'condition':condition,'test_size':len(z),'series_length':L,
                    'times':times.tolist(),'fixed':fixed,'frontiers':swept,'teaser_default_g':default_g,
                    'reproduction_checks':'probability, mixed alpha=.10, and TEASER matched cached per-case decisions',
                    'seconds':time.perf_counter()-started}
            save(path,record)
            print('FRONTIER_CONDITION_DONE',name,draw,condition,'seconds',round(record['seconds'],1),flush=True)
        print('FRONTIER_DRAW_DONE',name,draw,flush=True)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',required=True)
    parser.add_argument('--datasets',nargs='+',choices=DATASETS,required=True)
    parser.add_argument('--draws',nargs='+',type=int,choices=range(5),default=list(range(5)))
    parser.add_argument('--conditions',nargs='+',choices=CONDITIONS,default=list(CONDITIONS));args=parser.parse_args()
    start=time.perf_counter();errors=[]
    for name in args.datasets:
        try:run_dataset(name,args.draws,args.conditions)
        except Exception:
            err=f'{name}: '+traceback.format_exc();errors.append(err);print('DATASET_ERROR',err,flush=True)
    path=unique(OUT/f'worker_{args.worker}_status.json')
    save(path,{'worker':args.worker,'datasets':args.datasets,'draws':args.draws,'conditions':args.conditions,
               'errors':errors,'seconds':time.perf_counter()-start})
    print('WORKER_DONE',args.worker,'errors',len(errors),'seconds',round(time.perf_counter()-start,2),flush=True)
if __name__=='__main__':main()
