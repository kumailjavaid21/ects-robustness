"""ECTS robustness v2: fixed mixed calibration versus condition-matched oracle."""
import argparse,json,time,traceback,warnings
from pathlib import Path
import joblib,numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDOneClassSVM
from aeon.classification.early_classification import TEASER
from aeon.transformations.collection.convolution_based import MiniRocket
from .rocket_v2 import (load_arrow,split_indices,checkpoints,make_classifier,
                        calibrate_sequential,decisions,metrics)
from .run_v2 import sweep,select
from .robustness import CONDITIONS
from .robustness_v2 import corrupt,train_augmentation,mixed_calibration
from .anytime import normalize
from .teaser_rocket import REGISTRY,CachedRocketClassifier

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/ects_robust_v2'
DATASETS=('FordA','ECG5000','StarLightCurves','Wafer','ElectricDevices','Crop','TwoLeadECG','ItalyPowerDemand')
METHODS=('probability','TEASER','realistic','oracle');ALPHA=.10

def save_json(path,obj):
    with path.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)

def unique_path(path):
    if not path.exists():return path
    i=1
    while True:
        alt=path.with_name(path.stem+f'_resume{i}'+path.suffix)
        if not alt.exists():return alt
        i+=1

def proba(models,X,times):
    values=[]
    for model,clf,t in zip(models['transforms'],models['classifiers'],times):
        data=np.ascontiguousarray(X[:,:int(t),None].transpose(0,2,1),dtype=np.float32)
        values.append(clf.predict_proba(model.transform(data)).astype(np.float32))
    return np.stack(values,axis=1)

def train_models(name,draw,fit_x,y,times,folder,tag):
    transforms=[];classifiers=[];seen_warnings=[]
    for t in times:
        filename=f'{name}_{tag}_{int(t):04d}.joblib'
        path=folder/filename
        if path.exists():
            saved=joblib.load(path)
            if saved['dataset']!=name or saved['length']!=int(t) or saved['tag']!=tag:
                raise ValueError(f'Checkpoint metadata mismatch: {path}')
            model,clf=saved['transform'],saved['classifier']
        else:
            x=np.ascontiguousarray(fit_x[:,:int(t),None].transpose(0,2,1),dtype=np.float32)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always',ConvergenceWarning)
                model=MiniRocket(n_kernels=672,n_jobs=1,random_state=42)
                F=model.fit_transform(x);clf=make_classifier().fit(F,y)
            seen_warnings += [str(w.message) for w in caught if issubclass(w.category,ConvergenceWarning)]
            with path.open('xb') as f:joblib.dump({'dataset':name,'length':int(t),'tag':tag,
                         'transform':model,'classifier':clf},f,compress=3)
        transforms.append(model);classifiers.append(clf)
        print('MODEL_READY',name,'draw',draw,'tag',tag,'t',int(t),flush=True)
    return {'transforms':transforms,'classifiers':classifiers},seen_warnings

def get_base(name,X,y,T,old):
    row=next((x for x in old['records'] if x['dataset']==name and x['mode']=='causal'),None)
    norm=normalize(X,'causal');test=normalize(T,'causal')
    if row is not None:
        ind=row['indices'];fit=np.array(ind['fit']);val=np.array(ind['validation']);cal=np.array(ind['calibration']);unused=np.array(ind['unused'])
        times=np.array(row['checkpoint_lengths']);base={'transforms':[],'classifiers':[]}
        for t in times:
            z=joblib.load(ROOT/'runs/ects_v2/models'/f'{name}_causal_{int(t):04d}.joblib')
            base['transforms'].append(z['transform']);base['classifiers'].append(z['classifier'])
        threshold=float(row['selected']['threshold']['setting']);origin='v2 existing causal checkpoints';warnings_seen=[]
    else:
        fit,val,cal,unused=split_indices(y);times=checkpoints(X.shape[1])
        base,warnings_seen=train_models(name,-1,norm[fit],y[fit],times,OUT/'base_models','base')
        classes=base['classifiers'][0].classes_
        pval=proba(base,norm[val],times)
        threshold=float(select(sweep(pval,classes,y[val],times),'threshold')['setting'])
        origin='new clean causal checkpoints, same v2 fit/validation/calibration protocol'
    if set(fit)&set(val) or set(fit)&set(cal) or set(val)&set(cal):raise AssertionError('Split overlap')
    classes=base['classifiers'][0].classes_
    if any(not np.array_equal(c.classes_,classes) for c in base['classifiers']):raise ValueError('Base class mismatch')
    power=float(np.mean(X[fit].astype(np.float64)**2))
    metadata={'dataset':name,'train_shape':list(X.shape),'test_shape':list(T.shape),'base_origin':origin,
      'fit_count':len(fit),'validation_count':len(val),'calibration_count':len(cal),'unused_count':len(unused),
      'fit_indices':fit.tolist(),'validation_indices':val.tolist(),'calibration_indices':cal.tolist(),'unused_indices':unused.tolist(),
      'missing_cal_classes':sorted(set(map(int,np.unique(y)))-set(map(int,np.unique(y[cal])))),
      'checkpoint_lengths':times.tolist(),'classes':classes.tolist(),'base_probability_threshold':threshold,
      'fit_signal_power':power,'warnings':warnings_seen}
    path=OUT/'records'/f'{name}_base.json'
    if not path.exists():save_json(path,metadata)
    else:
        prior=json.loads(path.read_text())
        for k in ('fit_indices','validation_indices','calibration_indices','checkpoint_lengths','base_probability_threshold'):
            if prior[k]!=metadata[k]:raise ValueError('Resume base metadata differs: '+k)
    return base,fit,val,cal,times,classes,power,threshold,metadata,norm,test

def prepare_teaser(base,xfit,yfit,test_clean,pclean,times):
    REGISTRY.clear()
    for i,(m,c,t) in enumerate(zip(base['transforms'],base['classifiers'],times)):
        v=np.ascontiguousarray(xfit[:,:int(t),None].transpose(0,2,1),dtype=np.float32)
        REGISTRY[int(t)]={'fit_input':xfit[:,:int(t)],'fit_y':yfit,'fit_features':m.transform(v),
          'test_input':test_clean[:,:int(t)],'test_probs':pclean[:,i,:],'transform':m,'classifier':c}
    teaser=TEASER(estimator=CachedRocketClassifier(random_state=42),
      one_class_classifier=SGDOneClassSVM(nu=.1,max_iter=500,tol=.001,random_state=42),
      one_class_param_grid={'nu':[.1]},classification_points=list(map(int,times)),n_jobs=1,random_state=42)
    teaser.fit(xfit[:,None,:],yfit)
    return teaser

def teaser_decide(teaser,X,p,times):
    for i,t in enumerate(times):
        REGISTRY[int(t)]['test_input']=X[:,:int(t)];REGISTRY[int(t)]['test_probs']=p[:,i,:]
    pred,ok=teaser.predict(X[:,None,:])
    if not np.all(ok):raise RuntimeError('TEASER failed to decide all test cases')
    t=np.asarray(teaser.state_info[:,3],dtype=int)
    return pred,t,t==X.shape[1]

def metric_result(y,p,t,forced,length):
    result=metrics(y,p,t,length);err=1-result['accuracy']
    result.update({'test_error':err,'meets_reference_alpha':bool(err<=ALPHA+1e-12),
                   'forced_fraction':float(np.mean(forced)),'forced_count':int(np.sum(forced))})
    return result

def calibrate(p,y,classes,times):
    lam,support,err,risk=calibrate_sequential(p,y,classes,ALPHA)
    cp,ct,cf=decisions(p,classes,times,lam)
    detail={'thresholds':[float(x) if np.isfinite(x) else None for x in lam],
      'support':support.tolist(),'at_risk':risk.tolist(),'per_checkpoint_error':err.tolist(),
      'calibration_stop_error':float(np.mean(cp!=y)),'calibration_forced_fraction':float(np.mean(cf))}
    return lam,detail

def run_dataset(name,draws=range(5)):
    started=time.perf_counter();old=json.loads((ROOT/'runs/ects_v2/results.json').read_text())
    X,y=load_arrow(name,'train');T,z=load_arrow(name,'test')
    base,fit,val,cal,times,classes,power,threshold,base_info,norm,test_clean=get_base(name,X,y,T,old)
    xfit=norm[fit];rawfit=X[fit];rawcal=X[cal];yc=y[cal]
    print('BASE_READY',name,'sizes',len(fit),len(val),len(cal),'checkpoints',times.tolist(),flush=True)
    if all((OUT/'records'/f'{name}_draw{draw}.json').exists() for draw in draws):
        print('REUSE_DATASET',name,flush=True);return
    cleanp=proba(base,test_clean,times)
    teaser=prepare_teaser(base,xfit,y[fit],test_clean,cleanp,times)
    for draw in draws:
        draw_path=OUT/'records'/f'{name}_draw{draw}.json'
        if draw_path.exists():print('REUSE_DRAW',name,draw,flush=True);continue
        begin=time.perf_counter()
        aug,aug_y,aug_info=train_augmentation(rawfit,y[fit],name,draw,power)
        aug=normalize(aug,'causal')
        model,warns=train_models(name,draw,aug,aug_y,times,OUT/'models',f'aug_draw{draw}')
        if any(not np.array_equal(c.classes_,classes) for c in model['classifiers']):
            raise ValueError('Augmented class order changed')
        mixed,counts=mixed_calibration(rawcal,name,draw,power)
        mixp=proba(model,normalize(mixed,'causal'),times)
        realistic_lam,realistic_detail=calibrate(mixp,yc,classes,times)
        rows={};oracle_probs={}
        for condition in CONDITIONS:
            path=OUT/'records'/f'{name}_draw{draw}_{condition}.json'
            if path.exists():
                record=json.loads(path.read_text())
                if record['realistic_thresholds']!=realistic_detail['thresholds']:raise ValueError('Resume calibration mismatch')
                rows[condition]=record
                # Retrieve stored calibration probabilities to rebuild a complete draw artifact.
                with np.load(record['artifact']) as oldcond:oracle_probs[condition]=oldcond['oracle_cal_prob'].copy()
                print('REUSE_CONDITION',name,draw,condition,flush=True);continue
            cstart=time.perf_counter()
            test=normalize(corrupt(T,name,draw,'test',condition,power),'causal')
            calx=normalize(corrupt(rawcal,name,draw,'calibration',condition,power),'causal')
            basep=cleanp if condition=='clean' else proba(base,test,times)
            augp=proba(model,test,times);oraclep=proba(model,calx,times);oracle_probs[condition]=oraclep
            oracle_lam,oracle_detail=calibrate(oraclep,yc,classes,times)
            preds={};ts={};forced={}
            preds['probability'],ts['probability'],forced['probability']=decisions(basep,classes,times,np.full(len(times),threshold))
            preds['TEASER'],ts['TEASER'],forced['TEASER']=teaser_decide(teaser,test,basep,times)
            preds['realistic'],ts['realistic'],forced['realistic']=decisions(augp,classes,times,realistic_lam)
            preds['oracle'],ts['oracle'],forced['oracle']=decisions(augp,classes,times,oracle_lam)
            results={method:metric_result(z,preds[method],ts[method],forced[method],T.shape[1]) for method in METHODS}
            npz=unique_path(OUT/'predictions'/f'{name}_draw{draw}_{condition}.npz')
            fields={'y':z,'times':times,'classes':classes,'oracle_cal_prob':oraclep}
            for method in METHODS:
                fields[method+'_pred']=preds[method];fields[method+'_time']=ts[method];fields[method+'_forced']=forced[method]
            with npz.open('xb') as f:np.savez_compressed(f,**fields)
            record={'dataset':name,'draw':draw,'condition':condition,'results':results,
                'realistic_thresholds':realistic_detail['thresholds'],'oracle_calibration':oracle_detail,
                'artifact':str(npz),'seconds':time.perf_counter()-cstart}
            save_json(path,record);rows[condition]=record
            print('DRAW_CONDITION_DONE',name,draw,condition,'seconds',round(record['seconds'],1),
                'errors',json.dumps({k:round(v['test_error'],4) for k,v in results.items()}),flush=True)
        mix_path=unique_path(OUT/'predictions'/f'{name}_draw{draw}_calibration.npz')
        with mix_path.open('xb') as f:np.savez_compressed(f,cal_y=yc,classes=classes,times=times,mixed_prob=mixp)
        draw_record={'dataset':name,'draw':draw,'base':base_info,'mixture_counts':counts,
          'realistic_calibration':realistic_detail,'augmentation':aug_info,'warnings':warns,
          'calibration_artifact':str(mix_path),'conditions':rows,'seconds':time.perf_counter()-begin}
        save_json(draw_path,draw_record);print('DRAW_DONE',name,draw,'seconds',round(draw_record['seconds'],1),flush=True)
    print('DATASET_DONE',name,'seconds',round(time.perf_counter()-started,1),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',required=True)
    parser.add_argument('--datasets',nargs='+',choices=DATASETS,required=True);parser.add_argument('--draws',nargs='+',type=int,choices=range(5),default=list(range(5)));args=parser.parse_args()
    for sub in ('records','predictions','models','base_models'):(OUT/sub).mkdir(exist_ok=True)
    started=time.perf_counter();errors=[]
    for name in args.datasets:
        try:run_dataset(name,args.draws)
        except Exception:
            err=f'{name}: '+traceback.format_exc();errors.append(err);print('DATASET_ERROR',err,flush=True)
    status={'worker':args.worker,'datasets':args.datasets,'draws':args.draws,'seconds':time.perf_counter()-started,'errors':errors}
    target=unique_path(OUT/f'worker_{args.worker}_status.json');save_json(target,status)
    print('WORKER_DONE',args.worker,'errors',len(errors),'seconds',round(status['seconds'],1),flush=True)
if __name__=='__main__':main()
