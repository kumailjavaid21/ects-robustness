"""Robust ECTS evaluation with paired corruptions and matched calibration."""
import json,time,traceback,warnings
from pathlib import Path
import joblib,numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDOneClassSVM
from aeon.classification.early_classification import TEASER
from aeon.transformations.collection.convolution_based import MiniRocket
from .rocket_v2 import load_arrow,make_classifier,calibrate_sequential,decisions,metrics
from .anytime import normalize
from .robustness import CONDITIONS,streams,augment_fit
from .teaser_rocket import CachedRocketClassifier,REGISTRY

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/ects_robust'
DATASETS=('FordA','ECG5000','StarLightCurves','Wafer')
METHODS=('probability','TEASER','plain_clean_cal','plain_matched_cal','aug_matched_cal')
ALPHA=.10
def save_json(path,obj):
    with path.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)
def proba(models,X,times):
    rows=[]
    for model,clf,t in zip(models['transforms'],models['classifiers'],times):
        x=np.ascontiguousarray(X[:,:t,None].transpose(0,2,1),dtype=np.float32)
        rows.append(clf.predict_proba(model.transform(x)).astype(np.float32))
    return np.stack(rows,axis=1)
def result(y,pred,t,forced,length):
    v=metrics(y,pred,t,length);err=1-v['accuracy']
    v.update({'test_error':err,'meets_reference_alpha':err<=ALPHA+1e-12,
              'forced_fraction':float(np.mean(forced)),'forced_count':int(np.sum(forced))})
    return v
def calibration(y,probs,classes,times):
    lam,support,err,risk=calibrate_sequential(probs,y,classes,ALPHA)
    cp,ct,cf=decisions(probs,classes,times,lam)
    return lam,{'thresholds':[float(x) if np.isfinite(x) else None for x in lam],
      'support':support.tolist(),'at_risk':risk.tolist(),'per_checkpoint_error':err.tolist(),
      'calibration_stop_error':float(np.mean(cp!=y)),'calibration_forced_fraction':float(np.mean(cf))}
def train_augmented(name,fit_x,fit_y,times,classes):
    transforms=[];classifiers=[];warnings_seen=[];train_stats=[]
    for t in times:
        path=OUT/'models'/f'{name}_aug_{int(t):04d}.joblib';started=time.perf_counter()
        if path.exists():
            saved=joblib.load(path)
            if saved['dataset']!=name or saved['length']!=int(t):raise ValueError('Existing checkpoint mismatch')
            model=saved['transform'];clf=saved['classifier']
        else:
            x=np.ascontiguousarray(fit_x[:,:t,None].transpose(0,2,1),dtype=np.float32)
            with warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter('always',ConvergenceWarning)
                model=MiniRocket(n_kernels=672,n_jobs=1,random_state=42)
                features=model.fit_transform(x);clf=make_classifier().fit(features,fit_y)
            warnings_seen.extend(str(w.message) for w in seen if issubclass(w.category,ConvergenceWarning))
            with path.open('xb') as f:joblib.dump({'dataset':name,'length':int(t),'mode':'causal_aug',
               'transform':model,'classifier':clf},f,compress=3)
        if not np.array_equal(clf.classes_,classes):raise ValueError('Augmented classes differ from base')
        transforms.append(model);classifiers.append(clf)
        train_stats.append({'timepoint':int(t),'seconds':time.perf_counter()-started,
                           'iterations':clf[-1].n_iter_.tolist()})
        print('AUG_CHECKPOINT',name,int(t),'seconds',round(time.perf_counter()-started,2),flush=True)
    return {'transforms':transforms,'classifiers':classifiers},warnings_seen,train_stats
def make_teaser(xfit,yfit,test_clean,classes,times,base,clean_test_probs):
    REGISTRY.clear()
    for i,(model,clf,t) in enumerate(zip(base['transforms'],base['classifiers'],times)):
        v=np.ascontiguousarray(xfit[:,:t,None].transpose(0,2,1),dtype=np.float32)
        REGISTRY[int(t)]={'fit_input':xfit[:,:t],'fit_y':yfit,'fit_features':model.transform(v),
          'test_input':test_clean[:,:t],'test_probs':clean_test_probs[:,i,:],
          'transform':model,'classifier':clf}
    teaser=TEASER(estimator=CachedRocketClassifier(random_state=42),
      one_class_classifier=SGDOneClassSVM(nu=.1,max_iter=500,tol=.001,random_state=42),
      one_class_param_grid={'nu':[.1]},classification_points=list(map(int,times)),n_jobs=1,random_state=42)
    teaser.fit(xfit[:,None,:],yfit)
    return teaser
def evaluate_teaser(teaser,X,y,base_probs,times):
    for i,t in enumerate(times):
        REGISTRY[int(t)]['test_input']=X[:,:t];REGISTRY[int(t)]['test_probs']=base_probs[:,i,:]
    pred,accepted=teaser.predict(X[:,None,:])
    if not np.all(accepted):raise RuntimeError('TEASER did not finish at 100%')
    t=np.asarray(teaser.state_info[:,3],int);forced=t==X.shape[1]
    return result(y,pred,t,forced,X.shape[1]),pred,t,forced
def run_dataset(name,source):
    r=next(x for x in source['records'] if x['dataset']==name and x['mode']=='causal')
    X,y=load_arrow(name,'train');T,z=load_arrow(name,'test')
    fit=np.array(r['indices']['fit']);cal=np.array(r['indices']['calibration'])
    assert not set(fit)&set(cal)
    power=float(np.mean(X[fit].astype(np.float64)**2));times=np.array(r['checkpoint_lengths'],int)
    raw_fit=X[fit];raw_cal=X[cal];xfit=normalize(raw_fit,'causal');test_clean=normalize(T,'causal')
    aug_x,aug_y,augmentation=augment_fit(raw_fit,y[fit],name,power)
    aug_x=normalize(aug_x,'causal')
    with np.load(ROOT/'runs/ects_v2/predictions'/f'{name}_causal.npz') as prev:
        classes=prev['classes'].copy();original_cal=prev['cal_prob'].copy();original_test=prev['test_prob'].copy()
    base={'transforms':[],'classifiers':[]}
    for t in times:
        saved=joblib.load(ROOT/'runs/ects_v2/models'/f'{name}_causal_{int(t):04d}.joblib')
        base['transforms'].append(saved['transform']);base['classifiers'].append(saved['classifier'])
        if not np.array_equal(saved['classifier'].classes_,classes):raise ValueError('Base class order mismatch')
    aug,warns,training=train_augmented(name,aug_x,aug_y,times,classes)
    teaser=make_teaser(xfit,y[fit],test_clean,classes,times,base,original_test)
    base_conf=float(r['selected']['threshold']['setting'])
    saved_lam=np.array([v if v is not None else np.inf for v in r['calibration']['0.10']['thresholds']])
    entries={}
    for cond in CONDITIONS:
        result_path=OUT/'records'/f'{name}_{cond}.json'
        if result_path.exists():
            entries[cond]=json.loads(result_path.read_text());print('REUSE_CONDITION',name,cond,flush=True);continue
        started=time.perf_counter()
        test=normalize(streams(T,name,'test',cond,power),'causal')
        if cond=='clean':pbase=original_test;calbase=original_cal
        else:
            calx=normalize(streams(raw_cal,name,'calibration',cond,power),'causal')
            pbase=proba(base,test,times);calbase=proba(base,calx,times)
        if cond=='clean':calx=normalize(raw_cal,'causal')
        paug=proba(aug,test,times);calaug=proba(aug,calx,times)
        zpred,zt,zforced=decisions(pbase,classes,times,np.full(len(times),base_conf))
        outcomes={'probability':result(z,zpred,zt,zforced,T.shape[1])}
        outcomes['TEASER'],teaser_pred,teaser_t,teaser_forced=evaluate_teaser(teaser,test,z,pbase,times)
        zpred,zt,zforced=decisions(pbase,classes,times,saved_lam)
        outcomes['plain_clean_cal']=result(z,zpred,zt,zforced,T.shape[1])
        lam_base,metabase=calibration(y[cal],calbase,classes,times)
        zpred,zt,zforced=decisions(pbase,classes,times,lam_base)
        outcomes['plain_matched_cal']=result(z,zpred,zt,zforced,T.shape[1])
        lam_aug,metaaug=calibration(y[cal],calaug,classes,times)
        zpred,zt,zforced=decisions(paug,classes,times,lam_aug)
        outcomes['aug_matched_cal']=result(z,zpred,zt,zforced,T.shape[1])
        path=OUT/'predictions'/f'{name}_{cond}.npz'
        with path.open('xb') as f:np.savez_compressed(f,classes=classes,times=times,test_y=z,cal_y=y[cal],
          test_prob_base=pbase,test_prob_aug=paug,cal_prob_base=calbase,cal_prob_aug=calaug,
          teaser_pred=teaser_pred,teaser_t=teaser_t,teaser_forced=teaser_forced)
        row={'dataset':name,'condition':cond,'test_size':len(z),'cal_size':len(cal),
          'results':outcomes,'calibration':{'base_matched':metabase,'aug_matched':metaaug},
          'seconds':time.perf_counter()-started}
        save_json(result_path,row);entries[cond]=row
        print('CONDITION_DONE',name,cond,'seconds',round(row['seconds'],2),
          'errors',json.dumps({k:round(v['test_error'],4) for k,v in outcomes.items()}),flush=True)
    return {'dataset':name,'train_shape':list(X.shape),'test_shape':list(T.shape),'fit_count':len(fit),
       'calibration_count':len(cal),'missing_cal_classes':r['missing_cal_classes'],'fit_signal_power':power,
       'base_probability_threshold':base_conf,'augmentation':augmentation,'training':training,
       'warnings':warns,'conditions':entries}
def main():
    for name in ('models','records','predictions'):(OUT/name).mkdir(exist_ok=True)
    source=json.loads((ROOT/'runs/ects_v2/results.json').read_text());started=time.perf_counter();errors=[]
    for name in DATASETS:
        path=OUT/'records'/f'{name}_complete.json'
        if path.exists():print('REUSE_DATASET',name,flush=True);continue
        try:
            print('DATASET_BEGIN',name,flush=True)
            row=run_dataset(name,source);save_json(path,row)
            print('DATASET_COMPLETE',name,flush=True)
        except Exception:
            err=f'{name}: '+traceback.format_exc();errors.append(err);print('DATASET_ERROR',err,flush=True)
    save_json(OUT/'run_status.json',{'errors':errors,'seconds':time.perf_counter()-started})
    print('ROBUST_RUN_DONE',len(errors),'errors','seconds',round(time.perf_counter()-started,2),flush=True)
if __name__=='__main__':main()
