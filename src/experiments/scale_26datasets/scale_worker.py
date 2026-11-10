"""ECTS scale: reproducible corruption-agnostic evaluation of immutable v2 artifacts."""
import sys,json,time,traceback,argparse
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, joblib
from sklearn.model_selection import StratifiedGroupKFold,cross_val_predict
from sklearn.linear_model import SGDOneClassSVM
from aeon.classification.early_classification import TEASER
ROOT=Path('/home/ubuntu/host_share/kumail_mad');OUT=ROOT/'runs/ects_scale_v1'
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'runs/ects_frontier_v1'))
from ects.rocket_v2 import load_arrow,split_indices,checkpoints,calibrate_sequential,decisions,metrics,make_classifier
from ects.run_robust_v2 import train_models,proba,prepare_teaser
from ects.teaser_rocket import REGISTRY,CachedRocketClassifier
from ects.anytime import normalize
from ects.robustness import CONDITIONS
from ects.robustness_v2 import corrupt,train_augmentation,mixed_calibration
from fast_teaser import all_decisions

OLD={'FordA','ECG5000','StarLightCurves','Wafer','ElectricDevices','Crop','TwoLeadECG','ItalyPowerDemand'}
NEW=('FordB','NonInvasiveFetalECGThorax1','PhalangesOutlinesCorrect','HandOutlines','UWaveGestureLibraryX','TwoPatterns','Strawberry','MiddlePhalanxOutlineCorrect','EthanolLevel','MixedShapesRegularTrain','SwedishLeaf','FaceAll','ShapesAll','SyntheticControl','EOGHorizontalSignal','ChlorineConcentration','SemgHandMovementCh2','LargeKitchenAppliances','InsectWingbeatSound','CricketX')
MAIN=('FordA','ECG5000','StarLightCurves','Wafer','ElectricDevices','Crop')+NEW
CAVEAT=('TwoLeadECG','ItalyPowerDemand');TAUS=(.5,.6,.7,.8,.9,.95,.99)
ALPHAS=(.02,.05,.1,.15,.2,.3);GS=(2,3,4,5)
ECEC_TAUS=(.5,.6,.7,.8,.9,.95,.99,.999,.9999,.99999,1.)

def save(path,obj):
    with path.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)

def load_data(name,split):
    if name in OLD:return load_arrow(name,split)
    table=pq.read_table(OUT/'data'/name/(split+'.parquet'));meta=table.schema.metadata
    length=int(meta[b'sequence_length']);channels=int(meta[b'num_channels'])
    if channels!=1:raise ValueError('Expected univariate '+name)
    x=table.column('data').combine_chunks().values.values.to_numpy().reshape(-1,length).astype(np.float32)
    y=table.column('label').combine_chunks().to_numpy().astype(int)
    if not np.isfinite(x).all():raise ValueError('Nonfinite data '+name)
    return x,y

def metric(y,p,t,f,L):
    v=metrics(y,p,t,L);v.update(error=1-v['accuracy'],forced_fraction=float(np.mean(f)));return v

def modelset(paths):
    out={'transforms':[],'classifiers':[]}
    for p in paths:
        v=joblib.load(p);out['transforms'].append(v['transform']);out['classifiers'].append(v['classifier'])
    return out

class GroupedCachedRocketClassifier(CachedRocketClassifier):
    """OOF safety: paired original and augmented examples stay in the same fold."""
    def _get_train_probs(self,X,y):
        n=len(y)//2
        if len(y)!=2*n or not np.array_equal(y[:n],y[n:]):raise ValueError('Expected paired augmented examples')
        groups=np.tile(np.arange(n),2);folds=min(3,int(np.unique(y[:n],return_counts=True)[1].min()))
        if folds<2:raise ValueError('TEASER requires two distinct examples per class')
        cv=StratifiedGroupKFold(n_splits=folds,shuffle=True,random_state=42)
        return cross_val_predict(make_classifier(),self._entry['fit_features'],y,cv=cv.split(X,y,groups),method='predict_proba',n_jobs=1)

def fit_aug_teaser(base,xfit,yfit,times):
    REGISTRY.clear()
    for m,c,t in zip(base['transforms'],base['classifiers'],times):
        v=np.ascontiguousarray(xfit[:,:int(t),None].transpose(0,2,1),dtype=np.float32)
        REGISTRY[int(t)]={'fit_input':xfit[:,:int(t)],'fit_y':yfit,'fit_features':m.transform(v),
           'test_input':None,'test_probs':None,'transform':m,'classifier':c}
    model=TEASER(estimator=GroupedCachedRocketClassifier(random_state=42),
      one_class_classifier=SGDOneClassSVM(nu=.1,max_iter=500,tol=.001,random_state=42),
      one_class_param_grid={'nu':[.1]},classification_points=list(map(int,times)),n_jobs=1,random_state=42)
    model.fit(xfit[:,None,:],yfit)
    for e in model._estimators:
        for key in ('fit_input','fit_y','fit_features','test_input','test_probs'):e._entry[key]=None
    return model

def ecec_rates(p,y,classes):
    indices=p.argmax(axis=2);truth=np.searchsorted(classes,y);rates=np.zeros((p.shape[1],len(classes)));counts=np.zeros_like(rates,dtype=int)
    for j in range(p.shape[1]):
        for c in range(len(classes)):
            selected=indices[:,j]==c;counts[j,c]=selected.sum()
            if counts[j,c]:rates[j,c]=np.mean(truth[selected]==c)
    return rates,counts

def ecec_decide(p,classes,times,rates,tau):
    n,k,c=p.shape;idx=p.argmax(axis=2);failure=np.ones((n,c));scores=np.empty((n,k));ids=np.arange(n)
    for j in range(k):
        label=idx[:,j];failure[ids,label]*=1-rates[j,label];scores[:,j]=1-failure[ids,label]
    accept=scores>=tau;forced=~accept.any(axis=1);accept[:,-1]=True;stop=accept.argmax(axis=1)
    return classes[idx[ids,stop]],np.asarray(times)[stop],forced

def run_dataset(name,draws,conditions):
    if all((OUT/'records'/f'{name}_draw{d}_{c}.json').exists() for d in draws for c in conditions):
        print('REUSE_COMPLETE',name,flush=True);return
    X,y=load_data(name,'train');T,z=load_data(name,'test');L=X.shape[1]
    if T.shape[1]!=L:raise ValueError('Train/test length mismatch')
    normal=normalize(X,'causal');clean_test=normalize(T,'causal')
    if name in OLD:
        old=json.loads((ROOT/'runs/ects_robust_v2/records'/f'{name}_draw0.json').read_text())['base']
        fit=np.asarray(old['fit_indices']);val=np.asarray(old['validation_indices']);cal=np.asarray(old['calibration_indices'])
        times=np.asarray(old['checkpoint_lengths']);power=old['fit_signal_power'];classes=np.asarray(old['classes'])
        paths=[(ROOT/'runs/ects_v2/models'/f'{name}_causal_{t:04d}.joblib' if name not in CAVEAT else ROOT/'runs/ects_robust_v2/base_models'/f'{name}_base_{t:04d}.joblib') for t in times]
        base=modelset(paths);origin='v2 pretrained clean checkpoints'
    else:
        fit,val,cal,unused=split_indices(y);times=checkpoints(L)
        power=float(np.mean(X[fit].astype(np.float64)**2));classes=np.unique(y)
        base,_=train_models(name,-1,normal[fit],y[fit],times,OUT/'base_models','base');origin='new checkpoints, v2 settings'
    if len(X)<200 and name not in CAVEAT:raise ValueError('Train < 200: '+name)
    if not (set(fit).isdisjoint(val) and set(fit).isdisjoint(cal) and set(val).isdisjoint(cal)):raise ValueError('Split overlap')
    if not np.array_equal(base['classifiers'][0].classes_,classes):raise ValueError('Class mismatch')
    info={'dataset':name,'train_size':len(X),'test_size':len(T),'length':L,'classes':classes.tolist(),
       'n_classes':len(classes),'fit_count':len(fit),'validation_count':len(val),'calibration_count':len(cal),
       'missing_cal_classes':sorted(set(map(int,classes))-set(map(int,np.unique(y[cal])))),
       'fit_indices':fit.tolist(),'validation_indices':val.tolist(),'calibration_indices':cal.tolist(),
       'checkpoint_lengths':times.tolist(),'fit_signal_power':power,'origin':origin}
    path=OUT/'draw_meta'/f'{name}_base.json'
    if not path.exists():save(path,info)
    elif json.loads(path.read_text())!=info:raise ValueError('Base metadata changed on resume')
    pval=proba(base,normal[val],times);rates,counts=ecec_rates(pval,y[val],classes);cleanp=proba(base,clean_test,times)
    cpath=(ROOT/'runs/ects_frontier_v1/teaser_safety'/f'{name}.joblib' if name in OLD else OUT/'teaser_safety'/f'{name}_clean.joblib')
    if cpath.exists():teaser=joblib.load(cpath)
    else:
        teaser=prepare_teaser(base,normal[fit],y[fit],clean_test,cleanp,times)
        for e in teaser._estimators:
            for key in ('fit_input','fit_y','fit_features','test_input','test_probs'):e._entry[key]=None
        with cpath.open('xb') as f:joblib.dump(teaser,f,compress=3)
    print('DATASET_READY',name,'train',len(X),'test',len(T),'cal',len(cal),'classes',len(classes),flush=True)
    for draw in draws:
        if all((OUT/'records'/f'{name}_draw{draw}_{c}.json').exists() for c in conditions):continue
        started=time.monotonic();aug,ay,aug_info=train_augmentation(X[fit],y[fit],name,draw,power);aug=normalize(aug,'causal')
        if name in OLD:
            paths=[ROOT/'runs/ects_robust_v2/models'/f'{name}_aug_draw{draw}_{int(t):04d}.joblib' for t in times];augmented=modelset(paths)
        else:augmented,_=train_models(name,draw,aug,ay,times,OUT/'models',f'aug_draw{draw}')
        spath=OUT/'teaser_safety'/f'{name}_aug_draw{draw}.joblib'
        if spath.exists():ateg=joblib.load(spath)
        else:
            ateg=fit_aug_teaser(augmented,aug,ay,times)
            with spath.open('xb') as f:joblib.dump(ateg,f,compress=3)
        if name in OLD:
            ref=json.loads((ROOT/'runs/ects_robust_v2/records'/f'{name}_draw{draw}.json').read_text())
            with np.load(ref['calibration_artifact']) as d:
                cp=d['mixed_prob'].copy();cy=d['cal_y'].copy()
                if not np.array_equal(d['times'],times) or not np.array_equal(d['classes'],classes):raise ValueError('Old calibration mismatch')
        else:
            mixed,mixinfo=mixed_calibration(X[cal],name,draw,power);cp=proba(augmented,normalize(mixed,'causal'),times);cy=y[cal]
        thresholds={a:calibrate_sequential(cp,cy,classes,a)[0] for a in ALPHAS}
        dp=OUT/'draw_meta'/f'{name}_draw{draw}.json'
        details={'dataset':name,'draw':draw,'mixed_thresholds':{str(a):[float(x) if np.isfinite(x) else None for x in v] for a,v in thresholds.items()},
                 'ecec_val_rates':rates.tolist(),'ecec_val_class_prediction_counts':counts.tolist(),
                 'aug_teaser_g_selected':int(ateg._consecutive_predictions),'augmentation':aug_info}
        if not dp.exists():save(dp,details)
        for condition in conditions:
            path=OUT/'records'/f'{name}_draw{draw}_{condition}.json'
            if path.exists():continue
            before=time.monotonic();test=normalize(corrupt(T,name,draw,'test',condition,power),'causal')
            pa=proba(augmented,test,times);pb=cleanp if condition=='clean' else proba(base,test,times)
            previous=ROOT/'runs/ects_frontier_v1/records'/f'{name}_draw{draw}_{condition}.json'
            prior=json.loads(previous.read_text()) if previous.exists() else None
            if prior:
                fixed=prior['fixed'];f=prior['frontiers'];fronts={'probability':f['probability'],'TEASER':f['TEASER'],'ours':f['augmented']}
            else:
                fixed=[]
                for j,t in enumerate(times):
                    tt=np.full(len(z),int(t));no=np.zeros(len(z),bool)
                    fixed.append({'timepoint':int(t),'actual_fraction':float(t/L),
                      'clean':metric(z,classes[pb[:,j].argmax(1)],tt,no,L),
                      'augmented':metric(z,classes[pa[:,j].argmax(1)],tt,no,L)})
                fronts={'probability':[],'TEASER':[],'ours':[]}
                for tau in TAUS:
                    p,t,forced=decisions(pb,classes,times,np.full(len(times),tau))
                    fronts['probability'].append({'setting':tau,**metric(z,p,t,forced,L)})
                for a in ALPHAS:
                    p,t,forced=decisions(pa,classes,times,thresholds[a])
                    fronts['ours'].append({'setting':a,**metric(z,p,t,forced,L)})
                for g,(p,t,forced) in all_decisions(teaser,pb,times,GS).items():fronts['TEASER'].append({'setting':g,**metric(z,p,t,forced,L)})
            fronts['aug_probability']=[]
            for tau in TAUS:
                p,t,forced=decisions(pa,classes,times,np.full(len(times),tau))
                fronts['aug_probability'].append({'setting':tau,**metric(z,p,t,forced,L)})
            fronts['aug_TEASER']=[]
            for g,(p,t,forced) in all_decisions(ateg,pa,times,GS).items():fronts['aug_TEASER'].append({'setting':g,**metric(z,p,t,forced,L)})
            fronts['ECEC']=[]
            for tau in ECEC_TAUS:
                p,t,forced=ecec_decide(pb,classes,times,rates,tau)
                fronts['ECEC'].append({'setting':tau,**metric(z,p,t,forced,L)})
            save(path,{'dataset':name,'draw':draw,'condition':condition,'test_size':len(z),'series_length':L,
               'times':times.tolist(),'fixed':fixed,'frontiers':fronts,'reuse_previous':prior is not None,'seconds':time.monotonic()-before})
            print('CONDITION_DONE',name,draw,condition,'seconds',round(time.monotonic()-before,1),flush=True)
        print('DRAW_DONE',name,draw,'seconds',round(time.monotonic()-started,1),flush=True)
    print('DATASET_DONE',name,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',required=True);p.add_argument('--draws',nargs='+',type=int,default=list(range(5)));p.add_argument('--conditions',nargs='+',default=list(CONDITIONS));p.add_argument('--worker',required=True)
    args=p.parse_args();errors=[]
    for name in args.datasets:
        try:run_dataset(name,args.draws,args.conditions)
        except Exception:
            msg=traceback.format_exc();errors.append({'dataset':name,'error':msg});print('DATASET_ERROR',name,msg,flush=True)
    path=OUT/'logs'/('worker_'+args.worker+'_status.json')
    if not path.exists():save(path,{'worker':args.worker,'errors':errors,'datasets':args.datasets})
    print('WORKER_DONE',args.worker,'errors',len(errors),flush=True)

if __name__=='__main__':main()
