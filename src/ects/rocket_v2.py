"""Efficient per-checkpoint MiniRocket models and independent v2 splits."""
import numpy as np
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score,f1_score
from aeon.transformations.collection.convolution_based import MiniRocket
from .run_baselines import ROOT
from .metrics import evaluate

def load_arrow(name,split):
    tab=pq.read_table(ROOT/'data/ucr'/name/(split+'.parquet'))
    meta=tab.schema.metadata
    length=int(meta[b'sequence_length']);channels=int(meta[b'num_channels'])
    if channels!=1:raise ValueError('Expected univariate input')
    x=tab.column('data').combine_chunks().values.values.to_numpy().reshape(-1,length).astype(np.float32)
    y=tab.column('label').combine_chunks().to_numpy().astype(int)
    if not np.isfinite(x).all():raise ValueError('Nonfinite time series')
    return x,y

def split_indices(y,seed=42,fit_cap=4000,cal_cap=2000,val_cap=400):
    """Preserve >=2 rare-class examples for fitting; allocate others by class."""
    n=len(y);rng=np.random.default_rng(seed);cal_target=min(cal_cap,max(200,round(.25*n))) if n>=500 else max(2,round(.20*n))
    val_target=min(val_cap,max(len(np.unique(y)),round(.20*n)))
    fit=[];val=[];cal=[]
    for label in np.unique(y):
        idx=np.flatnonzero(y==label);rng.shuffle(idx);m=len(idx)
        if m<=3:
            fit.extend(idx.tolist());continue
        ncal=min(m-2,max(1,round(m*cal_target/n)))
        nval=min(m-ncal-2,max(1,round(m*val_target/n)))
        cal.extend(idx[:ncal].tolist());val.extend(idx[ncal:ncal+nval].tolist());fit.extend(idx[ncal+nval:].tolist())
    fit=np.array(fit,dtype=int);val=np.array(val,dtype=int);cal=np.array(cal,dtype=int)
    unused=np.array([],dtype=int)
    if len(fit)>fit_cap:
        fit,unused=train_test_split(fit,train_size=fit_cap,random_state=seed,stratify=y[fit])
    return np.sort(fit),np.sort(val),np.sort(cal),np.sort(unused)

def checkpoints(length):
    return np.unique(np.maximum(9,np.ceil(np.arange(2,11)*length/10).astype(int)))

def make_classifier():
    return make_pipeline(StandardScaler(),LogisticRegression(max_iter=400,C=0.1,tol=0.001,solver='lbfgs',random_state=42))

def fit_checkpoint(train,y,val,cal,test,length,n_kernels=672):
    tr=np.ascontiguousarray(train[:,:length,None].transpose(0,2,1),dtype=np.float32)
    model=MiniRocket(n_kernels=n_kernels,n_jobs=1,random_state=42)
    feat=model.fit_transform(tr)
    classifier=make_classifier().fit(feat,y)
    probs=[]
    for data in (val,cal,test):
        z=np.ascontiguousarray(data[:,:length,None].transpose(0,2,1),dtype=np.float32)
        probs.append(classifier.predict_proba(model.transform(z)).astype(np.float32))
    return model,classifier,feat,probs

def metrics(y,pred,t,length):
    output=evaluate(y,pred,t,length)
    output['balanced_accuracy']=float(balanced_accuracy_score(y,pred))
    output['macro_f1']=float(f1_score(y,pred,average='macro',zero_division=0))
    return output

def calibrate_fast(probs,y,classes,alpha):
    """Lowest inclusive confidence threshold with empirical accepted error <= alpha."""
    confidence=probs.max(axis=2);pred=classes[probs.argmax(axis=2)]
    thresholds=np.full(probs.shape[1],np.inf);supports=np.zeros(probs.shape[1],int);errs=np.zeros(probs.shape[1])
    for t in range(probs.shape[1]):
        order=np.argsort(-confidence[:,t],kind='stable')
        scores=confidence[order,t];error=(pred[order,t]!=y[order]).astype(int)
        cumulative=np.cumsum(error);ends=np.flatnonzero(np.r_[scores[:-1]!=scores[1:],True])
        eligible=ends[cumulative[ends]/(ends+1)<=alpha+1e-12]
        if len(eligible):
            end=int(eligible[-1]);thresholds[t]=scores[end];supports[t]=end+1;errs[t]=cumulative[end]/(end+1)
    return thresholds,supports,errs

def calibrate_sequential(probs,y,classes,alpha):
    # Calibrate each confidence set only on cases still waiting at that checkpoint.
    remaining=np.ones(len(y),dtype=bool);length=probs.shape[1]
    thresholds=np.full(length,np.inf);supports=np.zeros(length,dtype=int)
    errors=np.zeros(length);at_risk=np.zeros(length,dtype=int)
    confidence=probs.max(axis=2)
    for t in range(length):
        ids=np.flatnonzero(remaining);at_risk[t]=len(ids)
        if not len(ids):continue
        th,n,e=calibrate_fast(probs[ids,t:t+1,:],y[ids],classes,alpha)
        thresholds[t]=th[0];supports[t]=n[0];errors[t]=e[0]
        if np.isfinite(th[0]):remaining[ids[confidence[ids,t]>=th[0]]]=False
    return thresholds,supports,errors,at_risk

def decisions(probs,classes,times,thresholds):
    conf=probs.max(axis=2);good=conf>=np.asarray(thresholds)[None,:]
    forced=~good.any(axis=1);good[:,-1]=True;idx=good.argmax(axis=1);n=np.arange(len(idx))
    return classes[probs[n,idx].argmax(1)],np.asarray(times)[idx],forced
