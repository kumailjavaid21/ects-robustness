"""Seeded stream corruptions; masking and imputation never use future samples."""
import hashlib
import numpy as np

CONDITIONS=('clean','noise_20db','noise_10db','noise_5db','missing_10pct','missing_30pct','missing_50pct')
def seed_for(dataset,partition,kind):
    key=f'ects-robust-v1:{dataset}:{partition}:{kind}'.encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8],'big')

def causal_forward_fill(X,observed):
    x=np.asarray(X,dtype=np.float32);mask=np.asarray(observed,dtype=bool)
    if x.ndim!=2 or x.shape!=mask.shape:raise ValueError('Expected matching 2D arrays')
    # Running last observed value, with a fixed zero until a first sample is observed.
    idx=np.where(mask,np.arange(1,x.shape[1]+1)[None,:],0)
    idx=np.maximum.accumulate(idx,axis=1)
    padded=np.pad(x,((0,0),(1,0)),constant_values=0)
    return np.take_along_axis(padded,idx,axis=1)

def streams(X,dataset,partition,condition,fit_signal_power):
    x=np.asarray(X,dtype=np.float32)
    if condition=='clean':return x.copy()
    if fit_signal_power<=0:raise ValueError('Fit-signal power must be positive')
    if condition.startswith('noise_'):
        db=float(condition.split('_')[1][:-2])
        rng=np.random.default_rng(seed_for(dataset,partition,'noise'))
        z=rng.standard_normal(x.shape,dtype=np.float32)
        sigma=np.sqrt(fit_signal_power/(10**(db/10)))
        return x+z*np.float32(sigma)
    if condition.startswith('missing_'):
        rate=float(condition.split('_')[1][:-3])/100.
        rng=np.random.default_rng(seed_for(dataset,partition,'missing'))
        observed=rng.random(x.shape,dtype=np.float32)>=rate
        return causal_forward_fill(x,observed)
    raise ValueError('Unknown condition '+condition)

def augment_fit(X,y,dataset,fit_signal_power):
    x=np.asarray(X,dtype=np.float32);n=len(x)
    rng=np.random.default_rng(seed_for(dataset,'fit','augmentation'))
    db=rng.choice(np.array([np.inf,20.,10.,5.]),size=n)
    rates=rng.choice(np.array([0.,.1,.3,.5]),size=n)
    z=rng.standard_normal(x.shape,dtype=np.float32)
    sigma=np.where(np.isinf(db),0.,np.sqrt(fit_signal_power/(10**(db/10))))
    noisy=x+z*sigma[:,None].astype(np.float32)
    mask=rng.random(x.shape,dtype=np.float32)>=rates[:,None]
    augmented=causal_forward_fill(noisy,mask)
    return np.concatenate((x,augmented)),np.concatenate((y,y)),{
      'samples':n,'noise_counts':{str(k):int(np.count_nonzero(db==k)) for k in (np.inf,20.,10.,5.)},
      'missing_counts':{str(k):int(np.count_nonzero(rates==k)) for k in (0.,.1,.3,.5)}}
