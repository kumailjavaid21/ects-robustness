"""Recover actual acquisition masks from the unchanged seeded corruption recipe."""
import numpy as np
from settings import ROOT
import sys
sys.path.insert(0,str(ROOT/'src'))
from ects.robustness import seed_for,causal_forward_fill
from ects.robustness_v2 import token,corrupt,train_augmentation,mixed_calibration,CORRUPTED

def stream_with_mask(X,name,draw,partition,condition,power):
    value=corrupt(X,name,draw,partition,condition,power)
    if condition.startswith('missing_'):
        rate=float(condition.split('_')[1][:-3])/100.
        rng=np.random.default_rng(seed_for(token(name,draw),partition,'missing'))
        mask=rng.random(np.asarray(X).shape,dtype=np.float32)>=rate
        if not np.array_equal(causal_forward_fill(X,mask),value):raise AssertionError('Stream mask does not reproduce data')
    else:mask=np.ones(np.asarray(X).shape,dtype=bool)
    return value,mask

def augmented_with_mask(X,y,name,draw,power):
    value,labels,info=train_augmentation(X,y,name,draw,power)
    x=np.asarray(X,dtype=np.float32);n=len(x)
    rng=np.random.default_rng(seed_for(token(name,draw),'fit','augmentation'))
    db=rng.choice(np.array([np.inf,20.,10.,5.]),size=n)
    rates=rng.choice(np.array([0.,.1,.3,.5]),size=n)
    z=rng.standard_normal(x.shape,dtype=np.float32)
    sigma=np.where(np.isinf(db),0.,np.sqrt(power/(10**(db/10))))
    noisy=x+z*sigma[:,None].astype(np.float32)
    mask=rng.random(x.shape,dtype=np.float32)>=rates[:,None]
    reproduced=np.concatenate([x,causal_forward_fill(noisy,mask)])
    if not np.array_equal(value,reproduced):raise AssertionError('Augmentation mask/data mismatch')
    return value,labels,np.concatenate([np.ones_like(mask),mask]),info

def mixed_with_mask(X,name,draw,power):
    value,info=mixed_calibration(X,name,draw,power)
    n=len(X);rng=np.random.default_rng(seed_for(token(name,draw),'calibration','mix-assignment'))
    assignment=rng.permutation(len(CORRUPTED))[np.arange(n,dtype=int)%len(CORRUPTED)];rng.shuffle(assignment)
    mask=np.ones(np.asarray(X).shape,dtype=bool)
    for i,cond in enumerate(CORRUPTED):
        take=assignment==i
        if take.any():
            v,m=stream_with_mask(X,name,draw,'calibration',cond,power)
            if not np.array_equal(value[take],v[take]):raise AssertionError('Mixed calibration value mismatch')
            mask[take]=m[take]
    return value,mask,info
