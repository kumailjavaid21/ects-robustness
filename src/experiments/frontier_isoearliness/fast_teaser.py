"""Vectorized replay of aeon TEASER's fitted safety rules for multiple g values."""
import numpy as np
from sklearn.utils import check_random_state

def _random_tie_argmax(probs,rng):
    maxima=probs.max(axis=1)
    ties=probs==maxima[:,None]
    result=probs.argmax(axis=1)
    # RandomState.choice on a singleton consumes no randomness, so only tied rows need draws.
    for i in np.flatnonzero(ties.sum(axis=1)>1):
        result[i]=int(rng.choice(np.flatnonzero(ties[i])))
    return result

def all_decisions(teaser,probs,times,g_values):
    n,K,C=probs.shape
    if K!=len(teaser._estimators) or list(times)!=teaser._classification_points:
        raise ValueError('TEASER checkpoint mismatch')
    outer=check_random_state(teaser.random_state);pred_index=np.empty((n,K),dtype=int)
    safe=np.zeros((n,K),dtype=bool)
    for j in range(K):
        rng=check_random_state(outer.randint(np.iinfo(np.int32).max))
        p=probs[:,j,:]
        pred=_random_tie_argmax(p,rng);pred_index[:,j]=pred
        safety=teaser._one_class_classifiers[j]
        if j==K-1:safe[:,j]=True
        elif safety is not None:
            d=p.astype(np.float64,copy=True);other=d.copy();other[np.arange(n),pred]=-np.inf
            gap=np.minimum(1.,d[np.arange(n),pred]-other.max(axis=1))
            oc=np.column_stack((d,gap))
            safe[:,j]=safety.predict(oc)==1
    results={}
    for g in g_values:
        counts=np.zeros(n,dtype=int);prev=np.zeros(n,dtype=int)
        final_idx=np.full(n,-1,dtype=int);active=np.ones(n,dtype=bool)
        for j in range(K):
            update=np.flatnonzero(active);accepted=safe[update,j];current=pred_index[update,j]
            counts[update]=np.where(accepted & (current==prev[update]),counts[update]+1,np.where(accepted,1,0))
            prev[update]=current
            done=update[(counts[update]>=g) | (j==K-1)]
            final_idx[done]=j;active[done]=False
        if np.any(final_idx<0):raise AssertionError('Undecided TEASER case')
        p=probs[np.arange(n),final_idx,:]
        final=_random_tie_argmax(p,check_random_state(teaser.random_state))
        prediction=teaser.classes_[final];t=np.asarray(times)[final_idx]
        results[g]=(prediction,t,t==times[-1])
    return results
