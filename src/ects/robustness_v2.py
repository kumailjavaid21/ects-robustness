"""Five independent mixed-calibration robustness draws, paired within each draw."""
import numpy as np
from .robustness import CONDITIONS,streams,seed_for,augment_fit

CORRUPTED=CONDITIONS[1:]
def token(dataset,draw):
    if not 0<=draw<5:raise ValueError('Draw index must be 0-4')
    return f'{dataset}::draw={draw}'

def corrupt(X,dataset,draw,partition,condition,fit_signal_power):
    return streams(X,token(dataset,draw),partition,condition,fit_signal_power)

def train_augmentation(X,y,dataset,draw,power):
    return augment_fit(X,y,token(dataset,draw),power)

def mixed_calibration(X,dataset,draw,power):
    """Balanced mix: each calibration sample gets exactly one of six corruptions."""
    n=len(X);rng=np.random.default_rng(seed_for(token(dataset,draw),'calibration','mix-assignment'))
    assignments=rng.permutation(len(CORRUPTED))[np.arange(n,dtype=int)%len(CORRUPTED)];rng.shuffle(assignments)
    mixed=np.empty_like(np.asarray(X,dtype=np.float32))
    for i,condition in enumerate(CORRUPTED):
        mask=assignments==i
        if mask.any():
            full=corrupt(X,dataset,draw,'calibration',condition,power)
            mixed[mask]=full[mask]
    return mixed,{c:int(np.sum(assignments==i)) for i,c in enumerate(CORRUPTED)}
