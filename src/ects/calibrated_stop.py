"""Per-time empirical calibration; no distribution-free risk guarantee is claimed."""
import numpy as np

def calibrate(probs,y,classes,alpha):
    # probs: [sample,time,class], all examples have observed full calibration series.
    y=np.asarray(y);conf=probs.max(axis=2);pred=classes[probs.argmax(axis=2)]
    lambdas=np.full(probs.shape[1],np.inf);accepted=np.zeros(probs.shape[1],dtype=int);errors=np.zeros(probs.shape[1])
    for t in range(probs.shape[1]):
        # Tied confidence values must be selected together under the >= rule.
        for threshold in np.unique(conf[:,t]):
            mask=conf[:,t]>=threshold;err=float(np.mean(pred[mask,t]!=y[mask]))
            if err<=alpha:
                lambdas[t]=threshold;accepted[t]=int(mask.sum());errors[t]=err
                break
    return lambdas,accepted,errors

def predict(probs,classes,lambdas):
    reached=probs.max(axis=2)>=lambdas[None,:]
    stopped_before_forcing=reached.any(axis=1)
    reached[:,-1]=True
    t=reached.argmax(axis=1);idx=np.arange(len(t))
    return classes[probs[idx,t].argmax(axis=1)],t+1,~stopped_before_forcing
