"""Metrics for early time-series classification."""
import numpy as np

def evaluate(y_true,y_pred,decision_times,length):
    y_true=np.asarray(y_true);y_pred=np.asarray(y_pred);t=np.asarray(decision_times)
    if not len(y_true) or y_true.shape!=y_pred.shape or y_true.shape!=t.shape:
        raise ValueError('Labels, predictions and decision times must align and be nonempty')
    if int(length)<=0 or np.any(t<1) or np.any(t>length):
        raise ValueError('Decision times must be within the observed series')
    accuracy=float(np.mean(y_true==y_pred));earliness=float(np.mean(t/float(length)))
    speed=1-earliness
    return {'accuracy':accuracy,'earliness':earliness,
            'combined':2*accuracy*speed/(accuracy+speed) if accuracy+speed>0 else 0.}
