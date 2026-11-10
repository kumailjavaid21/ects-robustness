"""Recalibrate stored checkpoint predictions conditional on reaching each checkpoint."""
import json
import numpy as np
from .run_v2 import OUT,NAMES,MODES,ALPHAS,summarize
from .rocket_v2 import calibrate_sequential,decisions,metrics

def main():
    records=[];errors=[]
    (OUT/'records_sequential').mkdir(exist_ok=True)
    for name in NAMES:
        for mode in MODES:
            path=OUT/'records'/f'{name}_{mode}.json'
            if not path.exists():
                errors.append(f'Missing completed record: {name} {mode}');continue
            r=json.loads(path.read_text())
            with np.load(OUT/'predictions'/f'{name}_{mode}.npz') as p:
                classes=p['classes'];times=p['times'];calp=p['cal_prob'];yc=p['cal_y'];testp=p['test_prob'];z=p['test_y']
                for alpha in ALPHAS:
                    lam,support,err,at_risk=calibrate_sequential(calp,yc,classes,alpha)
                    pred,t,forced=decisions(testp,classes,times,lam)
                    m=metrics(z,pred,t,int(times[-1]));test_error=1-m['accuracy']
                    m.update({'baseline':'reliability','alpha':alpha,'test_error':test_error,
                       'meets_target':test_error<=alpha+1e-12,'forced_fraction':float(forced.mean()),'forced_count':int(forced.sum())})
                    r['results'][f'alpha_{alpha:.2f}']=m
                    cp,ct,cf=decisions(calp,classes,times,lam)
                    r['calibration'][f'{alpha:.2f}']={
                       'thresholds':[float(x) if np.isfinite(x) else None for x in lam],
                       'support':support.tolist(),'at_risk':at_risk.tolist(),'per_checkpoint_error':err.tolist(),
                       'calibration_error_at_stop':float(np.mean(cp!=yc)),
                       'calibration_forced_fraction':float(cf.mean())}
                    print('SEQUENTIAL',name,mode,alpha,'test_error',round(test_error,4),
                          'forced',round(float(forced.mean()),4),'cal_stop_error',round(float(np.mean(cp!=yc)),4),flush=True)
            with (OUT/'records_sequential'/f'{name}_{mode}.json').open('x') as f:json.dump(r,f,indent=2,allow_nan=False)
            records.append(r)
    seconds=sum(r['seconds'] for r in records)
    payload=json.dumps({'records':records,'errors':errors,'seconds':seconds,
                       'protocol':'Sequential calibration conditional on not yet stopped.'},indent=2,allow_nan=False)
    for filename in ('results_sequential.json','results.json'):
        with (OUT/filename).open('x') as f:f.write(payload)
    with (OUT/'summary.md').open('x') as f:f.write(summarize(records,errors,seconds))
    print('SEQUENTIAL_ALL_DONE',len(records),'errors',len(errors),'summed_seconds',round(seconds,2),flush=True)
if __name__=='__main__':main()
