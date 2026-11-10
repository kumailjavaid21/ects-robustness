"""Re-evaluate aeon TEASER using prefix-restricted cross-validation."""
import json,time,traceback
import numpy as np
from .run_v1 import OUT,make_summary,gpu_choice,teaser
from .run_baselines import load_split,FRACTIONS
from .prefix import prefix_times
from .anytime import normalize

def main():
    original=json.loads((OUT/'results_everytime.json').read_text());records=original['records']
    device=gpu_choice();started=time.perf_counter()
    for r in records:
        name=r['dataset'];mode=r['mode'];X,y=load_split(name,'train');T,z=load_split(name,'test')
        fit=np.array(r['split']['fit_indices']);times=prefix_times(X.shape[1],FRACTIONS)
        try:r['results']['TEASER']=teaser(normalize(X[fit],mode),y[fit],normalize(T,mode),z,times)
        except Exception:
            err=traceback.format_exc();r['results']['TEASER']={'baseline':'TEASER','error':err}
            print('TEASER_PREFIX_ERROR',name,mode,err,flush=True)
        print('TEASER_PREFIX_DONE',name,mode,json.dumps(r['results']['TEASER']),flush=True)
        with (OUT/f'{name}_{mode}_teaser_prefix_results.json').open('x') as f:json.dump(r,f,indent=2,allow_nan=False)
    elapsed=time.perf_counter()-started
    with (OUT/'results_teaser_prefix.json').open('x') as f:
        json.dump({'device':str(device),'seconds':original['seconds']+original['refinement_seconds']+elapsed,
                   'teaser_prefix_seconds':elapsed,'records':records},f,indent=2,allow_nan=False)
    summary=OUT/'summary.md'
    with (OUT/'summary_aeon_fullcv.md').open('xb') as f:f.write(summary.read_bytes())
    with summary.open('w') as f:f.write(make_summary(records,device,original['seconds']+original['refinement_seconds']+elapsed))
    print('TEASER_PREFIX_ALL_DONE',round(elapsed,2),flush=True)
if __name__=='__main__':main()
