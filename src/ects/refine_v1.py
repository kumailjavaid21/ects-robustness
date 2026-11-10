"""Evaluate fixed and probability rules at every timestep using saved v1 models."""
import json,time
from pathlib import Path
import numpy as np
import torch
from .run_baselines import load_split,select
from .anytime import AnytimeGRU,normalize,probabilities
from .run_v1 import ROOT,OUT,sweep_full,make_summary,gpu_choice

def main():
    input_path=OUT/'results.json';original=json.loads(input_path.read_text())
    device=gpu_choice();t0=time.perf_counter();records=original['records']
    for r in records:
        name=r['dataset'];mode=r['mode'];X,y=load_split(name,'train');T,z=load_split(name,'test')
        model_info=torch.load(r['model_path'],map_location=device,weights_only=True)
        classes=np.array(model_info['classes'])
        model=AnytimeGRU(len(classes),model_info['hidden']).to(device)
        model.load_state_dict(model_info['state_dict']);model.eval()
        val_idx=np.array(r['split']['validation_indices']);val=normalize(X[val_idx],mode);test=normalize(T,mode)
        valp=probabilities(model,val,device);testp=probabilities(model,test,device)
        val_rows=sweep_full(valp,classes,y[val_idx]);test_rows=sweep_full(testp,classes,z)
        r['validation_sweep']=val_rows;r['test_sweep']=test_rows
        for method in ('fixed','threshold'):
            selection=select(val_rows,method);r['selected'][method]=selection
            r['results'][method]=next(v for v in test_rows if v['baseline']==method and v['setting']==selection['setting'])
        path=OUT/f'{name}_{mode}_everytime_results.json'
        with path.open('x') as f:json.dump(r,f,indent=2,allow_nan=False)
        print('FAIR_SWEEP_DONE',mode,name,
              'fixed',r['results']['fixed'],'threshold',r['results']['threshold'],flush=True)
    elapsed=time.perf_counter()-t0
    with (OUT/'results_everytime.json').open('x') as f:
        json.dump({'device':str(device),'device_name':torch.cuda.get_device_name(device),
                   'seconds':original['seconds'],'refinement_seconds':elapsed,'records':records},f,indent=2,allow_nan=False)
    summary=OUT/'summary.md'
    with (OUT/'summary_grid10.md').open('xb') as f:f.write(summary.read_bytes())
    with summary.open('w') as f:f.write(make_summary(records,device,original['seconds']+elapsed))
    print('FAIR_SWEEP_COMPLETE',round(elapsed,2),flush=True)
if __name__=='__main__':main()
