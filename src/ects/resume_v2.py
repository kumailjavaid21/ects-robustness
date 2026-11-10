"""Finish interrupted v2 runs without changing completed dataset records."""
import json,time,traceback
from .run_v2 import OUT,NAMES,MODES,run_one
from .rocket_v2 import load_arrow,split_indices

def main():
    started=time.perf_counter();errors=[]
    for name in NAMES:
        paths=[OUT/'records'/f'{name}_{mode}.json' for mode in MODES]
        if all(p.exists() for p in paths):
            for p in paths:
                data=json.loads(p.read_text())
                assert (OUT/'predictions'/f"{data['dataset']}_{data['mode']}.npz").exists()
            print('REUSE_COMPLETED',name,flush=True);continue
        X,y=load_arrow(name,'train');T,z=load_arrow(name,'test');indices=split_indices(y)
        for mode,path in zip(MODES,paths):
            if path.exists():print('REUSE_COMPLETED',name,mode,flush=True);continue
            try:
                result=run_one(name,mode,X,y,T,z,indices)
                print('RESUMED_MODE_DONE',name,mode,'seconds',round(result['seconds'],2),flush=True)
            except Exception:
                e=f'{name} {mode}: '+traceback.format_exc();errors.append(e);print('RESUME_ERROR',e,flush=True)
    with (OUT/'resume_status.json').open('x') as f:json.dump({'errors':errors,'seconds':time.perf_counter()-started},f,indent=2)
    print('RESUME_DONE',len(errors),'errors',flush=True)
if __name__=='__main__':main()
