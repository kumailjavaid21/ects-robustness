"""Run exactly one dataset/draw worker on each of the 3 project-authorized GPUs."""
import os,sys,time,subprocess,json
from pathlib import Path
from settings import *
def save(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2)
def main():
 save(OUT/'config.json',CONFIG)
 # Spread long inputs and large fits across all three GPUs; all data pairing stays unchanged.
 order=('FordA','HandOutlines','NonInvasiveFetalECGThorax1','StarLightCurves','ElectricDevices','Crop','Wafer','ECG5000')
 jobs=[(n,d) for d in DRAWS for n in order];running={};done=[];failed=[]
 env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
    OPENBLAS_NUM_THREADS='2',NUMBA_NUM_THREADS='2',NUMBA_CACHE_DIR=str(OUT/'cache'),
    MPLCONFIGDIR=str(OUT/'cache'),TMPDIR=str(OUT/'tmp'),XDG_CACHE_HOME=str(OUT/'cache'),TORCH_HOME=str(OUT/'cache'))
 start=time.monotonic()
 while jobs or running:
  for gpu in range(3):
   if gpu not in running and jobs:
    name,draw=jobs.pop(0);log=(OUT/'logs'/f'{name}_draw{draw}.log').open('x')
    e=env.copy();e['CUDA_VISIBLE_DEVICES']=str(gpu)
    p=subprocess.Popen([sys.executable,'-B',str(OUT/'code/worker.py'),'--dataset',name,'--draw',str(draw)],
                       cwd=ROOT,env=e,stdout=log,stderr=subprocess.STDOUT)
    running[gpu]=(p,name,draw,log,time.monotonic());print('START',gpu,name,draw,p.pid,flush=True)
  for gpu,(p,name,draw,log,tic) in list(running.items()):
   code=p.poll()
   if code is not None:
    row={'dataset':name,'draw':draw,'gpu':gpu,'exit':code,'wall_seconds':time.monotonic()-tic}
    done.append(row);log.close();del running[gpu]
    if code:failed.append(row)
    print('FINISH',json.dumps(row),flush=True)
  time.sleep(3)
 save(OUT/'orchestration_status.json',{'jobs':done,'failed':failed,'wall_seconds':time.monotonic()-start})
 print('ALL_DONE','failed',len(failed),'wall_seconds',time.monotonic()-start,flush=True)
if __name__=='__main__':main()
