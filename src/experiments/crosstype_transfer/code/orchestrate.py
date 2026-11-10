import os,sys,time,json,subprocess
from settings import *
def save(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2)
def main():
 assert json.loads((OUT/'preflight.json').read_text())['corruption_causality']['passed']
 bins=[[] for _ in range(3)];loads=[0.]*3
 costs=[]
 for name in DATASETS:
  m=json.loads((SCALE/'draw_meta'/f'{name}_base.json').read_text())
  costs.append((m['length']*(m['test_size']+m['calibration_count']),name))
 for cost,name in sorted(costs,reverse=True):
  i=loads.index(min(loads));bins[i].append(name);loads[i]+=cost
 env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2',
  'NUMBA_NUM_THREADS':'2','NUMBA_CACHE_DIR':str(OUT/'cache'),'MPLCONFIGDIR':str(OUT/'cache'),
  'TMPDIR':str(OUT/'tmp'),'XDG_CACHE_HOME':str(OUT/'cache'),'TORCH_HOME':str(OUT/'cache')}
 workers=[];start=time.monotonic()
 for gpu,names in enumerate(bins):
  log=(OUT/'logs'/f'gpu{gpu}.log').open('x')
  p=subprocess.Popen([sys.executable,'-B',str(OUT/'code/worker.py'),'--worker',str(gpu),'--datasets',*names],
    cwd=ROOT,env={**env,'CUDA_VISIBLE_DEVICES':str(gpu)},stdout=log,stderr=subprocess.STDOUT)
  workers.append((gpu,p,log));print('WORKER_START',gpu,p.pid,names,flush=True)
 save(OUT/'worker_pids.json',[{'worker':i,'pid':p.pid,'datasets':bins[i]} for i,p,l in workers])
 for i,p,l in workers:
  status=p.wait();l.close();print('WORKER_EXIT',i,status,flush=True)
 statuses=[];errors=[]
 for i,p,l in workers:
  f=OUT/'logs'/f'worker_{i}_status.json'
  if f.exists():
   s=json.loads(f.read_text());statuses.append(s);errors.extend(s['errors'])
  else:errors.append({'worker':i,'error':'Missing worker status'})
 save(OUT/'orchestration_status.json',{'workers':statuses,'errors':errors,'wall_seconds':time.monotonic()-start,'training_performed':False})
 print('ALL_DONE',len(errors),'errors','seconds',time.monotonic()-start,flush=True)
if __name__=='__main__':main()
