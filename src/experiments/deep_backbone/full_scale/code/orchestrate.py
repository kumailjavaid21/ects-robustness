"""Three persistent GPU workers, balanced by training/inference work; fixed experiment."""
import os,sys,time,json,subprocess
from settings import *
def save(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2)
def main():
 assert json.loads((OUT/'preflight.json').read_text())['mask_causality']['passed']
 save(OUT/'config.json',CONFIG)
 jobs=[]
 for name in DATASETS:
  m=json.loads((SCALE/'draw_meta'/f'{name}_base.json').read_text())
  for d in DRAWS:
   reuse=(PILOT/'models'/f'{name}_draw{d}_C_l0.pt').exists()
   fits=1 if reuse else 2
   cost=m['length']*(fits*(m['fit_count']*2*33*3+m['validation_count']*2*33)+m['test_size']*(9 if reuse else 14))
   jobs.append((cost,name,d))
 bins=[[] for _ in range(3)];loads=[0.,0.,0.]
 for cost,n,d in sorted(jobs,reverse=True):
  i=loads.index(min(loads));bins[i].append([n,d]);loads[i]+=cost
 env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2',
      'NUMBA_NUM_THREADS':'2','NUMBA_CACHE_DIR':str(OUT/'cache'),'MPLCONFIGDIR':str(OUT/'cache'),
      'TMPDIR':str(OUT/'tmp'),'XDG_CACHE_HOME':str(OUT/'cache'),'TORCH_HOME':str(OUT/'cache')}
 workers=[];started=time.monotonic()
 for gpu,j in enumerate(bins):
  jf=OUT/'metadata'/f'jobs_gpu{gpu}.json';save(jf,j)
  log=(OUT/'logs'/f'gpu{gpu}.log').open('x')
  e={**env,'CUDA_VISIBLE_DEVICES':str(gpu)}
  p=subprocess.Popen([sys.executable,'-B',str(OUT/'code/worker.py'),'--jobs',str(jf),'--worker',str(gpu)],
                      cwd=ROOT,env=e,stdout=log,stderr=subprocess.STDOUT)
  workers.append((gpu,p,log));print('START',gpu,p.pid,'jobs',len(j),'first',j[0],flush=True)
 for gpu,p,log in workers:
  code=p.wait();log.close();print('WORKER_EXIT',gpu,code,flush=True)
 statuses=[];errors=[]
 for gpu,p,log in workers:
  f=OUT/'logs'/f'worker_{gpu}_status.json'
  if f.exists():
   s=json.loads(f.read_text());statuses.append(s);errors.extend(s['errors'])
  else:errors.append({'worker':gpu,'error':'Missing status file'})
 save(OUT/'orchestration_status.json',{'workers':statuses,'errors':errors,'wall_seconds':time.monotonic()-started,
     'planned_jobs':130,'planned_models':260,'planned_new_fits':236,'planned_pilot_reuse':24})
 print('ALL_DONE','errors',len(errors),'seconds',time.monotonic()-started,flush=True)
if __name__=='__main__':main()
