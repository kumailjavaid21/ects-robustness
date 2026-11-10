import os,sys,time,json,subprocess
from settings import *
while not (OUT/'orchestration_status.json').exists():time.sleep(15)
status=json.loads((OUT/'orchestration_status.json').read_text())
if status['failed']:raise RuntimeError('Worker failure: inspect logs')
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','MPLCONFIGDIR':str(OUT/'cache'),
     'XDG_CACHE_HOME':str(OUT/'cache'),'TMPDIR':str(OUT/'tmp'),'OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2'}
for s in ['report.py','audit.py']:
 print('FINALIZING',s,flush=True)
 subprocess.run([sys.executable,'-B',str(OUT/'code'/s)],cwd=ROOT,env=env,check=True)
with (OUT/'ANALYSIS_COMPLETE').open('x') as f:f.write('Report and audit complete. Hash after final review.\n')
print('ANALYSIS_COMPLETE',flush=True)
