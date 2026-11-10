import os,sys,time,json,subprocess
from settings import *
while not (OUT/'orchestration_status.json').exists():time.sleep(15)
s=json.loads((OUT/'orchestration_status.json').read_text())
if s['errors']:raise RuntimeError('Worker errors: inspect preserved logs before recovery')
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','MPLCONFIGDIR':str(OUT/'cache'),'XDG_CACHE_HOME':str(OUT/'cache'),
     'NUMBA_CACHE_DIR':str(OUT/'cache'),'TMPDIR':str(OUT/'tmp'),'OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2'}
for script in ['report.py','audit.py']:
 print('FINALIZING',script,flush=True)
 subprocess.run([sys.executable,'-B',str(OUT/'code'/script)],cwd=ROOT,env=env,check=True)
with (OUT/'ANALYSIS_COMPLETE').open('x') as f:f.write('Report and audit complete; awaiting final review/manifest.\n')
print('ANALYSIS_COMPLETE',flush=True)
