import os,sys,json,time,subprocess,traceback
from settings import *
def main():
 while not (OUT/'orchestration_status.json').exists():time.sleep(15)
 try:
  status=json.loads((OUT/'orchestration_status.json').read_text());assert not status['errors'],status['errors']
  for script in ('analysis.py','audit.py','report.py'):
   print('START',script,flush=True);subprocess.run([sys.executable,'-B',str(OUT/'code'/script)],check=True)
  with (OUT/'ANALYSIS_COMPLETE').open('x') as f:f.write('complete; report awaits human review and manifest\n')
  print('FINALIZATION_COMPLETE',flush=True)
 except Exception:
  err=traceback.format_exc()
  with (OUT/'logs/finalization_error.json').open('x') as f:json.dump({'error':err},f,indent=2)
  print(err,flush=True);raise
if __name__=='__main__':main()
