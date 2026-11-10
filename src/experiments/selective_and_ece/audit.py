"""Verify prior sources and the complete no-retraining replay."""
import sys,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE));from replay import sw,CONDITIONS
import numpy as np
from ects.rocket_v2 import calibrate_sequential
def digest(f):
    h=hashlib.sha256()
    with f.open('rb') as o:
        for chunk in iter(lambda:o.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
old=json.loads((HERE/'source_snapshot.json').read_text());now={}
for root in old['roots']:
    for p in sorted((ROOT/root).rglob('*')):
        if p.is_file() and not p.is_symlink():now[str(p.relative_to(ROOT))]=digest(p)
before=old['files']
audit={'files_hashed_before':len(before),'files_hashed_after':len(now),
 'changed':[p for p in before if p in now and before[p]!=now[p]],
 'removed':[p for p in before if p not in now],'added':[p for p in now if p not in before]}
(HERE/'source_audit.json').write_text(json.dumps(audit,indent=2))
print('SOURCE_AUDIT',json.dumps(audit),flush=True)
missing=[];count=0;cal=0;cert=0;threshold_checks=0;mismatched_thresholds=[]
for name in sw.MAIN:
 for draw in range(5):
  meta=HERE/'draw_meta'/f'{name}_draw{draw}.json';npz=HERE/'calibration'/f'{name}_draw{draw}.npz'
  if not(meta.exists() and npz.exists()):missing.append(str(meta));continue
  p=json.loads(meta.read_text());cal+=1;cert+=int(p['policy']['status']=='certified_on_mixed_calibration')
  prior=json.loads((ROOT/'runs/ects_scale_v1/draw_meta'/f'{name}_draw{draw}.json').read_text())
  with np.load(npz) as z:
   for alpha in sw.ALPHAS:
    got=calibrate_sequential(z['p_augmented'],z['y'],z['classes'],alpha)[0]
    expected=np.asarray([np.inf if x is None else x for x in prior['mixed_thresholds'][str(alpha)]])
    threshold_checks+=1
    if not np.allclose(got,expected,rtol=1e-6,atol=1e-7):mismatched_thresholds.append(f'{name} draw {draw} alpha {alpha}')
  for cond in CONDITIONS:
   path=HERE/'records'/f'{name}_draw{draw}_{cond}.json'
   if not path.exists():missing.append(str(path));continue
   rec=json.loads(path.read_text())
   if rec.get('replay_check')!='all clean and augmented checkpoint accuracies match prior record':raise ValueError('Replay check absent '+str(path))
   count+=1
result={'datasets':len(sw.MAIN),'draws':5,'conditions':len(CONDITIONS),'records_verified':count,
        'calibration_sets':cal,'certified_policies':cert,'prior_threshold_checks':threshold_checks,'mismatched_thresholds':mismatched_thresholds,'missing':missing,'retraining':False}
(HERE/'replay_audit.json').write_text(json.dumps(result,indent=2))
print('REPLAY_AUDIT',json.dumps(result),flush=True)
if missing or mismatched_thresholds or count!=len(sw.MAIN)*5*len(CONDITIONS):sys.exit(1)
