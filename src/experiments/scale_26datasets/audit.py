"""Audit source immutability, model convergence and evaluation completeness."""
import sys,json,hashlib
from pathlib import Path
import joblib,numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from scale_worker import MAIN,CAVEAT,OLD
def digest(f):
    h=hashlib.sha256()
    with f.open('rb') as o:
        for chunk in iter(lambda:o.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
old=json.loads((HERE/'source_snapshot.json').read_text());now={}
for root in old['roots']:
    for f in sorted((ROOT/root).rglob('*')):
        if f.is_file() and not f.is_symlink():now[str(f.relative_to(ROOT))]=digest(f)
before=old['files'];changed=[p for p in before if p in now and before[p]!=now[p]]
removed=[p for p in before if p not in now];added=[p for p in now if p not in before]
audit={'files_hashed_before':len(before),'files_hashed_after':len(now),'changed':changed,'removed':removed,'added':added}
(HERE/'source_audit.json').write_text(json.dumps(audit,indent=2))
print('SOURCE_AUDIT',json.dumps(audit),flush=True)
rows=[];failures=[]
for name in MAIN+CAVEAT:
    meta=json.loads((HERE/'draw_meta'/f'{name}_base.json').read_text())
    for draw in [-1,0,1,2,3,4]:
        for t in meta['checkpoint_lengths']:
            if name in OLD:
                if draw==-1:
                    path=(ROOT/'runs/ects_v2/models'/f'{name}_causal_{t:04d}.joblib' if name not in CAVEAT else ROOT/'runs/ects_robust_v2/base_models'/f'{name}_base_{t:04d}.joblib')
                else:path=ROOT/'runs/ects_robust_v2/models'/f'{name}_aug_draw{draw}_{t:04d}.joblib'
            else:
                path=(HERE/'base_models'/f'{name}_base_{t:04d}.joblib' if draw==-1 else HERE/'models'/f'{name}_aug_draw{draw}_{t:04d}.joblib')
            try:
                z=joblib.load(path);clf=z['classifier'].named_steps['logisticregression'];iters=int(np.max(clf.n_iter_))
                rows.append({'dataset':name,'draw':draw,'checkpoint':t,'iterations':iters,'max_iter':int(clf.max_iter),'hit_iteration_limit':iters>=clf.max_iter,'path':str(path.relative_to(ROOT))})
            except Exception as e:failures.append({'path':str(path),'error':repr(e)})
pd.DataFrame(rows).to_csv(HERE/'model_convergence_audit.csv',index=False)
warn=[]
for f in sorted((HERE/'logs').glob('*.log')):
    for i,line in enumerate(f.read_text(errors='replace').splitlines(),1):
        if 'ConvergenceWarning' in line or 'DATASET_ERROR' in line or 'Traceback' in line:warn.append({'file':str(f.relative_to(HERE)),'line':i,'text':line})
(HERE/'logs'/'warnings_and_errors.json').write_text(json.dumps({'notices':warn,'audit_failures':failures},indent=2))
print('MODEL_AUDIT models',len(rows),'iteration_limit',sum(x['hit_iteration_limit'] for x in rows),'read_failures',len(failures),'logged_notices',len(warn),flush=True)
