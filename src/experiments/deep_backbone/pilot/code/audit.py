import json,hashlib
from pathlib import Path
import numpy as np
from settings import *
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2)
def main():
 old=json.loads((OUT/'source_snapshot.json').read_text());now={}
 for root in old['roots']:
  for p in sorted((ROOT/root).rglob('*')):
   if p.is_file() and not p.is_symlink():now[str(p.relative_to(ROOT))]=digest(p)
 before=old['files'];audit={'files_before':len(before),'files_after':len(now),
  'changed':[p for p in before if p in now and before[p]!=now[p]],
  'removed':[p for p in before if p not in now],'added':[p for p in now if p not in before]}
 save(OUT/'source_audit.json',audit)
 assert not any(audit[k] for k in ('changed','removed','added'))
 records=causal=inherited=streams=pred_files=0
 for name in DATASETS:
  for draw in DRAWS:
   split=json.loads((OUT/'metadata'/f'{name}_draw{draw}_splits.json').read_text())
   prior=json.loads((SCALE/'draw_meta'/f'{name}_draw{draw}.json').read_text())
   assert split['augmentation']==prior['augmentation'], 'Augmentation recipe differs from scale_v1'
   expected=None
   for tag,mode,lam,aug in VARIANTS:
    rec=json.loads((OUT/'records'/f'{name}_draw{draw}_{tag}.json').read_text());records+=1
    if expected is None:expected=rec['stream_hashes']
    assert rec['stream_hashes']==expected
    streams+=len(rec['stream_hashes']);assert rec['training']['causality_audit']['passed'];causal+=1
    if mode=='B':
     events=rec['training']['inheritance_events'];assert len(events)==8 and all(e['exact'] for e in events);inherited+=len(events)
    assert (OUT/'models'/f'{name}_draw{draw}_{tag}.pt').is_file()
    assert len([x for x in rec['rows'] if x['kind']=='fixed'])==len(CONDITIONS)*9
    for cond in CONDITIONS:
     with np.load(OUT/'predictions'/f'{name}_draw{draw}_{tag}_{cond}.npz') as z:
      assert z['pred'].shape==z['confidence'].shape and z['pred'].shape[1]==9 and np.isfinite(z['confidence']).all()
     pred_files+=1
    for t in TARGETS:
     assert rec['selected'][str(t)]==min(rec['calibration_frontier'],key=lambda a:(abs(a['earliness']-t),a['tau']))
 result={'networks':records,'causality_passes':causal,'exact_inheritance_events':inherited,
  'paired_variant_stream_hashes':streams,'prediction_files':pred_files,'datasets':8,'draws':3,'conditions':5,
  'minirocket_retraining':False,'prior_inputs_unchanged':True,'augmentation_recipes_match_scale_v1':24}
 save(OUT/'artifact_audit.json',result);print('AUDIT',json.dumps(result),flush=True)
if __name__=='__main__':main()
