"""Verify readonly sources, all paired artifacts, calibration thresholds, and pilot reuse."""
import json,hashlib
import numpy as np
from settings import *
from ects.rocket_v2 import calibrate_sequential
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2)
def main():
 snap=json.loads((OUT/'source_snapshot.json').read_text());before=snap['files'];now={}
 for root in snap['roots']:
  for p in sorted((ROOT/root).rglob('*')):
   if p.is_file() and not p.is_symlink():now[str(p.relative_to(ROOT))]=digest(p)
 audit={'files_before':len(before),'files_after':len(now),
  'changed':[p for p in before if p in now and before[p]!=now[p]],
  'removed':[p for p in before if p not in now],'added':[p for p in now if p not in before]}
 save(OUT/'source_audit.json',audit)
 assert not any(audit[k] for k in ('changed','removed','added'))
 count=reused=predictions=causal=thresholds=pilot_predictions=0
 max_difference=0.
 for name in DATASETS:
  for draw in DRAWS:
   split=json.loads((OUT/'metadata'/f'{name}_draw{draw}_splits.json').read_text())
   prior=json.loads((SCALE/'draw_meta'/f'{name}_draw{draw}.json').read_text())
   assert split['augmentation']==prior['augmentation']
   expected=None;mask_expected=None
   for tag in VARIANTS:
    rec=json.loads((OUT/'records'/f'{name}_draw{draw}_{tag}.json').read_text());count+=1
    if expected is None:expected=rec['stream_hashes'];mask_expected=rec['mask_hashes']
    assert rec['stream_hashes']==expected and rec['mask_hashes']==mask_expected
    assert len(set(rec['mask_hashes'][c] for c in ['clean','noise_20db','noise_10db','noise_5db']))==1
    tr=rec['training'];ca=tr['causality_audit'];assert ca['passed'] and ca['future_value_and_mask_gradient_max']==0
    causal+=1;max_difference=max(max_difference,ca['max_prefix_logit_difference'])
    model=OUT/'models'/f'{name}_draw{draw}_{tag}.pt';assert model.exists()
    if tr['reused_from']:
     assert tag=='TCN_aug' and digest(model)==digest(Path(tr['reused_from']))
     assert split['pilot_inputs_match'];reused+=1;pilot_predictions+=len(rec['reused_pilot_prediction_conditions'])
    assert len(rec['rows'])==7*(9+6+7)
    with np.load(OUT/'predictions'/f'{name}_draw{draw}_{tag}_calval.npz') as z:
     probs=z['cal_prob'];cy=z['cal_y']
     for alpha in ALPHAS:
      got=calibrate_sequential(probs,cy,np.arange(probs.shape[-1]),alpha)[0]
      row=next(r for r in rec['calibration_frontiers'] if r['kind']=='mixed' and r['setting']==alpha)
      expected_th=np.array([np.inf if x is None else x for x in row['thresholds']])
      assert np.array_equal(got,expected_th);thresholds+=1
    for cond in CONDITIONS:
     with np.load(OUT/'predictions'/f'{name}_draw{draw}_{tag}_{cond}.npz') as z:
      assert z['pred'].shape==z['confidence'].shape and z['pred'].shape[1]==9
      assert np.isfinite(z['confidence']).all() and ((z['confidence']>=0)&(z['confidence']<=1)).all()
     predictions+=1
 assert count==260 and reused==24 and predictions==1820 and pilot_predictions==120
 result={'model_artifacts':count,'newly_fitted':count-reused,'reused_pilot_models':reused,
  'reused_pilot_test_prediction_files':pilot_predictions,'prediction_files':predictions,'causality_passes':causal,
  'max_prefix_logit_difference':max_difference,'zero_future_value_and_mask_gradients':True,
  'calibration_threshold_vectors_verified':thresholds,'augmentation_recipes_match_prior':130,
  'paired_signal_and_mask_hash_groups':130,'prior_sources_unchanged':True,'baseline_retraining':False}
 save(OUT/'artifact_audit.json',result);print('AUDIT',json.dumps(result),flush=True)
if __name__=='__main__':main()
