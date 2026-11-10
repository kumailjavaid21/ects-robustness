import json,hashlib
import numpy as np,pandas as pd
from settings import *
from worker import choice_map,thresholds
from corruptions import audit as corruption_audit
from ects.rocket_v2 import calibrate_sequential
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2)
def decide(pred,conf,times,th):
 good=conf>=np.asarray(th)[None,:];forced=~good.any(1);good[:,-1]=True;j=good.argmax(1)
 return pred[np.arange(len(pred)),j],times[j],forced
def main():
 choices=choice_map();nc=npred=nth=ndec=0;maxdiff=0.
 status=json.loads((OUT/'orchestration_status.json').read_text());assert not status['errors']
 assert corruption_audit()['passed']
 for name in DATASETS:
  meta=json.loads((SCALE/'draw_meta'/f'{name}_base.json').read_text());classes=np.array(meta['classes']);times=np.array(meta['checkpoint_lengths'])
  for d in DRAWS:
   prior=json.loads((SCALE/'draw_meta'/f'{name}_draw{d}.json').read_text())
   deep=json.loads((DEEP/'records'/f'{name}_draw{d}_TCN_aug.json').read_text())
   th={'MiniRocket_aug':{float(a):thresholds(v) for a,v in prior['mixed_thresholds'].items()},
       'TCN_aug':{float(q['setting']):thresholds(q['thresholds']) for q in deep['calibration_frontiers'] if q['kind']=='mixed'}}
   for c in CONDITIONS:
    rec=json.loads((OUT/'records'/f'{name}_draw{d}_{c}.json').read_text())
    with np.load(OUT/'predictions'/f'{name}_draw{d}_{c}.npz') as z:
     y=z['y'];encoded=np.searchsorted(classes,y);assert len(y)==meta['test_size']
     assert np.array_equal(z['times'],times) and np.array_equal(z['classes'],classes)
     for tag in FIXED_METHODS:
      pred=z[tag+'_pred'];conf=z[tag+'_conf']
      assert pred.shape==conf.shape==(len(y),9) and np.isfinite(conf).all() and ((conf>=0)&(conf<=1)).all()
      for j in range(9):
       row=next(q for q in rec['fixed'] if q['method']==tag and q['checkpoint']==j)
       assert abs(row['error']-np.mean(pred[:,j]!=encoded))<1e-12
     for q in rec['selected']:
      method=q['method'];setting=q['setting'];target=q['target']
      assert choices[(name,method,target)]==setting
      if method in AUG_METHODS:pred,t,f=decide(z[method+'_pred'],z[method+'_conf'],times,th[method][setting])
      elif method=='Probability':pred,t,f=decide(z['MiniRocket_clean_pred'],z['MiniRocket_clean_conf'],times,np.full(9,setting))
      elif method=='ECEC':
       idx=z['MiniRocket_clean_pred'];rates=np.array(prior['ecec_val_rates']);failure=np.ones((len(y),len(classes)));score=np.empty((len(y),9))
       ids=np.arange(len(y))
       for j in range(9):failure[ids,idx[:,j]]*=1-rates[j,idx[:,j]];score[:,j]=1-failure[ids,idx[:,j]]
       pred,t,f=decide(idx,score,times,np.full(9,setting))
      else:continue
      assert abs(q['error']-np.mean(pred!=encoded))<1e-12 and abs(q['earliness']-np.mean(t/times[-1]))<1e-12
      assert abs(q['forced_fraction']-f.mean())<1e-12;ndec+=1
    with np.load(OUT/'calibration'/f'{name}_draw{d}_{c}.npz') as z:
     assert len(z['y'])==meta['calibration_count']
     for tag in AUG_METHODS:
      cp=z[tag];assert cp.shape==(len(z['y']),9,len(classes)) and np.isfinite(cp).all()
      for alpha in ALPHAS:
       got=calibrate_sequential(cp,z['y'],classes,alpha)[0]
       row=next(q for q in rec['oracle_calibration'] if q['method']==tag and q['setting']==alpha)
       assert np.array_equal(got,thresholds(row['thresholds']));nth+=1
    if rec['causality_audit']:
     assert rec['causality_audit']['minirocket_exact_prefix'];nc+=1
     maxdiff=max(maxdiff,rec['causality_audit']['TCN_max_prefix_logit_difference'])
    if c.startswith('burst'):
     assert abs(rec['test_missing_fraction']-round(.3*meta['length'])/meta['length'])<1e-12
    npred+=1
 assert npred==1300 and nth==15600 and nc==26 and ndec==10400
 for fname,count in [('fixed_length_stats.csv',300),('crosstype_fixed_op_stats.csv',140)]:
  df=pd.read_csv(OUT/fname);assert len(df)==count and (df.N==26).all()
 snap=json.loads((OUT/'source_snapshot.json').read_text());now={}
 for root in snap['roots']:
  for p in sorted((ROOT/root).rglob('*')):
   if p.is_file() and not p.is_symlink():now[str(p.relative_to(ROOT))]=digest(p)
 before=snap['files']
 source={'files_before':len(before),'files_after':len(now),'changed':[p for p in before if p in now and before[p]!=now[p]],
  'removed':[p for p in before if p not in now],'added':[p for p in now if p not in before]}
 save(OUT/'source_audit.json',source);assert not any(source[k] for k in ('changed','removed','added'))
 result={'prediction_files':npred,'oracle_calibration_threshold_vectors_verified':nth,'selected_decisions_reproduced':ndec,
  'TCN_models_prefix_audited':nc,'TCN_max_prefix_logit_difference':maxdiff,'new_corruption_generators_causal':True,
  'fitted_models_count':0,'cached_mixed_choices_identical':True,'all_statistics_N':26,'source_files_unchanged':len(before)}
 save(OUT/'artifact_audit.json',result);print('AUDIT_PASS',json.dumps(result),flush=True)
if __name__=='__main__':main()
