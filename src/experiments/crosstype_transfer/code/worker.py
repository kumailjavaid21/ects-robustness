"""Cached-model inference only; all calibration decisions precede test-label metrics."""
import os,sys,json,time,hashlib,argparse,traceback,gc
import numpy as np,torch,joblib
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
from settings import *
from corruptions import corrupt
from ects.anytime import normalize
from ects.run_robust_v2 import proba
from ects.rocket_v2 import decisions,calibrate_sequential
from fast_teaser import all_decisions
def save(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def harray(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def model_paths(name,times,draw=None):
 if draw is None:
  return [(ROOT/'runs/ects_v2/models'/f'{name}_causal_{int(t):04d}.joblib' if name in sw.OLD else SCALE/'base_models'/f'{name}_base_{int(t):04d}.joblib') for t in times]
 folder=ROOT/'runs/ects_robust_v2/models' if name in sw.OLD else SCALE/'models'
 return [folder/f'{name}_aug_draw{draw}_{int(t):04d}.joblib' for t in times]
def load_teaser(name):
 p=ROOT/'runs/ects_frontier_v1/teaser_safety'/f'{name}.joblib' if name in sw.OLD else SCALE/'teaser_safety'/f'{name}_clean.joblib'
 return joblib.load(p)
def metrics(y,p,t,forced,L):
 acc=float(accuracy_score(y,p));early=float(np.mean(np.asarray(t)/L))
 return {'accuracy':acc,'error':1-acc,'balanced_accuracy':float(balanced_accuracy_score(y,p)),
  'macro_f1':float(f1_score(y,p,average='macro',zero_division=0)),'earliness':early,
  'combined':2*acc*(1-early)/(acc+1-early) if acc+1-early else 0.,'forced_fraction':float(np.mean(forced))}
def predict_deep(model,x,times,batch=128):
 out=[];model.eval()
 with torch.inference_mode():
  for i in range(0,len(x),batch):
   a=torch.as_tensor(np.ascontiguousarray(x[i:i+batch]),device='cuda')
   out.append(model(a,times).float().softmax(-1).cpu().numpy())
 return np.concatenate(out)
def thresholds(v):return np.array([np.inf if t is None else t for t in v])
def serial(v):return [float(t) if np.isfinite(t) else None for t in v]
def choice_map():
 import csv
 rows=list(csv.DictReader((DEEP/'fixedop_calibration_choices.csv').open()))
 return {(q['dataset'],q['method'],float(q['target'])):float(q['setting']) for q in rows if q['method'] in METHODS}
def check_prefix_models(base,model,values,times):
 # Transform the actual observed prefix only; suffix replacement cannot change its inputs.
 from copy import deepcopy
 changed=values[:2].copy();changed[:,int(times[3]):]=1000.
 a=base['transforms'][3].transform(np.ascontiguousarray(values[:2,:int(times[3]),None].transpose(0,2,1)))
 b=base['transforms'][3].transform(np.ascontiguousarray(changed[:,:int(times[3]),None].transpose(0,2,1)))
 assert np.array_equal(a,b)
 v=torch.tensor(np.ascontiguousarray(values[:2]),device='cuda')
 with torch.inference_mode():
  full=model(v,times);part=model(v[:,:int(times[3])],times[:4])
 diff=float((full[:,3]-part[:,3]).abs().max())
 assert diff<2e-4
 return {'minirocket_exact_prefix':True,'TCN_max_prefix_logit_difference':diff}
def run_dataset(name):
 start=time.monotonic();meta=json.loads((SCALE/'draw_meta'/f'{name}_base.json').read_text())
 raw,y=sw.load_data(name,'train');test,z=sw.load_data(name,'test')
 times=np.array(meta['checkpoint_lengths']);classes=np.array(meta['classes']);cal=np.array(meta['calibration_indices'])
 rawcal=raw[cal];cy=y[cal];L=raw.shape[1];choices=choice_map()
 base=sw.modelset(model_paths(name,times));teaser=load_teaser(name);detcache={}
 for d in DRAWS:
  done=OUT/'metadata'/f'{name}_draw{d}_complete.json'
  if done.exists():print('REUSE_COMPLETE',name,d,flush=True);continue
  tic=time.monotonic();aug=sw.modelset(model_paths(name,times,d))
  prior=json.loads((SCALE/'draw_meta'/f'{name}_draw{d}.json').read_text())
  mixed={'MiniRocket_aug':{float(a):thresholds(t) for a,t in prior['mixed_thresholds'].items()}}
  deep_rec=json.loads((DEEP/'records'/f'{name}_draw{d}_TCN_aug.json').read_text())
  mixed['TCN_aug']={float(q['setting']):thresholds(q['thresholds']) for q in deep_rec['calibration_frontiers'] if q['kind']=='mixed'}
  mpath=DEEP/'models'/f'{name}_draw{d}_TCN_aug.pt'
  ck=torch.load(mpath,map_location='cuda',weights_only=True)
  assert ck['classes']==classes.tolist() and ck['times']==times.tolist()
  model=AnytimeTCN(len(classes),'C',len(times)).cuda();model.load_state_dict(ck['state_dict']);model.eval()
  rates=np.array(prior['ecec_val_rates']);completed=[]
  for c in CONDITIONS:
   recpath=OUT/'records'/f'{name}_draw{d}_{c}.json'
   if recpath.exists():completed.append(c);continue
   before=time.monotonic()
   # Labels are never passed to corruption or model inference.
   tx,mask=corrupt(test,name,d,'test',c);cx,cmask=corrupt(rawcal,name,d,'calibration',c)
   tx=normalize(tx,'causal');cx=normalize(cx,'causal')
   if c.startswith('quantization') and c in detcache:pb=detcache[c]
   else:
    pb=proba(base,tx,times)
    if c.startswith('quantization'):detcache[c]=pb
   pa=proba(aug,tx,times);pt=predict_deep(model,tx,times,min(128,max(16,131072//L)))
   cap=proba(aug,cx,times);ctp=predict_deep(model,cx,times,min(128,max(16,131072//L)))
   assert pb.shape==pa.shape==pt.shape==(len(z),len(times),len(classes))
   oracle={};calrows=[]
   for tag,p in [('MiniRocket_aug',cap),('TCN_aug',ctp)]:
    oracle[tag]={}
    for alpha in ALPHAS:
     th,support,err,risk=calibrate_sequential(p,cy,classes,alpha);oracle[tag][alpha]=th
     pred,t,f=decisions(p,classes,times,th)
     calrows.append({'method':tag,'setting':alpha,'earliness':float(np.mean(t/L)),'thresholds':serial(th),
                    'support':support.tolist(),'at_risk':risk.tolist()})
   # Policies are now fully determined without test labels. Labels enter only metrics below.
   pp={'MiniRocket_clean':pb,'MiniRocket_aug':pa,'TCN_aug':pt}
   fixed=[]
   for tag,p in pp.items():
    for j,t in enumerate(times):
     fixed.append({'method':tag,'checkpoint':j,'nominal_fraction':(j+2)/10,
      **metrics(z,classes[p[:,j].argmax(1)],np.full(len(z),t),np.zeros(len(z),bool),L)})
   fronts=[]
   for tag,p in [('MiniRocket_aug',pa),('TCN_aug',pt)]:
    for regime,ths in [('mixed',mixed[tag]),('oracle',oracle[tag])]:
     for alpha in ALPHAS:
      pred,t,f=decisions(p,classes,times,ths[alpha])
      fronts.append({'method':tag,'regime':regime,'setting':alpha,**metrics(z,pred,t,f,L)})
   selected=[]
   gvalues=sorted(set(int(choices[(name,'TEASER',target)]) for target in TARGETS))
   teaser_out=all_decisions(teaser,pb,times,gvalues)
   for target in TARGETS:
    for tag in METHODS:
     setting=choices[(name,tag,target)]
     if tag in AUG_METHODS:
      p=pa if tag=='MiniRocket_aug' else pt;pred,t,f=decisions(p,classes,times,mixed[tag][setting])
     elif tag=='Probability':pred,t,f=decisions(pb,classes,times,np.full(len(times),setting))
     elif tag=='TEASER':pred,t,f=teaser_out[int(setting)]
     else:pred,t,f=sw.ecec_decide(pb,classes,times,rates,setting)
     selected.append({'method':tag,'target':target,'setting':setting,**metrics(z,pred,t,f,L)})
   audit=None
   if d==0 and c=='spike_005':audit=check_prefix_models(base,model,tx,times)
   fields={'y':z,'times':times,'classes':classes}
   for tag,p in pp.items():
    fields[tag+'_pred']=p.argmax(2).astype(np.uint16);fields[tag+'_conf']=p.max(2).astype(np.float32)
   with (OUT/'predictions'/f'{name}_draw{d}_{c}.npz').open('xb') as f:np.savez_compressed(f,**fields)
   with (OUT/'calibration'/f'{name}_draw{d}_{c}.npz').open('xb') as f:np.savez_compressed(f,MiniRocket_aug=cap,TCN_aug=ctp,y=cy,times=times,classes=classes)
   save(recpath,{'dataset':name,'draw':d,'condition':c,'type':TYPES[c],'fixed':fixed,'selected':selected,'frontiers':fronts,
        'oracle_calibration':calrows,'stream_sha256':harray(tx),'calibration_stream_sha256':harray(cx),
        'test_missing_fraction':float((~mask).mean()),'cal_missing_fraction':float((~cmask).mean()),
        'causality_audit':audit,'seconds':time.monotonic()-before})
   completed.append(c)
   print('CONDITION_DONE',name,d,c,'sec',round(time.monotonic()-before,2),flush=True)
  save(done,{'dataset':name,'draw':d,'conditions':completed,'seconds':time.monotonic()-tic,
       'clean_model_paths':[str(p) for p in model_paths(name,times)],
       'aug_model_paths':[str(p) for p in model_paths(name,times,d)],'TCN_model_path':str(mpath),
       'training_performed':False})
  print('DRAW_DONE',name,d,'sec',round(time.monotonic()-tic,1),flush=True)
  del model,aug;torch.cuda.empty_cache();gc.collect()
 print('DATASET_DONE',name,'sec',round(time.monotonic()-start,1),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',required=True);p.add_argument('--worker',required=True);a=p.parse_args()
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 done=[];errors=[]
 for name in a.datasets:
  try:run_dataset(name);done.append(name)
  except Exception:
   e=traceback.format_exc();errors.append({'dataset':name,'error':e});print('DATASET_ERROR',name,e,flush=True)
 save(OUT/'logs'/f'worker_{a.worker}_status.json',{'worker':a.worker,'datasets':done,'errors':errors})
 print('WORKER_DONE',a.worker,'errors',len(errors),flush=True)
