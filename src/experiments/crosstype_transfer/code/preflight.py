import json,time
import numpy as np,torch
from settings import *
from corruptions import audit
from worker import model_paths,load_teaser,predict_deep,choice_map
from ects.anytime import normalize
from ects.run_robust_v2 import proba
from fast_teaser import all_decisions
def main():
 result={'corruption_causality':audit(),'model_inventory':{},'training_performed':False}
 assert len(DATASETS)==26
 choices=choice_map();assert len(choices)==26*5*2
 for n in DATASETS:
  m=json.loads((SCALE/'draw_meta'/f'{n}_base.json').read_text());times=m['checkpoint_lengths']
  paths=model_paths(n,times)
  for d in DRAWS:paths+=model_paths(n,times,d)+[DEEP/'models'/f'{n}_draw{d}_TCN_aug.pt',DEEP/'records'/f'{n}_draw{d}_TCN_aug.json']
  assert all(p.exists() for p in paths),n
  result['model_inventory'][n]=len(paths)
 n='FordA';m=json.loads((SCALE/'draw_meta'/f'{n}_base.json').read_text())
 times=np.array(m['checkpoint_lengths']);classes=np.array(m['classes'])
 x,y=sw.load_data(n,'test');x=normalize(x[:16],'causal')
 base=sw.modelset(model_paths(n,times));aug=sw.modelset(model_paths(n,times,0))
 bp=proba(base,x,times);ap=proba(aug,x,times)
 ref=json.loads((SCALE/'records'/f'{n}_draw0_clean.json').read_text())
 ck=torch.load(DEEP/'models'/f'{n}_draw0_TCN_aug.pt',map_location='cuda',weights_only=True)
 model=AnytimeTCN(len(classes),'C',len(times)).cuda();model.load_state_dict(ck['state_dict']);model.eval()
 p=predict_deep(model,x,times)
 with np.load(DEEP/'predictions'/f'{n}_draw0_TCN_aug_clean.npz') as z:
  assert np.array_equal(p.argmax(2),z['pred'][:16])
  assert np.max(abs(p.max(2)-z['confidence'][:16]))<1e-6
 teaser=load_teaser(n);all_decisions(teaser,bp,times,[2,3])
 assert np.isfinite(bp).all() and np.isfinite(ap).all()
 result['cached_model_replay']='FordA TCN clean predictions reproduced; cached clean/aug MiniRocket and TEASER load and infer'
 result['choice_count']=len(choices);result['gpu_name']=torch.cuda.get_device_name(0)
 with (OUT/'config.json').open('x') as f:json.dump(CONFIG,f,indent=2)
 with (OUT/'preflight.json').open('x') as f:json.dump(result,f,indent=2)
 print('PREFLIGHT_PASS',json.dumps({k:v for k,v in result.items() if k!='model_inventory'}),flush=True)
if __name__=='__main__':main()
