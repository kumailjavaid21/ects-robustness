"""Post-training mask-use check; no fitting or parameter changes."""
import sys,json
import numpy as np,torch
from settings import *
from deep_model import MaskAnytimeTCN
from worker import pack,probs,to_gpu
from masks import stream_with_mask
from ects.anytime import normalize
rows=[]
for p in sorted((OUT/'models').glob('*_TCN_aug_mask.pt')):
 state=torch.load(p,map_location='cpu',weights_only=True)['state_dict']
 w=state['input.weight'];assert w.shape==(64,2,1)
 rows.append({'model':p.name,'mask_weight_l1':float(w[:,1].abs().sum()),'value_weight_l1':float(w[:,0].abs().sum())})
assert len(rows)==130 and all(r['mask_weight_l1']>1e-8 for r in rows)
name='FordA';draw=0;cond='missing_50pct'
meta=json.loads((SCALE/'draw_meta'/f'{name}_base.json').read_text())
raw,y=sw.load_data(name,'test');times=np.array(meta['checkpoint_lengths'])
v,m=stream_with_mask(raw,name,draw,'test',cond,meta['fit_signal_power']);v=normalize(v,'causal')
model=MaskAnytimeTCN(len(meta['classes']),'C',len(times)).cuda()
state=torch.load(OUT/'models'/f'{name}_draw{draw}_TCN_aug_mask.pt',map_location='cuda',weights_only=True)
model.load_state_dict(state['state_dict']);model.eval();torch.set_num_threads(2)
actual=probs(model,to_gpu(pack(v[:32],m[:32],True)),times,128)
all_observed=probs(model,to_gpu(pack(v[:32],np.ones_like(m[:32]),True)),times,128)
with np.load(OUT/'predictions'/f'{name}_draw{draw}_TCN_aug_mask_{cond}.npz') as z:
 assert np.array_equal(actual.argmax(2),z['pred'][:32])
 assert np.allclose(actual.max(2),z['confidence'][:32],atol=2e-6,rtol=1e-5)
delta=abs(actual-all_observed)
result={'mask_models_checked':len(rows),'all_mask_weights_nonzero':True,'weights':rows,
 'probe':{'dataset':name,'draw':draw,'condition':cond,'n':32,'saved_predictions_reproduced':True,
 'mean_probability_change_if_flags_forced_to_one':float(delta.mean()),
 'max_probability_change_if_flags_forced_to_one':float(delta.max()),
 'changed_checkpoint_predictions':int((actual.argmax(2)!=all_observed.argmax(2)).sum())},
 'training_performed':False}
assert delta.max()>1e-6
with (OUT/'mask_usage_audit.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps({k:v for k,v in result.items() if k!='weights'},indent=2))
