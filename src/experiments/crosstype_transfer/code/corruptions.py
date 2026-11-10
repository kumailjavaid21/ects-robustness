"""Frozen unseen corruptions. No labels or future signal values enter a transformation."""
import hashlib
import numpy as np
def seed(name,draw,part,kind,i):
 return int.from_bytes(hashlib.sha256(f'cross-type-v1:{name}:{draw}:{part}:{kind}:{i}'.encode()).digest()[:8],'big')
def forward_fill(x,mask):
 index=np.maximum.accumulate(np.where(mask,np.arange(x.shape[1])[None,:],-1),axis=1)
 return np.where(index>=0,np.take_along_axis(x,np.maximum(index,0),axis=1),0).astype(np.float32)
def corrupt(x,name,draw,partition,condition,horizon=None):
 x=np.asarray(x,np.float32);n,L=x.shape;H=int(horizon or L)
 if L>H:raise ValueError('Observed prefix exceeds declared horizon')
 kind=condition.split('_')[0];out=x.copy();time=np.arange(L,dtype=np.float64)
 masks=np.ones((n,L),bool)
 if kind=='quantization':
  levels=int(condition.split('_')[1]);step=6/(levels-1)
  out=(-3+step*np.rint((np.clip(x,-3,3)+3)/step)).astype(np.float32)
 elif kind in ('spike','drift','warp','burst'):
  for i in range(n):
   rng=np.random.default_rng(seed(name,draw,partition,kind,i))
   if kind=='spike':
    fraction=.05 if condition=='spike_005' else .15
    # Separate generators preserve prefixes if the function is called on shorter input.
    flags=rng.random(L)<fraction
    signs=np.random.default_rng(seed(name,draw,partition,'spike-sign',i)).integers(0,2,L)*2-1
    out[i]+=flags*signs*5
   elif kind=='drift':
    amplitude=.5 if condition=='drift_small' else 1.5
    sign=1 if rng.integers(0,2) else -1
    out[i]+=sign*amplitude*time/max(H-1,1)
   elif kind=='warp':
    amplitude=.4 if condition=='warp_moderate' else .8
    freq=rng.uniform(.5,1.5);phase=rng.uniform(0,2*np.pi)
    speed=1+amplitude*np.sin(2*np.pi*freq*time/max(H-1,1)+phase)
    source=np.cumsum(np.r_[0.,speed[1:]])
    source=np.minimum(time,source-np.maximum.accumulate(source-time))
    source=np.maximum(source,0.)
    lo=np.floor(source).astype(int);hi=np.minimum(np.ceil(source).astype(int),np.arange(L))
    w=source-lo;out[i]=(1-w)*x[i,lo]+w*x[i,hi]
   else:
    count=int(round(.3*H));runs=3 if condition=='burst_3x10' else 1
    sizes=np.full(runs,count//runs,int);sizes[:count%runs]+=1
    gaps=rng.multinomial(H-count,np.full(runs+1,1/(runs+1)))
    cursor=int(gaps[0])
    for j,length in enumerate(sizes):
     masks[i,cursor:min(cursor+int(length),L)]=False
     cursor+=int(length)+int(gaps[j+1])
  if kind=='burst':out=forward_fill(x,masks)
 else:raise ValueError(condition)
 if not np.isfinite(out).all():raise ValueError('Nonfinite corrupted stream')
 return out,masks
def audit():
 conditions=('spike_005','spike_015','warp_moderate','warp_strong','quantization_8','quantization_4','drift_small','drift_large','burst_3x10','burst_1x30')
 rng=np.random.default_rng(982);x=rng.normal(size=(4,100)).astype(np.float32);checks=[]
 from ects.anytime import normalize
 for c in conditions:
  a,m=corrupt(x,'audit',0,'test',c)
  b=x.copy();b[:,40:]=1000+rng.normal(size=b[:,40:].shape)
  bx,bm=corrupt(b,'audit',0,'test',c)
  prefix,pm=corrupt(x[:,:40],'audit',0,'test',c,horizon=100)
  assert np.array_equal(a[:,:40],bx[:,:40]) and np.array_equal(m[:,:40],bm[:,:40])
  assert np.array_equal(a[:,:40],prefix) and np.array_equal(m[:,:40],pm)
  assert np.array_equal(normalize(a,'causal')[:,:40],normalize(prefix,'causal'))
  if c.startswith('burst'):assert np.all((~m).sum(1)==30)
  if c.startswith('quantization'):assert len(np.unique(a))<=int(c.split('_')[1])
  checks.append({'condition':c,'suffix_perturbation':True,'truncated_input_same_prefix':True,'normalization_causal':True})
 return {'passed':True,'conditions':checks}
