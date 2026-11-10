"""One GPU, one dataset/draw; all 7 paired head/consistency ablations."""
import os,sys,time,json,hashlib,argparse,traceback,copy
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
from settings import *
sys.path.insert(0,str(SCALE));import scale_worker as sw
sys.path.insert(0,str(ROOT/'src'))
from ects.anytime import normalize
from ects.robustness_v2 import corrupt,train_augmentation,mixed_calibration
from ects.rocket_v2 import decisions
from deep_model import AnytimeTCN,objective

def save(path,obj):
    with path.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)
def seed_for(name,draw):
    return int.from_bytes(hashlib.sha256(f'deep-pilot-v1:{name}:{draw}'.encode()).digest()[:4],'big')
def harray(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def to_gpu(x):return torch.as_tensor(np.ascontiguousarray(x),device='cuda')
def metric(y,p,t,L):
    a=float(accuracy_score(y,p));e=float(np.mean(np.asarray(t)/L))
    return {'accuracy':a,'error':1-a,'balanced_accuracy':float(balanced_accuracy_score(y,p)),
      'macro_f1':float(f1_score(y,p,average='macro',zero_division=0)),
      'earliness':e,'combined':2*a*(1-e)/(a+1-e) if a+1-e else 0.}
def compact_prob(p):return p.argmax(2).astype(np.uint16),p.max(2).astype(np.float32)
def compact_decision(pred,confidence,times,tau):
    ok=confidence>=tau;forced=~ok.any(1);ok[:,-1]=True;j=ok.argmax(1);ii=np.arange(len(j))
    return pred[ii,j],np.asarray(times)[j],forced
def batch_probs(model,X,times,batch):
    model.eval();result=[]
    with torch.inference_mode():
        for i in range(0,len(X),batch):
            result.append(model(X[i:i+batch],times).float().softmax(-1).cpu().numpy())
    return np.concatenate(result)
def validation(model,X,y,times,batch):
    model.eval();loss=0.;hits=0.;n=0
    with torch.inference_mode():
        for i in range(0,len(X),batch):
            logits=model(X[i:i+batch],times).float()
            k=logits.shape[1];ce=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),
                y[i:i+batch,None].expand(-1,k).reshape(-1),reduction='sum')
            loss+=float(ce);hits+=int((logits.argmax(-1)==y[i:i+batch,None]).sum());n+=len(logits)*k
    return loss/n,hits/n

def causal_audit(model,raw,times):
    model.eval();x=normalize(raw[:2],'causal');a=to_gpu(x)
    with torch.no_grad():
        full=model(a,times);maximum=0.;normdiff=0.
        for j,t in enumerate(times[:-1]):
            changed=raw[:2].copy();changed[:,t:]=1000+np.arange(changed.shape[1]-t)
            cz=normalize(changed,'causal')
            normdiff=max(normdiff,float(np.max(abs(cz[:,:t]-x[:,:t]))))
            mutated=model(to_gpu(cz),times)
            truncated=model(a[:,:t],times[:j+1])
            maximum=max(maximum,float((mutated[:,j]-full[:,j]).abs().max()),
                        float((truncated[:,j]-full[:,j]).abs().max()))
    g=a.clone().requires_grad_(True);j=len(times)//2
    gradient=torch.autograd.grad(model(g,times)[:,j].sum(),g)[0]
    future_gradient=float(gradient[:,int(times[j]):].abs().max())
    if maximum>2e-4 or normdiff!=0 or future_gradient!=0:raise AssertionError('Causality failure')
    return {'max_prefix_logit_difference':maximum,'normalization_prefix_difference':normdiff,
            'future_input_gradient_max':future_gradient,'passed':True}

def fit_variant(name,draw,tag,mode,lam,aug,X,y,V,vy,times,classes,raw,seed,batch):
    path=OUT/'models'/f'{name}_draw{draw}_{tag}.pt'
    meta_path=OUT/'metadata'/f'{name}_draw{draw}_{tag}.json'
    if path.exists() and meta_path.exists():
        model=AnytimeTCN(len(classes),mode,len(times)).cuda()
        model.load_state_dict(torch.load(path,map_location='cuda',weights_only=True)['state_dict'])
        return model,json.loads(meta_path.read_text())
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed)
    model=AnytimeTCN(len(classes),mode,len(times)).cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
    scaler=torch.amp.GradScaler('cuda')
    generator=torch.Generator(device='cpu').manual_seed(seed)
    staged=len(times)*CONFIG['curriculum_epochs_per_checkpoint']
    total=staged+CONFIG['max_joint_epochs'];best=float('inf');best_state=None;best_epoch=None;bad=0
    rows=[];inheritance=[];last_k=0;start=time.monotonic()
    for epoch in range(total):
        k=min(len(times),epoch//CONFIG['curriculum_epochs_per_checkpoint']+1)
        if k!=last_k and mode=='B' and k>1:
            before={key:v.detach().clone() for key,v in model.heads[k-2].state_dict().items()}
            model.inherit(k-1)
            exact=all(torch.equal(v,model.heads[k-1].state_dict()[key]) for key,v in before.items())
            if not exact:raise AssertionError('Warm-start copy mismatch')
            inheritance.append({'checkpoint':k-1,'epoch':epoch,'copied_from':k-2,'exact':exact})
        last_k=k;model.train();order=torch.randperm(len(X),generator=generator)
        sums=np.zeros(3);seen=0;before=time.monotonic()
        for ids in order.split(batch):
            ix=ids.cuda();optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                # The active curriculum truncates the batch at the largest active checkpoint.
                logits=model(X[ix,:int(times[k-1])],times[:k])
                loss,ce,consistency=objective(logits,y[ix],lam)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),CONFIG['gradient_clip'])
            scaler.step(optimizer);scaler.update()
            sums+=np.array([float(loss),float(ce),float(consistency)])*len(ix);seen+=len(ix)
        vl,va=validation(model,V,vy,times[:k],batch)
        row={'epoch':epoch+1,'active_checkpoints':k,'train_loss':float(sums[0]/seen),
             'train_ce':float(sums[1]/seen),'train_consistency':float(sums[2]/seen),
             'val_ce':vl,'val_accuracy':va,'seconds':time.monotonic()-before}
        rows.append(row)
        # All variants get the full curriculum. Early stopping only compares all-head validation losses.
        if k==len(times):
            if vl<best-1e-5:
                best=vl;best_epoch=epoch+1;bad=0
                best_state={key:v.detach().cpu().clone() for key,v in model.state_dict().items()}
            else:bad+=1
        print('EPOCH',name,draw,tag,epoch+1,'k',k,'train',round(row['train_loss'],4),
              'val',round(vl,4),'acc',round(va,4),'sec',round(row['seconds'],2),flush=True)
        if epoch>=staged+5 and bad>=CONFIG['joint_patience']:break
    torch.cuda.synchronize();train_seconds=time.monotonic()-start
    if best_state is None:raise RuntimeError('No full-checkpoint validation state')
    model.load_state_dict(best_state);vl,va=validation(model,V,vy,times,batch)
    audit=causal_audit(model,raw,times)
    meta={'dataset':name,'draw':draw,'variant':tag,'mode':mode,'lambda':lam,'augmentation':aug,
      'epochs':len(rows),'best_epoch':best_epoch,'best_val_ce':vl,'best_val_accuracy':va,
      'train_seconds':train_seconds,'gpu_hours':train_seconds/3600,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
      'gpu_name':torch.cuda.get_device_name(0),'seed':seed,'batch_size':batch,
      'parameters':sum(p.numel() for p in model.parameters()),'history':rows,
      'inheritance_events':inheritance,'causality_audit':audit}
    with path.open('xb') as f:torch.save({'state_dict':best_state,'classes':classes.tolist(),'times':list(map(int,times)),
                                         'variant':tag,'config':CONFIG},f)
    save(meta_path,meta);return model,meta

def run(name,draw):
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    tic=time.monotonic();meta=json.loads((SCALE/'draw_meta'/f'{name}_base.json').read_text())
    raw,y=sw.load_data(name,'train');test,z=sw.load_data(name,'test')
    fit=np.array(meta['fit_indices']);val=np.array(meta['validation_indices']);cal=np.array(meta['calibration_indices'])
    if set(fit)&set(val) or set(fit)&set(cal) or set(val)&set(cal):raise AssertionError('Split overlap')
    times=np.array(meta['checkpoint_lengths']);classes=np.array(meta['classes']);power=meta['fit_signal_power']
    enc=lambda a:np.searchsorted(classes,a).astype(np.int64)
    seed=seed_for(name,draw);batch=min(128,max(16,131072//raw.shape[1]))
    cleanfit=normalize(raw[fit],'causal')
    af,ay,aug_info=train_augmentation(raw[fit],y[fit],name,draw,power);af=normalize(af,'causal')
    vm,vmix=mixed_calibration(raw[val],name+'::validation',draw,power)
    # Identical, corruption-agnostic validation mixture for ALL variants, including the clean reference.
    v=normalize(np.concatenate([raw[val],vm]),'causal');vy=np.tile(enc(y[val]),2)
    cm,cmix=mixed_calibration(raw[cal],name,draw,power);cm=normalize(cm,'causal')
    af_gpu=to_gpu(af);ay_gpu=to_gpu(enc(ay));clean_gpu=to_gpu(cleanfit);cy_gpu=to_gpu(enc(y[fit]))
    v_gpu=to_gpu(v);vy_gpu=to_gpu(vy);cm_gpu=to_gpu(cm)
    split_info={'dataset':name,'draw':draw,'base_metadata':str(SCALE/'draw_meta'/f'{name}_base.json'),
      'augmentation':aug_info,'mixed_calibration_counts':cmix,'validation_mixed_counts':vmix,
      'augmented_fit_sha256':harray(af),'validation_sha256':harray(v),'calibration_sha256':harray(cm),
      'fit_size':len(fit),'validation_size':len(val),'calibration_size':len(cal),'train_size':len(raw),
      'test_size':len(test),'length':raw.shape[1],'classes':classes.tolist(),'times':times.tolist()}
    spath=OUT/'metadata'/f'{name}_draw{draw}_splits.json'
    if not spath.exists():save(spath,split_info)
    for tag,mode,lam,is_aug in VARIANTS:
        record=OUT/'records'/f'{name}_draw{draw}_{tag}.json'
        if record.exists():print('REUSE',name,draw,tag,flush=True);continue
        model,training=fit_variant(name,draw,tag,mode,lam,is_aug,af_gpu if is_aug else clean_gpu,
              ay_gpu if is_aug else cy_gpu,v_gpu,vy_gpu,times,classes,raw[fit],seed,batch)
        probs=batch_probs(model,cm_gpu,times,batch);cp,cc=compact_prob(probs)
        vp=batch_probs(model,v_gpu,times,batch)
        calrows=[]
        for tau in TAUS:
            pred,t,f=compact_decision(cp,cc,times,tau)
            calrows.append({'tau':tau,'earliness':float(np.mean(t/times[-1]))})
        chosen={str(target):min(calrows,key=lambda a:(abs(a['earliness']-target),a['tau'])) for target in TARGETS}
        rows=[];stream_hashes={};causal=training['causality_audit']
        for cond in CONDITIONS:
            stream=normalize(corrupt(test,name,draw,'test',cond,power),'causal')
            stream_hashes[cond]=harray(stream)
            probs=batch_probs(model,to_gpu(stream),times,batch);pred,conf=compact_prob(probs)
            with (OUT/'predictions'/f'{name}_draw{draw}_{tag}_{cond}.npz').open('xb') as file:
                np.savez_compressed(file,pred=pred,confidence=conf,y=enc(z),times=times)
            for j,t in enumerate(times):
                rows.append({'kind':'fixed','condition':cond,'setting':int(j),'nominal_fraction':(j+2)/10,
                             **metric(enc(z),pred[:,j],np.full(len(z),t),times[-1])})
            for tau in TAUS:
                p,t,forced=compact_decision(pred,conf,times,tau)
                rows.append({'kind':'frontier','condition':cond,'setting':tau,
                    'forced_fraction':float(forced.mean()),**metric(enc(z),p,t,times[-1])})
            for target in TARGETS:
                ch=chosen[str(target)];p,t,forced=compact_decision(pred,conf,times,ch['tau'])
                rows.append({'kind':'calibration_target','condition':cond,'setting':target,'selected_tau':ch['tau'],
                    'calibration_earliness':ch['earliness'],'forced_fraction':float(forced.mean()),
                    **metric(enc(z),p,t,times[-1])})
        with (OUT/'predictions'/f'{name}_draw{draw}_{tag}_calval.npz').open('xb') as file:
            np.savez_compressed(file,cal_prob=batch_probs(model,cm_gpu,times,batch),cal_y=enc(y[cal]),
                               val_prob=vp,val_y=vy,times=times)
        save(record,{'dataset':name,'draw':draw,'variant':tag,'training':training,'calibration_frontier':calrows,
                     'selected':chosen,'stream_hashes':stream_hashes,'rows':rows})
        print('MODEL_DONE',name,draw,tag,'best_val',round(training['best_val_accuracy'],4),
              'train_sec',round(training['train_seconds'],1),flush=True)
        del model;torch.cuda.empty_cache()
    print('JOB_DONE',name,draw,'seconds',round(time.monotonic()-tic,1),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--draw',type=int,required=True)
    a=p.parse_args()
    try:run(a.dataset,a.draw)
    except Exception:
        e=traceback.format_exc();print(e,flush=True)
        path=OUT/'logs'/f'{a.dataset}_draw{a.draw}_error_{int(time.time())}.json'
        save(path,{'dataset':a.dataset,'draw':a.draw,'error':e});sys.exit(1)
