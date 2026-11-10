"""Full-scale paired TCNs. Reuse pilot C/lambda=0 checkpoints; never fit MiniRocket."""
import os,sys,time,json,hashlib,argparse,traceback,shutil,gc
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
from settings import *
from deep_model import AnytimeTCN,MaskAnytimeTCN,objective
from masks import augmented_with_mask,mixed_with_mask,stream_with_mask,causal_forward_fill
from ects.anytime import normalize
from ects.rocket_v2 import calibrate_sequential

def save(path,obj):
    with path.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)
def seed_for(name,draw):
    return int.from_bytes(hashlib.sha256(f'deep-pilot-v1:{name}:{draw}'.encode()).digest()[:4],'big')
def harray(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def to_gpu(x):return torch.as_tensor(np.ascontiguousarray(x),device='cuda')
def model_for(classes,tag,k):
    return (MaskAnytimeTCN(classes,'C',k) if tag=='TCN_aug_mask' else AnytimeTCN(classes,'C',k))
def pack(values,mask,masked):
    return np.stack([values,mask.astype(np.float32)],axis=1) if masked else values
def metric(y,p,t,L):
    a=float(accuracy_score(y,p));e=float(np.mean(np.asarray(t)/L))
    return {'accuracy':a,'error':1-a,'balanced_accuracy':float(balanced_accuracy_score(y,p)),
            'macro_f1':float(f1_score(y,p,average='macro',zero_division=0)),'earliness':e,
            'combined':2*a*(1-e)/(a+1-e) if a+1-e else 0.}
def compact(p):return p.argmax(2).astype(np.uint16),p.max(2).astype(np.float32)
def decide(pred,conf,times,threshold):
    ok=conf>=threshold;forced=~ok.any(1);ok[:,-1]=True;j=ok.argmax(1)
    return pred[np.arange(len(pred)),j],np.asarray(times)[j],forced
def probs(model,X,times,batch):
    model.eval();res=[]
    with torch.inference_mode():
        for i in range(0,len(X),batch):res.append(model(X[i:i+batch],times).float().softmax(-1).cpu().numpy())
    return np.concatenate(res)
def validation(model,X,y,times,batch):
    model.eval();loss=hits=n=0
    with torch.inference_mode():
        for i in range(0,len(X),batch):
            logits=model(X[i:i+batch],times).float();k=logits.shape[1]
            loss+=float(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y[i:i+batch,None].expand(-1,k).reshape(-1),reduction='sum'))
            hits+=int((logits.argmax(-1)==y[i:i+batch,None]).sum());n+=len(logits)*k
    return loss/n,hits/n
def causal_audit(model,raw,times,masked):
    model.eval();raw=raw[:2].copy()
    mask=np.random.default_rng(17).random(raw.shape)>.35 if masked else np.ones(raw.shape,bool)
    filled=causal_forward_fill(raw,mask);value=normalize(filled,'causal');x=pack(value,mask,masked);a=to_gpu(x)
    maximum=normdiff=0.
    with torch.no_grad():
        full=model(a,times)
        for j,t in enumerate(times[:-1]):
            changed=raw.copy();changed[:,t:]=1000+np.arange(raw.shape[1]-t)
            changed_mask=mask.copy()
            if masked:changed_mask[:,t:]=np.arange(raw.shape[1]-t)%2==0
            cz=normalize(causal_forward_fill(changed,changed_mask),'causal')
            nx=pack(cz,changed_mask,masked)
            normdiff=max(normdiff,float(np.max(abs(nx[...,:t]-x[...,:t]))))
            altered=model(to_gpu(nx),times);truncated=model(a[...,:t],times[:j+1])
            maximum=max(maximum,float((altered[:,j]-full[:,j]).abs().max()),float((truncated[:,j]-full[:,j]).abs().max()))
    g=a.clone().requires_grad_(True);j=len(times)//2
    grad=torch.autograd.grad(model(g,times)[:,j].sum(),g)[0]
    future=float(grad[...,int(times[j]):].abs().max())
    if maximum>2e-4 or normdiff!=0 or future!=0:raise AssertionError('Causality failed')
    return {'passed':True,'max_prefix_logit_difference':maximum,'normalization_and_mask_prefix_difference':normdiff,
            'future_value_and_mask_gradient_max':future,'masked':masked}

def train(name,draw,tag,X,y,V,vy,times,classes,raw,seed,batch):
    path=OUT/'models'/f'{name}_draw{draw}_{tag}.pt';mp=OUT/'metadata'/f'{name}_draw{draw}_{tag}.json'
    if path.exists() and mp.exists():
        model=model_for(len(classes),tag,len(times)).cuda()
        model.load_state_dict(torch.load(path,map_location='cuda',weights_only=True)['state_dict'])
        return model,json.loads(mp.read_text())
    pilot=PILOT/'models'/f'{name}_draw{draw}_C_l0.pt'
    if tag=='TCN_aug' and pilot.exists():
        pm=json.loads((PILOT/'metadata'/f'{name}_draw{draw}_C_l0.json').read_text())
        assert pm['mode']=='C' and pm['lambda']==0. and pm['augmentation'] and pm['seed']==seed
        model=model_for(len(classes),tag,len(times)).cuda();saved=torch.load(pilot,map_location='cuda',weights_only=True)
        assert saved['classes']==classes.tolist() and saved['times']==times.tolist()
        model.load_state_dict(saved['state_dict']);shutil.copy2(pilot,path)
        meta={**pm,'variant':tag,'reused_from':str(pilot),'model_sha256':digest(pilot),
              'original_train_seconds':pm['train_seconds'],'train_seconds':0.,'gpu_hours':0.,
              'evaluation_gpu':os.environ['CUDA_VISIBLE_DEVICES'],'causality_audit':causal_audit(model,raw,times,False)}
        save(mp,meta);print('REUSED_PILOT',name,draw,flush=True);return model,meta
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed)
    model=model_for(len(classes),tag,len(times)).cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
    scaler=torch.amp.GradScaler('cuda');generator=torch.Generator(device='cpu').manual_seed(seed)
    staged=len(times)*CONFIG['curriculum_epochs_per_checkpoint'];total=staged+CONFIG['max_joint_epochs']
    best=float('inf');best_state=None;best_epoch=None;bad=0;history=[];start=time.monotonic()
    for epoch in range(total):
        k=min(len(times),epoch//CONFIG['curriculum_epochs_per_checkpoint']+1)
        model.train();order=torch.randperm(len(X),generator=generator);summed=0.;seen=0;before=time.monotonic()
        for ids in order.split(batch):
            ix=ids.cuda();optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                logits=model(X[ix,...,:int(times[k-1])],times[:k]);loss,ce,cons=objective(logits,y[ix],0.)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),CONFIG['gradient_clip'])
            scaler.step(optimizer);scaler.update();summed+=float(loss)*len(ix);seen+=len(ix)
        vl,va=validation(model,V,vy,times[:k],batch)
        row={'epoch':epoch+1,'active_checkpoints':k,'train_ce':summed/seen,'val_ce':vl,'val_accuracy':va,'seconds':time.monotonic()-before}
        history.append(row)
        if k==len(times):
            if vl<best-1e-5:
                best=vl;best_epoch=epoch+1;bad=0
                best_state={key:v.detach().cpu().clone() for key,v in model.state_dict().items()}
            else:bad+=1
        print('EPOCH',name,draw,tag,epoch+1,'k',k,'train',round(row['train_ce'],4),'val',round(vl,4),'acc',round(va,4),'sec',round(row['seconds'],2),flush=True)
        if epoch>=staged+5 and bad>=CONFIG['joint_patience']:break
    torch.cuda.synchronize();seconds=time.monotonic()-start
    if best_state is None:raise RuntimeError('No validated all-checkpoint model')
    model.load_state_dict(best_state);vl,va=validation(model,V,vy,times,batch)
    meta={'dataset':name,'draw':draw,'variant':tag,'mode':'C','lambda':0.,'augmentation':True,
          'epochs':len(history),'best_epoch':best_epoch,'best_val_ce':vl,'best_val_accuracy':va,
          'train_seconds':seconds,'gpu_hours':seconds/3600,'gpu':os.environ['CUDA_VISIBLE_DEVICES'],
          'evaluation_gpu':os.environ['CUDA_VISIBLE_DEVICES'],'gpu_name':torch.cuda.get_device_name(0),
          'seed':seed,'batch_size':batch,'parameters':sum(p.numel() for p in model.parameters()),
          'history':history,'reused_from':None,'causality_audit':causal_audit(model,raw,times,tag=='TCN_aug_mask')}
    with path.open('xb') as f:torch.save({'state_dict':best_state,'classes':classes.tolist(),'times':times.tolist(),'variant':tag,'config':CONFIG},f)
    save(mp,meta);return model,meta

def run(name,draw):
    start=time.monotonic();meta=json.loads((SCALE/'draw_meta'/f'{name}_base.json').read_text())
    raw,y=sw.load_data(name,'train');test,z=sw.load_data(name,'test')
    fit=np.array(meta['fit_indices']);val=np.array(meta['validation_indices']);cal=np.array(meta['calibration_indices'])
    assert not(set(fit)&set(val) or set(fit)&set(cal) or set(val)&set(cal))
    times=np.array(meta['checkpoint_lengths']);classes=np.array(meta['classes']);power=meta['fit_signal_power']
    enc=lambda y:np.searchsorted(classes,y).astype(np.int64)
    seed=seed_for(name,draw);batch=min(128,max(16,131072//raw.shape[1]))
    af,ay,am,info=augmented_with_mask(raw[fit],y[fit],name,draw,power);af=normalize(af,'causal')
    vm,vmm,vinfo=mixed_with_mask(raw[val],name+'::validation',draw,power)
    v=normalize(np.concatenate([raw[val],vm]),'causal')
    vmask=np.concatenate([np.ones_like(vmm),vmm]);vy=np.tile(enc(y[val]),2)
    cm,cmm,cinfo=mixed_with_mask(raw[cal],name,draw,power);cm=normalize(cm,'causal');caly=enc(y[cal])
    prior=json.loads((SCALE/'draw_meta'/f'{name}_draw{draw}.json').read_text())
    assert info==prior['augmentation']
    split={'dataset':name,'draw':draw,'source_metadata':str(SCALE/'draw_meta'/f'{name}_base.json'),
           'fit_size':len(fit),'validation_size':len(val),'calibration_size':len(cal),'train_size':len(raw),'test_size':len(test),
           'length':raw.shape[1],'classes':classes.tolist(),'times':times.tolist(),'augmentation':info,
           'mixed_calibration_counts':cinfo,'validation_mixed_counts':vinfo,
           'augmented_fit_sha256':harray(af),'validation_sha256':harray(v),'calibration_sha256':harray(cm),
           'augmented_mask_sha256':harray(am),'validation_mask_sha256':harray(vmask),'calibration_mask_sha256':harray(cmm)}
    pmeta=PILOT/'metadata'/f'{name}_draw{draw}_splits.json'
    if pmeta.exists():
        old=json.loads(pmeta.read_text())
        for key in ['augmented_fit_sha256','validation_sha256','calibration_sha256']:
            assert split[key]==old[key], 'Pilot data reproduction failed '+key
        split['pilot_inputs_match']=True
    else:split['pilot_inputs_match']=None
    sp=OUT/'metadata'/f'{name}_draw{draw}_splits.json'
    if not sp.exists():save(sp,split)
    for tag in VARIANTS:
        recpath=OUT/'records'/f'{name}_draw{draw}_{tag}.json'
        if recpath.exists():print('REUSE_COMPLETE',name,draw,tag,flush=True);continue
        masked=tag=='TCN_aug_mask'
        X=to_gpu(pack(af,am,masked));Y=to_gpu(enc(ay));V=to_gpu(pack(v,vmask,masked));VY=to_gpu(vy)
        model,training=train(name,draw,tag,X,Y,V,VY,times,classes,raw[fit],seed,batch)
        del X,Y,V,VY
        pp=PILOT/'predictions'/f'{name}_draw{draw}_C_l0_calval.npz'
        if training['reused_from'] and pp.exists():
            with np.load(pp) as q:
                cp=q['cal_prob'].copy();vp=q['val_prob'].copy()
                assert np.array_equal(q['cal_y'],caly) and np.array_equal(q['times'],times)
        else:
            cp=probs(model,to_gpu(pack(cm,cmm,masked)),times,batch)
            vp=probs(model,to_gpu(pack(v,vmask,masked)),times,batch)
        calpred,calconf=compact(cp);thresholds={};calrows=[]
        for alpha in ALPHAS:
            th,support,errors,at_risk=calibrate_sequential(cp,caly,np.arange(len(classes)),alpha)
            thresholds[str(alpha)]=th;p,t,forced=decide(calpred,calconf,times,th)
            calrows.append({'kind':'mixed','setting':alpha,'earliness':float(np.mean(t/times[-1])),
                            'thresholds':[float(t) if np.isfinite(t) else None for t in th],
                            'support':support.tolist(),'remaining':at_risk.tolist()})
        for tau in TAUS:
            p,t,forced=decide(calpred,calconf,times,tau)
            calrows.append({'kind':'probability','setting':tau,'earliness':float(np.mean(t/times[-1]))})
        cvp=OUT/'predictions'/f'{name}_draw{draw}_{tag}_calval.npz'
        if not cvp.exists():
            with cvp.open('xb') as f:np.savez_compressed(f,cal_prob=cp,cal_y=caly,val_prob=vp,val_y=vy,times=times)
        rows=[];stream_hashes={};mask_hashes={};reused_predictions=[]
        for cond in CONDITIONS:
            values,mask=stream_with_mask(test,name,draw,'test',cond,power);values=normalize(values,'causal')
            stream_hashes[cond]=harray(values);mask_hashes[cond]=harray(mask)
            dest=OUT/'predictions'/f'{name}_draw{draw}_{tag}_{cond}.npz'
            oldp=PILOT/'predictions'/f'{name}_draw{draw}_C_l0_{cond}.npz'
            if training['reused_from'] and oldp.exists():
                oldr=json.loads((PILOT/'records'/f'{name}_draw{draw}_C_l0.json').read_text())
                assert stream_hashes[cond]==oldr['stream_hashes'][cond]
                with np.load(oldp) as q:
                    pred=q['pred'].copy();conf=q['confidence'].copy();assert np.array_equal(q['y'],enc(z))
                if not dest.exists():shutil.copy2(oldp,dest)
                reused_predictions.append(cond)
            elif dest.exists():
                with np.load(dest) as q:pred=q['pred'].copy();conf=q['confidence'].copy()
            else:
                pred,conf=compact(probs(model,to_gpu(pack(values,mask,masked)),times,batch))
                with dest.open('xb') as f:np.savez_compressed(f,pred=pred,confidence=conf,y=enc(z),times=times)
            for j,t in enumerate(times):
                rows.append({'kind':'fixed','condition':cond,'setting':j,'nominal_fraction':(j+2)/10,
                             **metric(enc(z),pred[:,j],np.full(len(z),t),times[-1])})
            for alpha in ALPHAS:
                p,t,forced=decide(pred,conf,times,thresholds[str(alpha)])
                rows.append({'kind':'mixed','condition':cond,'setting':alpha,'forced_fraction':float(forced.mean()),
                             **metric(enc(z),p,t,times[-1])})
            for tau in TAUS:
                p,t,forced=decide(pred,conf,times,tau)
                rows.append({'kind':'probability','condition':cond,'setting':tau,'forced_fraction':float(forced.mean()),
                             **metric(enc(z),p,t,times[-1])})
        save(recpath,{'dataset':name,'draw':draw,'variant':tag,'training':training,
          'calibration_frontiers':calrows,'rows':rows,'stream_hashes':stream_hashes,'mask_hashes':mask_hashes,
          'reused_pilot_prediction_conditions':reused_predictions})
        print('MODEL_DONE',name,draw,tag,'train_seconds',round(training['train_seconds'],1),'reused',bool(training['reused_from']),flush=True)
        del model;torch.cuda.empty_cache();gc.collect()
    print('JOB_DONE',name,draw,'seconds',round(time.monotonic()-start,1),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--jobs',required=True);a.add_argument('--worker',required=True);args=a.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    jobs=json.loads(Path(args.jobs).read_text());done=[];errors=[]
    for name,draw in jobs:
        try:
            start=time.monotonic();run(name,int(draw));done.append({'dataset':name,'draw':draw,'wall_seconds':time.monotonic()-start})
        except Exception:
            err=traceback.format_exc();errors.append({'dataset':name,'draw':draw,'error':err});print('JOB_ERROR',name,draw,err,flush=True)
    save(OUT/'logs'/f'worker_{args.worker}_status.json',{'worker':args.worker,'jobs':done,'errors':errors})
    print('WORKER_DONE',args.worker,'jobs',len(done),'errors',len(errors),flush=True)
