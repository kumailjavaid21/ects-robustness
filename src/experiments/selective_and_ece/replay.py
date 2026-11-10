"""Replay saved ECTS models; no fitting. Risk-bound selective stopping and ECE."""
import sys,json,time,traceback,argparse
from pathlib import Path
import numpy as np,joblib
from scipy.stats import beta,binom
ROOT=Path('/home/ubuntu/host_share/kumail_mad');OUT=ROOT/'runs/ects_select_v1';SCALE=ROOT/'runs/ects_scale_v1'
sys.path.insert(0,str(SCALE));import scale_worker as sw
sys.path.insert(0,str(ROOT/'src'))
from ects.robustness import CONDITIONS
from ects.robustness_v2 import corrupt,mixed_calibration
from ects.anytime import normalize
from ects.run_robust_v2 import proba
from ects.rocket_v2 import decisions
from fast_teaser import all_decisions
import __main__
__main__.GroupedCachedRocketClassifier=sw.GroupedCachedRocketClassifier
ALPHA=.10;DELTA=.10
BASES=(.50,.65,.75,.85,.90,.95,.98,.99,.995)
SLOPES=(0.,.10)
TAUS=sw.TAUS;GS=sw.GS
GRID=[(b,s) for b in BASES for s in SLOPES]
def save(path,obj):
    with path.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)
def clean_models(name,times):
    if name in sw.OLD:
        paths=[ROOT/'runs/ects_v2/models'/f'{name}_causal_{int(t):04d}.joblib' for t in times]
    else:paths=[SCALE/'base_models'/f'{name}_base_{int(t):04d}.joblib' for t in times]
    return sw.modelset(paths)
def aug_models(name,draw,times):
    path=ROOT/'runs/ects_robust_v2/models' if name in sw.OLD else SCALE/'models'
    return sw.modelset([path/f'{name}_aug_draw{draw}_{int(t):04d}.joblib' for t in times])
def load_teaser(name,draw=None):
    if draw is None:
        p=(ROOT/'runs/ects_frontier_v1/teaser_safety'/f'{name}.joblib' if name in sw.OLD else SCALE/'teaser_safety'/f'{name}_clean.joblib')
    else:p=SCALE/'teaser_safety'/f'{name}_aug_draw{draw}.joblib'
    return joblib.load(p)
def choose_thresholds(base,slope,times):
    return np.minimum(1.,base+slope*(1-np.asarray(times,dtype=float)/times[-1]))
def selective_decision(prob,classes,times,thresholds):
    n=len(prob);score=prob.max(2);good=score>=np.asarray(thresholds)[None,:]
    accepted=good.any(1);idx=good.argmax(1);idx[~accepted]=len(times)-1
    pred=classes[prob[np.arange(n),idx].argmax(1)];return pred,np.asarray(times)[idx],accepted,score[np.arange(n),idx]
def cp_upper(errors,n,delta):
    if n==0 or errors>=n:return 1.
    return float(beta.ppf(1-delta,errors+1,n-errors))
def select_policy(prob,y,classes,times):
    rows=[]
    for base,slope in GRID:
        th=choose_thresholds(base,slope,times);p,t,accepted,conf=selective_decision(prob,classes,times,th)
        n=int(accepted.sum());err=int(np.sum((p!=y)&accepted))
        ub=cp_upper(err,n,DELTA/len(GRID))
        pvalue=float(binom.cdf(err,n,ALPHA)) if n else 1.
        rows.append({'base':base,'slope':slope,'n':n,'errors':err,'upper':ub,'pvalue':pvalue,
          'coverage':n/len(y),'earliness':float(np.mean(t[accepted]/times[-1])) if n else None,
          'certified':bool(n and ub<=ALPHA)})
    ok=[x for x in rows if x['certified']]
    if ok:
        best=sorted(ok,key=lambda x:(-x['coverage'],x['earliness'],x['base'],x['slope']))[0]
        thresholds=choose_thresholds(best['base'],best['slope'],times)
        status='certified_on_mixed_calibration'
    else:
        best={'base':None,'slope':None,'n':0,'errors':0,'upper':None,'coverage':0.,'earliness':None}
        thresholds=np.full(len(times),np.inf);status='no_certificate_abstain_all'
    return {'status':status,'thresholds':[float(t) if np.isfinite(t) else None for t in thresholds],
            'chosen':best,'candidate_count':len(GRID),'alpha':ALPHA,'delta_familywise':DELTA,'candidates':rows}
def forced_prob(prob,classes,times,tau):
    pred,t,forced=decisions(prob,classes,times,np.full(len(times),tau))
    idx=np.searchsorted(times,t);confidence=prob[np.arange(len(prob)),idx].max(1)
    return pred,t,confidence
def threshold_for_coverage(conf,target):
    n=len(conf);k=int(round(target*n))
    if k<=0:return np.inf
    v=np.sort(conf)[::-1];ends=np.flatnonzero(np.r_[v[1:]!=v[:-1],True])+1
    best=int(ends[np.argmin(np.abs(ends-k))])
    return float(v[best-1])
def baseline_calibration(prob,classes,times,policy):
    cov=policy['chosen']['coverage'];target_e=policy['chosen']['earliness'];trials=[]
    for tau in TAUS:
        p,t,conf=forced_prob(prob,classes,times,tau);floor=threshold_for_coverage(conf,cov)
        selected=conf>=floor;actual=float(selected.mean());e=float(np.mean(t[selected]/times[-1])) if selected.any() else None
        distance_e=abs(e-target_e) if e is not None and target_e is not None else (0. if e is None and target_e is None else 2.)
        trials.append((abs(actual-cov),distance_e,float(tau),floor,actual,e))
    z=sorted(trials,key=lambda x:(x[0],x[1],x[2]))[0]
    lastconf=prob[:,-1].max(1);f=threshold_for_coverage(lastconf,cov)
    return {'tau':z[2],'floor':float(z[3]) if np.isfinite(z[3]) else None,'cal_coverage':z[4],'cal_earliness':z[5],
            'final_floor':float(f) if np.isfinite(f) else None}
def selective_metrics(y,p,t,accepted,L):
    n=int(accepted.sum());errors=int(np.sum((p!=y)&accepted))
    return {'n':len(y),'committed':n,'errors':errors,'coverage':n/len(y),
       'committed_error':errors/n if n else None,'earliness_committed':float(np.mean(t[accepted]/L)) if n else None}
def ece(prob,y,classes,bins=15):
    confidence=prob.max(1);correct=(classes[prob.argmax(1)]==y).astype(float)
    idx=np.minimum((confidence*bins).astype(int),bins-1);ece=0.
    for b in range(bins):
        mask=idx==b
        if mask.any():ece+=float(mask.mean()*abs(confidence[mask].mean()-correct[mask].mean()))
    return {'ece':ece,'signed_confidence_minus_accuracy':float(confidence.mean()-correct.mean()),
            'accuracy':float(correct.mean()),'mean_confidence':float(confidence.mean()),'bins':bins}
def cal_earliness(pb,pa,classes,times,clean_teaser,aug_teaser,info):
    n=len(pb);result={}
    result['probability']=[{'setting':x,'earliness':float(decisions(pb,classes,times,np.full(len(times),x))[1].mean()/times[-1])} for x in TAUS]
    result['aug_probability']=[{'setting':x,'earliness':float(decisions(pa,classes,times,np.full(len(times),x))[1].mean()/times[-1])} for x in TAUS]
    result['TEASER']=[{'setting':g,'earliness':float(t.mean()/times[-1])} for g,(p,t,f) in all_decisions(clean_teaser,pb,times,GS).items()]
    result['aug_TEASER']=[{'setting':g,'earliness':float(t.mean()/times[-1])} for g,(p,t,f) in all_decisions(aug_teaser,pa,times,GS).items()]
    rates=np.asarray(info['ecec_val_rates'])
    result['ECEC']=[{'setting':x,'earliness':float(sw.ecec_decide(pb,classes,times,rates,x)[1].mean()/times[-1])} for x in sw.ECEC_TAUS]
    result['ours']=[]
    for a in sw.ALPHAS:
        thresholds=np.array([np.inf if t is None else t for t in info['mixed_thresholds'][str(a)]])
        result['ours'].append({'setting':a,'earliness':float(decisions(pa,classes,times,thresholds)[1].mean()/times[-1])})
    return result
def run_dataset(name):
    tic=time.monotonic()
    if all((OUT/'records'/f'{name}_draw{d}_{c}.json').exists() for d in range(5) for c in CONDITIONS):
        print('REUSE_COMPLETE',name,flush=True);return
    meta=json.loads((SCALE/'draw_meta'/f'{name}_base.json').read_text())
    X,y=sw.load_data(name,'train');test,z=sw.load_data(name,'test')
    fit=np.asarray(meta['fit_indices']);cal=np.asarray(meta['calibration_indices']);times=np.asarray(meta['checkpoint_lengths'],int)
    classes=np.asarray(meta['classes']);power=float(meta['fit_signal_power'])
    base=clean_models(name,times);clean_teaser=load_teaser(name)
    for draw in range(5):
        cm=OUT/'draw_meta'/f'{name}_draw{draw}.json';npz=OUT/'calibration'/f'{name}_draw{draw}.npz'
        if cm.exists() and npz.exists():continue
        before=time.monotonic();aug=aug_models(name,draw,times);aug_teaser=load_teaser(name,draw)
        mixed,counts=mixed_calibration(X[cal],name,draw,power);mixed=normalize(mixed,'causal')
        pb=proba(base,mixed,times);pa=proba(aug,mixed,times);cy=y[cal]
        if name in sw.OLD:
            old=json.loads((ROOT/'runs/ects_robust_v2/records'/f'{name}_draw{draw}.json').read_text())
            with np.load(old['calibration_artifact']) as d:
                if not np.array_equal(d['cal_y'],cy) or not np.allclose(d['mixed_prob'],pa,rtol=1e-6,atol=1e-7):
                    raise ValueError('Saved mixed probabilities fail reproduction '+name+' draw '+str(draw))
        policy=select_policy(pa,cy,classes,times);basecal=baseline_calibration(pa,classes,times,policy)
        info=json.loads((SCALE/'draw_meta'/f'{name}_draw{draw}.json').read_text())
        early=cal_earliness(pb,pa,classes,times,clean_teaser,aug_teaser,info)
        if not npz.exists():
            with npz.open('xb') as f:np.savez_compressed(f,y=cy,classes=classes,times=times,p_clean=pb,p_augmented=pa)
        if not cm.exists():save(cm,{'dataset':name,'draw':draw,'mixed_counts':counts,'policy':policy,'aug_probability':basecal,'calibration_earliness':early})
        print('CAL_READY',name,draw,'coverage',policy['chosen']['coverage'],'status',policy['status'],'sec',round(time.monotonic()-before,1),flush=True)
    for draw in range(5):
        if all((OUT/'records'/f'{name}_draw{draw}_{c}.json').exists() for c in CONDITIONS):continue
        aug=aug_models(name,draw,times);cm=json.loads((OUT/'draw_meta'/f'{name}_draw{draw}.json').read_text());policy=cm['policy'];bc=cm['aug_probability']
        thresholds=np.array([np.inf if x is None else x for x in policy['thresholds']])
        floor=np.inf if bc['floor'] is None else bc['floor'];final_floor=np.inf if bc['final_floor'] is None else bc['final_floor']
        for condition in CONDITIONS:
            path=OUT/'records'/f'{name}_draw{draw}_{condition}.json'
            if path.exists():continue
            before=time.monotonic()
            data=normalize(corrupt(test,name,draw,'test',condition,power),'causal')
            pb=proba(base,data,times);pa=proba(aug,data,times)
            original=json.loads((SCALE/'records'/f'{name}_draw{draw}_{condition}.json').read_text())
            for j,q in enumerate(original['fixed']):
                for tag,p in [('clean',pb),('augmented',pa)]:
                    accuracy=float(np.mean(classes[p[:,j].argmax(1)]==z))
                    if abs(accuracy-q[tag]['accuracy'])>1e-10:raise ValueError('Test replay differs from cached '+name+' '+condition+' '+tag+' '+str(j))
            pred,t,accept,conf=selective_decision(pa,classes,times,thresholds)
            chosen=selective_metrics(z,pred,t,accept,len(test[0]))
            bp,bt,bconf=forced_prob(pa,classes,times,bc['tau']);baccept=bconf>=floor
            final_p=classes[pa[:,-1].argmax(1)];final_t=np.full(len(z),times[-1]);fconf=pa[:,-1].max(1)
            results={'selective':chosen,
                'aug_probability_calmatched':selective_metrics(z,bp,bt,baccept,len(test[0])),
                'naive_final_calmatched':selective_metrics(z,final_p,final_t,fconf>=final_floor,len(test[0])),
                'naive_final_090':selective_metrics(z,final_p,final_t,fconf>=.9,len(test[0]))}
            # Offline diagnostic only: test-batch coverage match uses no labels but is transductive.
            order=np.argsort(-bconf,kind='stable');offline=np.zeros(len(z),bool);offline[order[:chosen['committed']]]=True
            results['aug_probability_testmatched_OFFLINE']=selective_metrics(z,bp,bt,offline,len(test[0]))
            ec=[]
            for j,frac in [(2,.4),(4,.6),(8,1.)]:
                if j>=len(times):raise ValueError('Missing required checkpoint')
                for tag,p in [('clean',pb),('augmented',pa)]:
                    ec.append({'model':tag,'nominal_fraction':frac,'actual_fraction':float(times[j]/times[-1]),**ece(p[:,j],z,classes)})
            save(path,{'dataset':name,'draw':draw,'condition':condition,'results':results,'ece':ec,
               'replay_check':'all clean and augmented checkpoint accuracies match prior record',
               'seconds':time.monotonic()-before})
            print('TEST_DONE',name,draw,condition,'ours_cov',round(chosen['coverage'],3),'err',chosen['committed_error'],'sec',round(time.monotonic()-before,1),flush=True)
    print('DATASET_DONE',name,'sec',round(time.monotonic()-tic,1),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',required=True);p.add_argument('--worker',required=True);args=p.parse_args();errors=[]
    for n in args.datasets:
        try:run_dataset(n)
        except Exception:
            e=traceback.format_exc();errors.append({'dataset':n,'error':e});print('DATASET_ERROR',n,e,flush=True)
    save(OUT/'logs'/('worker_'+args.worker+'_status.json'),{'worker':args.worker,'datasets':args.datasets,'errors':errors})
    print('WORKER_DONE',args.worker,'errors',len(errors),flush=True)
if __name__=='__main__':main()
