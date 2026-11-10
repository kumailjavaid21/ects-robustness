"""Audit isolation, fixed mixed thresholds, saved predictions, and five draws."""
import hashlib,json,joblib
import numpy as np
from .run_robust_v2 import ROOT,OUT,DATASETS,METHODS,ALPHA
from .robustness import CONDITIONS
from .robustness_v2 import corrupt,train_augmentation,mixed_calibration
from .rocket_v2 import load_arrow,calibrate_sequential,decisions,metrics
from .anytime import normalize

def main():
    files=0;calibrations=0;comparisons=0;v2=json.loads((ROOT/'runs/ects_v2/results.json').read_text())
    manifest=json.loads((ROOT/'data/ucr/manifest_v2.json').read_text())
    for e in manifest['datasets']:
        if e['name'] in DATASETS:
            for f in e['files']:
                assert hashlib.sha256(__import__('pathlib').Path(f['path']).read_bytes()).hexdigest()==f['sha256'];files+=1
    # TwoLeadECG and ItalyPowerDemand were previously loaded from the original v1 manifest.
    m1=json.loads((ROOT/'data/ucr/manifest.json').read_text())
    for e in m1['datasets']:
        if e['name'] in ('TwoLeadECG','ItalyPowerDemand'):
            for f in e['files']:
                assert hashlib.sha256(__import__('pathlib').Path(f['path']).read_bytes()).hexdigest()==f['sha256'];files+=1
    for name in DATASETS:
        X,y=load_arrow(name,'train');T,z=load_arrow(name,'test');base=None;corruption_samples=[];training_samples=[]
        for draw in range(5):
            r=json.loads((OUT/'records'/f'{name}_draw{draw}.json').read_text());b=r['base']
            indices=[set(b[k+'_indices']) for k in ('fit','validation','calibration','unused')]
            assert sum(map(len,indices))==len(set.union(*indices))==len(y)
            fit=np.array(b['fit_indices']);cal=np.array(b['calibration_indices'])
            power=b['fit_signal_power'];rawcal=X[cal]
            sample=corrupt(T[:4],name,draw,'test','noise_5db',power)
            corruption_samples.append(sample)
            train,aug_y,aug_info=train_augmentation(X[fit],y[fit],name,draw,power)
            assert aug_info==r['augmentation'] and len(aug_y)==2*len(fit)
            training_samples.append(train[len(fit):len(fit)+4].copy())
            mixed,counts=mixed_calibration(rawcal,name,draw,power)
            assert counts==r['mixture_counts'] and sum(counts.values())==len(cal)
            assert max(counts.values())-min(counts.values())<=1
            with np.load(r['calibration_artifact']) as d:
                assert np.array_equal(d['cal_y'],y[cal])
                timepoint=int(d['times'][0])
                model=joblib.load(OUT/'models'/f'{name}_aug_draw{draw}_{timepoint:04d}.joblib')
                prefix=np.ascontiguousarray(normalize(mixed,'causal')[:,:timepoint,None].transpose(0,2,1),dtype=np.float32)
                recomputed=model['classifier'].predict_proba(model['transform'].transform(prefix))
                assert np.allclose(recomputed,d['mixed_prob'][:,0,:],atol=1e-6)
                real,s,e,risk=calibrate_sequential(d['mixed_prob'],d['cal_y'],d['classes'],ALPHA)
                expected=[float(x) if np.isfinite(x) else None for x in real]
                assert expected==r['realistic_calibration']['thresholds'] and s.tolist()==r['realistic_calibration']['support']
                calibrations+=1
            for condition in CONDITIONS:
                c=r['conditions'][condition]
                assert c['realistic_thresholds']==expected
                with np.load(c['artifact']) as d:
                    assert np.array_equal(d['y'],z)
                    oracle,s,e,risk=calibrate_sequential(d['oracle_cal_prob'],y[cal],d['classes'],ALPHA)
                    assert [float(x) if np.isfinite(x) else None for x in oracle]==c['oracle_calibration']['thresholds']
                    assert s.tolist()==c['oracle_calibration']['support'] and all(v<=ALPHA+1e-12 for v in e)
                    for method in METHODS:
                        pred=d[method+'_pred'];t=d[method+'_time'];forced=d[method+'_forced']
                        m=metrics(z,pred,t,int(d['times'][-1]));saved=c['results'][method]
                        for k in ('accuracy','balanced_accuracy','macro_f1','earliness','combined'):
                            assert abs(m[k]-saved[k])<1e-12,(name,draw,condition,method,k)
                        assert int(forced.sum())==saved['forced_count']
                        comparisons+=1
                    calibrations+=1
                if condition=='clean' and name in ('FordA','ECG5000','StarLightCurves','Wafer','ElectricDevices','Crop'):
                    old=next(x for x in v2['records'] if x['dataset']==name and x['mode']=='causal')
                    for method,oldmethod in [('probability','threshold'),('TEASER','TEASER')]:
                        for k in ('accuracy','balanced_accuracy','macro_f1','earliness','combined'):
                            assert abs(c['results'][method][k]-old['results'][oldmethod][k])<1e-10,(name,method,k)
            if base is None:base=b
            else:
                for key in ('fit_indices','validation_indices','calibration_indices','base_probability_threshold'):
                    assert b[key]==base[key]
        for i in range(5):
            for j in range(i):
                assert not np.array_equal(corruption_samples[i],corruption_samples[j]),(name,i,j,'test draws')
                assert not np.array_equal(training_samples[i],training_samples[j]),(name,i,j,'training augmentation')
        sample=T[:4];changed=sample.copy();changed[:,10:]+=999
        a=normalize(corrupt(sample,name,0,'test','noise_5db',power),'causal')
        b=normalize(corrupt(changed,name,0,'test','noise_5db',power),'causal')
        assert np.allclose(a[:,:10],b[:,:10])
    result={'datasets':len(DATASETS),'draws_per_dataset':5,'conditions_per_draw':len(CONDITIONS),
            'file_hashes':files,'calibrations':calibrations,'metric_comparisons':comparisons,
            'clean_v2_reproduction':'passed','causal_invariance':'passed','draw_independence':'passed'}
    with (OUT/'audit.json').open('x') as f:json.dump(result,f,indent=2)
    print('AUDIT_PASS',json.dumps(result),flush=True)
if __name__=='__main__':main()
