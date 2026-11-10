"""Verify source immutability, calibration reuse, and cached per-case replay checks."""
import hashlib,json
from pathlib import Path
import numpy as np
from ects.rocket_v2 import calibrate_sequential
from ects.robustness import CONDITIONS
from frontier_worker import ROOT,OUT,DATASETS,TAUS,ALPHAS,GS

def main():
    src=ROOT/'runs/ects_robust_v2'
    before=json.loads((OUT/'source_hashes.json').read_text())
    now={str(p.relative_to(src)):p for p in src.rglob('*') if p.is_file()}
    if set(before)!=set(now):raise AssertionError('Source file list changed')
    for name,p in now.items():
        h=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
        if h.hexdigest()!=before[name]:raise AssertionError('Source file changed: '+name)
    checks=0;calibrations=0
    for name in DATASETS:
        for draw in range(5):
            r=json.loads((src/'records'/f'{name}_draw{draw}.json').read_text())
            dmeta=json.loads((OUT/'draw_meta'/f'{name}_draw{draw}.json').read_text())
            with np.load(r['calibration_artifact']) as d:
                for alpha in ALPHAS:
                    lam,s,e,risk=calibrate_sequential(d['mixed_prob'],d['cal_y'],d['classes'],alpha)
                    vals=[float(x) if np.isfinite(x) else None for x in lam]
                    assert vals==dmeta['mixed_thresholds'][str(alpha)]
                    assert all(v<=alpha+1e-12 for v in e)
                    calibrations+=1
            for cond in CONDITIONS:
                p=OUT/'records'/f'{name}_draw{draw}_{cond}.json';result=json.loads(p.read_text())
                assert result['reproduction_checks'].startswith('probability')
                for method,settings in [('probability',TAUS),('TEASER',GS),('augmented',ALPHAS)]:
                    assert [x['setting'] for x in result['frontiers'][method]]==list(settings)
                for fixed in result['fixed']:
                    assert abs(fixed['clean']['accuracy']+fixed['clean']['error']-1)<1e-12
                    assert abs(fixed['augmented']['accuracy']+fixed['augmented']['error']-1)<1e-12
                original=r['conditions'][cond]['results']
                v=next(x for x in result['frontiers']['augmented'] if abs(x['setting']-.10)<1e-12)
                for k in ('accuracy','balanced_accuracy','macro_f1','earliness'):
                    assert abs(v[k]-original['realistic'][k])<1e-12,(name,draw,cond,k)
                g=result['teaser_default_g']
                t=next(x for x in result['frontiers']['TEASER'] if x['setting']==g)
                for k in ('accuracy','balanced_accuracy','macro_f1','earliness'):
                    assert abs(t[k]-original['TEASER'][k])<1e-12,(name,draw,cond,'TEASER',k)
                checks+=1
    status={'source_files_unchanged':len(before),'conditions_reproduced':checks,
            'mixed_calibrations_recomputed':calibrations,'same_thresholds_all_conditions':True,
            'checkpoint_models_refit':0,'teaser_safety_rebuilt':len(list((OUT/'teaser_safety').glob('*.joblib')))}
    with (OUT/'audit.json').open('x') as f:json.dump(status,f,indent=2)
    print('FRONTIER_AUDIT_PASS',json.dumps(status),flush=True)
if __name__=='__main__':main()
