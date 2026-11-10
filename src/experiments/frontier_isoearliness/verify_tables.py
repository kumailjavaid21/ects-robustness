"""Independently check raw and aggregated CSV counts and five-draw statistics."""
import csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from frontier_worker import OUT,DATASETS,TAUS,ALPHAS,GS
from ects.robustness import CONDITIONS

def read(name):
    with (OUT/name).open(newline='') as f:return list(csv.DictReader(f))

def main():
    raw=read('fixed_length_curves_draws.csv');fixed=read('fixed_length_curves.csv')
    assert len(raw)==4970 and len(fixed)==1120,(len(raw),len(fixed))
    actual=[r for r in fixed if r['dataset']=='ItalyPowerDemand' and r['checkpoint_kind']=='actual_only']
    assert len(actual)==14 and all(r['timepoint']=='9' for r in actual)
    group=defaultdict(list)
    for r in raw:
        if r['nominal_fraction']:
            key=(r['condition'],r['model'],round(float(r['nominal_fraction']),1),int(r['draw']))
            group[key].append(r)
    for r in fixed:
        if r['scope']!='macro':continue
        frac=round(float(r['nominal_fraction']),1)
        expected_n=7 if frac in (.2,.3) else 8
        assert int(r['n_datasets'])==expected_n
        for metric in ('accuracy','balanced_accuracy','macro_f1','error'):
            vals=[]
            for d in range(5):
                items=group[(r['condition'],r['model'],frac,d)];assert len(items)==expected_n
                vals.append(np.mean([float(x[metric]) for x in items]))
            assert abs(float(r[metric+'_mean'])-np.mean(vals))<1e-12
            assert abs(float(r[metric+'_sd'])-np.std(vals,ddof=1))<1e-12
    rawfront=read('frontier_points_draws.csv');front=read('frontier_points.csv')
    assert len(rawfront)==4760 and len(front)==1071,(len(rawfront),len(front))
    groups=defaultdict(list)
    for r in rawfront:
        groups[(r['condition'],r['method'],float(r['setting']),int(r['draw']))].append(r)
    for r in front:
        if r['scope']!='macro':continue
        assert int(r['n_datasets'])==8 and int(r['n_draws'])==5
        for metric in ('accuracy','balanced_accuracy','macro_f1','error','earliness'):
            vals=[]
            for d in range(5):
                items=groups[(r['condition'],r['method'],float(r['setting']),d)];assert len(items)==8
                vals.append(np.mean([float(x[metric]) for x in items]))
            assert abs(float(r[metric+'_mean'])-np.mean(vals))<1e-12
            assert abs(float(r[metric+'_sd'])-np.std(vals,ddof=1))<1e-12
    comp=json.loads((OUT/'comparison_details.json').read_text())
    assert set(comp)==set(CONDITIONS)
    assert all(len(comp[c]['targets'])==3 for c in CONDITIONS)
    status={'fixed_raw_rows':len(raw),'fixed_summary_rows':len(fixed),
       'frontier_raw_rows':len(rawfront),'frontier_summary_rows':len(front),
       'every_condition_three_shared_levels':True,'italy_actual_only_rows':len(actual)}
    with (OUT/'table_audit.json').open('x') as f:json.dump(status,f,indent=2)
    print('TABLE_AUDIT_PASS',json.dumps(status))
if __name__=='__main__':main()
