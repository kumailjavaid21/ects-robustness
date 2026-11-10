import sys,json,os
from pathlib import Path
ROOT=Path('/home/ubuntu/host_share/kumail_mad')
OUT=ROOT/'runs/ects_crosstype_v1'
SCALE=ROOT/'runs/ects_scale_v1'
DEEP=ROOT/'runs/ects_deep_scale_v1'
SELECT=ROOT/'runs/ects_select_v1'
sys.path.insert(0,str(SCALE))
import scale_worker as sw
sys.path.append(str(DEEP/'code'))
sys.path.insert(0,str(OUT/'code'))
from deep_model import AnytimeTCN
DATASETS=tuple(sw.MAIN)
DRAWS=tuple(range(5))
CONDITIONS=('spike_005','spike_015','warp_moderate','warp_strong','quantization_8','quantization_4','drift_small','drift_large','burst_3x10','burst_1x30')
TYPES={c:c.split('_')[0] for c in CONDITIONS}
TARGETS=(.5,.6);ALPHAS=(.02,.05,.1,.15,.2,.3)
AUG_METHODS=('MiniRocket_aug','TCN_aug')
METHODS=('MiniRocket_aug','TCN_aug','Probability','TEASER','ECEC')
FIXED_METHODS=('MiniRocket_clean','MiniRocket_aug','TCN_aug')
METRICS=('accuracy','error','balanced_accuracy','macro_f1','earliness','combined','forced_fraction')
CONFIG={'datasets':DATASETS,'draws':DRAWS,'conditions':CONDITIONS,'training':False,
 'normalization':'unchanged causal running normalization AFTER corruption in archived normalized units',
 'spike':'independent Bernoulli fractions .05/.15; additive random-sign amplitude 5 archive units',
 'warp':'one-sided causal clock; speed 1+a*sin(2*pi*f*t/(L-1)+phase), a=.4/.8, f~U(.5,1.5), phase~U(0,2pi); source=min(t,previous+speed); linear interpolate source<=t',
 'quantization':'fixed 8/4 uniform reconstruction levels on [-3,3]; clip outside; no per-series range estimation',
 'drift':'add signed linear ramp 0 to .5/1.5 archive units over declared horizon',
 'burst':'exact round(.3*L) missing positions; split into 3 contiguous runs or 1 run; random observed-gap allocation; forward-fill, initial holes=0',
 'pairing':'per-sample deterministic random seeds; same random mechanisms across severity; quantization deterministic (clean model draws identical)',
 'selection':'reuse deep_scale fixedop_calibration_choices.csv without alteration; single mixed calibration setting held across all unseen conditions',
 'oracle':'diagnostic only: new type-matched CALIBRATION inputs; same alpha as mixed, plus separately reported type-matched-calibration earliness selection; no test labels for either',
 'fixed_primary_holm':'60 tests = 3 model pairings x 10 conditions x prefixes .5/.6',
 'fixed_other_holm':'210 tests = 3 pairings x 10 conditions x remaining 7 prefixes',
 'fixed_curve_holm':'30 tests = 3 pairings x 10 conditions, averaged over all nine prefixes',
 'fixedop_holm':'140 tests = (2 augmented vs 3 clean baselines + TCN vs augmented MiniRocket) x 10 conditions x 2 targets',
 'small_oracle_gap_descriptive':'absolute macro error gap <= .02 AND absolute macro earliness gap <= .05; not an error guarantee'}
