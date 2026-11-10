import sys,json
from pathlib import Path
ROOT=Path('/home/ubuntu/host_share/kumail_mad')
OUT=ROOT/'runs/ects_deep_scale_v1'
SCALE=ROOT/'runs/ects_scale_v1'
PILOT=ROOT/'runs/ects_deep_pilot_v1'
SELECT=ROOT/'runs/ects_select_v1'
sys.path.insert(0,str(SCALE));import scale_worker as sw
DATASETS=tuple(sw.MAIN)
CONDITIONS=('clean','noise_20db','noise_10db','noise_5db','missing_10pct','missing_30pct','missing_50pct')
DRAWS=tuple(range(5));VARIANTS=('TCN_aug','TCN_aug_mask')
ALPHAS=(.02,.05,.1,.15,.2,.3)
TAUS=(.5,.6,.7,.8,.9,.95,.99)
TARGETS=(.5,.6)
CONFIG=json.loads((PILOT/'config.json').read_text())
CONFIG.update({'variants':list(VARIANTS),'heads':'independent; mode C','lambda':0.,
 'draws':5,'datasets':list(DATASETS),'conditions':list(CONDITIONS),
 'mask':'binary observed=1, forward-filled/missing=0; never normalized; clean/noise observed=1',
 'mask_initialization':'copy paired plain TCN initialization; extra input weights zero; restore RNG state',
 'stopping':'same empirical sequential mixed-calibration alpha sweep as scale_v1; force final decision',
 'fixedop_selection':'one setting per method/dataset/target minimizing distance of FIVE-DRAW MEAN mixed calibration earliness; held constant across conditions',
 'primary_fixedlength_holm_family':'28 tests = 2 TCNs x 7 conditions x 2 prefixes (0.5,0.6)',
 'fixedop_holm_family':'112 tests = 2 TCNs x 4 baselines x 7 conditions x 2 targets',
 'pilot_checkpoint_reuse':24})
