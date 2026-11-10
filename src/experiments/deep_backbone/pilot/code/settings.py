from pathlib import Path
ROOT=Path('/home/ubuntu/host_share/kumail_mad')
OUT=ROOT/'runs/ects_deep_pilot_v1'
SCALE=ROOT/'runs/ects_scale_v1'
DATASETS=('FordA','ECG5000','StarLightCurves','Wafer','ElectricDevices','Crop','NonInvasiveFetalECGThorax1','HandOutlines')
CONDITIONS=('clean','noise_10db','noise_5db','missing_30pct','missing_50pct')
DRAWS=(0,1,2)
VARIANTS=(('A_l0','A',0.,True),('B_l0','B',0.,True),('C_l0','C',0.,True),
          ('A_l01','A',.1,True),('B_l01','B',.1,True),('C_l01','C',.1,True),
          ('A_clean','A',0.,False))
TAUS=(0.,.1,.2,.3,.4,.5,.6,.7,.8,.85,.9,.925,.95,.975,.99,.995,.999,1.01)
TARGETS=(.5,.6)
CONFIG={'backbone':'causal TCN','channels':64,'blocks':5,'kernel':3,'dilations':[1,2,4,8,16],
 'convs_per_block':2,'dropout':.1,'head':'linear on causal 1/2/4-bin prefix pyramid plus last feature',
 'normalization':'existing causal running normalization; LayerNorm over channels at each time',
 'curriculum_epochs_per_checkpoint':2,'max_joint_epochs':22,'joint_patience':6,
 'optimizer':'Adam','learning_rate':.001,'weight_decay':.0001,'gradient_clip':1.,
 'batch_size_rule':'min(128,max(16,131072//series_length))',
 'consistency':'adjacent Jensen-Shannon divergence; lambda in {0,0.1}',
 'class_weighting':'none: same empirical-frequency objective as existing unweighted logistic regression',
 'early_stopping':'clean+mixed validation mean checkpoint cross-entropy; min full-head epochs=6',
 'selection':'best augmented variant per dataset/draw by validation mean checkpoint accuracy; no test selection',
 'stopping':'confidence sweep, thresholds selected by mixed calibration earliness ONLY; fixed across conditions',
 'seed_rule':'sha256 dataset + draw; same initialization seed and minibatch order across variants',
 'precision':'FP16 automatic mixed precision for training; float32 normalization/cumulative pooling/CE',
 'gpu_workers':3,'draws':3}
