"""Aggregate fixed-point, selective-risk and calibration diagnostics."""
import sys,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
ROOT=Path('/home/ubuntu/host_share/kumail_mad');HERE=ROOT/'runs/ects_select_v1';SCALE=ROOT/'runs/ects_scale_v1'
sys.path.insert(0,str(HERE));import fixed_ops as f
from replay import CONDITIONS
MAIN=f.MAIN
METHODS=['selective','aug_probability_calmatched','naive_final_calmatched','naive_final_090','aug_probability_testmatched_OFFLINE']
NAMES={'selective':'Selective risk control','aug_probability_calmatched':'Aug + probability, calibration matched','naive_final_calmatched':'Final max-probability, calibration matched','naive_final_090':'Final max-probability, threshold .90','aug_probability_testmatched_OFFLINE':'Aug + probability, exact test-coverage match (OFFLINE)'}
def pm(a,b,d=4):
    return f'{a:.{d}f} +/- {b:.{d}f}' if np.isfinite(a) and np.isfinite(b) else 'n/a'
def table(head,rows):
    def c(x):return str(x).replace('|','\\|').replace('\n',' ')
    return '| '+' | '.join(map(c,head))+' |\n| '+' | '.join(['---']*len(head))+' |\n'+''.join('| '+' | '.join(map(c,row))+' |\n' for row in rows)
def calibration_choices():
    rows=[]
    for name in MAIN:
        for draw in range(5):
            m=json.loads((HERE/'draw_meta'/f'{name}_draw{draw}.json').read_text())
            for method,settings in m['calibration_earliness'].items():
                for v in settings:rows.append({'dataset':name,'draw':draw,'method':method,**v})
    d=pd.DataFrame(rows);a=d.groupby(['dataset','method','setting'],as_index=False).agg(earliness=('earliness','mean'),n_draws=('draw','nunique'))
    if not (a.n_draws==5).all():raise ValueError('Calibration selection incomplete')
    choices=[]
    for (name,method),g in a.groupby(['dataset','method'],sort=False):
        for target in f.TARGETS:
            q=g.assign(distance=(g.earliness-target).abs()).sort_values(['distance','setting'],kind='stable').iloc[0]
            for condition in CONDITIONS:
                choices.append({'dataset':name,'method':method,'target':target,'condition':condition,'setting':q.setting,
                                'selection_earliness':q.earliness,'selection':'mixed_calibration_fixed'})
    return pd.DataFrame(choices),d
def read_records():
    rows=[];ece=[];policies=[]
    for name in MAIN:
        for draw in range(5):
            m=json.loads((HERE/'draw_meta'/f'{name}_draw{draw}.json').read_text())
            p=m['policy'];policies.append({'dataset':name,'draw':draw,'status':p['status'],'cal_coverage':p['chosen']['coverage'],
              'cal_upper':p['chosen']['upper'],'cal_earliness':p['chosen']['earliness'],'base':p['chosen']['base'],'slope':p['chosen']['slope']})
            for cond in CONDITIONS:
                pth=HERE/'records'/f'{name}_draw{draw}_{cond}.json'
                if not pth.exists():raise ValueError('Missing requested condition '+str(pth))
                d=json.loads(pth.read_text())
                if (d['dataset'],d['draw'],d['condition'])!=(name,draw,cond):raise ValueError('Mismatched record')
                for method,x in d['results'].items():rows.append({'dataset':name,'draw':draw,'condition':cond,'method':method,**x})
                for x in d['ece']:ece.append({'dataset':name,'draw':draw,'condition':cond,**x})
    return pd.DataFrame(rows),pd.DataFrame(ece),pd.DataFrame(policies)
def aggregate_selective(d):
    keys=['dataset','condition','method'];cols=['coverage','committed_error','earliness_committed']
    each=d.groupby(keys,as_index=False).agg(**{k+'_mean':(k,'mean') for k in cols},**{k+'_sd':(k,'std') for k in cols},
      n_draws=('draw','nunique'),n_error_draws=('committed_error','count'),test_size=('n','first'))
    each['level']='dataset'
    perdraw=d.groupby(['condition','method','draw'],as_index=False).agg(**{k:(k,'mean') for k in cols},
       n_datasets=('dataset','nunique'),n_evaluable=('committed_error','count'),sum_errors=('errors','sum'),sum_committed=('committed','sum'))
    perdraw['pooled_committed_error']=perdraw.sum_errors/perdraw.sum_committed.replace(0,np.nan)
    macro=perdraw.groupby(['condition','method'],as_index=False).agg(**{k+'_mean':(k,'mean') for k in cols},
       **{k+'_sd':(k,'std') for k in cols},n_draws=('draw','nunique'),n_evaluable_min=('n_evaluable','min'),
       n_evaluable_max=('n_evaluable','max'),pooled_committed_error_mean=('pooled_committed_error','mean'),
       pooled_committed_error_sd=('pooled_committed_error','std'))
    macro['dataset']='__macro__';macro['level']='macro'
    return pd.concat([each,macro],ignore_index=True,sort=False),macro,perdraw
def compare_selective(d):
    out=[]
    for cond in CONDITIONS:
        a=d[d.condition==cond];ours=a[a.method=='selective'].set_index(['dataset','draw'])
        for method in METHODS[1:]:
            b=a[a.method==method].set_index(['dataset','draw']);pair=ours.join(b,lsuffix='_ours',rsuffix='_base',how='inner')
            pair=pair.dropna(subset=['committed_error_ours','committed_error_base','earliness_committed_ours','earliness_committed_base'])
            pair['margin']=pair.committed_error_base-pair.committed_error_ours
            pair['coverage_delta']=pair.coverage_base-pair.coverage_ours
            pair['earliness_delta']=pair.earliness_committed_base-pair.earliness_committed_ours
            for match in ('all_evaluable','within_0.05_coverage_and_earliness'):
                q=pair if match=='all_evaluable' else pair[(pair.coverage_delta.abs()<=.05)&(pair.earliness_delta.abs()<=.05)]
                per=q.reset_index().groupby('dataset').agg(margin=('margin','mean'),count=('draw','nunique'))
                per=per[per['count']==5];v=per.margin.to_numpy();n=len(v)
                if n<2:stat=p=np.nan
                elif np.all(np.abs(v)<=1e-14):stat=0.;p=1.
                else:
                    z=wilcoxon(v,zero_method='wilcox',alternative='two-sided',method='auto');stat=float(z.statistic);p=float(z.pvalue)
                perdraw=q.reset_index().groupby('draw').margin.mean()
                out.append({'condition':cond,'baseline':method,'comparison':match,'n_datasets_all_five_draws':n,
                   'n_draw_pairs':len(q),'mean_margin_baseline_minus_ours':float(perdraw.mean()) if len(perdraw) else np.nan,
                   'margin_sd_draw_means':float(perdraw.std(ddof=1)) if len(perdraw)>1 else np.nan,
                   'mean_coverage_difference_baseline_minus_ours':float(q.coverage_delta.mean()) if len(q) else np.nan,
                   'mean_earliness_difference_baseline_minus_ours':float(q.earliness_delta.mean()) if len(q) else np.nan,
                   'wins':int(np.sum(v>1e-12)),'ties':int(np.sum(np.abs(v)<=1e-12)),'losses':int(np.sum(v< -1e-12)),
                   'wilcoxon_stat':stat,'p_raw':p})
    result=pd.DataFrame(out);result['p_holm']=np.nan
    idx=result.index[result.p_raw.notna()].tolist();idx.sort(key=lambda i:result.loc[i,'p_raw']);running=0.
    for k,i in enumerate(idx):
        running=max(running,min(1.,(len(idx)-k)*result.loc[i,'p_raw']));result.loc[i,'p_holm']=running
    return result
def aggregate_ece(e):
    keys=['dataset','condition','model','nominal_fraction']
    per=e.groupby(keys,as_index=False).agg(ece_mean=('ece','mean'),ece_sd=('ece','std'),
        signed_gap_mean=('signed_confidence_minus_accuracy','mean'),signed_gap_sd=('signed_confidence_minus_accuracy','std'),
        accuracy_mean=('accuracy','mean'),confidence_mean=('mean_confidence','mean'),n_draws=('draw','nunique'))
    per['level']='dataset'
    draws=e.groupby(['condition','model','nominal_fraction','draw'],as_index=False).agg(ece=('ece','mean'),
       signed_gap=('signed_confidence_minus_accuracy','mean'),accuracy=('accuracy','mean'),confidence=('mean_confidence','mean'),n_datasets=('dataset','nunique'))
    macro=draws.groupby(['condition','model','nominal_fraction'],as_index=False).agg(ece_mean=('ece','mean'),ece_sd=('ece','std'),
       signed_gap_mean=('signed_gap','mean'),signed_gap_sd=('signed_gap','std'),accuracy_mean=('accuracy','mean'),
       confidence_mean=('confidence','mean'),n_draws=('draw','nunique'),n_datasets=('n_datasets','min'))
    macro['dataset']='__macro__';macro['level']='macro'
    return pd.concat([per,macro],ignore_index=True,sort=False),macro
def plot_selective(macro):
    fig,axes=plt.subplots(2,1,figsize=(13,8),sharex=True)
    colors=['#d95f02','#2878b5','#4b9a56','#8f65b1','#999999']
    for method,color in zip(METHODS,colors):
        z=macro[macro.method==method].set_index('condition').reindex(CONDITIONS)
        x=np.arange(len(CONDITIONS))
        for ax,key in zip(axes,['coverage','committed_error']):
            y=z[key+'_mean'].to_numpy(float);sd=z[key+'_sd'].to_numpy(float)
            good=np.isfinite(y)
            if good.any():ax.errorbar(x[good],y[good],yerr=np.nan_to_num(sd[good]),label=NAMES[method],marker='o',ms=4,capsize=2,color=color)
    axes[0].set_ylabel('Coverage (all 26 datasets)');axes[1].set_ylabel('Error among commitments\n(datasets with decisions)')
    axes[1].axhline(.1,color='black',linestyle='--',label='Target .10');axes[1].set_xticks(np.arange(len(CONDITIONS)),[x.replace('_',' ') for x in CONDITIONS],rotation=18)
    for ax in axes:ax.set_ylim(0,1.02);ax.grid(alpha=.2)
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=2,fontsize=8)
    fig.suptitle('Selective stopping: macro means +/- sample SD over five draws');fig.tight_layout(rect=(0,.14,1,.96))
    fig.savefig(HERE/'selective_coverage_error.png',dpi=180);plt.close(fig)
def main():
    raw,ece,policy=read_records()
    raw.to_csv(HERE/'selective_draws.csv',index=False);ece.to_csv(HERE/'ece_draws.csv',index=False);policy.to_csv(HERE/'calibration_certificates.csv',index=False)
    result,macro,perdraw=aggregate_selective(raw);result.to_csv(HERE/'selective_results.csv',index=False)
    cmp=compare_selective(raw);cmp.to_csv(HERE/'selective_vs_baselines.csv',index=False)
    e,em=aggregate_ece(ece);e.to_csv(HERE/'ece_clean_vs_aug.csv',index=False)
    choices,cal=calibration_choices();choices.to_csv(HERE/'calibration_fixed_choices.csv',index=False);cal.to_csv(HERE/'calibration_earliness.csv',index=False)
    old=f.load();selected=f.selected(old,choices);selected.to_csv(HERE/'calibration_fixed_selected_draws.csv',index=False)
    stat=f.statistics(selected,'mixed_calibration_fixed');post=pd.read_csv(HERE/'posthoc_stats.csv')
    fixed=pd.concat([post,stat],ignore_index=True);fixed.to_csv(HERE/'fixed_op_stats.csv',index=False)
    ranks=f.diagrams(selected,'mixed_calibration_fixed');ranks.to_csv(HERE/'cd_diagrams/calibration_ranks.csv',index=False)
    plot_selective(macro)
    build_summary(macro,raw,policy,cmp,em,fixed,ranks)
    print('REPORT_COMPLETE',len(raw),'selective rows',len(ece),'ECE rows',len(fixed),'fixed tests',flush=True)
def build_summary(macro,raw,policy,cmp,em,fixed,ranks):
    out=['# ECTS selective decision pilot\n\n']
    out.append('## Scope and interpretation\n\nAll 26 main UCR datasets, five paired draws and seven conditions were replayed from the saved clean and augmented checkpoint models with causal running normalization. No model was retrained. Cached mixed-calibration indices and deterministic corruption streams were reused; all checkpoint accuracies were checked against the scale run. An initial fixed-point run stopped before writing results because two nearby ECEC thresholds were treated as equal; the match tolerance was corrected and the analysis rerun. The two small-calibration contrast datasets remain outside this main cohort.\n\n')
    out.append('A condition-agnostic method never receives test-condition identity or test labels. Two explicitly descriptive diagnostics below are not deployable policies: choosing a stopping setting separately using the observed earliness of each test condition, and selecting the highest-confidence fraction of a test batch to exactly match observed coverage. These diagnostics use no test labels for their selection, but they do use test-batch information and are not causal online procedures.\n\n')
    out.append('## Fixed operating points: all 26 datasets\n\nFor each dataset and method, the mean earliness for each grid setting over the five draws is compared with T in {0.5, 0.6}; the closest is selected, with lower setting breaking ties. The condition-posthoc version uses test stopping times for that condition and is descriptive only. The mixed-calibration-fixed version selects once using the five mixed-calibration draws and reuses the same setting under every condition. The latter satisfies the deployment constraint. Neither selection uses test error. Nevertheless, selecting on test stopping times is post-selection on the test sample and can affect error-dependent inference indirectly; post-hoc p-values are exploratory, whereas the mixed-calibration-fixed selection avoids that specific test-sample choice.\n\n')
    out.append('The selected settings may still fall far from T or from each other. Thus N=26 fixes the missing-overlap sample problem, but does not establish equal-earliness comparisons when the realized times differ. The table reports the actual mean earliness for both methods, the mean absolute gap across datasets, and the number of datasets with a gap greater than 0.05. Wilcoxon tests use one five-draw mean error per dataset; Holm correction covers 70 condition x target x baseline tests separately within each selection analysis. W/T/L is baseline error minus ours, with tolerance 1e-12.\n\n')
    rr=[]
    for _,x in fixed.iterrows():
        rr.append([x.selection,x.condition,f'{x.target:.1f}',f.NAMES[x.baseline],f'{x.ours_earliness:.3f}',f'{x.baseline_earliness:.3f}',f'{x.mean_absolute_earliness_gap:.3f}',int(x.datasets_with_earliness_gap_over_005),
         pm(x.margin_baseline_minus_ours,x.margin_sd_across_draw_means),f'{int(x.wins)}/{int(x.ties)}/{int(x.losses)}',f'{x.p_raw:.4g}',f'{x.p_holm:.4g}'])
    out.append(table(['Selection','Condition','T','Comparator','Ours earliness','Comparator earliness','Mean absolute gap','Datasets gap > .05','Error margin +/- draw SD','W/T/L','p raw','p Holm'],rr)+'\n')
    out.append('Friedman and Nemenyi diagrams use all 26 datasets at T=0.5. Horizontal groups are shown only when the Friedman omnibus test rejects; within a group, rank differences do not exceed the Nemenyi critical difference, which is not evidence of equivalence. The post-hoc diagrams have condition-aware setting selection and must not be interpreted as deployable results; the mixed-calibration-fixed diagrams use a single setting across conditions.\n\n')
    post=pd.read_csv(HERE/'cd_diagrams/posthoc_ranks.csv')
    for _,z in pd.concat([post,ranks],ignore_index=True).iterrows():
        out.append(f"- [{z.condition}, {z.selection}](cd_diagrams/{z.condition}_{z.selection}.png): N=26; Friedman p={z.friedman_p:.4g}; Nemenyi CD={z.cd:.3f}.\n")
    out.append('\n## Selective stopping rule\n\nThe augmented model may DECIDE at any checkpoint when confidence exceeds its fixed time-dependent threshold, WAIT otherwise, and ABSTAIN if no checkpoint passes. The candidate family is prespecified: 9 base confidence levels x 2 time slopes = 18 rules. For each candidate, its committed calibration errors receive a one-sided Clopper-Pearson upper bound with Bonferroni delta/18, where alpha=0.10 and delta=0.10. Among certified candidates, select maximal mixed-calibration coverage and break ties by earlier decisions. If none is certified, abstain on every test case. No test condition or label enters this policy.\n\n')
    out.append('This is a Learn-then-Test-style finite-candidate procedure, not a universal error guarantee: the exact bound assumes an appropriate independent/exchangeable calibration-to-test sampling model and a conditional binomial model. The original mixed calibration uses balanced fixed corruption quotas, and separate test corruptions are shifted from that mixture, so those assumptions and per-condition risk control are not assured. The bound is per dataset/draw, not simultaneous across all datasets and draws. See [Learn then Test](https://arxiv.org/abs/2110.01052).\n\n')
    count=int((policy.status=='certified_on_mixed_calibration').sum());out.append(f'Calibration certified a rule for {count} of {len(policy)} dataset-draw pairs; the rest abstain completely. A conditional error is undefined when there are no commitments and is never replaced with zero.\n\n')
    rr=[]
    for cond in CONDITIONS:
        x=macro[(macro.condition==cond)&(macro.method=='selective')].iloc[0]
        rr.append([cond,pm(x.coverage_mean,x.coverage_sd),pm(x.committed_error_mean,x.committed_error_sd),
            pm(x.earliness_committed_mean,x.earliness_committed_sd),f'{int(x.n_evaluable_min)}-{int(x.n_evaluable_max)}',
            pm(x.pooled_committed_error_mean,x.pooled_committed_error_sd)])
    out.append(table(['Condition','Coverage, all 26','Committed error','Earliness on commitments','Datasets with commitments/draw','Pooled committed error'],rr)+'\n')
    out.append('Coverage averages all 26 datasets within each draw. Committed-error and committed-earliness averages exclude dataset-draws with no decisions; their denominator can vary. Pooled error weights datasets by their number of committed test cases. The +/- values are sample SDs of the five draw-level aggregates, not dataset-sampling confidence intervals.\n\n')
    out.append('### Comparisons and significance\n\nThe deployable augmented-probability baseline selects its stopping threshold and abstention floor on mixed calibration to approximate our calibration coverage and committed earliness; the final-only max-probability baseline matches calibration coverage, while a separate naive rule uses a fixed 0.90 final confidence threshold. Realized test coverage and earliness can differ. The exact test-coverage baseline is OFFLINE and transductive: it ranks the entire unlabeled test batch to match our realized count, so it cannot be deployed as an online condition-agnostic policy.\n\n')
    out.append('The strict matching subset requires absolute coverage and earliness gaps each <= 0.05 in every included draw. Selection uses no errors, but it is post-hoc and test-feature dependent, so p-values on this selected subset are exploratory. Wilcoxon uses only datasets with decisions for both methods in all five included draws; no error is invented for zero-coverage cases. Holm correction covers all valid condition x baseline x matching-subset comparisons.\n\n')
    rr=[]
    for _,x in cmp.iterrows():
        rr.append([x.condition,NAMES[x.baseline],x.comparison,int(x.n_datasets_all_five_draws),pm(x.mean_margin_baseline_minus_ours,x.margin_sd_draw_means),
         f'{x.mean_coverage_difference_baseline_minus_ours:+.3f}' if np.isfinite(x.mean_coverage_difference_baseline_minus_ours) else 'n/a',
         f'{x.mean_earliness_difference_baseline_minus_ours:+.3f}' if np.isfinite(x.mean_earliness_difference_baseline_minus_ours) else 'n/a',
         f'{int(x.wins)}/{int(x.ties)}/{int(x.losses)}',f'{x.p_holm:.4g}' if np.isfinite(x.p_holm) else 'n/a'])
    out.append(table(['Condition','Comparator','Subset','Datasets','Error margin +/- SD','Coverage gap','Earliness gap','W/T/L','p Holm'],rr)+'\n')
    out.append('## Calibration mechanism: top-label ECE\n\nECE uses 15 equal-width confidence bins at the stated checkpoint. Signed confidence minus accuracy > 0 indicates aggregate overconfidence; a small ECE does not establish conditional or per-class calibration. All values below are 26-dataset means +/- sample SD across draws.\n\n')
    rr=[]
    for cond in CONDITIONS:
        for frac in (.4,.6,1.):
            clean=em[(em.condition==cond)&(em.model=='clean')&np.isclose(em.nominal_fraction,frac)].iloc[0]
            aug=em[(em.condition==cond)&(em.model=='augmented')&np.isclose(em.nominal_fraction,frac)].iloc[0]
            rr.append([cond,frac,pm(clean.ece_mean,clean.ece_sd),pm(aug.ece_mean,aug.ece_sd),
                       f'{clean.signed_gap_mean:+.4f}',f'{aug.signed_gap_mean:+.4f}'])
    out.append(table(['Condition','Prefix','Clean ECE','Augmented ECE','Clean signed gap','Augmented signed gap'],rr)+'\n')
    out.append('## Verdict\n\n')
    for cond in CONDITIONS:
        x=macro[(macro.condition==cond)&(macro.method=='selective')].iloc[0]
        if x.n_evaluable_min==26 and x.coverage_mean>=.8 and np.isfinite(x.committed_error_mean) and x.committed_error_mean<=.1:
            msg='macro coverage >= 0.8 and macro committed error <= 0.10; this does not imply every dataset meets the target'
        else:msg='high-coverage, <= 0.10 committed error across all 26 datasets is not established'
        out.append(f'- {cond}: {msg}.\n')
    out.append('\nA result is not evidence of a stopping-layer improvement at matched coverage and earliness unless the reported strict-matching subset is sufficiently large and its corrected paired comparison supports it. A zero-coverage policy may have no measured error; that is not successful high-coverage risk control.\n\n')
    out.append('The UCR train/test and calibration splits are fixed across draws. Dataset correlations, small calibration sets, test-condition-specific diagnostics, and possible upstream archive preprocessing limit external validity and causal claims. Detailed per-dataset results, certificates, code, fixed-point settings, and uncertainty summaries are included in the CSV files.\n')
    audit=json.loads((HERE/'source_audit.json').read_text())
    out.append(f"\nPrior-source audit: {audit['files_hashed_before']} files before and {audit['files_hashed_after']} after; {len(audit['changed'])} changed, {len(audit['removed'])} removed, {len(audit['added'])} added.\n")
    replay=json.loads((HERE/'replay_audit.json').read_text())
    out.append(f"Replay audit: {replay['records_verified']} condition records checked against saved checkpoint accuracies; {replay['prior_threshold_checks']} prior mixed-calibration threshold vectors rederived, with {len(replay['mismatched_thresholds'])} mismatches. No models were retrained.\n")
    (HERE/'summary.md').write_text(''.join(out),encoding='utf-8')
if __name__=='__main__':main()
