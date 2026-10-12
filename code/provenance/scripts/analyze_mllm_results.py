"""Build figures/tables exclusively from recorded responses; no placeholder scores."""
from pathlib import Path
import json,itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,pearsonr,rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'mllm_material';FIG=OUT/'reports/figures';FIG.mkdir(parents=True,exist_ok=True)
ann=pd.read_csv(BASE/'annotations/gloss_annotations.csv');h=ann.human_gloss_score.to_numpy();c=ann.physical_pellacini_c.to_numpy()
shape=ann.shape_id.to_numpy().astype(int)-1;light=ann.illumination_id.to_numpy().astype(int)-1
rng=np.random.RandomState(20260927)
boot=np.load(OUT/'statistics/human_bootstrap_means.npy',mmap_mode='r')
# Pigeonhole bootstrap resamples crossed shape and illumination factors.
sw=rng.multinomial(36,np.ones(36)/36,size=2000);lw=rng.multinomial(36,np.ones(36)/36,size=2000);weights=sw[:,shape]*lw[:,light]
def corr(a,b):
    m=np.isfinite(a)&np.isfinite(b)
    return float(spearmanr(a[m],b[m])[0]) if m.sum()>2 and np.ptp(a[m]) and np.ptp(b[m]) else np.nan
def interval(v):return list(map(float,np.nanpercentile(v,[2.5,97.5]))) if np.isfinite(v).any() else [np.nan,np.nan]
def savefig(fig,name):
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(FIG/f'{name}.{ext}',dpi=250)
    plt.close(fig)
def residual_crossfit(y,groups):
    x=np.column_stack([np.ones(len(c)),c,c*c,c*c*c]);out=np.full(len(c),np.nan)
    for group in np.unique(groups):
        train=(groups!=group)&np.isfinite(y);test=(groups==group)&np.isfinite(y)
        coef=np.linalg.lstsq(x[train],y[train],rcond=None)[0];out[test]=y[test]-x[test]@coef
    return out
summaries=[];quality=[];subgroups=[];sensitivity=[];pairrows=[];vectors={}
requests=[json.loads(l) for l in (OUT/'manifests/requests.jsonl').read_text().splitlines()]
expected={r['request_id'] for r in requests}
for folder in sorted((OUT/'responses').glob('*')):
    path=folder/'full.jsonl'
    if not path.exists():continue
    answers={r['request_id']:r for r in map(json.loads,path.read_text().splitlines())};rows=list(answers.values());name=folder.name
    complete=expected<=set(answers)
    controlpath=folder/'input_control.jsonl'
    if controlpath.exists():
        controls={r['request_id']:r for r in map(json.loads,controlpath.read_text().splitlines())};rows.extend(controls.values());complete=complete and len(controls)==864
    else:complete=False
    for (task,condition),rs in itertools.groupby(sorted(rows,key=lambda r:(r['task'],r['condition'])),key=lambda r:(r['task'],r['condition'])):
        rs=list(rs);quality.append(dict(model=name,task=task,condition=condition,n=len(rs),valid=sum(r['parse_status']=='valid' for r in rs),unrateable=sum(r['parse_status']=='unrateable' for r in rs)))
    main=[r for r in rows if r['task']=='T1' and r['condition']=='main']
    if len(main)!=3888:continue
    scores=np.full(3888,np.nan)
    for r in main:
        if r['parse_status']=='valid':scores[r['indices'][0]]=r['parsed']['glossiness']
    vectors[name]=scores;valid=np.isfinite(scores)
    bs=[]
    for b,w in enumerate(weights):
        ix=np.repeat(np.flatnonzero(valid),w[valid]);bs.append(corr(scores[ix],boot[b,ix]))
    low,high=interval(bs);rho=corr(scores,h)
    permutations=np.array([corr(scores,rng.permutation(h)) for _ in range(2000)])
    np.save(OUT/'statistics'/f'{name}_human_rho_bootstrap.npy',bs)
    summary=dict(model=name,complete_all_tasks=complete,n=int(valid.sum()),human_spearman=rho,human_pearson=float(pearsonr(scores[valid],h[valid])[0]),human_rho_ci_low=low,human_rho_ci_high=high,physical_spearman=corr(scores,c),unique_scores=int(len(np.unique(scores[valid]))),image_permutation_p=float((1+(permutations>=rho).sum())/2001),h1_bootstrap_centered_p=float((1+(abs(np.array(bs)-rho)>=abs(rho)).sum())/2001))
    if not np.isfinite(rho):summary.update(image_permutation_p=np.nan,h1_bootstrap_centered_p=np.nan)
    for group,gids in [('shape',shape),('illumination',light)]:
        rm=residual_crossfit(scores,gids)
        rh=residual_crossfit(h,gids);summary[group+'_physical_control_residual_rho']=corr(rm,rh)
        for g in np.unique(gids):subgroups.append(dict(model=name,factor=group,group=int(g+1),rho=corr(scores[gids==g],h[gids==g]),n=int(np.sum(valid&(gids==g)))))
    summaries.append(summary)
    for task,condition in [('T1b','main'),('T1c','main'),('T1','repeat1'),('T1','repeat2'),('T1','no_image'),('T1','original224'),('T1','author_gray224')]:
        v=np.full(3888,np.nan)
        for r in rows:
            if r['task']==task and r['condition']==condition and r['parse_status']=='valid':v[r['indices'][0]]=r['parsed']['glossiness']
        m=np.isfinite(v)&valid
        sensitivity.append(dict(model=name,task=task,condition=condition,n=int(m.sum()),human_rho=corr(v,h),main_score_rho=corr(v,scores),exact_repeat_fraction=float(np.mean(v[m]==scores[m])) if m.any() else np.nan))
    pd.DataFrame(dict(image_id=ann.image_id,human=h,physical=c,model_score=scores)).to_csv(OUT/'statistics'/f'{name}_image_scores.csv',index=False)
    fig,axs=plt.subplots(1,3,figsize=(12,3.6))
    axs[0].hexbin(h[valid],scores[valid],gridsize=30,mincnt=1,cmap='viridis');axs[0].set(xlabel='Human matched gloss',ylabel='MLLM gloss score (0–100)',title=f'Human alignment: rho={rho:.3f}')
    axs[1].hexbin(c[valid],scores[valid],gridsize=30,mincnt=1,cmap='viridis');axs[1].set(xlabel='Physical gloss c',ylabel='MLLM gloss score',title=f'Physical alignment: rho={corr(scores,c):.3f}')
    axs[2].hist(scores[valid],bins=np.arange(-.5,101.5),color='#4477aa');axs[2].set(xlabel='Reported gloss score',ylabel='Images',title=f'{summary["unique_scores"]} distinct scores')
    fig.suptitle(name);savefig(fig,name+'_behavior')
    pairs=pd.read_csv(OUT/'manifests/pairs2000_analysis_only.csv');pairanswers={}
    for r in rows:
        if r['task']=='T2':pairanswers.setdefault(r['group'],[]).append(r)
    for n,p in pairs.iterrows():
        values=[];selected=[]
        for r in pairanswers.get(f'pair{n}',[]):
            if r['parse_status']!='valid':continue
            choice=r['parsed']['choice'];chosen=None if choice=='tie' else r['indices'][ord(choice)-65]
            humanwinner=int(p.a if p.human_direction>0 else p.b)
            values.append(.5 if chosen is None else float(chosen==humanwinner));selected.append(chosen)
        pairrows.append(dict(model=name,pair=n,a=int(p.a),b=int(p.b),stratum=p.stratum,follow_human=float(np.mean(values)) if values else np.nan,valid_presentations=len(values),swap_consistent=(selected[0]==selected[1]) if len(selected)==2 else np.nan,missing_lower=(sum(values))/2,missing_upper=(sum(values)+2-len(values))/2))
pd.DataFrame(summaries).to_csv(OUT/'statistics/model_summary.csv',index=False)
pd.DataFrame(quality).to_csv(OUT/'statistics/response_quality.csv',index=False)
pd.DataFrame(subgroups).to_csv(OUT/'statistics/group_alignment.csv',index=False)
pd.DataFrame(sensitivity).to_csv(OUT/'statistics/prompt_and_repeat_sensitivity.csv',index=False)
if subgroups:
    gdf=pd.DataFrame(subgroups);fig,axs=plt.subplots(1,2,figsize=(12,4))
    for ax,factor in zip(axs,['shape','illumination']):
        for model,d in gdf[gdf.factor==factor].groupby('model'):ax.plot(d.group,d.rho,'o-',ms=3,lw=.8,label=model)
        ax.axhline(0,color='gray',lw=.6);ax.set(xlabel=factor+' ID',ylabel='Within-group human Spearman rho');ax.legend(fontsize=8)
    savefig(fig,'shape_illumination_generalization')
if sensitivity:
    sdf=pd.DataFrame(sensitivity);fig,ax=plt.subplots(figsize=(9,4));x=np.arange(len(sdf));ax.bar(x,sdf.human_rho,color='#4477aa');ax.set_xticks(x);ax.set_xticklabels([r.model+'\n'+r.task+'/'+r.condition for r in sdf.itertuples()],rotation=35,ha='right',fontsize=7);ax.set(ylabel='Human Spearman rho',title='All predefined wording/repeat/no-image conditions');savefig(fig,'prompt_sensitivity')
pairdf=pd.DataFrame(pairrows);pairdf.to_csv(OUT/'statistics/pair_level_results.csv',index=False)
ps=[]
if len(pairdf):
    for (model,stratum),d in pairdf.groupby(['model','stratum']):
        a=d.a.to_numpy();b=d.b.to_numpy();v=d.follow_human.to_numpy();m=np.isfinite(v);bootpair=[]
        for k in range(2000):
            # Shared images and shared shape/illumination receive the same bootstrap weight.
            w=weights[k,a]*weights[k,b]
            bootpair.append(np.average(v[m],weights=w[m]) if w[m].sum() else np.nan)
        ci=interval(bootpair);estimate=float(np.nanmean(v));ps.append(dict(model=model,stratum=stratum,n=len(d),valid_pairs=int(m.sum()),follow_human=estimate,ci_low=ci[0],ci_high=ci[1],missing_lower=d.missing_lower.mean(),missing_upper=d.missing_upper.mean(),swap_consistency=d.swap_consistent.mean(),bootstrap_centered_p=float((1+(abs(np.array(bootpair)-estimate)>=abs(estimate-.5)).sum())/2001),inference='approximate_centered_crossed_bootstrap_not_iid_pairs'))
    for result in ps:
        if result['valid_pairs']==0:result['bootstrap_centered_p']=np.nan
    pd.DataFrame(ps).to_csv(OUT/'statistics/pair_summary.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,4))
    for i,r in enumerate(ps):ax.errorbar(i,r['follow_human'],yerr=[[max(0,r['follow_human']-r['ci_low'])],[max(0,r['ci_high']-r['follow_human'])]],fmt='o',color='#4477aa')
    ax.axhline(.5,ls='--',color='gray');ax.set_xticks(range(len(ps)));ax.set_xticklabels([r['model']+'\n'+r['stratum'] for r in ps],fontsize=8);ax.set(ylabel='Fraction following human-induced ordering',ylim=(0,1));savefig(fig,'pair_human_bias')
primary=[dict(model=s['model'],hypothesis='H1',p=s['h1_bootstrap_centered_p']) for s in summaries if np.isfinite(s['h1_bootstrap_centered_p'])]
primary.extend(dict(model=s['model'],hypothesis='H2',p=s['bootstrap_centered_p']) for s in ps if s['stratum']=='discordant' and s['valid_pairs']>0)
if primary:
    order=np.argsort([r['p'] for r in primary]);previous=0.
    for rank,i in enumerate(order):
        value=min(1.,max(previous,(4-rank)*primary[i]['p']));primary[i]['p_holm']=value;primary[i]['prespecified_family_size']=4;previous=value
    pd.DataFrame(primary).to_csv(OUT/'statistics/primary_hypotheses_holm.csv',index=False)
comparison=[]
for a,b in itertools.combinations(vectors,2):
    m=np.isfinite(vectors[a])&np.isfinite(vectors[b]);delta=[]
    for k,w in enumerate(weights):
        ix=np.repeat(np.flatnonzero(m),w[m]);delta.append(corr(vectors[a][ix],boot[k,ix])-corr(vectors[b][ix],boot[k,ix]))
    lo,hi=interval(delta);comparison.append(dict(model_a=a,model_b=b,common_n=int(m.sum()),rho_difference=corr(vectors[a][m],h[m])-corr(vectors[b][m],h[m]),ci_low=lo,ci_high=hi))
pd.DataFrame(comparison).to_csv(OUT/'statistics/paired_model_comparisons.csv',index=False)
report=['# Gloss 实验结果（自动更新）','只报告原始响应文件中实际存在的结果。全流程完成状态见下表；未完成不能作为最终论文结果。','']
for s in summaries:report.append(f"- {s['model']}：有效单图 {s['n']}/3888；人类 Spearman rho={s['human_spearman']:.4f}，95% CI [{s['human_rho_ci_low']:.4f}, {s['human_rho_ci_high']:.4f}]；所有任务完成={s['complete_all_tasks']}。")
report.extend(['','## 解释边界','评分相关、共享偏差及 SPoSE 留出预测须分别判断。作者小网络发布结果单列。人类比较/三选一是既有匹配评分诱导参照，不是真实采集的人类选择。仅两款模型不能证明随规模变化的涌现规律，gloss 单数据集也不能替代尚未执行的 Serrano 外推。'])
(OUT/'reports/RESULTS.md').write_text('\n\n'.join(report),encoding='utf-8')
print('Actual-result tables and figures updated:',len(summaries),'models')
