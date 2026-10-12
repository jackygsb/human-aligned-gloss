"""Exploratory adjusted shape/illumination effects; no new model responses."""
from pathlib import Path
import os
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='1'
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,pearsonr,t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'mllm_material/matching_full_v1/shared_condition_effects';OUT.mkdir(exist_ok=True)
df=pd.read_csv(OUT.parent/'image_summary.csv').merge(pd.read_csv(BASE/'annotations/gloss_annotations.csv')[['image_id','hue','chroma']],on='image_id',validate='one_to_one')
rng=np.random.default_rng(2026092807)

def design(data,conditions=True):
    parts=[np.ones((len(data),1))];names=['intercept']
    dims=['physical_level','viewpoint_id','chroma']+(['shape_id','illumination_id'] if conditions else [])
    for dim in dims:
        levels=sorted(df[dim].unique())
        for level in levels[1:]:parts.append((data[dim].to_numpy()==level).astype(float)[:,None]);names.append(f'{dim}:{level}')
    angle=np.deg2rad(data.hue.to_numpy())
    for k in [1,2]:
        parts.extend([np.sin(k*angle)[:,None],np.cos(k*angle)[:,None]]);names.extend([f'hue_sin{k}',f'hue_cos{k}'])
    return np.concatenate(parts,axis=1),names

df['physical_level']=np.rint(df.physical_c/.1487*51).astype(int)
Y=df[['human_c','model_c']].to_numpy();scale=Y.std(axis=0,ddof=1);Y=(Y-Y.mean(axis=0))/scale
X,names=design(df);pinv=np.linalg.pinv(X);B=pinv@Y;res=Y-X@B
N,nuisance_names=design(df,False);nr=Y-N@np.linalg.lstsq(N,Y,rcond=None)[0]
summary={'n_images':len(df),'analysis':'Exploratory analysis of all images; additive adjusted associations, not causal effects or equality of original and reconstructed c scales',
 'controls':'50 categorical physical-c levels, categorical viewpoint and chroma, first two circular hue harmonics; shape and illumination jointly included',
 'design_columns':X.shape[1],'design_rank':int(np.linalg.matrix_rank(X)),
 'residual_pearson_after_physical_color_viewpoint':float(pearsonr(nr[:,0],nr[:,1]).statistic),
 'residual_pearson_after_all_additive_controls':float(pearsonr(res[:,0],res[:,1]).statistic),'conditions':{}}
assert summary['design_rank']==X.shape[1]
fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
tables={}
for ax,dim,other,title in zip(axes,['shape_id','illumination_id'],['illumination_id','shape_id'],['Shape','Illumination']):
    levels=sorted(df[dim].unique());C=np.zeros((len(levels),len(names)))
    for i,level in enumerate(levels[1:],1):C[i,names.index(f'{dim}:{level}')]=1
    C-=C.mean(axis=0)
    effect=C@B
    # Paired cluster multiplier bootstrap of OLS influence functions, clustering
    # on the other condition; preserves shared human/model response dependence.
    clusters=sorted(df[other].unique());infl=[]
    for c in clusters:
        ix=df[other].to_numpy()==c
        infl.append(C@pinv[:,ix]@res[ix])
    infl=np.stack(infl);G=len(clusters);correction=np.sqrt(G/(G-1)*(len(df)-1)/(len(df)-X.shape[1]))
    delta=np.einsum('bg,glk->blk',rng.choice([-1.,1.],size=(4000,G)),infl)*correction
    boot=effect[None,:,:]+delta
    ci=np.quantile(boot,[.025,.975],axis=0)
    corr=float(spearmanr(effect[:,0],effect[:,1]).statistic)
    bootcorr=np.array([spearmanr(v[:,0],v[:,1]).statistic for v in boot])
    null=np.array([spearmanr(effect[:,0],rng.permutation(effect[:,1])).statistic for _ in range(9999)])
    rows=[]
    for i,level in enumerate(levels):
        joint='both_higher' if np.all(ci[0,i]>0) else 'both_lower' if np.all(ci[1,i]<0) else 'not_resolved'
        rows.append(dict(condition_id=int(level),human_adjusted_sd=effect[i,0],model_adjusted_sd=effect[i,1],human_ci_low=ci[0,i,0],human_ci_high=ci[1,i,0],model_ci_low=ci[0,i,1],model_ci_high=ci[1,i,1],same_direction=bool(effect[i,0]*effect[i,1]>0),pointwise_joint_direction=joint,human_adjusted_c=effect[i,0]*scale[0],model_adjusted_c=effect[i,1]*scale[1]))
    table=pd.DataFrame(rows);tables[dim]=table;table.to_csv(OUT/f'{dim}_adjusted_effects.csv',index=False,encoding='utf-8-sig')
    # Single common 95% band across both outcomes and all 36 condition levels.
    se=delta.std(axis=0,ddof=1);critical=np.quantile(np.max(np.abs(delta/np.maximum(se,1e-12)),axis=(1,2)),.95)
    robust_high=levels and [int(levels[i]) for i in range(len(levels)) if np.all(effect[i]-critical*se[i]>0)]
    robust_low=[int(levels[i]) for i in range(len(levels)) if np.all(effect[i]+critical*se[i]<0)]
    table['human_simultaneous_low']=effect[:,0]-critical*se[:,0];table['human_simultaneous_high']=effect[:,0]+critical*se[:,0]
    table['model_simultaneous_low']=effect[:,1]-critical*se[:,1];table['model_simultaneous_high']=effect[:,1]+critical*se[:,1]
    table.to_csv(OUT/f'{dim}_adjusted_effects.csv',index=False,encoding='utf-8-sig')
    summary['conditions'][dim]={'effect_spearman':corr,'cluster_multiplier_bootstrap95':np.quantile(bootcorr,[.025,.975]).tolist(),'label_permutation_p_descriptive':float((1+np.sum(abs(null)>=abs(corr)))/10000),'same_direction_count':int(table.same_direction.sum()),'n_conditions':len(levels),'simultaneous_both_higher_ids':robust_high,'simultaneous_both_lower_ids':robust_low,'ci_cluster_unit':other}
    ax.errorbar(effect[:,0],effect[:,1],xerr=np.maximum(0,np.array([effect[:,0]-ci[0,:,0],ci[1,:,0]-effect[:,0]])),yerr=np.maximum(0,np.array([effect[:,1]-ci[0,:,1],ci[1,:,1]-effect[:,1]])),fmt='o',markersize=4,color='#246c91',ecolor='#cad9e3',elinewidth=.8)
    for i,level in enumerate(levels):ax.annotate(str(level),(effect[i,0],effect[i,1]),xytext=(3,3),textcoords='offset points',fontsize=7)
    ax.axhline(0,color='gray',lw=.7);ax.axvline(0,color='gray',lw=.7);ax.set(xlabel='Human adjusted effect (SD)',ylabel='Model adjusted effect (SD)',title=f'{title}: rho = {corr:.3f}');ax.grid(alpha=.12)
fig.savefig(OUT/'shared_effects.png',dpi=200);fig.savefig(OUT/'shared_effects.svg');plt.close(fig)

# Sensitivity to excluding the 39 protocol-development images, and to choice of
# physical nuisance model. These are reported, never used to choose the headline.
summary['sensitivity']={}
for tag,mask,categorical in [('heldout_only',df.cohort.eq('heldout_matching').to_numpy(),True),('linear_physical',np.ones(len(df),bool),False)]:
    xx=X[mask].copy();nn=names.copy()
    if not categorical:
        keep=[i for i,n in enumerate(nn) if not n.startswith('physical_level:')];xx=np.c_[xx[:,keep],df.physical_c.to_numpy()[mask]];nn=[nn[i] for i in keep]+['physical_c']
    bb=np.linalg.lstsq(xx,Y[mask],rcond=None)[0];s={}
    for dim in ['shape_id','illumination_id']:
        e=np.zeros((36,2))
        for i in range(2,37):e[i-1]=bb[nn.index(f'{dim}:{i}')]
        e-=e.mean(axis=0);s[dim]=float(spearmanr(e[:,0],e[:,1]).statistic)
    summary['sensitivity'][tag]=s
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
lines=['# 人类与 InternVL 的共同形状、照明判断规律','',f'使用全部 {len(df)} 张图的两次匹配均值；属于采集后的探索性分析。',
 '同时控制物理光泽的 50 个离散档位、视角、色度、色相，以及另一类条件。人类与模型分别按自身标准差标准化，因此比较的是条件效应的方向和排序，不是假设两者 c 标尺相同。',
 '效应为相对于 36 个条件平均值的调整后差异。图中右上象限：两者都偏高；左下象限：两者都偏低。误差线为逐条件 95% 区间；下列明确列出的条件使用更严格的同时置信带（每类条件覆盖两种响应和全部 36 个条件）。','']
for dim,label in [('shape_id','形状'),('illumination_id','照明')]:
    s=summary['conditions'][dim]
    lines.extend([f'## {label}',f"调整后效应排序相关：{s['effect_spearman']:.3f}；对另一条件聚类的重采样区间：{s['cluster_multiplier_bootstrap95']}。",f"同方向条件：{s['same_direction_count']}/36。",f"同时置信带支持共同升高的 ID：{s['simultaneous_both_higher_ids']}。",f"同时置信带支持共同降低的 ID：{s['simultaneous_both_lower_ids']}。",''])
lines.extend([f"控制物理值、颜色、视角后的图像残差 Pearson：{summary['residual_pearson_after_physical_color_viewpoint']:.3f}；进一步控制形状与照明加性效应后：{summary['residual_pearson_after_all_additive_controls']:.3f}。",'',
 '这些是同一刺激集上的条件关联，不是通过控制实验得到的因果效应；未控制全部图像统计量，也未传播原始人类个体数据的不确定性。加性模型不识别形状×照明交互。边界匹配全部保留，因此效应描述包含序列范围约束。聚类区间条件于已观察到的另一组因素；标签置换 p 只作为条件对应关系的描述性检查。',
 '本轮未新增采集，未按一致性筛选图像或调整提示词。开发集剔除与线性物理值控制的敏感性结果见 summary.json。'])
(OUT/'共同判断规律分析.md').write_text('\n\n'.join(lines),encoding='utf-8')
html='<meta charset="utf-8"><title>共同形状与照明规律</title><style>body{font:17px system-ui;max-width:1150px;margin:40px auto;line-height:1.7}img{width:100%}table{border-collapse:collapse;font-size:13px}td,th{padding:6px;border:1px solid #ddd}</style>'
html+='<h1>人类与 InternVL：共同形状和照明规律</h1><img src="shared_effects.png">'+''.join('<p>'+s+'</p>' for s in lines)
for dim,tab in tables.items():html+='<h2>'+dim+'</h2>'+tab.round(4).to_html(index=False)
html+='<p><a href="summary.json">完整统计</a> · <a href="shape_id_adjusted_effects.csv">形状数据</a> · <a href="illumination_id_adjusted_effects.csv">照明数据</a></p>'
(OUT/'report.html').write_text(html,encoding='utf-8')
print(json.dumps(summary,indent=2))
