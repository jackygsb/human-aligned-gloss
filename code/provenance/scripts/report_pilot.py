# -*- coding: utf-8 -*-
"""Render scientific pilot figures and a result summary from saved out-of-fold results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'pilot/results'
LABELS={'physical_calibrated':'Physical c (calibrated)','image_stats':'Image statistics (with masks)',
        'physical_plus_stats':'Physical + statistics','clip_b32':'CLIP ViT-B/32','dinov2_small':'DINOv2-S (native crop)',
        'dinov2_same_fov':'DINOv2-S (full field)','physical_stats_plus_clip':'Control + CLIP',
        'physical_stats_plus_dino':'Control + DINOv2 (native)','physical_stats_plus_dino_same_fov':'Control + DINOv2 (full field)'}
ZH={'physical_calibrated':'物理参数（训练集校准）','image_stats':'图像统计（含遮罩）','physical_plus_stats':'物理参数＋图像统计',
    'clip_b32':'CLIP ViT-B/32','dinov2_small':'DINOv2-S（原生裁剪）','dinov2_same_fov':'DINOv2-S（完整视野）',
    'physical_stats_plus_clip':'控制基线＋CLIP','physical_stats_plus_dino':'控制基线＋DINOv2（原生）',
    'physical_stats_plus_dino_same_fov':'控制基线＋DINOv2（完整视野）'}

def main():
    s=pd.read_csv(OUT/'model_summary.csv');detail=pd.read_csv(OUT/'fold_metrics.csv');allpred=pd.read_csv(OUT/'all_oof_predictions.csv')
    increments=pd.read_csv(OUT/'incremental_diagnostics.csv')
    plt.rcParams.update({'font.size':10,'axes.spines.right':False,'axes.spines.top':False})
    fig,axs=plt.subplots(1,2,figsize=(14,7),sharey=True)
    for ax,factor,title in zip(axs,['shape','illumination'],['Held-out shapes','Held-out illuminations']):
        rows=s[s.split==factor].set_index('model').loc[list(LABELS)]
        xx=np.arange(len(rows));ax.barh(xx,rows.mean_fold_spearman,color=['#999999']*3+['#348ca5','#cd8841','#dfb775']+['#7c9d7c']*3)
        for i,model in enumerate(rows.index):
            vals=detail[(detail.split==factor)&(detail.model==model)].spearman.to_numpy()
            ax.scatter(vals,np.full(len(vals),i)+np.linspace(-.12,.12,len(vals)),s=12,c='#222222',alpha=.8,zorder=3)
        ax.set(yticks=xx,yticklabels=[LABELS[m] for m in rows.index],xlabel='Mean test-fold Spearman correlation',title=title,xlim=(0,1));ax.invert_yaxis()
    fig.suptitle('Exploratory grouped evaluation | dots: 12 folds, not independent confidence intervals',fontsize=12)
    fig.tight_layout();fig.savefig(BASE/'figures/07_pilot_model_comparison.png',dpi=160,bbox_inches='tight');plt.close(fig)
    # Descriptive relative error reductions on identical held-out observations.
    add=['physical_stats_plus_clip','physical_stats_plus_dino','physical_stats_plus_dino_same_fov']
    changes=[]
    for factor in ['shape','illumination']:
        baseline=s[(s.split==factor)&(s.model=='physical_plus_stats')].iloc[0].pooled_oof_rmse**2
        for model in add:
            row=increments[(increments.split==factor)&(increments.model==model)].iloc[0]
            changes.append(dict(split=factor,model=model,relative_mse_reduction=float(row.delta_mse_vs_physical_stats/baseline)))
    pd.DataFrame(changes).to_csv(OUT/'relative_mse_reduction.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,4.7))
    for i,factor in enumerate(['shape','illumination']):
        vals=[100*r['relative_mse_reduction'] for r in changes if r['split']==factor]
        ax.bar(np.arange(3)+(i-.5)*.36,vals,width=.36,label=factor)
    ax.axhline(0,color='black',lw=.8);ax.set(xticks=range(3),xticklabels=['Add CLIP','Add DINOv2 native','Add DINOv2 full field'],ylabel='Held-out MSE reduction over control (%)',title='Increment over physical + image-statistics control');ax.legend()
    fig.tight_layout();fig.savefig(BASE/'figures/08_pilot_incremental_information.png',dpi=160,bbox_inches='tight');plt.close(fig)
    # Examples are selected after evaluation solely to inspect model disagreement.
    p=allpred[allpred.split=='illumination'].pivot(index='image_id',columns='model',values='prediction')
    p['abs_disagreement']=abs(p.clip_b32-p.dinov2_same_fov)
    chosen=p.sort_values('abs_disagreement',ascending=False).head(6)
    a=pd.read_csv(BASE/'annotations/gloss_annotations.csv').set_index('image_id')
    selected=a.loc[chosen.index].join(chosen[['clip_b32','dinov2_same_fov','abs_disagreement']])
    selected.to_csv(OUT/'exploratory_disagreement_examples.csv',index_label='image_id')
    fig,axs=plt.subplots(2,3,figsize=(11,9))
    for ax,(idx,r) in zip(axs.flat,selected.iterrows()):
        with Image.open(BASE/r.image_224_path) as im:ax.imshow(im)
        ax.axis('off');ax.set_title(f'{idx} | human={r.human_gloss_score:.3f}\nCLIP={r.clip_b32:.3f}, DINO={r.dinov2_same_fov:.3f}',fontsize=10)
    fig.suptitle('Exploratory examples selected for disagreement | not new controlled experiments',fontsize=12)
    fig.subplots_adjust(top=.90,bottom=.03,hspace=.30,wspace=.12)
    fig.savefig(BASE/'figures/09_exploratory_disagreements.png',dpi=150,bbox_inches='tight');plt.close(fig)
    lines=['# 第一轮模型先导比较结果','','本轮使用 Morimoto 3,888 张带背景图像和作者人类均值，完成 12 折形状留出与 12 折照明留出。CLIP/DINOv2 冻结，岭回归读出器在训练集拟合；每个外层训练集再按对应因素做 3 折内层验证选正则化强度。没有将该过程称为零样本预测。','','## 主要结果','','主指标：12 个测试折 Spearman 相关的非加权均值。下表为描述性比较，不报告独立折 t 检验或显著性。','','| 方法 | 新形状 | 新照明 |','|---|---:|---:|']
    for model in LABELS:
        rr=s[s.model==model].set_index('split')
        lines.append(f'| {ZH[model]} | {rr.loc["shape","mean_fold_spearman"]:.3f} | {rr.loc["illumination","mean_fold_spearman"]:.3f} |')
    lines+=['','控制基线使用物理参数及作者遮罩，具有额外信息。CLIP/DINO 单模型只输入 RGB。这不是同等输入条件下的算法排行榜。','','## 在基线之外增加多少信息？','','在相同外层测试样本上，以 pooled out-of-fold MSE 计算相对下降；正数表示改善，负数表示变差。不同于只比較相关系数，该指标也惩罚数值预测误差。','','| 增量模型 | 划分 | 相对 MSE 下降 |','|---|---|---:|']
    for r in changes:lines.append(f'| {ZH[r["model"]]} | {r["split"]} | {r["relative_mse_reduction"]:.2%} |')
    lines+=['','这些结果不等于解释了同等比例的“人类独有偏差”，也不是因果效应。增加特征还改变了读出器维度；需要更多稳健性对照和独立验证。',
        '', '## 预处理与尺度', '', 'CLIP 使用作者原生预处理，DINOv2 原生先放大到 256 再裁剪 224。下载后发现该差异，在查看模型评分之前记录并加入 DINOv2 完整 224 视野的敏感性分析。不能把原生裁剪造成的差异归因于语义训练。',
        '', '图像统计为全图、物体、背景三个区域的线性 RGB 均值/标准差、线性亮度矩、分位数、梯度和高亮比例，另含物体面积，共 58 维。遮罩由作者提供，缩放到 224 时采用最近邻。物理校准为 c、c²、c³ 的岭回归。所有特征标准化仅拟合训练部分。',
        '', '预测保持连续值、不裁剪到标签范围；这允许错误暴露而不靠测试标签调节。MAE、RMSE、Pearson、每折 alpha、逐图预测都另存。训练均值基线在单折内为常数，因此相关未定义；其跨折拼接相关可能受不同训练均值影响，不作科学解释。',
        '', '## 限制与尚未完成', '', '- 这是单个公开数据集上的探索，已有标签此前被查看。不是新实验或确认性证据。',
        '- CLIP 与 DINOv2 的结构、训练数据、容量和目标都不同，差值不能隔离语言监督的作用。',
        '- 该数据的物理参数、色彩、形状和照明不是逐对全部固定；相关比较不能替代可控的同材料操纵。',
        '- tiny-network 尚无满足本项目新划分的可审计对照，原因见 `TINY_NETWORK_AUDIT.md`。不能宣称超过原论文或达到人类水平。',
        '- Serrano 仍只有 7 张图像，本轮没有将其加入模型比较；没有大图下载或付费 API。',
        '- 分歧示例为事后挑选，用于设计研究，不是独立证据；现有图像变化后的模型预测不等于新的人类判断。',
        '', '## 下一步应解决的问题', '', '先检验结论是否依赖背景、裁剪、读出器容量及物理/色彩混杂，并补上严格 tiny-network 对照。随后用新刺激让候选解释给出方向不同的预测，在新参与者数据上检验。若模型优势或分歧不稳健，应调整问题，而不是直接进入 Nature 子刊投稿叙事。',
        '', '通俗说明见 `WHAT_WE_ARE_STUDYING.md`。本轮没有发现或证明新的神经机制。',
        '', '## 文件和复现', '', '- `results/model_summary.csv`、`fold_metrics.csv`：整体和逐折指标。',
        '- `results/all_oof_predictions.csv`：全部图像的外层留出预测。',
        '- `results/incremental_diagnostics.csv`、`relative_mse_reduction.csv`：增量诊断。',
        '- `features/*_manifest.json`：模型仓库固定版本、权重哈希、预处理与硬件。',
        '- `results/validation.json`：校验结果。',
        '- `../protocols/pilot_v1_config.json`：首次比较前固定的主要设置。',
        '', '依次运行 `scripts/run_pilot_features.ps1`、`python scripts/evaluate_pilot.py`、`python scripts/report_pilot.py`（脚本路径相对于本基准目录）。运行器引用本机已有 GPU 环境，额外库仅安装在本项目 runtime_packages；无需修改系统环境。']
    (BASE/'pilot/PILOT_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':main()
