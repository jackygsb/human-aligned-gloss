"""Reproduce published Figure 3 row-selection exactly; do not claim retraining."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parents[1]
SRC=BASE.parent/'gloss_tinynetworks-main/data'
OUT=BASE/'mllm_material/baselines'
OUT.mkdir(parents=True,exist_ok=True)
rows=[]
for arch,kernels in [('onelayer',[1,2,4,9]),('threelayer',[9,16,32,64])]:
    for k in kernels:
        for target in ['human','groundtruth']:
            for split in ['shape','lighting']:
                for fold in range(1,13):
                    p=SRC/'networks'/f'{arch}_models'/f'{target}_kernelN{k}_{split}{fold}'
                    h=np.loadtxt(str(p/'corrs_human.csv'),delimiter=',',skiprows=1,ndmin=2)
                    g=np.loadtxt(str(p/'corrs_gt.csv'),delimiter=',',skiprows=1,ndmin=2)
                    idx=np.argmax((h if target=='human' else g)[:,0])
                    rows.append(dict(architecture=arch,kernels=k,training_target=target,split=split,fold=fold,
                                     selected_row_1based=int(idx+1),human_r=h[idx,0],physical_r=g[idx,0],
                                     provenance='author_released_correlations_author_row_selection',source=str(p)))
df=pd.DataFrame(rows);df.to_csv(OUT/'author_fig3_all_folds.csv',index=False)
summary=df.groupby(['architecture','kernels','training_target'])[['human_r','physical_r']].agg(['mean','std'])
summary.to_csv(OUT/'author_fig3_summary.csv')
fig,ax=plt.subplots(figsize=(7,5))
for target,marker in [('human','o'),('groundtruth','s')]:
    for arch,color in [('onelayer','#4477aa'),('threelayer','#cc6677')]:
        d=df[(df.training_target==target)&(df.architecture==arch)].groupby('kernels')[['human_r','physical_r']].mean()
        ax.plot(d.physical_r,d.human_r,marker=marker,color=color,label=f'{arch}, trained on {target}')
        for k,r in d.iterrows():ax.annotate(str(k),(r.physical_r,r.human_r),xytext=(4,4),textcoords='offset points',fontsize=8)
ax.set(xlabel='Correlation with physical gloss (Pearson r)',ylabel='Correlation with human judgments (Pearson r)',title='Author-released tiny-network results\nOriginal Figure 3 selection rule; not new training')
ax.legend(fontsize=8);fig.tight_layout()
for ext in ['png','svg']:fig.savefig(OUT/f'author_tiny_networks.{ext}',dpi=200)
ann=pd.read_csv(BASE/'annotations/gloss_annotations.csv')
pred=pd.read_csv(SRC/'pred_validation/prediction_3888imgs.csv')
assert len(pred)==len(ann)==3888
pd.concat([ann[['image_id','human_gloss_score','physical_pellacini_c']],pred],axis=1).to_csv(OUT/'author_released_image_predictions.csv',index=False)
metrics=[]
for c in pred:
    metrics.append(dict(model=c,n=len(pred),human_pearson=pearsonr(pred[c],ann.human_gloss_score)[0],human_spearman=spearmanr(pred[c],ann.human_gloss_score)[0],physical_pearson=pearsonr(pred[c],ann.physical_pellacini_c)[0],provenance='released_prediction_file_fold_provenance_unverified'))
pd.DataFrame(metrics).to_csv(OUT/'author_prediction_descriptive_metrics.csv',index=False)
(OUT/'README.md').write_text('''# 作者小网络对照溯源

Figure 3 的数值直接来自作者发布的 corrs CSV，逐折按原 MATLAB 代码选择训练目标相关最高的行，再读取该行另一目标的相关。该选择过程照原样复现，不等同于我们完成了独立嵌套验证或重新训练。

逐图 prediction_3888imgs.csv 原值完整保留，按源图索引连接。其具体检查点和交叉验证折映射尚未由发布文件充分确认，因此逐图相关只能暂作描述，不能冒称本项目严格留出预测。

rankN_lightprobeANDshape.mat 是按人类与物理相关排序的展示顺序（见 figS4AND5），不能误当训练折划分。单层发布 kernel.mat，三层对应目录仅见相关 CSV；不能声称三层完整推理权重已提供。
''',encoding='utf-8')
print(summary.to_string())
