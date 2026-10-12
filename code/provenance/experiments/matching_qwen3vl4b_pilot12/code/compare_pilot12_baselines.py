"""Same-image descriptive comparison, with explicit training/provenance strata."""
from pathlib import Path
import json, hashlib, html, shutil
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parents[1];ROOT=BASE/'mllm_material/matching_qwen3vl4b_pilot12';OUT=ROOT/'comparison';OUT.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def metrics(x,y):
    return dict(n=len(x),spearman=float(spearmanr(x,y).statistic),pearson=float(pearsonr(x,y).statistic),mae=float(np.mean(abs(np.array(x)-np.array(y)))))
gpt=BASE/'mllm_material/matching_subscription_cli_v1/matching_pilot12/image_summary.csv'
d=pd.read_csv(gpt).rename(columns={'model_c':'gpt_c'})
ann=pd.read_csv(BASE/'annotations/gloss_annotations.csv')
predfile=BASE.parent/'gloss_tinynetworks-main/data/pred_validation/prediction_3888imgs.csv'
pred=pd.read_csv(predfile)
assert len(pred)==3888 and set(ann.source_image_index)==set(range(1,3889))
# Explicit source index binding; never assume the annotations are in source row order.
pred['source_image_index']=np.arange(1,3889)
released=ann[['image_id','source_image_index','human_gloss_score']].merge(pred,on='source_image_index',validate='one_to_one')
for c in ['onelayer_lighting','onelayer_shape','threelayer_lighting','threelayer_shape']:
    d=d.merge(released[['image_id',c]],on='image_id',validate='one_to_one')
tinyfile=BASE/'robustness_v2/tiny_paper_lr/all_predictions.csv'
audit=[]
if tinyfile.exists():
    tiny=pd.read_csv(tinyfile)
    for p in sorted(tinyfile.parent.glob('*_audit.json')):
        a=json.loads(p.read_text());tr=set(a['train_image_ids']);te=set(a['test_image_ids']);assert not tr&te and len(tr)==3564 and len(te)==324
        factor='shape' if p.name.startswith('shape_') else 'illumination'
        groups=ann.set_index('image_id')[factor+'_id']
        assert set(groups.loc[list(tr)]).isdisjoint(set(groups.loc[list(te)]))
        audit.append(dict(file=p.name,train_n=len(tr),test_n=len(te),groups_disjoint=True,learning_rate=a['learning_rate'],selected_epoch=a['selected_epoch']))
    assert len(audit)==48
    for (split,seed),part in tiny.groupby(['split','seed']):assert len(part)==3888 and part.image_id.nunique()==3888
    for split in ['shape','illumination']:
        s=tiny[tiny.split==split].groupby('image_id').prediction.mean().rename('independent_tiny_'+split)
        d=d.merge(s,on='image_id',validate='one_to_one')
q=ROOT/'matches.csv'
if q.exists():
    qd=pd.read_csv(q);counts=qd.groupby('image_id').size();ids=counts[counts==2].index
    qm=qd[qd.image_id.isin(ids)].groupby('image_id').matching_c.mean().rename('qwen_c')
    d=d.merge(qm,on='image_id',how='left',validate='one_to_one')
d.to_csv(OUT/'same12_predictions.csv',index=False)
rows=[]
for col in ['gpt_c','internvl_c','qwen_c','physical_c','onelayer_lighting','onelayer_shape','threelayer_lighting','threelayer_shape','independent_tiny_shape','independent_tiny_illumination']:
    if col not in d:continue
    v=d[['human_c',col]].dropna()
    if len(v)<3:continue
    group='no_task_label_training' if col in ['gpt_c','internvl_c','qwen_c'] else 'physical_parameter' if col=='physical_c' else 'independent_supervised_group_OOF' if col.startswith('independent') else 'author_predictions_fold_mapping_unverified'
    rows.append(dict(model=col,provenance=group,**metrics(v.human_c,v[col])))
pd.DataFrame(rows).to_csv(OUT/'same12_metrics.csv',index=False)
# Recompute author Figure 3 using precisely the released MATLAB selection rule.
author=[];source=BASE.parent/'gloss_tinynetworks-main/data/networks'
for arch,ks in [('onelayer',[1,2,4,9]),('threelayer',[9,16,32,64])]:
    for kernels in ks:
        for target in ['human','groundtruth']:
            for split in ['shape','lighting']:
                for fold in range(1,13):
                    folder=source/f'{arch}_models'/f'{target}_kernelN{kernels}_{split}{fold}'
                    h=np.loadtxt(folder/'corrs_human.csv',delimiter=',',skiprows=1,ndmin=2);gt=np.loadtxt(folder/'corrs_gt.csv',delimiter=',',skiprows=1,ndmin=2)
                    idx=np.argmax((h if target=='human' else gt)[:,0])
                    author.append(dict(architecture=arch,kernels=kernels,training_target=target,split=split,fold=fold,selected_row=int(idx+1),human_pearson=h[idx,0],physical_pearson=gt[idx,0]))
ad=pd.DataFrame(author);ad.to_csv(OUT/'author_fig3_reproduced_folds.csv',index=False)
ad.groupby(['architecture','kernels','training_target','split'])[['human_pearson','physical_pearson']].agg(['mean','std']).to_csv(OUT/'author_fig3_by_split.csv')
full=[]
for col in pred.columns:
    if col=='source_image_index':continue
    full.append(dict(model=col,provenance='author_predictions_fold_mapping_unverified',**metrics(released.human_gloss_score,released[col])))
pd.DataFrame(full).to_csv(OUT/'author_full3888_descriptive.csv',index=False)
(OUT/'independent_tiny_split_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
fig,axes=plt.subplots(1,3,figsize=(12,3.6))
for ax,col,title in zip(axes,['gpt_c','internvl_c','qwen_c'],['GPT pilot','InternVL same images','Qwen3 same images']):
    if col in d and d[col].notna().sum()>=3:
        v=d[['human_c',col]].dropna();ax.scatter(v.human_c,v[col]);ax.set_title(title+f' (n={len(v)}, rho={spearmanr(v.human_c,v[col]).statistic:.3f})')
    else:ax.text(.5,.5,'Collection pending',ha='center',transform=ax.transAxes);ax.set_title(title)
    ax.set(xlabel='Human matched c',ylabel='Model matched c',xlim=(-.005,.16),ylim=(-.005,.16));ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(OUT/'same12_comparison.png',dpi=180);fig.savefig(OUT/'same12_comparison.svg');plt.close(fig)
note='''# 作者小网络对照：可比性审计

本次重新读取作者源文件，复现 Figure 3 的 384 个逐折结果（2 种架构、各 4 个宽度、2 种训练目标、2 种划分、12 折），同时完成与 GPT / InternVL 固定 12 张图的逐图对照。Qwen 完成后同一脚本自动加入结果，不改变样本。

## 已核实的方法

- 作者分别以人类光泽判断和物理光泽参数为监督标签。Figure 3 各折按训练目标对应的相关 CSV 第一列最大行选取，再读取同一行另一目标的相关；我们照原 MATLAB 代码复现，不能将其改写成独立嵌套验证。
- 论文报告两组 12 折验证：训练 33 个形状/照明条件（3564 图），测试 3 个未见条件（324 图）；Figure 3 汇总的是 24 个折内 Pearson 相关的均值。不能与 12 图 Spearman 或 3888 图合并相关直接作为优劣比较。
- 论文训练方法：单层 30 epochs，三层 90 epochs，Adam、初始学习率 0.01、MAE、水平翻转和随机裁剪。论文形状验证段中有“3 novel lighting environments”的用词，但同段明确全部形状轮流作为测试；按整体设计记为形状划分，同时保留原文，不将此笔误当成具体折索引。
- 作者小网络使用去背景图，当前多模态匹配使用带场景图并与重建参考比较。输入、监督标签使用方式、输出任务均不同；这是不同方法的行为预测对照，不能归因成纯架构或参数量效应。

## 原始发布文件的边界

`prediction_3888imgs.csv` 含单层/三层 × shape/lighting 四列。本次通过 source_image_index 显式连接原始行号，保证同图对齐。当前本地发布包未找到该文件逐行对应的检查点、核数、训练/测试成员完整映射，不能仅凭列名断言每行是本项目已核验的 OOF 预测。全部同图指标标记为描述性，保留四列，不选最高列作为唯一基线。

本地仓库 README 主要承诺数据处理及图表生成。单层折目录可见 kernel.mat 和相关 CSV；卷积核并不等于含回归头的完整推理检查点。当前未找到可独立重跑所有作者原始训练的完整 Python 训练入口和三层完整检查点。因此“已复现作者发布的结果”与“重新训练作者原始网络”明确区分。

## 另有可审计的独立小网络对照

本项目已有单核小网络重训，学习率 0.01 版本保存了 2 seeds × 2 划分 × 12 折共 48 份训练/测试审计和检查点。本次逐折核验图像不重叠、形状/照明组不重叠、3564/324 数量正确；每图每 seed 每划分仅出现一次测试预测。终末 epoch 30 固定，未以测试分数选检查点。

该网络仍是独立同族实现，目标缩放、增强暴露量等与作者不完全相同，不能冒称作者精确复现。两 seed 的测试预测平均后加入同图表，并单独标记为 supervised group OOF，不与作者发布预测混名。

## 12 张图的用途

足以检查完整匹配能否运行、同图效应方向、粗略相关和重复可靠性；不足以精确估计模型差异或 36 形状 × 36 照明效应。24 次匹配仍只有 12 个不同刺激，不是 24 个独立刺激。后续应根据目标效应估计精度决定样本量，不能以本轮显著性作为充分样本量论证。

## 文件与复现

- same12_predictions.csv：固定图像逐图的人类、物理值及各模型预测。
- same12_metrics.csv：相同刺激、相同统计定义的描述指标，并列明来源分组。
- author_fig3_reproduced_folds.csv / author_fig3_by_split.csv：原始逐折规则的重算结果。
- author_full3888_descriptive.csv：作者发布预测的全量描述结果。
- independent_tiny_split_audit.json：48 个独立重训折的核验。
- 同目录 manifest.json：分析输入与代码的 SHA256。

从项目根目录运行 mllm_runtime/Scripts/python.exe scripts/compare_pilot12_baselines.py 可更新所有表和图。原始作者资源位于项目同级 gloss_tinynetworks-main。原始数据遵从作者许可证；公开包不含账户认证、密钥或浏览器资料。
'''
(OUT/'小网络对照审计.md').write_text(note,encoding='utf-8')
files=[gpt,predfile,BASE/'annotations/gloss_annotations.csv',BASE/'scripts/compare_pilot12_baselines.py',BASE.parent/'gloss_tinynetworks-main/fig3_model_comparison.m',BASE.parent/'gloss_tinynetworks-main/README.md',BASE/'scripts/train_tiny_control.py',BASE/'protocols/gloss_group_folds.csv']
if tinyfile.exists():files.append(tinyfile)
(OUT/'manifest.json').write_text(json.dumps({str(p.relative_to(BASE.parent)):sha(p) for p in files},indent=2),encoding='utf-8')
report='<meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>同图比较与小网络审计</title><style>body{font:17px system-ui;margin:35px;line-height:1.7}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>固定 12 图：模型比较与小网络可比性审计</h1><p>同图指标已统一；模型训练与输入条件不同，作者发布逐图预测的折映射尚未核实，不能据此作严格性能排名。</p><img src="same12_comparison.png">'+pd.DataFrame(rows).round(4).to_html(index=False)+'<details><summary>展开小网络方法与划分审计</summary><pre>'+html.escape(note)+'</pre></details><p><a href="same12_predictions.csv">逐图数据</a> · <a href="same12_metrics.csv">统计表</a> · <a href="小网络对照审计.md">审计报告</a> · <a href="../progress.html">Qwen采集进度</a></p>'
report+='<p><a href="../reproducibility_snapshot.zip">本地复现快照 ZIP（采集完成后自动更新）</a> · <a href="../release_manifest.json">快照状态和哈希</a></p>'
(OUT/'report.html').write_text(report,encoding='utf-8')
print(pd.DataFrame(rows).round(4).to_string(index=False))
