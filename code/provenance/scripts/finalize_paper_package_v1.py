"""Validate exports, add condition-cluster uncertainty and package allowed research files."""
from pathlib import Path
import json,hashlib,zipfile,subprocess,sys,shutil
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
B=Path(__file__).resolve().parents[1];M=B/'mllm_material';O=M/'paper_figures_v1';F=M/'matching_full_v1'
d=pd.read_csv(O/'source_data/image_level_all3888.csv');h=d[d.cohort=='heldout_matching'].set_index('image_id');pred=pd.read_csv(O/'source_data/continuous_cross_task_oof.csv');summary=json.loads((O/'summary.json').read_text(encoding='utf-8'))
rng=np.random.default_rng(20260928);records=[]
for (split,source,target),g in pred.groupby(['split','source','target']):
    assert len(g)==3849 and g.image_id.nunique()==3849
    labels=h.loc[g.image_id];y=g.observed.to_numpy();p=g.predicted.to_numpy();c=labels[split+'_id'].to_numpy()-1
    sums=np.zeros((36,4));np.add.at(sums,c,np.column_stack([np.ones(len(y)),y,y*y,(y-p)**2]))
    w=rng.multinomial(36,np.ones(36)/36,size=4000);b=w@sums;n=b[:,0];variance=b[:,2]/n-(b[:,1]/n)**2;r2=1-(b[:,3]/n)/variance
    records.append(dict(metric='cross_task_oof_r2',split=split,source=source,target=target,estimate=float(1-np.mean((y-p)**2)/np.var(y)),ci_low=float(np.quantile(r2,.025)),ci_high=float(np.quantile(r2,.975)),method='Paired held-out-condition cluster bootstrap, 4000 draws; fixed predictions, not refitted training uncertainty'))
for factor in ['shape','illumination']:
    c=h[factor+'_id'].to_numpy();groups=[np.flatnonzero(c==v) for v in sorted(set(c))];vals=[]
    x=h.single_score.to_numpy();y=h.model_c.to_numpy()
    for _ in range(2000):
        ix=np.concatenate([groups[i] for i in rng.integers(0,36,36)]);vals.append(spearmanr(x[ix],y[ix]).statistic)
    records.append(dict(metric='single_matching_spearman',split=factor,source='single_score',target='model_c',estimate=float(spearmanr(x,y).statistic),ci_low=float(np.quantile(vals,.025)),ci_high=float(np.quantile(vals,.975)),method='Paired condition cluster bootstrap, 2000 draws'))
pd.DataFrame(records).to_csv(O/'supplement/S3_cross_task_uncertainty.csv',index=False)
conditions=[]
for dim in ['shape_id','illumination_id']:
    z=pd.read_csv(O/'source_data'/f'Fig3_{dim}.csv');z.insert(0,'factor',dim);conditions.append(z)
pd.concat(conditions).to_csv(O/'tables/Table6_shared_condition_effects.csv',index=False)
# Check pair membership, two-order reconstruction and original human-reference summaries.
p=pd.read_csv(O/'source_data/pair_cross_task_predictions.csv');assert len(p)==2000 and p.pair_id.nunique()==2000
assert np.allclose((p.order1_a+p.order2_a)/2,p.t2_probability_a)
assert all(set(p.pair_cv_fold)==set(range(5)) for _ in [0])
assert np.isclose(p.swap_consistent.mean(),.804)
for stratum,expected in [('random',.719),('discordant',.688)]:
    z=p[p.stratum==stratum];actual=(.5+np.sign(z.delta_human)*(z.t2_probability_a-.5)).mean();assert np.isclose(actual,expected)
for r in summary['continuous_cross_task']:
    g=pred[(pred.split==r['split'])&(pred.source==r['source'])];assert np.isclose(1-np.mean((g.observed-g.predicted)**2)/np.var(g.observed),r['r2'])
sources=json.loads((O/'methods/input_manifest.json').read_text(encoding='utf-8'))
assert all(hashlib.sha256((B/p).read_bytes()).hexdigest()==v for p,v in sources.items())
freeze=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True)
(O/'methods/requirements.lock.txt').write_text(freeze.stdout,encoding='utf-8')
shutil.copyfile(Path(__file__),O/'code'/Path(__file__).name)
note='''# 主论文贡献与 Results 初稿

建议工作题目：**Human-like gloss judgments emerge in a pretrained multimodal language model**。

这里的 emerge 操作化为“冻结预训练模型、未使用本任务人类评分微调，经任务提示即可表现出一致性”；没有检验模型规模阈值或训练过程的能力突现。题目使用 gloss judgments，而不是尚未验证的全部材料属性或内部表征。

## 三项可写入正文的贡献

1. **可复现的视觉问答测量流程。** 将光泽判断转化为单图评分、直接双图比较和离散参考匹配，明确图片呈现、系统指令、任务提示、响应格式、搜索控制与数值转换。公开原始响应及逐步匹配轨迹。当前没有提示词优越性消融，因此贡献是可复现方案及所得证据，不宣称最优提示词。
2. **与人类一致且具有条件依赖的光泽判断。** 在既有刺激和行为数据上检验总体一致性、物理排序冲突，以及共同形状和照明规律。
3. **跨问答任务的可预测性。** 从单图和参考匹配输出预测独立采集的双图选择，并在留出条件上相互预测两种连续输出；相关校准不读取人类评分。

## Results draft

We evaluated a frozen InternVL3.5-4B model using three visual judgment protocols: scalar gloss ratings, direct image-pair comparisons, and adaptive reference matching (Fig. 1). The dataset comprised 3,888 scene images with previously collected human gloss judgments. Direct comparisons covered 2,000 image pairs presented in both orders. Reference matching used two starting levels per image and a reconstructed 52-level reference sequence. The model received no human ratings in its prompts and was not fine-tuned on these behavioral labels.

On the 3,849 images excluded from matching-protocol development, single-image ratings correlated with human judgments at Spearman rho = 0.696, and reference matches at rho = 0.587 (Fig. 2). The matching–human association remained positive after controlling linear physical gloss (partial Pearson r = 0.524). In direct comparisons, agreement with the ordering implied by human mean ratings was 71.9% for random pairs and 68.8% for pairs whose human and physical orderings disagreed. These references were derived from existing matching judgments, rather than independently measured human binary choices.

Exploratory condition-adjusted analyses of the complete dataset revealed corresponding illumination and shape effects (Fig. 3). Human and model illumination effects correlated at rho = 0.809, and shape effects at rho = 0.504, after adjustment for physical gloss levels, viewpoint, chroma, hue and the other condition factor. These associations indicate shared condition-dependent judgment patterns without identifying a common causal mechanism.

The model's judgments were also consistent across task formats (Fig. 4). Single-image ratings and reference matches correlated at rho = 0.782. Monotonic mappings fitted only to model outputs predicted reference matches from ratings with held-out R² = 0.652 for shape splits and 0.654 for illumination splits. Reverse prediction yielded R² = 0.689 and 0.691. Without fitting human labels, the ordering of single-image ratings agreed with the independently collected pair choices in 76.7% of responses, and matching-value ordering in 79.7%, assigning half credit to predictor ties. Five-fold held-out-pair logistic prediction yielded Brier errors of 0.0925 for ratings, 0.0864 for matches and 0.0727 for both signals together, against a constant-half baseline of 0.2010. Pair identities were held out, while image identities could recur across folds.

Together, these findings support convergent, human-like gloss judgments across distinct response formats in a pretrained multimodal model. They do not establish identical human and model mechanisms or general material perception beyond gloss. Reconstructed reference images, boundary responses and the exploratory nature of the analyses define the scope of the inference. Cross-model pilot results and tiny-network comparisons are reported as supplementary evidence.

## 投稿定位

现有结果能够组成有明确行为主线的完整结果初稿，已经超出单一相关性展示。问答设计与采集方案、跨任务一致性、共同条件规律可以共同支撑投稿叙事。仅凭当前数值无法保证 Nature 系列期刊的接收；相对于已有材料知觉/多模态知觉研究的新颖性仍须在写作时论证。当前不自动新增采集或位置交换核验。
'''
(O/'methods/Contributions_and_Results_draft.md').write_text(note,encoding='utf-8')
page=O/'report.html';text=page.read_text(encoding='utf-8').split('<!-- finalized-package -->')[0]
text+='<!-- finalized-package --><h2>正文草稿与补充不确定性</h2><p><a href="methods/Contributions_and_Results_draft.md">论文贡献与英文 Results 初稿</a> · <a href="supplement/S3_cross_task_uncertainty.csv">跨任务结果的条件重采样区间</a> · <a href="tables/Table6_shared_condition_effects.csv">完整条件效应表</a></p>'
page.write_text(text,encoding='utf-8')
savefiles=[f for f in O.rglob('*') if f.is_file() and f.suffix not in ['.zip','.log'] and f.name!='artifact_manifest.json']
external=[B/p for p in sources]+[B/'scripts/build_paper_figures_v1.py',B/'scripts/finalize_paper_package_v1.py',B.parent/'gloss_tinynetworks-main/LICENSE']
external += [F/f'shared_condition_effects/{dim}_adjusted_effects.csv' for dim in ['shape_id','illumination_id']]
external += [F/'protocol.json',F/'summary.json',M/'matching_qwen3vl4b_pilot12/comparison/same12_metrics.csv',F/'incremental_tiny/overall_metrics.csv',F/'incremental_tiny/summary.json']
# Save only example stimulus images used in Fig.1; remaining scatter figures use exported numeric data.
ex=json.loads((O/'source_data/Fig1_examples.json').read_text(encoding='utf-8'))
examplepaths=ex['single']['image_paths']+ex['pair']['image_paths']+[ex['matching']['image_path'],f"mllm_material/reference_reconstruction_v3/output/reference_{ex['reference_index']:02d}.png"]
external += [B/p for p in examplepaths]
files={p.relative_to(B.parent).as_posix():p for p in savefiles+external}
manifest={n:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for n,p in files.items()}
(O/'artifact_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
with zipfile.ZipFile(O/'paper_figures_reproducibility.zip','w',zipfile.ZIP_DEFLATED) as z:
    for name,p in files.items():z.write(p,name)
    z.writestr('MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
with zipfile.ZipFile(O/'paper_figures_reproducibility.zip') as z:assert z.testzip() is None
print(pd.DataFrame(records).to_string(index=False))
print('Validated source hashes, 4 x 3849 cross-task predictions, 2000 paired choices; packaged',len(files),'files')
