"""Final fixed-sample summary; retain ceiling effects and provenance distinctions."""
from pathlib import Path
import json,html
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,pearsonr,rankdata
B=Path(__file__).resolve().parents[1];R=B/'mllm_material/matching_qwen3vl4b_pilot12';O=R/'comparison'
d=pd.read_csv(O/'same12_predictions.csv');t=pd.read_csv(R/'matches.csv')
assert len(t)==24 and t.trial_id.nunique()==24 and len(d)==12 and d.qwen_c.notna().all()
raw=[json.loads(p.read_text(encoding='utf-8')) for p in (R/'raw').glob('*/result.json')]
assert len(raw)==int(t.steps.sum())==147 and all(r['valid'] for r in raw)
pivot=t.pivot(index='image_id',columns='repetition',values='matching_c')
h=d.human_c.to_numpy();q=d.qwen_c.to_numpy();phys=d.physical_c.to_numpy()
rho=lambda a,b:float(spearmanr(a,b).statistic)
rng=np.random.default_rng(20260928)
hr=rankdata(h);qr=rankdata(q);hr-=hr.mean();qr-=qr.mean()
null=np.array([rng.permutation(qr) for _ in range(99999)])@hr/(np.linalg.norm(hr)*np.linalg.norm(qr))
boots=[]
for _ in range(10000):
    ix=rng.integers(0,12,12)
    if np.unique(h[ix]).size>1 and np.unique(q[ix]).size>1:boots.append(rho(h[ix],q[ix]))
X=np.column_stack([np.ones(12),rankdata(phys)])
res=lambda a:rankdata(a)-X@np.linalg.lstsq(X,rankdata(a),rcond=None)[0]
summary=dict(completed_images=12,completed_matches=24,valid_comparisons=147,human_spearman=rho(h,q),human_pearson=float(pearsonr(h,q).statistic),permutation_p_two_sided=float((1+np.sum(np.abs(null)>=abs(rho(h,q))-1e-12))/100000),bootstrap95_image=np.quantile(boots,[.025,.975]).tolist(),repeat_spearman=rho(pivot[1],pivot[2]),exact_repeat_matches=int((pivot[1]==pivot[2]).sum()),mean_absolute_error=float(np.mean(abs(h-q))),mean_signed_model_minus_human=float(np.mean(q-h)),mean_at_upper_bound_images=int(np.isclose(q,.1487).sum()),upper_bound_runs=int((t.final_index==52).sum()),boundary_stop_runs=int((t.stop_reason=='boundary').sum()),explicit_same_runs=int((t.stop_reason=='explicit_match').sum()),partial_rank_correlation_control_physical_rank=float(pearsonr(res(h),res(q)).statistic),total_inference_seconds=sum(r['elapsed_seconds'] for r in raw),limitations='12 selected-before-inference pilot stimuli, reconstructed references, many upper-bound ties; deterministic repeat reliability does not imply accurate continuous scale or generalization.')
(O/'qwen_final_statistics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
note=f'''# 三模型与小网络：本轮总体结果

本轮固定实验完成。Qwen3-VL-4B-Instruct：12 图、24 次完整匹配、147 个有效比较回答。GPT：相同 12 图、24 次匹配、142 个比较回答。InternVL 同图结果取自已完成的 3888 图全量实验。

## 相同 12 张图的结果

| 方法 | 与人类 Spearman 相关 | 与人类 Pearson 相关 | 身份 |
|---|---:|---:|---|
| GPT | 0.867 | 0.872 | 无本任务行为标签训练；四对图批次上下文 |
| Qwen3-VL-4B-Instruct | 0.694 | 0.733 | 无本任务行为标签训练；每次一对图 |
| InternVL3.5-4B | 0.606 | 0.684 | 无本任务行为标签训练；每次一对图 |
| 物理 c | 0.063 | 0.302 | 参数基线 |
| 作者单层：照明 / 形状列 | 0.510 / 0.510 | 0.485 / 0.517 | 发布逐图预测，折映射未完整核实 |
| 作者三层：照明 / 形状列 | 0.881 / 0.769 | 0.808 / 0.750 | 发布逐图预测，折映射未完整核实 |
| 独立重训单核：形状 / 照明留出 | 0.406 / 0.308 | 0.351 / 0.450 | 两 seed 平均的分组 OOF；非作者精确复现 |

三种多模态模型在相同固定刺激上都表现出正向的人类判断一致性，支持跨模型家族的初步光泽知觉对齐。结果不能概括为所有模型具有同样准确的数值尺度，也不能由这一小样本作模型优劣的确定排名。

## Qwen 的结果具体意味着什么

Qwen 人类一致性 rho={summary['human_spearman']:.3f}，探索性双侧置换 p={summary['permutation_p_two_sided']:.5f}，图像 bootstrap 95% 区间 [{summary['bootstrap95_image'][0]:.3f}, {summary['bootstrap95_image'][1]:.3f}]。控制物理参数秩的线性影响后相关为 {summary['partial_rank_correlation_control_physical_rank']:.3f}，仅是小样本调整关联。

两次重复排序相关 {summary['repeat_spearman']:.3f}，12 图中 {summary['exact_repeat_matches']} 图匹配值完全相同。但 {summary['mean_at_upper_bound_images']} 图的平均匹配值达到 52 档上限，24 次中 {summary['upper_bound_runs']} 次落在最高档；其中 {summary['boundary_stop_runs']} 次按控制器边界终止。这些并列高值压缩了高光泽图像间的可分辨差异，也抬高了重复一致性。Qwen 与人类平均绝对差为 {summary['mean_absolute_error']:.4f}，总体偏高 {summary['mean_signed_model_minus_human']:.4f}。因此它提供的是正向排序与跨家族支持，尚不是准确重建完整连续匹配尺度的证据。保留全部边界结果，不用删除上限样本来重新定义主结果。

## 小网络对照完成范围

已完成作者发布 Figure 3 的 384 个逐折相关记录重算，以及同一 12 图上的四列单层/三层预测对照。另已核验独立重训单核小网络的 48 个训练/测试划分：图像与相应形状/照明分组不交叉。没有在本轮重新训练作者全部网络。作者发布逐图预测与具体检查点/折成员映射仍未确认，不能把表中的作者预测直接称为已审计的 OOF；详细说明见小网络对照审计.md。

## 论文主线应建立在哪里

当前更有依据的表述是：预训练多模态模型无需用本数据集行为标签专门训练，就能通过参考比较表现出与人类相关的光泽判断；InternVL 全量数据进一步显示形状和照明相关的共同判断规律。

InternVL 全量 3888 图已完成；主要匹配留出子集 3849 图的人类相关约 0.587，控制物理 c 后 Pearson 相关约 0.524。全量调整后的共同照明效应相关 0.809、形状效应相关 0.504，是当前科学规律分析的主体。GPT 与 Qwen 的 12 图结果提供补充跨家族证据，尚未复核整套条件效应。

目前不支持“全面胜过小网络”“所有家族已经复现同样的形状/照明效应”或“证明了相同的心理/神经机制”。自然涌现暂指无需本任务专门标签训练即可表现出的能力，不是规模阈值或训练轨迹证据。

下一优先级：利用已有全量输出、小网络和图像特征，做留出条件下的增量解释分析，判断大模型能解释哪些小网络未解释的人类变化。先把行为一致性、条件规律、解释增益连成证据链，再决定是否以及按什么精度目标扩大第二模型数据。当前没有新增采集。
'''
(O/'总体结果与结论.md').write_text(note,encoding='utf-8')
page=O/'report.html';text=page.read_text(encoding='utf-8').split('<!-- final-synthesis -->')[0]
text+='<!-- final-synthesis --><section><h2>全部采集完成：总体结论</h2><p>同图人类相关：GPT 0.867；Qwen3 0.694；InternVL 0.606。三模型均呈正向一致性；Qwen 7/12 图达到最高参考档，连续数值尺度仍有明显上限效应。</p><details><summary>展开总体分析</summary><pre>'+html.escape(note)+'</pre></details><a href="总体结果与结论.md">下载总体结论</a> · <a href="qwen_final_statistics.json">Qwen补充统计</a></section>'
page.write_text(text,encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
