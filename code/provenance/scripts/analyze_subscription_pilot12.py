from pathlib import Path
import json, csv, itertools
import numpy as np
from scipy.stats import rankdata, spearmanr, pearsonr
BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'mllm_material/matching_subscription_cli_v1/matching_pilot12'
rows=list(csv.DictReader((OUT/'image_summary.csv').open(encoding='utf-8-sig')))
arr=lambda k:np.array([float(r[k]) for r in rows])
h,m,p,i=map(arr,['human_c','model_c','physical_c','internvl_c'])
r1,r2=map(arr,['repeat1_c','repeat2_c'])
rho=lambda x,y:float(spearmanr(x,y).statistic)
rng=np.random.default_rng(20260928)
hr=rankdata(h);mr=rankdata(m);hc=hr-hr.mean();mc=mr-mr.mean()
perms=np.array([rng.permutation(mc) for _ in range(99999)])
null=(perms@hc)/np.linalg.norm(hc)/np.linalg.norm(mc)
pval=float((1+(np.abs(null)>=abs(rho(h,m))-1e-12).sum())/100000)
X=np.column_stack([np.ones(len(h)),rankdata(p)])
res=lambda a:rankdata(a)-X@np.linalg.lstsq(X,rankdata(a),rcond=None)[0]
bootdiff=[]
for _ in range(10000):
    ix=rng.integers(0,len(h),len(h))
    if all(np.unique(x[ix]).size>1 for x in [h,m,i]):bootdiff.append(rho(h[ix],m[ix])-rho(h[ix],i[ix]))
matches=list(csv.DictReader((OUT/'matches.csv').open(encoding='utf-8-sig')))
results=[json.loads(f.read_text()) for f in sorted(OUT.glob('batch_*/result.json'))]
assert len(rows)==12 and len(matches)==24 and len({t['trial_id'] for t in matches})==24
assert all(r['valid'] and r['tool_items']==0 for r in results)
nonend=(m>0)&(m<.1487)
d=dict(n_images=12,n_matches=24,requests=len(results),comparisons=sum(len(r['answer']) for r in results),all_requests_valid=True,
 human_model_spearman=rho(h,m),human_model_pearson=float(pearsonr(h,m).statistic),permutation_p_two_sided=pval,permutations=99999,
 partial_rank_correlation_control_linear_physical_rank=float(pearsonr(res(h),res(m)).statistic),
 human_physical_spearman=rho(h,p),model_physical_spearman=rho(m,p),human_internvl_same_images_spearman=rho(h,i),
 gpt_minus_internvl_rho_difference=rho(h,m)-rho(h,i),paired_bootstrap_rho_difference_ci95=np.quantile(bootdiff,[.025,.975]).tolist(),
 leave_one_image_out_rho_range=[min(rho(np.delete(h,j),np.delete(m,j)) for j in range(12)),max(rho(np.delete(h,j),np.delete(m,j)) for j in range(12))],
 endpoint_exclusion_sensitivity_n=int(nonend.sum()),endpoint_exclusion_sensitivity_rho=rho(h[nonend],m[nonend]),
 pairwise_order_agreement=sum((h[a]-h[b])*(m[a]-m[b])>0 for a,b in itertools.combinations(range(12),2)),pairwise_order_total=66,
 repeat_spearman=rho(r1,r2),repeat_mean_absolute_difference=float(np.mean(abs(r1-r2))),repeat_mean_absolute_difference_reference_steps=float(np.mean(abs(r1-r2))/(.1487/51)),
 model_human_mean_absolute_error=float(np.mean(abs(m-h))),model_human_mean_signed_error=float(np.mean(m-h)),
 boundary_stop_runs=sum(t['stop_reason']=='boundary' for t in matches),endpoint_runs=sum(int(t['final_index']) in (1,52) for t in matches),explicit_same_runs=sum(t['stop_reason']=='explicit_match' for t in matches),
 total_request_minutes=sum(r['elapsed_seconds'] for r in results)/60,
 interpretation='Exploratory n=12; pairwise orders are derived from matching values, not 66 independent collected trials; rank residual control is linear only; reconstructed references limit absolute-scale interpretation; no shape/illumination effect validation at this sample size.')
d={k:v.item() if isinstance(v,np.generic) else v for k,v in d.items()}
(OUT/'analysis.json').write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(d,indent=2,ensure_ascii=False))
note='''# 完整匹配小样本：结果解读

已完成 12 张固定测试图、每图 2 次匹配，24 次完整匹配共包含 142 个比较回答、41 次请求，推理请求耗时约 83 分钟。全部请求有效且无工具调用。使用 gpt-6-astra / low，经 ChatGPT 订阅 CLI 调用，与网页 Pro 是独立实验条件。

| 指标 | 结果 |
|---|---:|
| GPT 匹配值与人类评分的 Spearman 相关 | 0.867 |
| Pearson 相关 | 0.872 |
| 双侧置换检验（99,999 次） | p = 0.00057 |
| 两次重复的 Spearman 相关 | 0.958 |
| 人类评分与物理光泽值的 Spearman 相关 | 0.063 |
| 控制物理光泽秩的线性影响后的偏相关 | 0.883 |
| 同一批图上 InternVL 与人类的 Spearman 相关 | 0.606 |

这批数据支持的核心发现是：模型通过观察测试物体与参考物体、逐步选择更有光泽的一方，得到的最终匹配值与人类判断高度一致。在这 12 张图上，一致性并不能由物理光泽值的简单排序解释。这是 human-like gloss perception 的初步正向证据；尚不能凭此认定已复现共同的形状、照明效应或证明能力如何涌现。

例如，morimoto_1490 的物理 c 为 0.0554，但人类匹配值仅为 0.00219，GPT 为 0；morimoto_1394 的物理 c 更低（0.0292），人类却觉得更有光泽（0.0940），GPT 也给出更高的匹配值（0.1225）。这是同向偏离物理参数排序的直观例子。全样本数据均保留，例子不替代整体统计。

逐一去掉任意一张图后，人类一致性仍为 0.827–0.936。仅作为敏感性分析，去掉均值位于参考序列两端的两张图后，剩余 10 图相关为 0.770。12 图推导出的 66 个两两排序中，57 个与人类一致（86.4%）；这些并非独立采集的 66 次双图试验。

两次重复平均相差 0.0109，约 3.75 个参考档位，因此高排序稳定性不等于数值完全重现。24 次匹配中 2 次明确回答 SAME，5 次边界终止，17 次搜索区间耗尽；6 次最终值落在序列端点。匹配值主要是搜索得到的阈值，不能都称为模型明确认定相等。GPT 与人类的平均绝对差为 0.0210；参考图是重建版本，绝对尺度的一致性需谨慎解释。

人类一致性的探索性 bootstrap 95% 区间为 [0.469, 1.000]，样本仍小。GPT 的相关数值高于同图 InternVL，但两者差值的配对 bootstrap 95% 区间为 [-0.128, 0.665]，目前不足以宣称显著优于 InternVL。

建议下一步保持本轮提示词和匹配规则不变，在新增且未用于本轮分析的图上进行较大样本验证，并在形状、照明条件间平衡采样。12 图作为先导样本单列；新增样本先验证总体人类一致性，再检验预先规定的形状、照明效应。当前仅完成分析，没有自动扩大采集。
'''
(OUT/'结果解读.md').write_text(note,encoding='utf-8')
page=OUT/'report.html';content=page.read_text(encoding='utf-8')
marker='<!-- completed-analysis -->'
content=content.split(marker)[0]
import html
content+=marker+'<section><h2>完整结果解读</h2><p><b>人类一致性 ρ = 0.867；重复一致性 ρ = 0.958；全部 24 次匹配完成。</b></p><p>控制物理光泽秩的线性影响后，相关仍为 0.883；逐一去掉任意一张图，相关保持在 0.827–0.936。支持初步人类式光泽判断，仍需扩大样本验证。</p><details><summary>展开完整分析及下一步建议</summary><pre>'+html.escape(note)+'</pre></details><p><a href="analysis.json">补充统计数据</a> · <a href="结果解读.md">下载结果解读</a></p></section>'
page.write_text(content,encoding='utf-8')
