from pathlib import Path
import json,csv,collections,hashlib
import numpy as np
from scipy.stats import spearmanr,pearsonr
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'mllm_material/matching_paper_pilot'
s=(BASE/'scripts/analyze_matching_pilot.py').read_text(encoding='utf-8').replace('matching_pilot_v1','matching_paper_pilot')
s=s.replace("decisions=[json.loads(l) for l in (OUT/'decisions.jsonl').read_text().splitlines()]","decisions=[json.loads(l) for l in (OUT/'decisions.jsonl').read_text().splitlines() if json.loads(l)['phase']=='main']")
for a,b in [('low_start_c','session1_c'),('high_start_c','session2_c'),('high_minus_low_start_mean_c','session2_minus_session1_mean_c'),('Low-start matched c','Session 1 matched c'),('High-start matched c','Session 2 matched c')]:s=s.replace(a,b)
exec(compile(s,str(BASE/'scripts/generated_matching_paper_analysis.py'),'exec'),{'__file__':str(BASE/'scripts/generated_matching_paper_analysis.py')})
rows=list(csv.DictReader((OUT/'image_summary.csv').open(encoding='utf-8-sig')))
h=np.array([float(r['human_c']) for r in rows]);g=np.array([float(r['physical_c']) for r in rows]);m=np.array([float(r['model_c']) for r in rows]);n=len(rows)
summary=json.loads((OUT/'summary.json').read_text())
records=[json.loads(l) for l in (OUT/'all_trials.jsonl').read_text().splitlines()]
summary['practice_trials']=sum(r['phase']=='practice' for r in records)
summary['catch_results']=[{'trial_id':r['trial_id'],'passed':r['passed'],'raw':r['raw']} for r in records if r['phase']=='catch']
summary['n_repeated_images_at_boundary']=sum(int(r['boundary_runs'])>0 for r in rows)
summary['repetition_caveat']='same deterministic model, not independent human observers; repeat reliability tests initial state/order only'
X=np.c_[np.ones(n),g];hr=h-X@np.linalg.lstsq(X,h,rcond=None)[0];mr=m-X@np.linalg.lstsq(X,m,rcond=None)[0]
rng=np.random.default_rng(47);ci=[]
for _ in range(3000):
    ix=rng.integers(0,n,n);xx=X[ix];hh=h[ix]-xx@np.linalg.lstsq(xx,h[ix],rcond=None)[0];mm=m[ix]-xx@np.linalg.lstsq(xx,m[ix],rcond=None)[0]
    if np.std(mm)>0:ci.append(float(pearsonr(hh,mm).statistic))
summary['partial_pearson_bootstrap95']=np.quantile(ci,[.025,.975]).tolist()
summary['spearman_permutation_p_two_sided']=(1+sum(abs(spearmanr(h,rng.permutation(m)).statistic)>=abs(summary['human_model_spearman']) for _ in range(4999)))/5000
old={}
for line in (BASE/'mllm_material/responses/internvl35_4b/full.jsonl').open():
    r=json.loads(line)
    if r['task']=='T1' and r['condition']=='main' and r['parse_status']=='valid':old[r['image_ids'][0]]=r['parsed']['glossiness']
scores=np.array([old[r['image_id']] for r in rows])
summary['existing_single_image_human_rho_same_36']=float(spearmanr(h,scores).statistic)
summary['matching_vs_existing_single_image_rho']=float(spearmanr(m,scores).statistic)
summary['protocol_sha256']=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()
summary['model_manifest_sha256']=hashlib.sha256((BASE/'models/internvl35_4b/manifest.json').read_bytes()).hexdigest()
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
lo,hi=summary['human_model_spearman_bootstrap95']
doc=f'''# InternVL参考图匹配先导结果

## 已完成

36张原始带背景测试图，覆盖36种形状及36种光照。两轮独立随机顺序、每次随机非端点起始档位（2–51），3次练习，每轮1次无图注意检查。共72次主匹配、{summary['n_decisions']}次主试验逐轮判断。两次注意检查均通过：{all(r['passed'] for r in summary['catch_results'])}。

模型：InternVL3.5-4B，BF16，确定性解码。每轮以Image A为测试图、Image B为参考图；每张512×512原图编码为一个448×448图块。用同一重建v3参考序列。此次为适配先导，不称完全复现原实验。

## 人类一致性证据

- 模型匹配阈值与人类评分：Spearman rho={summary['human_model_spearman']:.4f}，按图像bootstrap的95%区间[{lo:.4f}, {hi:.4f}]，双侧置换p={summary['spearman_permutation_p_two_sided']:.4f}。
- Pearson r={summary['human_model_pearson']:.4f}。
- 去除测试物理c的线性趋势后，残差Pearson r={summary['partial_pearson_controlling_linear_physical_c']:.4f}，bootstrap区间{summary['partial_pearson_bootstrap95']}。它是线性控制下的探索性关联，不是已证明所有人类错觉均被复现。
- 两轮匹配排序rho={summary['repeat_spearman']:.4f}。这是同一确定性模型对起点及顺序的稳健性，不是独立被试间一致性。
- 相同36图上已有单图评分与人类rho={summary['existing_single_image_human_rho_same_36']:.4f}；当前匹配阈值与已有单图评分rho={summary['matching_vs_existing_single_image_rho']:.4f}。

## 额外采集的数据

保留每轮两张输入路径、参考档位、上下搜索边界、完整原始回答、动作方向、模型推理耗时；每次匹配记录起点、终点、完整路径、停止原因。另有练习与catch记录。不将推理耗时解释成人类反应时。

## 方法开发过程（全部保留）

matching_pilot_v1采用INCREASE/DECREASE/MATCH指令；同一参考球控制只通过3/9，未作为主证据。改为A/B/SAME直接比较后，同样控制通过9/9，随后程序映射为调整动作。选用提示词依据控制任务结果，而非人类相关性。

matching_pilot_v2在相同36图上检验高/低起点：rho=0.4964，重复rho=0.9983。当前matching_paper_pilot按原文要求改为两轮独立随机起点、独立随机顺序，并加入练习及catch。两轮开发与复核使用相同36张图，并非独立确认集。

## 必须保留的适配说明

- 原作每组72独有图+12共同图，本次是36图先导。
- 参考球是按图示重建，实际几何、底色、显示转换尚未证明等同作者原刺激，不能直接把两种c的数值差解释成人类/模型错觉差异。
- 通过有序图像占位符呈现两图，没有真的展示鼠标滑块。模型回答哪张更光亮，程序用二分搜索估计判断切换点；这属于自适应比较得到的匹配阈值，不是每次模型主动宣布相等。
- 停止原因：{summary['stop_counts']}。边界输出为受限估计，保留在主结果并明确标注，不作为精确匹配成功。
- 屏幕尺寸校准、暗室、一分钟视觉适应和人类反应时排除阈值不适用于该无状态模型接口。未将模型快速响应按人类标准排除。
- 每种形状及光照仅选1图，不能从本轮估计独立形状效应或光照效应。

## 输出

report.html、pilot_results.png/svg、image_summary.csv、summary.json、decisions.jsonl、trials.jsonl、all_trials.jsonl、protocol.json、schedule.json。
'''
(OUT/'先导结果说明.md').write_text(doc,encoding='utf-8')
report=OUT/'report.html';html=report.read_text(encoding='utf-8').replace('模型匹配','模型匹配阈值')
html=html.replace('<h1>InternVL · 参考图匹配试验</h1>',f'<h1>InternVL · 参考图匹配先导</h1><p>三次练习、两轮随机顺序与随机内部起点、两次注意检查均已完成。原始鼠标调节适配为模型两图比较和程序搜索。</p><p>去除物理c线性趋势后，模型—人类残差相关 r={summary["partial_pearson_controlling_linear_physical_c"]:.3f}。同一确定性模型两轮排序一致性 rho={summary["repeat_spearman"]:.3f}。<a href="先导结果说明.md">完整方法和结果说明</a></p>')
report.write_text(html,encoding='utf-8');print(json.dumps(summary,indent=2))
