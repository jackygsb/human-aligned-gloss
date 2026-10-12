from pathlib import Path
import json,html
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'mllm_material/matching_gemini31pro_v1'
s=json.loads((OUT/'status.json').read_text());rs=[json.loads(l) for l in (OUT/'decisions.jsonl').read_text().splitlines()]
trials=[json.loads(l) for l in (OUT/'trials.jsonl').read_text().splitlines()] if (OUT/'trials.jsonl').exists() else []
main=[r for r in rs if r['phase']=='main'];mean=sum(r['observed_balance_delta'] for r in main)/len(main) if main else None
estimate={}
if mean:
 for n in [36,360,3888]:estimate[str(n)]={'images':n,'matches':2*n,'points_estimate_range':[round(n*2*6*mean),round(n*2*7*mean)]}
out={'status':s,'completed_matching_trials':trials,'main_request_mean_observed_points':mean,'budget_projection':estimate,'note':'Pilot feasibility only. No human-alignment statistic estimated. Account balance deltas may include concurrent use; costs vary by input and reasoning length.'}
(OUT/'pilot_summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
text='''# 第二模型验证：Gemini 3.1 Pro

本轮选择来自 Google 的商用视觉模型，与 InternVL 的模型家族不同。沿用相同的参考序列、测试图、提示词、A/B角色、起始档位、最多7步搜索及c转换；上传512像素原图，由服务端编码，无法声称与InternVL的448像素内部编码完全一致。温度请求为0，但云端重复运行不保证逐字确定性。不使用人类评分作为模型输入。

先导的4个参考自检和3张测试图的最多6次匹配，均在采集前固定；本次消费设置约2500积分目标，达到2200后不再启动新请求。预算不足时保留未完成试次的逐步回答，不将其计为完成匹配。

## 正式分析目标（尚未完成）

1. 沿用InternVL的全部3888张排程、每图两次，单独标记39张开发图；报告3849张匹配验证图的人类Spearman相关、重复一致性及边界比例。
2. 使用相同加性模型：50档物理光泽、视角、色度、色相、形状、照明；比较人类与Gemini的36个形状效应和36个照明效应。
3. 对另一条件进行聚类重采样，同时置信带标出共同升高和降低的条件；报告与InternVL效应向量的对应关系。
4. 全部边界值和不一致结果保留；不依据相关性选择图像、停止位置或改变提示词。接口先导不作为显著性证据。

## 当前采集与预算

'''+json.dumps(out,ensure_ascii=False,indent=2)+'''

预算估计来自本次请求的账户余额差，不等于报价；实际消耗随输入、推理长度与其他账户使用变化。正式全量尚未开始；当前积分不足时不自动充值。

接口依据：https://creator.poe.com/docs/external-applications/openai-compatible-api 。不依赖该兼容接口可能忽略的专有推理参数；原始响应与usage已保存。
'''
(OUT/'第二模型复核方案与先导结果.md').write_text(text,encoding='utf-8')
(OUT/'report.html').write_text('<meta charset="utf-8"><title>Gemini 第二模型验证</title><style>body{font:16px system-ui;max-width:1000px;margin:40px auto;line-height:1.7}pre{white-space:pre-wrap}</style><h1>Gemini 3.1 Pro · 第二模型验证</h1><pre>'+html.escape(text)+'</pre><a href="decisions.jsonl">全部响应</a> · <a href="pilot_summary.json">统计与预算</a>',encoding='utf-8')
print(json.dumps(out,indent=2))
