from pathlib import Path
import os,json,csv,collections
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'mllm_material/matching_pilot_v2'
os.environ['MPLCONFIGDIR']=str(BASE/'mllm_runtime/matplotlib_cache')
import numpy as np
from scipy.stats import spearmanr,pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
labels={r['image_id']:r for r in csv.DictReader((BASE/'annotations/gloss_annotations.csv').open(encoding='utf-8-sig'))}
trials=[json.loads(l) for l in (OUT/'trials.jsonl').read_text().splitlines()]
decisions=[json.loads(l) for l in (OUT/'decisions.jsonl').read_text().splitlines()]
by=collections.defaultdict(list)
for r in trials:by[r['image_id']].append(r)
rows=[]
for key,rs in by.items():
    if len(rs)!=2 or not all(x['valid'] for x in rs):continue
    rs=sorted(rs,key=lambda x:x['repetition']);a=labels[key]
    cs=[r['matching_c'] for r in rs]
    rows.append(dict(image_id=key,human_c=float(a['human_gloss_score']),physical_c=float(a['physical_pellacini_c']),model_c=float(np.mean(cs)),
      low_start_c=cs[0],high_start_c=cs[1],repeat_abs_difference=float(abs(cs[1]-cs[0])),shape_id=a['shape_id'],illumination_id=a['illumination_id'],
      boundary_runs=sum(r['stop_reason']=='boundary' for r in rs),explicit_match_runs=sum(r['stop_reason']=='explicit_match' for r in rs),
      human_sem=float(a['human_gloss_sem']),n_observers=int(a['n_observers'])))
with (OUT/'image_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
h=np.array([r['human_c'] for r in rows]);g=np.array([r['physical_c'] for r in rows]);m=np.array([r['model_c'] for r in rows])
a=np.array([r['low_start_c'] for r in rows]);b=np.array([r['high_start_c'] for r in rows])
def rho(x,y):return float(spearmanr(x,y).statistic) if np.std(x)>0 and np.std(y)>0 else None
rng=np.random.default_rng(20260928);boots=[]
for _ in range(3000):
    ix=rng.integers(0,len(h),len(h));r=rho(h[ix],m[ix])
    if r is not None:boots.append(r)
stops=dict(collections.Counter(r['stop_reason'] for r in trials));acts=dict(collections.Counter(r['action'] for r in decisions))
# Descriptive partial correlation after removing a linear physical-c trend.
design=np.c_[np.ones(len(g)),g]
hr=h-design@np.linalg.lstsq(design,h,rcond=None)[0];mr=m-design@np.linalg.lstsq(design,m,rcond=None)[0]
summary={'n_images':len(rows),'n_trials':len(trials),'n_decisions':len(decisions),
 'human_model_spearman':rho(h,m),'human_model_spearman_bootstrap95':np.quantile(boots,[.025,.975]).tolist() if boots else None,
 'human_model_pearson':float(pearsonr(h,m).statistic) if np.std(m)>0 else None,
 'human_physical_spearman':rho(h,g),'model_physical_spearman':rho(m,g),
 'repeat_spearman':rho(a,b),'repeat_median_abs_c':float(np.median(abs(a-b))),
 'high_minus_low_start_mean_c':float(np.mean(b-a)),
 'partial_pearson_controlling_linear_physical_c':float(pearsonr(hr,mr).statistic) if np.std(mr)>0 else None,
 'stop_counts':stops,'action_counts':acts,'boundary_trial_fraction':stops.get('boundary',0)/len(trials),
 'explicit_match_fraction':stops.get('explicit_match',0)/len(trials),
 'distinct_final_indices':len(set(r['final_presented_index'] for r in trials)),
 'median_decision_latency_s':float(np.median([r['elapsed_s'] for r in decisions])),
 'interpretation':'exploratory; boundary outputs are censored; c scales not proven equivalent; confidence interval resamples images, not observers'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
boundary=np.array([r['boundary_runs']>0 for r in rows])
axes[0].scatter(h,m,c=np.where(boundary,'#cf7343','#4969ae'));axes[0].set(xlabel='Human matched c',ylabel='Model matched c',title=f"Human alignment: rho={summary['human_model_spearman']:.3f}")
axes[1].scatter(g,h,label='Human',alpha=.75);axes[1].scatter(g,m,label='Model',alpha=.75);axes[1].set(xlabel='Test physical c',ylabel='Matched c',title='Physical vs matched appearance');axes[1].legend()
axes[2].scatter(a,b,c='#4969ae');axes[2].set(xlabel='Low-start matched c',ylabel='High-start matched c',title='Different initial reference levels')
for ax in axes:ax.plot([0,.1487],[0,.1487],color='#bbb',ls='--',lw=1);ax.set_xlim(-.004,.153);ax.set_ylim(-.004,.153);ax.grid(alpha=.15)
fig.savefig(OUT/'pilot_results.png',dpi=170);fig.savefig(OUT/'pilot_results.svg');plt.close(fig)
table=''.join(f"<tr><td>{r['image_id']}</td><td>{r['human_c']:.4f}</td><td>{r['physical_c']:.4f}</td><td>{r['model_c']:.4f}</td><td>{r['low_start_c']:.4f} / {r['high_start_c']:.4f}</td><td>{r['boundary_runs']}</td></tr>" for r in rows)
html=f'''<!doctype html><meta charset="utf-8"><title>InternVL参考匹配试验</title><style>body{{font-family:system-ui;max-width:1150px;margin:35px auto;padding:20px;color:#243040}}img{{width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}p{{line-height:1.8}}</style><h1>InternVL · 参考图匹配试验</h1><p>{len(rows)}张测试图，{len(trials)}次匹配，{len(decisions)}次逐轮判断。模型：InternVL3.5-4B。使用重建参考序列v3。</p><p>人类一致性 Spearman = {summary['human_model_spearman']:.3f}；95%图像重采样区间 {summary['human_model_spearman_bootstrap95']}。边界响应 {stops.get('boundary',0)}/{len(trials)}，明确回答已匹配 {stops.get('explicit_match',0)}/{len(trials)}。</p><img src="pilot_results.png"><p>左图橙点至少一次到达参考范围边界。边界值是受限输出，不代表已找到等感知匹配。初始档位检查、原始轨迹和全部响应均保留。本轮为探索性试验，不能把重建参考c与人类尺度视为已严格等价。</p><p><a href="summary.json">统计结果</a> · <a href="image_summary.csv">逐图数据</a> · <a href="trials.jsonl">匹配记录</a> · <a href="decisions.jsonl">全部回答</a> · <a href="protocol.json">冻结方案与提示词</a></p><table><tr><th>测试图</th><th>人类c</th><th>物理c</th><th>模型平均c</th><th>两次匹配c</th><th>边界次数</th></tr>{table}</table>'''
(OUT/'report.html').write_text(html,encoding='utf-8')
print(json.dumps(summary,indent=2))
