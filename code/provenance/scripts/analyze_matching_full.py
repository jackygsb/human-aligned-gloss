from pathlib import Path
import os,json,csv,collections,time
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'mllm_material/matching_full_v1'
os.environ['MPLCONFIGDIR']=str(BASE/'mllm_runtime/matplotlib_cache')
import numpy as np
from scipy.stats import spearmanr,pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def readlines(p):
    return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()] if p.exists() else []
def corr(x,y,rank=True):
    return float((spearmanr if rank else pearsonr)(x,y).statistic) if len(x)>2 and np.std(x)>0 and np.std(y)>0 else None
def savecsv(path,rows):
    if rows:
        with path.open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
def main():
    labels={r['image_id']:r for r in csv.DictReader((BASE/'annotations/gloss_annotations.csv').open(encoding='utf-8-sig'))}
    cfg=json.loads((OUT/'protocol.json').read_text());trials=readlines(OUT/'trials.jsonl');alltrials=readlines(OUT/'all_trials.jsonl')
    groups=collections.defaultdict(dict)
    for t in trials:groups[t['image_id']][t['repetition']]=t
    t1={}
    for x in readlines(BASE/'mllm_material/responses/internvl35_4b/full.jsonl'):
        if x.get('task')=='T1' and x.get('condition')=='main' and x.get('parse_status')=='valid':
            t1[x['image_ids'][0]]=x['parsed']['glossiness']
    rows=[]
    for key,reps in groups.items():
        if set(reps)!={1,2} or not all(t['valid'] for t in reps.values()):continue
        a=labels[key];r,s=reps[1],reps[2]
        rows.append(dict(image_id=key,cohort=r['cohort'],human_c=float(a['human_gloss_score']),physical_c=float(a['physical_pellacini_c']),model_c=(r['matching_c']+s['matching_c'])/2,session1_c=r['matching_c'],session2_c=s['matching_c'],shape_id=a['shape_id'],illumination_id=a['illumination_id'],viewpoint_id=a['viewpoint_id'],boundary_runs=sum(t['stop_reason']=='boundary' for t in reps.values()),single_image_score=t1.get(key,'')))
    savecsv(OUT/'image_summary.csv',rows)
    stats={};rng=np.random.default_rng(2026092902)
    for cohort,data in [('heldout_matching',[r for r in rows if r['cohort']=='heldout_matching']),('all',rows)]:
        if len(data)<3:continue
        h,m,g=[np.array([r[k] for r in data]) for k in ['human_c','model_c','physical_c']]
        X=np.c_[np.ones(len(g)),g];hr=h-X@np.linalg.lstsq(X,h,rcond=None)[0];mr=m-X@np.linalg.lstsq(X,m,rcond=None)[0]
        st=dict(n_images=len(data),human_model_spearman=corr(h,m),human_model_pearson=corr(h,m,False),human_physical_spearman=corr(h,g),model_physical_spearman=corr(m,g),partial_pearson_linear_physical_c=corr(hr,mr,False),repeat_spearman=corr([r['session1_c'] for r in data],[r['session2_c'] for r in data]),boundary_trial_fraction=sum(r['boundary_runs'] for r in data)/(2*len(data)))
        for unit in ['image','shape_id','illumination_id']:
            clusters=[np.array([i for i,r in enumerate(data) if r[unit]==v]) for v in sorted({r[unit] for r in data})] if unit!='image' else None
            vals=[]
            for _ in range(1000):
                ix=rng.integers(0,len(data),len(data)) if clusters is None else np.concatenate([clusters[i] for i in rng.integers(0,len(clusters),len(clusters))])
                v=corr(h[ix],m[ix])
                if v is not None:vals.append(v)
            st['bootstrap95_'+unit]=np.quantile(vals,[.025,.975]).tolist() if vals else None
        paired=[r for r in data if r['single_image_score']!='']
        st['single_image_same_targets_spearman']=corr([r['human_c'] for r in paired],[r['single_image_score'] for r in paired])
        stats[cohort]=st
    summary=dict(updated_unix=time.time(),completed_main_trials=len(trials),completed_paired_images=len(rows),target_images=3888,total_scheduled=cfg['total_scheduled'],completed_scheduled=len(alltrials),statistics=stats,stop_counts=dict(collections.Counter(t['stop_reason'] for t in trials)),catch_passed=sum(t.get('passed',False) for t in alltrials if t['phase']=='catch'),catch_completed=sum(t['phase']=='catch' for t in alltrials),interpretation='Interim until all scheduled trials complete; reconstructed reference; deterministic repeats are not independent observers; residual linear-c association is descriptive, not proof of shared causal biases')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (OUT/f"checkpoint_{len(rows):04d}.json").write_text(json.dumps(summary,indent=2))
    grouped=[]
    for dim in ['shape_id','illumination_id']:
        for v in sorted({r[dim] for r in rows}):
            subset=[r for r in rows if r[dim]==v]
            grouped.append(dict(dimension=dim,group=v,n=len(subset),human_mean_c=np.mean([r['human_c'] for r in subset]),model_mean_c=np.mean([r['model_c'] for r in subset]),human_model_spearman=corr([r['human_c'] for r in subset],[r['model_c'] for r in subset])))
    savecsv(OUT/'group_summary.csv',grouped)
    data=[r for r in rows if r['cohort']=='heldout_matching']
    if data:
        fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
        axes[0].scatter([r['human_c'] for r in data],[r['model_c'] for r in data],s=10,alpha=.5);axes[0].set(xlabel='Human matched c',ylabel='Model threshold c',title='Human alignment (matching-heldout)')
        axes[1].scatter([r['physical_c'] for r in data],[r['human_c'] for r in data],s=10,alpha=.35,label='Human');axes[1].scatter([r['physical_c'] for r in data],[r['model_c'] for r in data],s=10,alpha=.35,label='Model');axes[1].set(xlabel='Physical c',ylabel='Judged c');axes[1].legend()
        axes[2].scatter([r['session1_c'] for r in data],[r['session2_c'] for r in data],s=10,alpha=.4);axes[2].set(xlabel='Session 1',ylabel='Session 2',title='Repeat consistency')
        for ax in axes:ax.grid(alpha=.2)
        fig.savefig(OUT/'results.png',dpi=180);fig.savefig(OUT/'results.svg');plt.close(fig)
    stathtml=json.dumps(stats,ensure_ascii=False,indent=2)
    (OUT/'report.html').write_text('''<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>InternVL 全量匹配进度</title><style>body{font:17px system-ui;max-width:1100px;margin:40px auto;padding:20px;color:#243040}img{width:100%}pre{white-space:pre-wrap;background:#f1f5f9;padding:18px}p{line-height:1.8}</style><h1>InternVL · 全量参考匹配</h1><p id="live">读取实时进度…</p><p>目标：3,888 张图 × 2 次匹配；每 72 张完成后更新统计和图。39 张开发/练习图单独标记。首个中期节点 360 张。</p><p>使用冻结提示词与重建参考 v3，当前结果为中期结果，未按相关性筛选图像。模型输出是比较阈值估计；重建参考与人类原参考数值标尺尚不能视为严格等价。</p>'''+f'<h2>最近分析：{len(rows)} 张完成两次匹配</h2><img src="results.png"><pre>{stathtml}</pre>'+'''<p><a href="image_summary.csv">逐图结果</a> · <a href="group_summary.csv">形状/照明分组</a> · <a href="summary.json">统计数据</a> · <a href="protocol.json">冻结方案</a> · <a href="trials.jsonl">匹配轨迹</a> · <a href="decisions.jsonl">全部模型响应</a></p><script>async function tick(){try{let s=await(await fetch('status.json?t='+Date.now())).json();document.getElementById('live').textContent='状态：'+s.state+'；已完成 '+s.completed+' / '+s.total+' 个排程项目（含练习与检查）。最近任务：'+(s.last_trial||'全部完成');}catch(e){}}tick();setInterval(tick,5000);</script>''',encoding='utf-8')
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
