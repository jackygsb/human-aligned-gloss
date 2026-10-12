"""CPU-only descriptive audit of frozen single-image results; no new inference."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

base = Path(__file__).resolve().parents[1]
out = base / 'mllm_material'
dest = out / 'reports/interim_review'
dest.mkdir(exist_ok=True)
ann = pd.read_csv(base / 'annotations/gloss_annotations.csv')
scores = pd.read_csv(out / 'statistics/internvl35_4b_image_scores.csv')
assert ann.image_id.tolist() == scores.image_id.tolist()
pairs = pd.read_csv(out / 'manifests/pairs2000_analysis_only.csv')
pairs = pairs[pairs.stratum == 'discordant'].copy()
a, b = pairs.a.to_numpy(), pairs.b.to_numpy()
delta = scores.model_score.to_numpy()[a] - scores.model_score.to_numpy()[b]
pairs['follow'] = np.where(delta == 0, .5, (np.sign(delta) == pairs.human_direction).astype(float))
pairs['category'] = np.where(delta == 0, 'tie', np.where(pairs.follow == 1, 'human', 'physical'))
checks = []
for factor in ['shape_id', 'illumination_id']:
    values = ann[factor].to_numpy()
    for group in sorted(set(values)):
        keep = (values[a] != group) & (values[b] != group)
        checks.append(dict(factor=factor,excluded_group=int(group),remaining_pairs=int(keep.sum()),follow_human=float(pairs.loc[keep,'follow'].mean())))
pd.DataFrame(checks).to_csv(dest / 'leave_one_group_out_descriptive.csv',index=False)
pairs.to_csv(dest / 'discordant_pairs_categorized.csv',index_label='frozen_pair_index')
counts = pairs.category.value_counts().to_dict()
summary = dict(n=len(pairs),counts=counts,follow_human=float(pairs.follow.mean()),leave_one_group_out={factor:dict(min=float(g.follow_human.min()),max=float(g.follow_human.max())) for factor,g in pd.DataFrame(checks).groupby('factor')},scope='descriptive frozen-pair-pool analysis; no retraining, no confidence interval, not independent validation')
(dest / 'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
labels={'human':'模型跟随人类','physical':'模型跟随物理排序','tie':'模型给出相同分数'}
parts=['''<!doctype html><html lang="zh"><meta charset="utf-8"><title>光泽结果：实例核查</title><style>body{font:17px system-ui;max-width:1100px;margin:35px auto;padding:0 20px;background:#f6f7fa;color:#182333}article{background:white;padding:20px;margin:24px 0;border-radius:12px}.pair{display:flex;gap:20px}.item{width:48%}img{width:100%;max-width:400px}small{color:#536170}button{padding:10px;cursor:pointer}.values{display:none}body.show .values{display:block}</style><h1>InternVL 光泽结果：实例核查</h1><p>本页用于研究者检查已有刺激和结果，不构成新的人类实验。样例来自冻结的 1,000 对冲突图像：每类按固定随机种子抽取 6 对，没有按视觉效果或效应大小挑图。</p><p>先看物体、高光和背景，再显示数值。请记录可能的刺激或输入问题；不要据此调整主提示并在同一数据上选择最优结果。</p><button onclick="document.body.classList.toggle('show')">显示 / 隐藏结果</button>''']
    
for category in ['human','physical','tie']:
    pool=pairs[pairs.category==category]
    sample=pool.sample(n=min(6,len(pool)),random_state=20260927)
    parts.append(f'<h2 class="values">{labels[category]}：全量 {len(pool)} 对</h2>')
    for idx,row in sample.iterrows():
        parts.append(f'<article><h3>冻结图像对 #{idx}</h3><div class="pair">')
        for position in ['a','b']:
            ix=int(row[position]); r=ann.iloc[ix]; s=scores.iloc[ix]
            path='../../../'+r.image_512_path
            assert (base/r.image_512_path).is_file()
            parts.append(f'<div class="item"><img src="{path}" alt="{r.image_id}"><p>{position.upper()} · {r.image_id}</p><div class="values">人类匹配评分：{s.human:.5f}<br>物理参数 c：{s.physical:.5f}<br>模型评分：{s.model_score:.0f}<br>形状 {r.shape_id} / 照明 {r.illumination_id}</div></div>')
        parts.append('</div><p class="values">注意：人类匹配评分、物理参数与模型 0–100 评分的单位不同，只比较排序。</p></article>')
parts.append('</html>')
(dest/'实例核查.html').write_text('\n'.join(parts),encoding='utf-8')
print(json.dumps(summary,indent=2))
