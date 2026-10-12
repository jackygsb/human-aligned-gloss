from pathlib import Path
import json,collections
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'mllm_material';result={}
for model in ['qwen25vl3b','internvl35_4b']:
    d=pd.read_csv(OUT/'statistics'/f'{model}_image_scores.csv');counts=d.model_score.value_counts().sort_index()
    raw=(OUT/'responses'/model/'full.jsonl').read_text();lines=raw.splitlines();lines=lines if raw.endswith('\n') else lines[:-1];answers=[json.loads(l) for l in lines]
    r=dict(score_counts={str(k):int(v) for k,v in counts.items()},modal_score_fraction=float(counts.max()/len(d)),request_counts=dict(collections.Counter(x['task']+'/'+x['condition'] for x in answers)))
    for task,condition,key in [('T1','no_image','glossiness'),('T2','main','choice'),('T2','swap','choice'),('T3','main','odd')]:
        r[task+'_'+condition+'_outputs']=dict(collections.Counter(str(x['parsed'].get(key)) for x in answers if x['task']==task and x['condition']==condition and x['parsed']))
    groups=pd.read_csv(OUT/'statistics/group_alignment.csv');groups=groups[groups.model==model]
    r['group_alignment']={factor:dict(median=float(g.rho.median()),min=float(g.rho.min()),max=float(g.rho.max()),positive=int((g.rho>0).sum()),defined=int(g.rho.notna().sum())) for factor,g in groups.groupby('factor')}
    checks=pd.read_csv(OUT/'statistics/direct_choice_vs_score.csv');checks=checks[checks.model==model]
    r['T1_pair_follow_human']={s:float(g.t1_follow_human.mean()) for s,g in checks.groupby('stratum')}
    result[model]=r
(OUT/'reports/interim_findings_detail.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
