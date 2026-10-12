"""Analyze saved trials at any sample size; never require the full schedule."""
from pathlib import Path
import json, collections, datetime
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parents[1]; OUT=BASE/'mllm_material'
h=pd.read_csv(BASE/'annotations/gloss_annotations.csv').human_gloss_score.to_numpy()
pairs=pd.read_csv(OUT/'manifests/pairs2000_analysis_only.csv')
result={'timestamp':datetime.datetime.now().isoformat(),'scope':'Descriptive interim results, acquisition-order prefix; no independent human triplet choices.'}
for model in ['qwen25vl3b','internvl35_4b']:
    raw=(OUT/'responses'/model/'full.jsonl').read_bytes()
    lines=raw.splitlines(); lines=lines if raw.endswith(b'\n') else lines[:-1]
    rows=list({r['request_id']:r for r in map(json.loads,lines)}.values())
    r={'total':len(rows),'tasks':dict(collections.Counter(x['task'] for x in rows))}
    two=[x for x in rows if x['task']=='T2' and x['parse_status']=='valid']
    r['T2_position']=dict(collections.Counter(x['parsed']['choice'] for x in two))
    grouped=collections.defaultdict(list)
    for x in two:grouped[x['group']].append(x)
    details=[]
    for key,rs in grouped.items():
        ix=int(key.removeprefix('pair')); p=pairs.iloc[ix]
        if len(rs)!=2:continue
        selected=[None if x['parsed']['choice']=='tie' else x['indices'][ord(x['parsed']['choice'])-65] for x in rs]
        winner=int(p.a) if p.human_direction>0 else int(p.b)
        details.append(dict(pair=ix,stratum=p.stratum,follow_human=float(np.mean([.5 if s is None else s==winner for s in selected])),swap_consistent=selected[0]==selected[1]))
    df=pd.DataFrame(details)
    r['T2_complete_pairs']={s:dict(n=len(g),follow_human=float(g.follow_human.mean()),swap_consistency=float(g.swap_consistent.mean())) for s,g in df.groupby('stratum')} if len(df) else {}
    if len(df):df.to_csv(OUT/'statistics'/f'{model}_live_pair_details.csv',index=False)
    three=[x for x in rows if x['task']=='T3' and x['condition']=='main' and x['parse_status']=='valid']
    r['T3_position']=dict(collections.Counter(x['parsed']['odd'] for x in three))
    if three:
        labels=[]; actual=[]
        for x in three:
            a,b,c=h[x['indices']]; distances=np.array([abs(b-c),abs(a-c),abs(a-b)])
            if (distances==distances.min()).sum()!=1:continue
            labels.append(int(distances.argmin()));actual.append(ord(x['parsed']['odd'])-65)
        lab=np.array(labels);act=np.array(actual)
        r['T3_human_scalar_proxy']=dict(n=len(lab),agreement=float((lab==act).mean()),always_A=float((lab==0).mean()),proxy_label_counts=dict(collections.Counter(map(str,lab))),note='nearest pair in scalar human mean ratings; not observed human odd-one-out behavior')
    result[model]=r
(OUT/'reports/live_choice_analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
