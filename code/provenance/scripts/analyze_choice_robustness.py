from pathlib import Path
import json
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'mllm_material';ann=pd.read_csv(BASE/'annotations/gloss_annotations.csv');pairs=pd.read_csv(OUT/'manifests/pairs2000_analysis_only.csv');rows=[];pairchecks=[];thresholds=[]
for folder in (OUT/'responses').iterdir():
    p=folder/'full.jsonl'
    if not p.exists():continue
    answers={r['request_id']:r for r in map(json.loads,p.read_text().splitlines())};data=list(answers.values());main={r['group']:r for r in data if r['task']=='T3' and r['condition']=='main' and r['parse_status']=='valid'}
    def selected(r):return r['image_ids'][ord(r['parsed']['odd'])-65]
    for condition in ['permutation','wording']+[f'sample{k}' for k in range(5)]:
        comparisons=[dict(group=r['group'],condition=condition,consistent=selected(r)==selected(main[r['group']])) for r in data if r['condition']==condition and r['task'] in ['T3','T3b'] and r['parse_status']=='valid' and r['group'] in main]
        for r in comparisons:rows.append(dict(model=folder.name,**r))
    position=pd.Series([r['parsed']['odd'] for r in main.values()]).value_counts().reindex(['A','B','C'],fill_value=0);position.rename_axis('odd_position').to_csv(OUT/'statistics'/f'{folder.name}_odd_position_counts.csv',header=['count'])
    scores=np.full(3888,np.nan);direct_groups={}
    for r in data:
        if r['task']=='T1' and r['condition']=='main' and r['parse_status']=='valid':scores[r['indices'][0]]=r['parsed']['glossiness']
        if r['task']=='T2' and r['parse_status']=='valid':direct_groups.setdefault(r['group'],[]).append(r)
    for n,pair in pairs.iterrows():
        a,b=int(pair.a),int(pair.b)
        if not np.isfinite(scores[[a,b]]).all():continue
        delta=scores[a]-scores[b];winner=None if delta==0 else a if delta>0 else b
        direct=direct_groups.get(f'pair{n}',[])
        agreement=[]
        for r in direct:
            choice=r['parsed']['choice'];actual=None if choice=='tie' else r['indices'][ord(choice)-65];agreement.append(actual==winner)
        pairchecks.append(dict(model=folder.name,pair=n,stratum=pair.stratum,score_tie=delta==0,t1_follow_human=.5 if delta==0 else float(np.sign(delta)==pair.human_direction),t2_vs_t1_consistency=float(np.mean(agreement)) if agreement else np.nan))
    subset=pd.DataFrame([r for r in pairchecks if r['model']==folder.name])
    if len(subset):
        joined=pairs.merge(subset,on=['pair','stratum']) if 'pair' in pairs else pairs.reset_index(names='pair').merge(subset,on=['pair','stratum'])
        for diff in [.0025,.005,.01]:
            for confidence in [.9,.95]:
                a=joined.a.to_numpy();b=joined.b.to_numpy();h=ann.human_gloss_score.to_numpy();c=ann.physical_pellacini_c.to_numpy();m=(abs(c[a]-c[b])>=diff)&((h[a]-h[b])*(c[a]-c[b])<0)&(joined.human_direction_bootstrap>=confidence)
                thresholds.append(dict(model=folder.name,physical_difference_min=diff,human_confidence_min=confidence,n=int(m.sum()),t1_follow_human=float(joined.loc[m,'t1_follow_human'].mean()),scope='sensitivity within frozen 2000-pair pool, not newly representative sampling'))
pd.DataFrame(rows).to_csv(OUT/'statistics/triplet_consistency_detail.csv',index=False)
if rows:pd.DataFrame(rows).groupby(['model','condition']).consistent.agg(['count','mean']).to_csv(OUT/'statistics/triplet_consistency_summary.csv')
pd.DataFrame(pairchecks).to_csv(OUT/'statistics/direct_choice_vs_score.csv',index=False);pd.DataFrame(thresholds).to_csv(OUT/'statistics/discordance_threshold_sensitivity.csv',index=False)
print('Choice robustness summaries updated.')
