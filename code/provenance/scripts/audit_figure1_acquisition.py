"""Count primary task records and save deterministic illustration examples."""
from pathlib import Path
from collections import Counter
import json,hashlib
import pandas as pd
B=Path(__file__).resolve().parents[1];M=B/'mllm_material';O=M/'figure1_english_v3';O.mkdir(exist_ok=True)
def rows(p):
    with p.open(encoding='utf-8') as f:return [json.loads(l) for l in f]
requests=rows(M/'manifests/requests.jsonl')
requests=[r for r in requests if (r['task']=='T1' and r['condition']=='main') or r['task']=='T2']
assert len(requests)==7888
keys={r['request_id'] for r in requests}
ir=rows(M/'responses/internvl35_4b/full.jsonl')
idata={r['request_id']:r for r in ir if r['request_id'] in keys}
assert set(idata)==keys and all(r['parse_status']=='valid' for r in idata.values())
trials=rows(M/'matching_full_v1/trials.jsonl')
decisions=[r for r in rows(M/'matching_full_v1/decisions.jsonl') if r['phase']=='main']
assert len(trials)==7776 and len({(r['trial_id'],r['step']) for r in decisions})==len(decisions)
assert len(decisions)==sum(r['decisions'] for r in trials)
qm=pd.read_csv(M/'qwen3_full_v1/matches.csv')
qpaths=list((M/'qwen3_full_v1/raw').glob('*.json'))
qt={p.stem for p in qpaths if p.name.startswith('T')}
qmatch={p.stem for p in qpaths if p.name.startswith('morimoto')}
assert qt==keys
assert qmatch=={f'{r.trial_id}_step{k:02d}' for r in qm.itertuples() for k in range(1,r.steps+1)}
counts={}
for model,n in [('InternVL',len(decisions)),('Qwen3',len(qmatch))]:
    counts[model]={'test_images':3888,'single_responses':3888,'unique_pairs':2000,'pair_orders':2,'pair_responses':4000,'complete_matches':7776,'matching_responses':n,'total_responses':3888+4000+n}
total=sum(r['total_responses'] for r in counts.values())
assert total==108732
audit={'models':counts,'total_primary_responses':total,'scope':'Primary three-task records only; practice, catch, pilots, other tasks and interventions excluded. Responses are model calls, not independent stimuli or observers.','count_checks':'InternVL unique request IDs and unique trial/step keys; Qwen request-file keys and all expected matching step-file keys.','count_sources':[]}
for rel in ['manifests/requests.jsonl','responses/internvl35_4b/full.jsonl','matching_full_v1/trials.jsonl','matching_full_v1/decisions.jsonl','qwen3_full_v1/matches.csv']:
    p=M/rel;audit['count_sources'].append({'path':str(p.relative_to(B)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(O/'acquisition_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
pd.DataFrame(counts).T.to_csv(O/'acquisition_counts.csv',index_label='model')
# Select illustration examples by manifest order and shape/view metadata, never by agreement.
ann=pd.read_csv(M/'mechanisms_v1/stimuli.csv')
ids=[ann[(ann.shape_id==shape)&(ann.illumination_id==1)&(ann.viewpoint_id==1)].iloc[0].image_id for shape in [1,4,7,10,13,16,19,22,25,28,31,34]]
single=[next(r for r in idata.values() if r['task']=='T1' and r['image_ids']==[iid]) for iid in ids]
pairrows=[r for r in requests if r['task']=='T2']
groups=list(dict.fromkeys(r['group'] for r in pairrows))[:3]
pairs=[[idata[r['request_id']] for r in pairrows if r['group']==g] for g in groups]
assert all(len(p)==2 and p[0]['image_ids']==p[1]['image_ids'][::-1] for p in pairs)
selected=trials[:3]
examples={'selection':'Single: illumination 1, viewpoint 1, shapes 1,4,...,34. Pairs: first three groups in the fixed manifest, both orders. Matches: first three main trials, first three recorded decisions.','single':single,'pairs':pairs,'matching':[{'trial':t,'decisions':[r for r in decisions if r['trial_id']==t['trial_id']][:3]} for t in selected]}
(O/'acquisition_examples.json').write_text(json.dumps(examples,indent=2),encoding='utf-8')
print(json.dumps(counts));print('Total primary responses:',total)
