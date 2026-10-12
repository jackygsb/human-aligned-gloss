"""Freeze label-blind panels/triplets and human-physical discordance sampling."""
from pathlib import Path
import json,hashlib,itertools
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'mllm_material'
for p in ['protocol','manifests','responses','statistics','reports','spose']:(OUT/p).mkdir(parents=True,exist_ok=True)
rng=np.random.RandomState(20260926)
ann=pd.read_csv(BASE/'annotations/gloss_annotations.csv').sort_values('source_image_index').reset_index(drop=True)
SYSTEM='Make a visual judgment using the supplied image. Follow the response format exactly.'
T1='''Rate the apparent glossiness of the main object in this image.
Glossiness means how matte or shiny the surface looks.
Use an integer from 0 to 100: 0 = not at all glossy; 100 = extremely glossy.
Judge the visible appearance, not the material name or its physical reflectance.
Return only {"glossiness": <integer>, "status": "ok"}.
If the image is unavailable or the main object cannot be identified, return only
{"glossiness": null, "status": "unrateable"}.'''
T2='''Compare the apparent glossiness of the main objects in Image A and Image B.
Which object's surface looks glossier? Judge visible matte versus shiny appearance,
not the object category or the material's physical properties.
Choose A, B, or tie if they appear equally glossy.
Return only a JSON object with keys "choice" and "status".
The value of "choice" must be "A", "B", or "tie"; "status" must be "ok".
If an image is unavailable or its main object cannot be identified, return only
{"choice": null, "status": "unrateable"}.'''
T3='''Consider only the apparent glossiness of the main object in each image.
Which two objects look most similar in glossiness, and which remaining image
is the odd one out? Glossiness means how matte or shiny the surface looks.
Do not choose based on object category or shape alone.
Choose the odd-one-out image A, B, or C. If the differences are small, give your best judgment.
Return only a JSON object with keys "odd" and "status".
The value of "odd" must be "A", "B", or "C"; "status" must be "ok".
If an image is unavailable or its main object cannot be identified, return only
{"odd": null, "status": "unrateable"}.'''
prompts=dict(system=SYSTEM,T1=T1,T1b='How glossy does the main object\'s surface appear? Judge the visible degree of matte versus shiny appearance.\n'+'\n'.join(T1.splitlines()[2:]),T1c='Assess the perceived surface gloss of the main object. Consider how matte or shiny it appears in the image.\n'+'\n'.join(T1.splitlines()[2:]),T2=T2,T3=T3,T3b=T3.replace('Which two objects look most similar in glossiness, and which remaining image\nis the odd one out?','Select the image whose object differs most in glossiness from the other two.'))
(OUT/'protocol/prompts.json').write_text(json.dumps(prompts,indent=2),encoding='utf-8')
shapes=rng.permutation(sorted(ann.shape_id.unique()));lights=rng.permutation(sorted(ann.illumination_id.unique()))
offsets=rng.choice(36,12,replace=False);panel=[]
for j,s in enumerate(shapes):
    for d in offsets:
        eligible=ann.index[(ann.shape_id==s)&(ann.illumination_id==lights[(j+d)%36])].to_numpy()
        panel.append(int(rng.choice(eligible)))
panel=np.array(sorted(panel));assert len(set(panel))==432
ann.iloc[panel].to_csv(OUT/'manifests/panel432.csv',index=False)
bootfile=OUT/'statistics/human_bootstrap_means.npy'
if not bootfile.exists():
    bootstrap_rng=np.random.RandomState();bootstrap_rng.set_state(rng.get_state())
    trials=pd.read_csv(BASE/'annotations/gloss_trials.csv');trials=trials[trials.is_main_image.astype(str).str.lower()=='true']
    boot=np.empty((2000,len(ann)),dtype=np.float32)
    index={v:i for i,v in enumerate(ann.image_id)}
    for group,t in trials.groupby('group_id'):
        a=t.pivot_table(index='participant_id',columns='image_id',values='human_pellacini_c',aggfunc='mean')
        assert not a.isna().any().any()
        selections=bootstrap_rng.randint(0,len(a),size=(2000,len(a)))
        boot[:,[index[k] for k in a.columns]]=a.to_numpy()[selections].mean(axis=1)
    np.save(bootfile,boot)
else:boot=np.load(bootfile)
pairs=[];seen=set();usage=np.zeros(len(ann),int)
h=ann.human_gloss_score.to_numpy();c=ann.physical_pellacini_c.to_numpy()
# Select robust discordant pairs first, without seeing model responses.
for _ in range(1000000):
    a,b=sorted(rng.choice(len(ann),2,replace=False));key=(int(a),int(b))
    if key in seen or max(usage[a],usage[b])>=10:continue
    if abs(c[a]-c[b])<.005 or (c[a]-c[b])*(h[a]-h[b])>=0:continue
    q=float(np.mean((boot[:,a]-boot[:,b])*np.sign(h[a]-h[b])>0))
    if q<.95:continue
    seen.add(key);usage[a]+=1;usage[b]+=1
    pairs.append(dict(a=int(a),b=int(b),stratum='discordant',human_direction=int(np.sign(h[a]-h[b])),human_direction_bootstrap=q))
    if len(pairs)==1000:break
assert len(pairs)==1000,'Insufficient robust discordant pairs; revise transparently.'
while len(pairs)<2000:
    a,b=sorted(rng.choice(len(ann),2,replace=False));key=(int(a),int(b))
    if key in seen:continue
    seen.add(key);pairs.append(dict(a=int(a),b=int(b),stratum='random',human_direction=int(np.sign(h[a]-h[b])),human_direction_bootstrap=float(np.mean((boot[:,a]-boot[:,b])*np.sign(h[a]-h[b])>0))))
pd.DataFrame(pairs).to_csv(OUT/'manifests/pairs2000_analysis_only.csv',index=False)
triples=[];seen=set()
while len(triples)<40000:
    key=tuple(sorted(rng.choice(panel,3,replace=False).tolist()))
    if key in seen:continue
    seen.add(key);triples.append(rng.permutation(key).tolist())
records=[]
def add(task,indices,condition='main',split='',sample=False,group=''):
    indices=list(indices)
    records.append(dict(request_id=f'{task}_{condition}_{len(records):06d}',task=task,condition=condition,indices=list(map(int,indices)),image_ids=ann.iloc[indices].image_id.tolist(),image_paths=ann.iloc[indices].image_512_path.tolist(),split=split,sample=sample,group=group))
for i in range(len(ann)):add('T1',[i])
for i in panel:
    for task in ['T1b','T1c']:add(task,[i])
    for rep in [1,2]:add('T1',[i],f'repeat{rep}')
    add('T1',[i],'no_image')
for n,p in enumerate(pairs):
    add('T2',[p['a'],p['b']],group=f'pair{n}')
    add('T2',[p['b'],p['a']],'swap',group=f'pair{n}')
for n,t in enumerate(triples):add('T3',t,split='train' if n<32000 else 'validation' if n<36000 else 'test',group=f'triple{n}')
for n in range(512):
    for rep in range(5):add('T3',triples[n],f'sample{rep}',sample=True,group=f'triple{n}')
    add('T3b',triples[n],'wording',group=f'triple{n}')
for n in range(256):
    for p in itertools.permutations(triples[n]):
        if list(p)!=triples[n]:add('T3',p,'permutation',group=f'triple{n}')
# Separate engineering requests; no label-based prompt selection.
engineering=[]
erng=np.random.RandomState(18031);remaining=list(erng.permutation(panel));chosen=[];ss=set();ls=set()
while len(chosen)<96:
    i=max(remaining,key=lambda i:int(ann.iloc[i].shape_id not in ss)+int(ann.iloc[i].illumination_id not in ls))
    chosen.append(i);remaining.remove(i);ss.add(ann.iloc[i].shape_id);ls.add(ann.iloc[i].illumination_id)
assert len(ss)==len(ls)==36
for task,nimgs in [('T1',1),('T2',2),('T3',3)]:
    for n,i in enumerate(chosen):
        ix=[int(i)]+list(map(int,erng.choice(panel[panel!=i],nimgs-1,replace=False)))
        engineering.append(dict(request_id=f'engineering_balanced_{task}_{n:03d}',task=task,condition='engineering',indices=ix,image_ids=ann.iloc[ix].image_id.tolist(),image_paths=ann.iloc[ix].image_512_path.tolist(),split='',sample=False,group=''))
for name,data in [('requests.jsonl',records),('engineering.jsonl',engineering)]:
    with (OUT/'manifests'/name).open('w',encoding='utf-8') as f:
        for r in data:f.write(json.dumps(r)+'\n')
files=['protocol/prompts.json','manifests/requests.jsonl','manifests/engineering.jsonl','manifests/panel432.csv','manifests/pairs2000_analysis_only.csv']
freeze=dict(seed=20260926,requests_per_model=len(records),engineering_per_model=len(engineering),image_input='original_512_with_background',labels_in_requests=False,sha256={p:hashlib.sha256((OUT/p).read_bytes()).hexdigest() for p in files})
(OUT/'protocol/freeze.json').write_text(json.dumps(freeze,indent=2))
print(json.dumps(freeze,indent=2))
