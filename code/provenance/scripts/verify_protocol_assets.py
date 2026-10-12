from pathlib import Path
import json,hashlib,collections
from PIL import Image
import pandas as pd
import numpy as np
from scipy.sparse.csgraph import connected_components
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'mllm_material'
ann=pd.read_csv(BASE/'annotations/gloss_annotations.csv');panel=pd.read_csv(OUT/'manifests/panel432.csv')
assert len(ann)==3888 and ann.image_id.is_unique
assert (panel.groupby('shape_id').size()==12).all() and (panel.groupby('illumination_id').size()==12).all()
assets=[]
for r in ann.itertuples():
    path=BASE/r.image_512_path
    with Image.open(path) as im:
        assert im.size==(512,512)
        mode=im.mode
    assets.append(dict(image_id=r.image_id,path=r.image_512_path,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),width=512,height=512,mode=mode))
pd.DataFrame(assets).to_csv(OUT/'manifests/image_hashes.csv',index=False)
requests=[json.loads(l) for l in (OUT/'manifests/requests.jsonl').read_text().splitlines()]
assert len({r['request_id'] for r in requests})==len(requests)
seen={}
for r in requests:
    assert r['image_ids']==ann.iloc[r['indices']].image_id.tolist()
    assert r['image_paths']==ann.iloc[r['indices']].image_512_path.tolist()
    if r['task']=='T3' and r['condition']=='main':
        key=tuple(sorted(r['indices']));assert key not in seen;seen[key]=r['split']
counts=collections.Counter((r['task'],r['condition'],r['split']) for r in requests)
ids={int(v)-1:i for i,v in enumerate(panel.source_image_index)};adj=np.zeros((432,432),dtype=np.uint8);occ=np.zeros((432,3),int)
for r in requests:
    if r['task']=='T3' and r['condition']=='main':
        for p,i in enumerate(r['indices']):occ[ids[i],p]+=1
        if r['split']=='train':
            for a in r['indices']:
                for b in r['indices']:adj[ids[a],ids[b]]=1
assert connected_components(adj,directed=False)[0]==1
assert occ.sum(1).min()>=180 and occ.sum(1).max()<=400
pd.DataFrame(dict(image_id=panel.image_id,A=occ[:,0],B=occ[:,1],C=occ[:,2],total=occ.sum(1))).to_csv(OUT/'manifests/triplet_image_exposure.csv',index=False)
report=dict(image_binding_verified=True,images=3888,requests_per_model=len(requests),panel_balanced=True,panel_viewpoint_counts={str(k):int(v) for k,v in panel.viewpoint_id.value_counts().items()},train_comparison_graph_connected=True,main_triplet_image_count_min=int(occ.sum(1).min()),main_triplet_image_count_max=int(occ.sum(1).max()),unique_main_triplets=len(seen),split_counts=dict(collections.Counter(seen.values())),request_counts={'/'.join(k):v for k,v in counts.items()})
(OUT/'protocol/asset_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
