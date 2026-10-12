from pathlib import Path
import json,hashlib
import pandas as pd
from PIL import Image
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'mllm_material';panel=pd.read_csv(OUT/'manifests/panel432.csv');records=[]
for row in panel.itertuples():
    i=int(row.source_image_index)-1
    for condition,path in [('original224',row.image_224_path),('author_gray224',f'../gloss_tinynetworks-main/data/imgs/imgs3888_nobg_png/img{i+1}.png')]:
        p=BASE/path
        with Image.open(p) as im:assert im.size==(224,224)
        records.append(dict(request_id=f'input_control_{condition}_{i:04d}',task='T1',condition=condition,indices=[i],image_ids=[row.image_id],image_paths=[path],image_sha256=[hashlib.sha256(p.read_bytes()).hexdigest()],split='',sample=False,group=''))
path=OUT/'manifests/input_control.jsonl'
path.write_text(''.join(json.dumps(r)+'\n' for r in records))
freeze=json.loads((OUT/'protocol/freeze.json').read_text());freeze['sha256']['manifests/input_control.jsonl']=hashlib.sha256(path.read_bytes()).hexdigest();freeze['additional_input_control_per_model']=len(records);(OUT/'protocol/freeze.json').write_text(json.dumps(freeze,indent=2))
print(len(records),'additional requests: matched 224 original vs author-gray; full background retained in main 512 condition.')
