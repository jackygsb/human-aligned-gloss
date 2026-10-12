"""Paired input controls, with identical full 224 field for both frozen models."""
from pathlib import Path
import sys,os,json,time,hashlib
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'runtime_packages'));os.environ['HF_HOME']=str(BASE/'models/hf_cache');os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['HF_HUB_OFFLINE']='1'
import torch,numpy as np,pandas as pd
from PIL import Image
from transformers import CLIPVisionModelWithProjection,AutoModel,AutoImageProcessor
OUT=BASE/'robustness_v2/features';OUT.mkdir(parents=True,exist_ok=True)

def main():
    torch.set_num_threads(3);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    a=pd.read_csv(BASE/'annotations/gloss_annotations.csv');a[['image_id']].to_csv(OUT/'index.csv',index=False)
    for name in ['clip','dino']:
        path=BASE/'models'/('clip_b32' if name=='clip' else 'dinov2_small')
        model=(CLIPVisionModelWithProjection if name=='clip' else AutoModel).from_pretrained(path,local_files_only=True).eval().cuda()
        processor=AutoImageProcessor.from_pretrained(path,local_files_only=True);processor.do_resize=False;processor.do_center_crop=False
        for variant in ['nobg','gray']:
            dest=OUT/f'{name}_{variant}.npy'
            if dest.exists():continue
            features=[];start=time.time()
            for offset in range(0,len(a),32):
                imgs=[]
                for r in a.iloc[offset:offset+32].itertuples():
                    p=BASE.parent/'gloss_tinynetworks-main/data/imgs/imgs3888_nobg_png'/f'img{r.source_image_index}.png' if variant=='nobg' else BASE/r.image_224_path
                    with Image.open(p) as im:imgs.append(im.convert('L').convert('RGB') if variant=='gray' else im.convert('RGB'))
                inp=processor(images=imgs,return_tensors='pt')['pixel_values'].cuda()
                with torch.inference_mode():
                    result=model(pixel_values=inp);x=result.image_embeds if name=='clip' else result.last_hidden_state[:,0]
                    features.append(torch.nn.functional.normalize(x,dim=1).cpu().numpy())
                if offset%1024==0:print(name,variant,offset,round(time.time()-start,1),'sec',flush=True)
            x=np.concatenate(features);assert len(x)==3888 and np.isfinite(x).all();np.save(dest,x)
            (OUT/f'{name}_{variant}_manifest.json').write_text(json.dumps(dict(shape=list(x.shape),variant=variant,processor=processor.to_dict(),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),model_manifest=json.loads((path/'manifest.json').read_text()),seconds=time.time()-start),indent=2))
            print(name,variant,'complete',flush=True)
        del model;torch.cuda.empty_cache()

if __name__=='__main__':main()
