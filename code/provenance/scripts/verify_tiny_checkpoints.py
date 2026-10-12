"""Reload all saved weights and independently reproduce every held-out prediction."""
from train_tiny_control import Tiny,BASE
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import numpy as np,pandas as pd,torch,json,hashlib

def main():
    a=pd.read_csv(BASE/'annotations/gloss_annotations.csv');indices={v:i for i,v in enumerate(a.image_id)}
    def read(r):
        path=BASE.parent/'gloss_tinynetworks-main/data/imgs/imgs3888_nobg_png'/f'img{r.source_image_index}.png'
        with Image.open(path) as im:arr=np.array(im.convert('RGB'))
        assert arr.shape==(224,224,3)
        return arr,dict(image_id=r.image_id,source_path=str(path.relative_to(BASE.parent)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with ThreadPoolExecutor(max_workers=8) as pool:rows=list(pool.map(read,a.itertuples()))
    x=torch.from_numpy(np.stack([r[0] for r in rows])).permute(0,3,1,2).contiguous().cuda()
    pd.DataFrame([r[1] for r in rows]).to_csv(BASE/'robustness_v2/nobg_source_manifest.csv',index=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    records=[]
    for folder in ['tiny','tiny_paper_lr']:
        for path in sorted((BASE/'robustness_v2'/folder).glob('*_fold*.csv')):
            d=pd.read_csv(path);model=Tiny().cuda();model.load_state_dict(torch.load(path.with_suffix('.pt'),map_location='cuda'));model.eval();pred=[]
            ids=[indices[i] for i in d.image_id]
            with torch.inference_mode():
                for start in range(0,len(ids),128):pred.extend((model(x[ids[start:start+128]].float()/255)*.1487).cpu().tolist())
            err=float(np.max(np.abs(np.array(pred)-d.prediction.to_numpy())))
            assert err<1e-7,(str(path),err)
            records.append(dict(file=str(path.relative_to(BASE)),max_abs_error=err,checkpoint_sha256=hashlib.sha256(path.with_suffix('.pt').read_bytes()).hexdigest()))
    assert len(records)==96
    (BASE/'robustness_v2/checkpoint_validation.json').write_text(json.dumps(dict(status='passed',checkpoints=96,predictions=31104,torch_version=torch.__version__,cuda_version=torch.version.cuda,gpu=torch.cuda.get_device_name(0),records=records),indent=2))
    print('All 96 checkpoints reproduced 31104 predictions; no-background source images hashed.')

if __name__=='__main__':main()
