"""Frozen features and deterministic image-statistic baselines in annotation order."""
from pathlib import Path
import sys,os,json,hashlib,argparse,time
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'runtime_packages'))
os.environ['HF_HOME']=str(BASE/'models/hf_cache')
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['HF_HUB_OFFLINE']='1'
import numpy as np
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
OUT=BASE/'pilot/features'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def stats(r):
    with Image.open(BASE/r.image_224_path) as im:rgb=np.asarray(im,dtype=np.float64)/255
    maskfile=BASE.parent/'gloss_tinynetworks-main/data/imgs/obj_mask'/f'img{r.source_image_index}_mask.png'
    with Image.open(maskfile) as im:mask=np.asarray(im.convert('L').resize((224,224),Image.Resampling.NEAREST if hasattr(Image,'Resampling') else Image.NEAREST))>127
    linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
    lum=linear@np.array([.2126,.7152,.0722])
    gy,gx=np.gradient(lum);edge=np.hypot(gx,gy)
    d={}
    for name,region in [('whole',np.ones_like(mask)),('object',mask),('background',~mask)]:
        assert region.sum()>20
        y=lum[region];pixels=linear[region];std=y.std();center=(y-y.mean())/(std+1e-12)
        vals={'luma_mean':y.mean(),'luma_std':std,'luma_skew':(center**3).mean(),'luma_kurtosis':(center**4).mean(),
            'edge_mean':edge[region].mean(),'edge_q90':np.quantile(edge[region],.9),'bright_fraction':(y>.8).mean()}
        for q in [.05,.25,.5,.75,.95,.99]:vals[f'luma_q{int(q*100)}']=np.quantile(y,q)
        for j,c in enumerate('rgb'):
            vals[c+'_mean']=pixels[:,j].mean();vals[c+'_std']=pixels[:,j].std()
        d.update({name+'_'+k:float(v) for k,v in vals.items()})
    d['object_area_fraction']=float(mask.mean())
    return d

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stats-only',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    a=pd.read_csv(BASE/'annotations/gloss_annotations.csv')
    a[['image_id']].to_csv(OUT/'index.csv',index=False)
    if not (OUT/'image_stats.npy').exists():
        with ThreadPoolExecutor(max_workers=8) as pool:records=list(pool.map(stats,a.itertuples()))
        frame=pd.DataFrame(records);assert np.isfinite(frame.to_numpy()).all()
        np.save(OUT/'image_stats.npy',frame.to_numpy())
        pd.DataFrame({'column':range(frame.shape[1]),'feature':frame.columns}).to_csv(OUT/'image_stats_columns.csv',index=False)
        print('Image statistics ready:',frame.shape,flush=True)
    if args.stats_only:return
    import torch,transformers
    from transformers import CLIPVisionModelWithProjection,AutoModel,AutoImageProcessor
    assert torch.cuda.is_available()
    torch.set_num_threads(4);torch.manual_seed(20260926)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    for name in ['clip_b32','dinov2_small','dinov2_same_fov']:
        if (OUT/(name+'.npy')).exists():print(name,'features already present',flush=True);continue
        path=BASE/'models'/('dinov2_small' if name=='dinov2_same_fov' else name)
        source=json.loads((path/'manifest.json').read_text())
        for f,rec in source['files'].items():assert sha(path/f)==rec['sha256']
        processor=AutoImageProcessor.from_pretrained(path,local_files_only=True)
        if name=='dinov2_same_fov':
            processor.do_resize=False
            processor.do_center_crop=False
        model=(CLIPVisionModelWithProjection if name=='clip_b32' else AutoModel).from_pretrained(path,local_files_only=True).eval().cuda()
        embeddings=[];start=time.time()
        for offset in range(0,len(a),32):
            images=[]
            for r in a.iloc[offset:offset+32].itertuples():
                with Image.open(BASE/r.image_224_path) as im:images.append(im.convert('RGB'))
            inp=processor(images=images,return_tensors='pt')['pixel_values'].cuda()
            with torch.inference_mode():
                outputs=model(pixel_values=inp)
                feat=outputs.image_embeds if name=='clip_b32' else outputs.last_hidden_state[:,0]
                feat=torch.nn.functional.normalize(feat,p=2,dim=1)
                embeddings.append(feat.cpu().numpy())
            if offset%512==0:print(name,offset,'/',len(a),round(time.time()-start,1),'seconds',flush=True)
        array=np.concatenate(embeddings)
        assert array.shape[0]==3888 and np.isfinite(array).all()
        assert np.allclose(np.linalg.norm(array,axis=1),1,atol=1e-5)
        np.save(OUT/(name+'.npy'),array)
        manifest=dict(model=source,feature_shape=list(array.shape),processor=processor.to_dict(),
            torch=torch.__version__,transformers=transformers.__version__,device=torch.cuda.get_device_name(0),
            seconds=time.time()-start,feature_sha256=sha(OUT/(name+'.npy')),
            annotation_sha256=sha(BASE/'annotations/gloss_annotations.csv'),config_sha256=sha(BASE/'protocols/pilot_v1_config.json'))
        (OUT/(name+'_manifest.json')).write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        print(name,'complete',array.shape,flush=True)
        del model;torch.cuda.empty_cache()

if __name__=='__main__':main()
