"""Leakage-auditable independently specified tiny control, not exact paper replication."""
import os
os.environ['OMP_NUM_THREADS']='4'
from pathlib import Path
import numpy as np,pandas as pd,torch,json,time,hashlib,argparse
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import torch.nn.functional as F
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'robustness_v2/tiny';OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4)

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.conv=torch.nn.Conv2d(3,1,15,bias=False);self.readout=torch.nn.Linear(1,1)
    def forward(self,x):return self.readout(self.conv(x).flatten(2).amax(2)).squeeze(1)

def augment(x,generator):
    n=len(x);side=.8+.2*torch.rand(n,device='cuda',generator=generator)
    flip=torch.where(torch.rand(n,device='cuda',generator=generator)<.5,-1.,1.)
    theta=torch.zeros(n,2,3,device='cuda');theta[:,0,0]=side*flip;theta[:,1,1]=side
    theta[:,:,2]=(torch.rand(n,2,device='cuda',generator=generator)*2-1)*(1-side[:,None])
    grid=F.affine_grid(theta,x.size(),align_corners=False)
    return F.grid_sample(x,grid,align_corners=False,padding_mode='border')

def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--paper-lr',action='store_true');args=parser.parse_args()
    lr=.01 if args.paper_lr else .001
    if args.paper_lr:
        OUT=BASE/'robustness_v2/tiny_paper_lr';OUT.mkdir(parents=True,exist_ok=True)
    assert torch.cuda.is_available();torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=json.loads((BASE/'protocols/robustness_v2_config.json').read_text())
    a=pd.read_csv(BASE/'annotations/gloss_annotations.csv');folds=pd.read_csv(BASE/'protocols/gloss_group_folds.csv')
    assert a.image_id.tolist()==folds.image_id.tolist()
    def read(r):
        with Image.open(BASE.parent/'gloss_tinynetworks-main/data/imgs/imgs3888_nobg_png'/f'img{r.source_image_index}.png') as im:return np.array(im.convert('RGB'))
    with ThreadPoolExecutor(max_workers=8) as pool:arr=np.stack(list(pool.map(read,a.itertuples())))
    images=torch.from_numpy(arr).permute(0,3,1,2).contiguous().cuda()
    target=torch.tensor(a.human_gloss_score.to_numpy()/.1487,dtype=torch.float32,device='cuda')
    records=[];log=[];start=time.time()
    for factor in ['shape','illumination']:
        for seed in cfg['tiny']['seeds']:
            for f in range(12):
                file=OUT/f'{factor}_seed{seed}_fold{f}.csv'
                if file.exists():continue
                tr=np.flatnonzero(folds[factor+'_fold'].to_numpy()!=f);te=np.flatnonzero(folds[factor+'_fold'].to_numpy()==f)
                assert set(folds.iloc[tr][factor+'_id']).isdisjoint(folds.iloc[te][factor+'_id'])
                torch.manual_seed(seed+f);gen=torch.Generator(device='cuda').manual_seed(seed+f)
                model=Tiny().cuda();opt=torch.optim.Adam(model.parameters(),lr=lr)
                tri=torch.tensor(tr,device='cuda');history=[]
                for epoch in range(30):
                    model.train();perm=tri[torch.randperm(len(tri),device='cuda',generator=gen)];losses=[]
                    for offset in range(0,len(perm),128):
                        ix=perm[offset:offset+128];x=images[ix].float()/255
                        x=augment(x,gen);pred=model(x);loss=F.l1_loss(pred,target[ix]);opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
                    history.append(float(np.mean(losses)))
                model.eval();predictions=[]
                with torch.inference_mode():
                    for offset in range(0,len(te),128):
                        ids=te[offset:offset+128]
                        predictions.extend((model(images[ids].float()/255)*.1487).cpu().tolist())
                assert np.isfinite(predictions).all()
                pd.DataFrame(dict(image_id=a.iloc[te].image_id,split=factor,fold=f,seed=seed,human=a.iloc[te].human_gloss_score,prediction=predictions)).to_csv(file,index=False)
                torch.save(model.state_dict(),OUT/f'{factor}_seed{seed}_fold{f}.pt')
                (OUT/f'{factor}_seed{seed}_fold{f}_audit.json').write_text(json.dumps(dict(train_image_ids=a.iloc[tr].image_id.tolist(),test_image_ids=a.iloc[te].image_id.tolist(),train_loss_history=history,selected_epoch=30,seed=seed,learning_rate=lr,parameters=sum(p.numel() for p in model.parameters()),config_sha256=hashlib.sha256((BASE/'protocols/robustness_v2_config.json').read_bytes()).hexdigest()),indent=2))
                print(factor,seed,'fold',f,'train MAE',round(history[-1],4),'elapsed',round(time.time()-start,1),flush=True)
    allrows=pd.concat([pd.read_csv(p) for p in sorted(OUT.glob('*_fold*.csv'))],ignore_index=True)
    assert len(allrows)==3888*4;allrows.to_csv(OUT/'all_predictions.csv',index=False)

if __name__=='__main__':main()
