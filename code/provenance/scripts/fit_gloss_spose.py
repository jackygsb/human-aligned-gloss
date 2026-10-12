"""SPoSE upstream model, explicit train/validation/test; separate induced references."""
from pathlib import Path
import argparse,json,importlib.util,itertools,os
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(2)
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'mllm_material'
spec=importlib.util.spec_from_file_location('upstream_spose',BASE.parent/'SPoSE-master/models/model.py')
upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)

def logits(w,tri):
    a,b,c=w[tri].unbind(1)
    return torch.stack([(b*c).sum(1),(a*c).sum(1),(a*b).sum(1)],1)

def fit(tri,q,split,dim,lam,seed,dest,evaluate_test=False,projection=False):
    resultpath=dest/f'd{dim}_l{lam}_s{seed}.json';weightpath=resultpath.with_suffix('.npy')
    if resultpath.exists() and weightpath.exists():
        result=json.loads(resultpath.read_text())
        if evaluate_test and 'test_ce' not in result:
            m=split=='test'
            with torch.no_grad():prob=F.softmax(logits(torch.tensor(np.load(weightpath)),torch.tensor(tri[m],dtype=torch.long)),1).numpy()
            result.update(test_n=int(m.sum()),test_ce=float(-(q[m]*np.log(prob.clip(1e-12))).sum(1).mean()),test_expected_agreement=float(q[m][np.arange(m.sum()),prob.argmax(1)].mean()))
            resultpath.write_text(json.dumps(result,indent=2))
        return result
    torch.manual_seed(seed);device=os.environ.get('SPOSE_DEVICE','cuda' if torch.cuda.is_available() else 'cpu')
    model=upstream.SPoSE(432,dim).to(device);optim=torch.optim.Adam(model.parameters(),lr=.001)
    t=torch.tensor(tri,dtype=torch.long,device=device);y=torch.tensor(q,dtype=torch.float32,device=device)
    tr=torch.tensor(np.flatnonzero(split=='train'),device=device);va=torch.tensor(np.flatnonzero(split=='validation'),device=device)
    best=float('inf');stale=0;bestw=None
    for epoch in range(500):
        order=tr[torch.randperm(len(tr),device=device)]
        for ix in order.split(128):
            optim.zero_grad();w=model.fc.weight.T
            ce=-(y[ix]*F.log_softmax(logits(w,t[ix]),1)).sum(1).mean()
            loss=ce+lam/432*model.fc.weight.abs().sum()+(0 if projection else .01*F.relu(-model.fc.weight).sum())
            loss.backward();optim.step()
            if projection:
                with torch.no_grad():model.fc.weight.clamp_(min=0)
        with torch.no_grad():v=float(-(y[va]*F.log_softmax(logits(model.fc.weight.T,t[va]),1)).sum(1).mean())
        if v<best-1e-6:best=v;stale=0;bestw=model.fc.weight.detach().cpu().numpy().T.copy();bestepoch=epoch+1
        else:stale+=1
        if stale>=30:break
    np.save(weightpath,bestw)
    with torch.no_grad():
        prob=F.softmax(logits(torch.tensor(bestw),torch.tensor(tri,dtype=torch.long)),1).numpy()
    metrics=dict(dim=dim,lam=lam,seed=seed,best_epoch=bestepoch,active_dimensions=int((bestw.max(0)>.1).sum()),negative_weight_fraction=float((bestw<0).mean()),torch_version=torch.__version__,device=device)
    for name in ['train','validation']+(['test'] if evaluate_test else []):
        m=split==name;metrics[name+'_n']=int(m.sum());metrics[name+'_ce']=float(-(q[m]*np.log(prob[m].clip(1e-12))).sum(1).mean());metrics[name+'_expected_agreement']=float(q[m][np.arange(m.sum()),prob[m].argmax(1)].mean())
    resultpath.write_text(json.dumps(metrics,indent=2));print(dest.name,metrics,flush=True);return metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,choices=['qwen25vl3b','internvl35_4b','human_rating_induced','synthetic_1d']);args=p.parse_args()
    panel=pd.read_csv(OUT/'manifests/panel432.csv');mapping={int(x)-1:i for i,x in enumerate(panel.source_image_index)}
    records=[json.loads(l) for l in (OUT/'manifests/requests.jsonl').read_text().splitlines()]
    records=[r for r in records if r['task']=='T3' and r['condition']=='main']
    tri=np.array([[mapping[i] for i in r['indices']] for r in records]);split=np.array([r['split'] for r in records])
    if args.source in ['qwen25vl3b','internvl35_4b']:
        path=OUT/'responses'/args.source/'full.jsonl';answers={r['request_id']:r for r in map(json.loads,path.read_text().splitlines())}
        if any(r['request_id'] not in answers for r in records):raise RuntimeError('Main T3 response collection incomplete')
        valid=np.array([answers[r['request_id']]['parse_status']=='valid' for r in records]);q=np.eye(3)[[ord(answers[r['request_id']]['parsed']['odd'])-65 for r,v in zip(records,valid) if v]];tri=tri[valid];split=split[valid]
    elif args.source=='human_rating_induced':
        boot=np.load(OUT/'statistics/human_bootstrap_means.npy')[:,panel.source_image_index.to_numpy()-1];q=np.empty((len(tri),3))
        for a in range(0,len(tri),256):
            v=boot[:,tri[a:a+256]];d=np.stack([abs(v[:,:,1]-v[:,:,2]),abs(v[:,:,0]-v[:,:,2]),abs(v[:,:,0]-v[:,:,1])],2)
            tied=d==d.min(2,keepdims=True);q[a:a+256]=(tied/tied.sum(2,keepdims=True)).mean(0)
    else:
        scalar=np.random.RandomState(905).uniform(0,1,432);v=scalar[tri];d=np.stack([abs(v[:,1]-v[:,2]),abs(v[:,0]-v[:,2]),abs(v[:,0]-v[:,1])],1);q=np.eye(3)[d.argmin(1)]
    dest=OUT/'spose'/args.source;dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'fitting_data.npz',triples=tri,targets=q,split=split)
    results=[fit(tri,q,split,d,l,s,dest) for d,l,s in itertools.product([32,64],[.001,.003,.01,.03,.1],[0,1,2])]
    df=pd.DataFrame(results);df.to_csv(dest/'search.csv',index=False)
    winner=df.groupby(['dim','lam']).validation_ce.mean().idxmin();dim,lam=int(winner[0]),float(winner[1])
    stable=[fit(tri,q,split,dim,lam,s,dest,evaluate_test=True) for s in range(20)]
    pd.DataFrame(stable).to_csv(dest/'selected_20seeds.csv',index=False)
    agreements=[]
    ws=[np.load(dest/f'd{dim}_l{lam}_s{s}.npy') for s in range(20)]
    for a,b in itertools.combinations(range(20),2):
        wa=ws[a][:,ws[a].max(0)>.1];wb=ws[b][:,ws[b].max(0)>.1]
        if min(wa.shape[1],wb.shape[1])==0:continue
        corr=np.array([[spearmanr(x,y)[0] for y in wb.T] for x in wa.T]);corr=np.nan_to_num(corr);i,j=linear_sum_assignment(-corr)
        agreements.append(dict(seed_a=a,seed_b=b,matched_dimension_r=float(corr[i,j].mean()),matched_n=len(i)))
    pd.DataFrame(agreements).to_csv(dest/'dimension_stability.csv',index=False)
    (dest/'interpretation.md').write_text('SPoSE 用上游 SPoSE 类及原 L1/负值惩罚，索引权重等价于 one-hot 前向。独立验证集选参数，测试集仅报告。已知图像上的新三元组预测属于 transductive 分析。人类评分诱导结构及一维合成校准不是独立人类三选一证据。',encoding='utf-8')
if __name__=='__main__':main()
