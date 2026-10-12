"""Predefined learning curves, constraint check, 1D comparator, validation truncation."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment
from fit_gloss_spose import fit,logits,BASE,OUT

def ce_prob(w,tri):
    with torch.no_grad():return F.softmax(logits(torch.tensor(w),torch.tensor(tri,dtype=torch.long)),1).numpy()
def free1d(tri,q,split,dest):
    path=dest/'free1d.json'
    if path.exists():return
    t=torch.tensor(tri,dtype=torch.long);y=torch.tensor(q,dtype=torch.float32);train=np.flatnonzero(split=='train');val=np.flatnonzero(split=='validation');test=np.flatnonzero(split=='test')
    def pred(z,ix):
        a,b,c=z[t[ix]].unbind(1);return torch.stack([-abs(b-c),-abs(a-c),-abs(a-b)],1)
    fits=[]
    for seed in range(3):
        torch.manual_seed(seed);z=torch.nn.Parameter(torch.randn(432)*.1);opt=torch.optim.Adam([z],lr=.01);best=float('inf');stale=0
        for epoch in range(500):
            for ix in torch.tensor(train)[torch.randperm(len(train))].split(128):
                opt.zero_grad();loss=-(y[ix]*F.log_softmax(pred(z,ix),1)).sum(1).mean();loss.backward();opt.step()
            with torch.no_grad():v=float(-(y[val]*F.log_softmax(pred(z,val),1)).sum(1).mean())
            if v<best-1e-6:best=v;bestz=z.detach().clone();stale=0
            else:stale+=1
            if stale>=30:break
        fits.append((best,seed,bestz))
    best,seed,z=min(fits,key=lambda x:x[0]);prob=F.softmax(pred(z,test),1).numpy();np.save(dest/'free1d_positions.npy',z.numpy())
    path.write_text(json.dumps(dict(seed_selected_on_validation=seed,validation_ce=best,test_n=len(test),test_ce=float(-(q[test]*np.log(prob.clip(1e-12))).sum(1).mean()),test_expected_agreement=float(q[test][np.arange(len(test)),prob.argmax(1)].mean())),indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);args=p.parse_args();folder=OUT/'spose'/args.source
    d=pd.read_csv(folder/'selected_20seeds.csv');dim=int(d.dim.iloc[0]);lam=float(d.lam.iloc[0]);data=np.load(folder/'fitting_data.npz');tri=data['triples'];q=data['targets'];split=data['split'];train=np.flatnonzero(split=='train')
    dest=folder/'additional_checks';dest.mkdir(exist_ok=True)
    # Dimensions ordered without reference to human scores or final-test answers.
    w=np.load(folder/f'd{dim}_l{lam}_s0.npy');order=np.argsort(-(w*w).sum(0));val=split=='validation';test=split=='test';losses=[];curves=[]
    for k in range(1,dim+1):
        prob=ce_prob(w[:,order[:k]],tri[val]);loss=-(q[val]*np.log(prob.clip(1e-12))).sum(1);losses.append(loss);curves.append(dict(dimensions=k,validation_ce=float(loss.mean())))
    losses=np.array(losses);best=int(losses.mean(1).argmin());rng=np.random.RandomState(18723);nodeweights=rng.multinomial(432,np.ones(432)/432,size=2000);vw=nodeweights[:,tri[val]].prod(2);averages=vw@losses[best]/vw.sum(1);se=float(averages.std(ddof=1));chosen=int(np.flatnonzero(losses.mean(1)<=losses[best].mean()+se)[0]+1)
    prob=ce_prob(w[:,order[:chosen]],tri[test]);result=dict(selected_dimensions=chosen,best_validation_dimensions=best+1,validation_structure_bootstrap_se=se,test_ce=float(-(q[test]*np.log(prob.clip(1e-12))).sum(1).mean()),test_expected_agreement=float(q[test][np.arange(test.sum()),prob.argmax(1)].mean()),selection='minimum dimensions within one validation SE; seed0 chosen a priori')
    (dest/'dimension_truncation.json').write_text(json.dumps(result,indent=2));pd.DataFrame(curves).to_csv(dest/'dimension_truncation_curve.csv',index=False)
    trainprop=q[train].mean(0);trainprop=trainprop/trainprop.sum();(dest/'position_baseline.json').write_text(json.dumps(dict(odd_probabilities=trainprop.tolist(),test_ce=float(-(q[test]*np.log(trainprop.clip(1e-12))).sum(1).mean()),test_expected_agreement=float(q[test][:,trainprop.argmax()].mean()))))
    learning=[]
    for n in [4000,8000,16000,32000]:
        take=(split!='train');take[train[:n]]=True;sub=dest/f'learning_{n}';sub.mkdir(exist_ok=True)
        for seed in range(3):learning.append(dict(training_limit=n,**fit(tri[take],q[take],split[take],dim,lam,seed,sub,evaluate_test=True)))
    pd.DataFrame(learning).to_csv(dest/'learning_curves.csv',index=False)
    sub=dest/'projection';sub.mkdir(exist_ok=True)
    pd.DataFrame([fit(tri,q,split,dim,lam,s,sub,evaluate_test=True,projection=True) for s in range(3)]).to_csv(dest/'projection_check.csv',index=False)
    # Two disjoint training subsets; shared validation/test for comparable predictions.
    for half in [0,1]:
        take=split!='train';take[train[half::2]]=True;sub=dest/f'independent_training_half{half}';sub.mkdir(exist_ok=True);fit(tri[take],q[take],split[take],dim,lam,0,sub,evaluate_test=True)
    wa=np.load(dest/'independent_training_half0'/f'd{dim}_l{lam}_s0.npy');wb=np.load(dest/'independent_training_half1'/f'd{dim}_l{lam}_s0.npy');wa=wa[:,wa.max(0)>.1];wb=wb[:,wb.max(0)>.1]
    if min(wa.shape[1],wb.shape[1]):
        cor=np.nan_to_num(np.array([[spearmanr(a,b)[0] for b in wb.T] for a in wa.T]));a,b=linear_sum_assignment(-cor);(dest/'independent_training_stability.json').write_text(json.dumps(dict(matched_dimensions=len(a),mean_matched_rho=float(cor[a,b].mean()),matched_above_point7=int((cor[a,b]>.7).sum())),indent=2))
    free1d(tri,q,split,dest)
    (dest/'COMPLETE.json').write_text(json.dumps(dict(source=args.source,completed=True)))
if __name__=='__main__':main()
