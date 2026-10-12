"""Post-hoc uncertainty analyses for manuscript v8; no response collection or refitting."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='2'
import numpy as np,pandas as pd,json,hashlib
from scipy.stats import spearmanr,rankdata
B=Path(__file__).resolve().parents[1];M=B/'mllm_material';O=M/'nmi_revision_v8/source';T=O/'tables';T.mkdir(exist_ok=True)
seed=20261001;N=2000;P=99999;rng=np.random.default_rng(seed)
plan={'status':'Post-hoc analysis of archived responses; specified before this script calculates results, not a preregistration','seed':seed,'bootstrap':N,'permutations':P,'agreement':'Crossed shape/illumination multinomial bootstrap; paired draws for model contrasts, all views retained','pairs':'Image-node multinomial bootstrap, edge weights product of endpoint multiplicities; both display orders grouped','cross_task':'Crossed condition bootstrap of fixed out-of-fold predictions; no refit','condition_tests':'Two-sided condition-label permutation, four tests BH','intervention_tests':'Average orders and images within shape; paired shape-level sign randomization, eight candidate-versus-control contrasts BH','scope':'Intervals conditional on saved model outputs and selected stimulus conditions; not human participant or training-run uncertainty'}
(O/'statistical_analysis_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
def bh(p):
 p=np.asarray(p);ix=np.argsort(p);q=np.empty(len(p));q[ix]=np.minimum.accumulate((p[ix]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1];return np.minimum(q,1)
def summary(vals,estimate,**meta):
 lo,hi=np.quantile(vals,[.025,.975]);return dict(**meta,estimate=float(estimate),ci_low=float(lo),ci_high=float(hi))
def weights(d):
 s=pd.Categorical(d.shape_id).codes;l=pd.Categorical(d.illumination_id).codes
 a=rng.multinomial(s.max()+1,np.ones(s.max()+1)/(s.max()+1),N)
 b=rng.multinomial(l.max()+1,np.ones(l.max()+1)/(l.max()+1),N)
 return (a[:,s]*b[:,l]).astype(np.float64)
def weighted_ranks(x,w):
 _,inv=np.unique(x,return_inverse=True);mass=np.bincount(inv,weights=w);return (np.cumsum(mass)-mass/2)[inv]
def wrho(x,y,w):
 x=weighted_ranks(x,w);y=weighted_ranks(y,w);x=x-np.average(x,weights=w);y=y-np.average(y,weights=w)
 return np.sum(w*x*y)/np.sqrt(np.sum(w*x*x)*np.sum(w*y*y))
def r2(y,p,w):
 total=w.sum(axis=1);mean=(w@y)/total;ss=w@(y*y)-total*mean*mean;return 1-(w@((y-p)**2))/ss
joint=pd.read_csv(M/'qwen3_full_v1/joint_model_predictions.csv');ii=pd.read_csv(M/'paper_figures_v1/source_data/image_level_all3888.csv')
d=joint[joint.cohort!='development'].merge(ii[['image_id','single_score']].rename(columns={'single_score':'intern_single'}),on='image_id',validate='one_to_one');W=weights(d)
out=[];draw={};y=d.human_c.to_numpy()
for name,col in [('InternVL_single','intern_single'),('InternVL_matching','internvl_c'),('Qwen3_single','single_score'),('Qwen3_matching','model_c'),('physical','physical_c')]:
 x=d[col].to_numpy();v=np.array([wrho(x,y,w) for w in W]);draw[name]=v;out.append(summary(v,spearmanr(x,y).statistic,measure=name,n=len(d)))
pd.DataFrame(out).to_csv(T/'v8_agreement_ci.csv',index=False)
delta=[]
for a,b in [(x,'physical') for x in draw if x!='physical']+[('Qwen3_matching','InternVL_matching'),('Qwen3_single','InternVL_single')]:
 est={r['measure']:r['estimate'] for r in out};delta.append(summary(draw[a]-draw[b],est[a]-est[b],contrast=a+' minus '+b))
pd.DataFrame(delta).to_csv(T/'v8_agreement_contrasts.csv',index=False)
print('Agreement intervals complete',flush=True)
ip=pd.read_csv(M/'paper_figures_v1/source_data/pair_cross_task_predictions.csv');qp=pd.read_csv(M/'qwen3_full_v1/pair_predictions.csv').merge(ip[['pair_id','a_id','b_id']],on='pair_id')
nodes=np.unique(np.r_[ip.a_id,ip.b_id]);lookup={x:i for i,x in enumerate(nodes)};cnt=rng.multinomial(len(nodes),np.ones(len(nodes))/len(nodes),N);ew=cnt[:,ip.a_id.map(lookup)]*cnt[:,ip.b_id.map(lookup)]
out=[]
for name,df,col in [('InternVL',ip,'t2_probability_a'),('Qwen3',qp,'choice_a')]:
 df=df.set_index('pair_id').loc[ip.pair_id].reset_index();prob=df[col].to_numpy()
 for pred in ['human','single','matching']:
  x=df['delta_'+pred].to_numpy();sgn=np.where(abs(x)<1e-10,0,np.sign(x));score=.5+sgn*(prob-.5)
  for st in ['all','random','discordant']:
   mask=np.ones(len(df),bool) if st=='all' else df.stratum.eq(st).to_numpy();w=ew[:,mask];v=(w@score[mask])/w.sum(axis=1);out.append(summary(v,score[mask].mean(),model=name,predictor=pred,stratum=st,n_pairs=int(mask.sum()),n_nodes=len(nodes)))
pd.DataFrame(out).to_csv(T/'v8_pair_ci.csv',index=False)
cv=pd.read_csv(M/'paper_figures_v1/source_data/continuous_cross_task_oof.csv');out=[]
for (split,source),z in cv.groupby(['split','source']):
 z=d[['image_id']].merge(z,on='image_id',validate='one_to_one');yy=z.observed.to_numpy();pp=z.predicted.to_numpy();out.append(summary(r2(yy,pp,W),1-np.mean((yy-pp)**2)/np.var(yy),model='InternVL',split=split,source=source))
for split in ['shape','illumination']:
 for source in ['single_score','model_c']:
  z=pd.read_csv(M/f'qwen3_full_v1/cross_{split}_{source}.csv');z=d[['image_id']].merge(z,on='image_id',validate='one_to_one');yy=z.target.to_numpy();pp=z.prediction.to_numpy();out.append(summary(r2(yy,pp,W),1-np.mean((yy-pp)**2)/np.var(yy),model='Qwen3',split=split,source=source))
pd.DataFrame(out).to_csv(T/'v8_cross_task_ci.csv',index=False)
out=[]
for name in ['InternVL','Qwen3']:
 for fac in ['shape_id','illumination_id']:
  if name=='InternVL': z=pd.read_csv(M/'paper_figures_v1/tables/Table6_shared_condition_effects.csv');z=z[z.factor==fac]
  else: z=pd.read_csv(M/f'qwen3_full_v1/{fac}_effects.csv')
  x=rankdata(z.human_adjusted_sd);y=rankdata(z.model_adjusted_sd);x-=x.mean();y-=y.mean();den=np.linalg.norm(x)*np.linalg.norm(y);obs=x@y/den;null=np.array([x@rng.permutation(y)/den for _ in range(P)]);idx=rng.integers(0,36,(N,36));boot=np.array([spearmanr(x[k],y[k]).statistic for k in idx]);out.append(summary(boot,obs,model=name,factor=fac,p=(1+int((abs(null)>=abs(obs)-1e-12).sum()))/(P+1)))
res=pd.DataFrame(out);res['q_bh4']=bh(res.p);res.to_csv(T/'v8_condition_tests.csv',index=False)
print('Condition tests complete',flush=True)
iv=pd.read_csv(M/'mechanisms_v1/tables/interventions.csv').merge(joint[['image_id','shape_id']],on='image_id',validate='many_to_one')
g=iv.groupby(['model','shape_id','direction','dose']).delta_p.mean().unstack(['direction','dose']);out=[];raw=[]
for name in ['InternVL','Qwen3']:
 z=g.loc[name];a=z[('candidate',1)]-z[('candidate',-1)]
 for control in ['brightness','random0','random1','random2']:
  v=(a-z[(control,1)]+z[(control,-1)]).to_numpy();obs=v.mean();null=[]
  for start in range(0,P,5000):null.extend((rng.choice([-1,1],(min(5000,P-start),len(v)))*v).mean(axis=1))
  p=(1+int((np.abs(null)>=abs(obs)-1e-12).sum()))/(P+1);boot=v[rng.integers(0,len(v),(N,len(v)))].mean(axis=1);out.append(summary(boot,obs,model=name,control=control,n_shapes=len(v),p=p))
  raw.extend(dict(model=name,control=control,shape_id=int(s),contrast=float(val)) for s,val in zip(z.index,v))
res=pd.DataFrame(out);res['q_bh8']=bh(res.p);res.to_csv(T/'v8_intervention_tests.csv',index=False);pd.DataFrame(raw).to_csv(T/'v8_intervention_shape_values.csv',index=False)
dose_rows=[]
for name in ['InternVL','Qwen3']:
 z=g.loc[name]
 for direction,dose in z.columns:
  v=z[(direction,dose)].dropna().to_numpy();boot=v[rng.integers(0,len(v),(N,len(v)))].mean(axis=1)
  dose_rows.append(summary(boot,v.mean(),model=name,direction=direction,dose=dose,n_shapes=len(v)))
pd.DataFrame(dose_rows).rename(columns={'estimate':'mean_delta_p'}).to_csv(T/'v8_intervention_dose_ci.csv',index=False)
np.savez_compressed(O/'tables/v8_agreement_bootstrap.npz',**draw)
print(res.to_string(index=False),flush=True)
print('All post-hoc analyses complete',flush=True)
