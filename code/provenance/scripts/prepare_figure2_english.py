"""Prepare an English multi-panel result figure from archived responses."""
from pathlib import Path
import json,shutil,hashlib
import numpy as np,pandas as pd
from scipy.stats import spearmanr
B=Path(__file__).resolve().parents[1];M=B/'mllm_material';O=M/'figure2_english_v1';T=O/'data';T.mkdir(parents=True,exist_ok=True)
plan={'claim':'Shared human-like gloss judgments across two frozen models, including deviations from physical ordering.','cohort':'3849 images excluding the 39 original matching-development images. Prior Qwen pilot images remain; not a new confirmatory cohort.','new_analysis':'Descriptive within-physical-level Spearman correlations (all levels retained); pooled Pearson association after subtracting the mean within each exact physical-c level. Residual association CI uses 2000 crossed shape/illumination bootstrap draws, refitting weighted physical-level means in every draw.','gallery':'Human empirical thirds versus equal-width physical-c thirds. Two examples nearest each cell center in the two rank coordinates, preferring distinct shapes. No model-agreement filter. Cell counts and median model percentile ranks use all images.','conflict_example':'First archived discordant pair by pair_id for which both models choose the human-preferred stimulus in both presentation orders, with both images in the common cohort; explicitly illustrative. Overall inference uses all 1000 conflict pairs.','seed':20261004,'bootstrap':2000,'scope':'Post-hoc reanalysis of existing image-level means, not human participant inference. No new responses collected.'}
(O/'analysis_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
sources=[]
def read(rel):
    p=M/rel;sources.append(p);return pd.read_csv(p)
j=read('qwen3_full_v1/joint_model_predictions.csv');orig=read('paper_figures_v1/source_data/image_level_all3888.csv')
d=j[j.cohort=='heldout_matching'].merge(orig[['image_id','single_score','image_512_path']].rename(columns={'single_score':'intern_single'}),on='image_id',validate='one_to_one').sort_values('image_id').reset_index(drop=True)
assert len(d)==3849 and d.image_id.is_unique
for key in ['human_c','internvl_c','model_c','physical_c']:d[key+'_rank']=d[key].rank(pct=True)*100
d['human_bin']=pd.qcut(d.human_c,3,labels=False);d['physical_bin']=pd.cut(d.physical_c,[-1e-10,.1487/3,2*.1487/3,.1487+1e-10],labels=False)
rho=[]
for c,z in d.groupby('physical_c'):
    for name,col in [('InternVL','internvl_c'),('Qwen3','model_c')]:rho.append({'model':name,'physical_c':c,'n_images':len(z),'rho':spearmanr(z.human_c,z[col]).statistic})
pd.DataFrame(rho).to_csv(T/'within_physical_levels.csv',index=False)
codes=pd.Categorical(d.physical_c).codes;shape=pd.Categorical(d.shape_id).codes;light=pd.Categorical(d.illumination_id).codes
Y=d[['human_c','internvl_c','model_c']].to_numpy()
def adjusted(w):
    total=np.bincount(codes,weights=w,minlength=50)
    means=np.stack([np.divide(np.bincount(codes,weights=w*Y[:,i],minlength=50),total,out=np.zeros_like(total),where=total>0) for i in range(3)],axis=1)
    R=Y-means[codes];cov=R.T@(R*w[:,None]);return np.array([cov[0,k]/np.sqrt(cov[0,0]*cov[k,k]) for k in [1,2]])
point=adjusted(np.ones(len(d)));rng=np.random.default_rng(plan['seed']);boot=[]
for _ in range(plan['bootstrap']):
    w=rng.multinomial(36,np.ones(36)/36)[shape]*rng.multinomial(36,np.ones(36)/36)[light];boot.append(adjusted(w))
boot=np.asarray(boot);np.savez_compressed(T/'within_physical_bootstrap.npz',draws=boot)
summary=[]
for k,name in enumerate(['InternVL','Qwen3']):
    lo,hi=np.quantile(boot[:,k],[.025,.975]);summary.append({'model':name,'estimate':point[k],'ci_low':lo,'ci_high':hi,'n':len(d)})
pd.DataFrame(summary).to_csv(T/'within_physical_association.csv',index=False)
gallery=[]
for (hb,pb),cell in d.groupby(['human_bin','physical_bin']):
    score=((cell.human_c_rank-cell.human_c_rank.median())/100)**2+((cell.physical_c_rank-cell.physical_c_rank.median())/100)**2
    cell=cell.assign(distance=score).sort_values(['distance','image_id']);picked=[];shapes=set()
    for r in cell.itertuples():
        if r.shape_id not in shapes:picked.append(r.image_id);shapes.add(r.shape_id)
        if len(picked)==2:break
    gallery.append({'human_bin':int(hb),'physical_bin':int(pb),'n':len(cell),'intern_rank_median':cell.internvl_c_rank.median(),'qwen_rank_median':cell.model_c_rank.median(),'image_ids':picked})
(T/'gallery_cells.json').write_text(json.dumps(gallery,indent=2),encoding='utf-8')
ip=read('paper_figures_v1/source_data/pair_cross_task_predictions.csv');qp=read('qwen3_full_v1/pair_predictions.csv')
pairs=ip.merge(qp[['pair_id','choice_a','swap_consistent']].rename(columns={'swap_consistent':'qwen_swap'}),on='pair_id',validate='one_to_one')
win=(pairs.delta_human>0).astype(float)
eligible=pairs[(pairs.stratum=='discordant')&(pairs.t2_probability_a==win)&(pairs.choice_a==win)&pairs.swap_consistent&pairs.qwen_swap&pairs.a_id.isin(d.image_id)&pairs.b_id.isin(d.image_id)].sort_values('pair_id')
example=eligible.iloc[0].to_dict();(T/'conflict_example.json').write_text(json.dumps(example,indent=2),encoding='utf-8')
pairs.to_csv(T/'pair_level_data.csv',index=False)
d.to_csv(T/'image_level_data.csv',index=False)
for name in ['v8_agreement_ci.csv','v8_agreement_contrasts.csv','v8_pair_ci.csv']:
    p=M/'nmi_revision_v8/source/tables'/name;sources.append(p);shutil.copy2(p,T/name)
images=O/'images';images.mkdir(exist_ok=True);selected=set([v for g in gallery for v in g['image_ids']]+[example['a_id'],example['b_id']])
for iid in selected:
    p=B/d.set_index('image_id').loc[iid,'image_512_path'];shutil.copy2(p,images/f'{iid}.png');sources.append(p)
(O/'source_manifest.json').write_text(json.dumps([{'path':str(p.relative_to(B)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],indent=2),encoding='utf-8')
print('Within-physical-level residual associations:',summary)
print('Level summaries:',pd.DataFrame(rho).groupby('model').rho.agg(['median','min','max']).to_dict())
print('Illustrative pair:',example['pair_id'],example['a_id'],example['b_id'])
