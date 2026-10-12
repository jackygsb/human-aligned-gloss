"""Independent behavioral condition spaces, nested view validation, and label-permutation RSA.
Not SPoSE, not neural activations. Run from the supplied project environment.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import json, hashlib, shutil, platform, itertools, zipfile
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

B=Path(__file__).resolve().parents[1]; M=B/'mllm_material'; O=M/'behavioral_spaces_v1'
SEED=20260929; NPERM=4999; NBOOT=2000
AGENTS={'Human':'human_c','InternVL':'internvl_c','Qwen3':'model_c'}
RANKS=[0,1,2,4,8]; SHRINK=[0.25,0.5,1.0]

def save_json(name,x): (O/name).write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def csv(name,x): pd.DataFrame(x).to_csv(O/name,index=False)
def center(a): return a-a.mean(0,keepdims=True)-a.mean(1,keepdims=True)+a.mean()
def corr(a,b):
 a=rankdata(a); b=rankdata(b); a-=a.mean(); b-=b.mean()
 return float(a@b/np.sqrt((a@a)*(b@b)))
def fit_nuisance(X,y,mask): return np.linalg.lstsq(X[mask],y[mask],rcond=None)[0]
def decompose(r,s,l,mask):
 mat=np.zeros((36,36)); count=np.zeros((36,36))
 np.add.at(mat,(s[mask],l[mask]),r[mask]);np.add.at(count,(s[mask],l[mask]),1)
 assert count.min()>0
 mat/=count; interaction=center(mat); additive=mat-interaction
 u,sv,vt=np.linalg.svd(interaction,full_matrices=False)
 return additive,u,sv,vt
def component(f,k,a=1):
 base,u,sv,vt=f
 return base+(u[:,:k]*sv[:k])@vt[:k]*a
def distances(x): return squareform(pdist(x,'sqeuclidean'))/x.shape[1]
def bh(p):
 p=np.asarray(p);ix=np.argsort(p);out=np.empty(len(p));out[ix]=np.minimum.accumulate((p[ix]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1].clip(0,1);return out
def rsa_stats(a,b,rng):
 ix=np.triu_indices(36,1);av=a[ix]; bv=b[ix]; obs=corr(av,bv)
 ar=rankdata(av); ar-=ar.mean(); ar/=np.linalg.norm(ar)
 br=np.zeros((36,36));r=rankdata(bv);r-=r.mean();r/=np.linalg.norm(r);br[ix]=r;br[(ix[1],ix[0])]=r
 null=np.empty(NPERM)
 for t in range(NPERM):
  p=rng.permutation(36);null[t]=ar@br[p[ix[0]],p[ix[1]]]
 boot=[]
 for t in range(NBOOT):
  ids=rng.integers(0,36,36);j,k=ids[ix[0]],ids[ix[1]];keep=j!=k
  boot.append(corr(a[j[keep],k[keep]],b[j[keep],k[keep]]))
 return obs,(1+np.sum(np.abs(null)>=abs(obs)))/(NPERM+1),np.quantile(boot,[.025,.975]),null

def main():
 for sub in ['inputs','code','tables','matrices','figures','embeddings']: (O/sub).mkdir(parents=True,exist_ok=True)
 source=M/'qwen3_full_v1/joint_model_predictions.csv'; shutil.copy2(source,O/'inputs/joint_model_predictions.csv')
 shutil.copy2(__file__,O/'code'/Path(__file__).name)
 d=pd.read_csv(source).sort_values('source_image_index').reset_index(drop=True)
 assert len(d)==3888 and d.image_id.is_unique and not d[list(AGENTS.values())].isna().any().any()
 shapes=sorted(d.shape_id.unique());lights=sorted(d.illumination_id.unique());views=sorted(d.viewpoint_id.unique())
 s=pd.Categorical(d.shape_id,categories=shapes).codes;l=pd.Categorical(d.illumination_id,categories=lights).codes;v=d.viewpoint_id.to_numpy()
 assert d.groupby(['shape_id','illumination_id','viewpoint_id']).size().eq(1).all()
 X=np.column_stack([np.ones(len(d)),pd.get_dummies(d.physical_c).to_numpy()[:,1:],pd.get_dummies(d.chroma).to_numpy()[:,1:]]+[f(n*np.deg2rad(d.hue.to_numpy())) for n in [1,2] for f in [np.sin,np.cos]])
 plan={'created_before_analysis':True,'seed':SEED,'data_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'agents':AGENTS,'n_images':len(d),'primary_task':'mean of two asymmetric reference matches','rsa_entities':{'shape':36,'illumination':36},'rsa_distance':'mean squared Euclidean profile distance','rsa_statistic':'Spearman upper triangle; two-sided node-label permutation; BH across all 18 full-profile comparisons','n_permutations':NPERM,'n_node_bootstraps':NBOOT,'spaces':['raw','physical_color_adjusted','interaction_only'],'nuisance':'OLS: 50-level physical c, chroma categories, sine/cosine hue harmonics 1 and 2; fit only on training views. No held-out view coefficient estimated.','residuals':'out-of-view nuisance residuals; all subjects fit separately','low_rank':'sequential nuisance regression then doubly centered shape-by-illumination residual matrix SVD; signed, not sparse/nonnegative SPoSE','ranks':RANKS,'shrinkage':SHRINK,'validation':'outer leave-one-view-out; inner two directional single-view validation on remaining views; choose minimum MSE, tie lower rank/shrinkage. Refits on two training views. All shape-light cells seen in train; unseen-view prediction, not unseen shape/light generalization.','primary_comparisons':'RSA interaction_only; nested prediction against additive baseline','scope':'Exploratory condition-profile behavior spaces, not neural representations or independent semantic/material similarity judgments. No new acquisitions. Full dataset includes prior development/pilot images; not a newly collected confirmatory cohort.'}
 save_json('analysis_plan.json',plan)
 rng=np.random.default_rng(SEED); cubes={};cvrows=[];predrows=[];selection=[];spectra=[];recon={};dimrows=[];fullspaces={}
 for agent,col in AGENTS.items():
  print('Fitting',agent,flush=True);y=d[col].to_numpy();res=np.zeros(len(d));pred={name:np.zeros(len(d)) for name in ['nuisance','additive','selected']+[f'rank{k}' for k in RANKS[1:]]}
  for testview in views:
   tr=v!=testview;te=~tr;inner=[]
   for k in RANKS:
    for alpha in ([1.] if k==0 else SHRINK):
     errors=[]
     for valview in views:
      if valview==testview:continue
      it=tr&(v!=valview);iv=tr&(v==valview)
      beta=fit_nuisance(X,y,it);r=y-X@beta;f=decompose(r,s,l,it);p=X[iv]@beta+component(f,k,alpha)[s[iv],l[iv]];errors.extend((y[iv]-p)**2)
     inner.append((float(np.mean(errors)),k,alpha))
   best=sorted(inner)[0];_,kbest,abest=best;selection.append(dict(agent=agent,test_view=int(testview),rank=kbest,shrinkage=abest,inner_mse=best[0]))
   for mse,k,a in inner:cvrows.append(dict(agent=agent,outer_view=int(testview),stage='inner',rank=k,shrinkage=a,mse=mse))
   beta=fit_nuisance(X,y,tr);r=y-X@beta;res[te]=r[te];f=decompose(r,s,l,tr)
   pred['nuisance'][te]=X[te]@beta;pred['additive'][te]=X[te]@beta+component(f,0)[s[te],l[te]];pred['selected'][te]=X[te]@beta+component(f,kbest,abest)[s[te],l[te]]
   for k in RANKS[1:]:pred[f'rank{k}'][te]=X[te]@beta+component(f,k)[s[te],l[te]]
  for name,p in pred.items():
   cvrows.append(dict(agent=agent,outer_view='pooled',stage='outer',rank=name,shrinkage=1,mse=float(np.mean((p-y)**2)),r2=r2_score(y,p)))
  for i,row in d.iterrows():predrows.append(dict(image_id=row.image_id,agent=agent,view=int(v[i]),shape_id=int(row.shape_id),illumination_id=int(row.illumination_id),target=y[i],nuisance_residual=res[i],**{name:p[i] for name,p in pred.items()}))
  raw=np.empty((3,36,36));adjusted=raw.copy()
  for j,view in enumerate(views):
   mask=v==view;raw[j,s[mask],l[mask]]=y[mask];adjusted[j,s[mask],l[mask]]=res[mask]
  cubes[agent]={'raw':raw,'physical_color_adjusted':adjusted,'interaction_only':np.array([center(x) for x in adjusted])}
  mean=cubes[agent]['interaction_only'].mean(0);u,sv,vt=np.linalg.svd(mean,full_matrices=False);rsel=int(pd.Series([x['rank'] for x in selection if x['agent']==agent]).mode().iloc[0]);display_rank=max(2,rsel)
  for j,z in enumerate(sv):spectra.append(dict(agent=agent,dimension=j+1,variance_fraction=z*z/(sv@sv),cumulative=float(np.sum(sv[:j+1]**2)/(sv@sv)),nested_rank_mode=rsel))
  for factor,coords,ids in [('shape',u*sv,shapes),('illumination',vt.T*sv,lights)]:
   tab=pd.DataFrame(coords,columns=[f'dim_{i+1}' for i in range(36)]);tab.insert(0,'condition_id',ids);tab.to_csv(O/f'embeddings/{agent}_{factor}_coordinates.csv',index=False)
   fullspaces[agent,factor]=coords
   for j in range(min(4,display_rank)):
    order=np.argsort(coords[:,j]);dimrows.append(dict(agent=agent,factor=factor,dimension=j+1,low_ids=';'.join(map(str,np.asarray(ids)[order[:5]])),high_ids=';'.join(map(str,np.asarray(ids)[order[-5:]]))))
  delta=(pred['additive']-y)**2-(pred['selected']-y)**2;boots=[]
  for _ in range(NBOOT):
   w=rng.multinomial(36,np.ones(36)/36)[s]*rng.multinomial(36,np.ones(36)/36)[l];boots.append(np.average(delta,weights=w))
  recon[agent]={'selected_ranks':[x['rank'] for x in selection if x['agent']==agent],'r2_nuisance':r2_score(y,pred['nuisance']),'r2_additive':r2_score(y,pred['additive']),'r2_selected':r2_score(y,pred['selected']),'mse_reduction_vs_additive':float(delta.mean()/np.mean((pred['additive']-y)**2)),'delta_mse_ci95':np.quantile(boots,[.025,.975]).tolist()}
 csv('tables/predictions.csv',predrows);csv('tables/cv_all_candidates.csv',cvrows);csv('tables/selected_dimensions.csv',selection);csv('tables/spectrum.csv',spectra);csv('tables/dimension_extremes.csv',dimrows)
 save_json('prediction_summary.json',recon)
 rs=[];rdms={};stability=[];cross=[]
 for space in ['raw','physical_color_adjusted','interaction_only']:
  for factor in ['shape','illumination']:
   profiles={a:(c[space] if factor=='shape' else c[space].transpose(0,2,1)) for a,c in cubes.items()}
   for a,x in profiles.items():
    R=distances(x.mean(0));rdms[space,factor,a]=R
    np.save(O/f'matrices/{space}_{factor}_{a}_profiles_by_view.npy',x);np.savetxt(O/f'matrices/{space}_{factor}_{a}_rdm.csv',R,delimiter=',')
    for i,j in itertools.combinations(range(3),2):stability.append(dict(space=space,factor=factor,agent=a,view1=i+1,view2=j+1,rsa=corr(squareform(distances(x[i])),squareform(distances(x[j])))))
   for a,b in itertools.combinations(AGENTS,2):
    obs,p,ci,null=rsa_stats(rdms[space,factor,a],rdms[space,factor,b],rng);rs.append(dict(space=space,factor=factor,agent1=a,agent2=b,rsa=obs,p_permutation=p,ci_low=ci[0],ci_high=ci[1]))
    np.save(O/f'matrices/null_{space}_{factor}_{a}_{b}.npy',null)
    vals=[corr(squareform(distances(profiles[a][i])),squareform(distances(profiles[b][j]))) for i in range(3) for j in range(3) if i!=j]
    cross.append(dict(space=space,factor=factor,agent1=a,agent2=b,mean_cross_view_rsa=float(np.mean(vals)),min=float(min(vals)),max=float(max(vals))))
   print('RSA',space,factor,flush=True)
 rs=pd.DataFrame(rs);rs['q_bh_18']=bh(rs.p_permutation);rs.to_csv(O/'tables/rsa_results.csv',index=False);csv('tables/cross_view_rsa.csv',cross);csv('tables/view_stability.csv',stability)
 # Descriptive truncated latent-space RSA: dimension selected using nested behavior prediction.
 latent=[]
 for factor in ['shape','illumination']:
  for a,b in itertools.combinations(AGENTS,2):
   ka=int(pd.Series(recon[a]['selected_ranks']).mode().iloc[0]);kb=int(pd.Series(recon[b]['selected_ranks']).mode().iloc[0])
   value=corr(squareform(distances(fullspaces[a,factor][:,:ka])),squareform(distances(fullspaces[b,factor][:,:kb]))) if min(ka,kb)>0 else None
   latent.append(dict(factor=factor,agent1=a,agent2=b,rank1=ka,rank2=kb,rsa=value,scope='descriptive full-data embedding; full-profile permutation RSA is primary'))
 csv('tables/latent_rsa.csv',latent)
 # Figures, with identically ordered labels but independently estimated spaces.
 plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
 for factor in ['shape','illumination']:
  fig,axes=plt.subplots(2,3,figsize=(12,7.5),constrained_layout=True)
  for j,a in enumerate(AGENTS):
   x=cubes[a]['interaction_only'].mean(0);x=x if factor=='shape' else x.T
   im=axes[0,j].imshow(x/np.std(x),cmap='RdBu_r',vmin=-3,vmax=3);axes[0,j].set_title(a+' | interaction profiles');axes[0,j].set_xlabel('Illumination' if factor=='shape' else 'Shape');axes[0,j].set_ylabel(factor.title())
   r=rdms['interaction_only',factor,a];axes[1,j].imshow(r/np.mean(r[np.triu_indices(36,1)]),cmap='magma',vmin=0,vmax=3);axes[1,j].set_title('Relative behavioral dissimilarity');axes[1,j].set_xlabel(factor.title());axes[1,j].set_ylabel(factor.title())
  for ext in ['png','svg','pdf']:fig.savefig(O/f'figures/Fig1_{factor}_profiles_RDM.{ext}',dpi=180)
  plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
 for j,a in enumerate(AGENTS):
  t=pd.DataFrame(cvrows);t=t[(t.agent==a)&(t.stage=='outer')].set_index('rank');ys=[t.loc['additive','r2']]+[t.loc[f'rank{k}','r2'] for k in RANKS[1:]];axes[j].plot(RANKS,ys,'o-',label='Fixed rank, no shrinkage');axes[j].axhline(recon[a]['r2_selected'],color='tab:red',ls='--',label='Nested selection');axes[j].axhline(recon[a]['r2_nuisance'],color='gray',ls=':',label='Physical/color baseline');axes[j].set(title=a,xlabel='Interaction rank (0 = additive)',ylabel='Held-view R²');axes[j].legend(fontsize=8)
 for ext in ['png','svg','pdf']:fig.savefig(O/f'figures/Fig2_held_view_prediction.{ext}',dpi=180)
 plt.close(fig)
 fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
 for i,factor in enumerate(['shape','illumination']):
  for j,a in enumerate(AGENTS):
   z=fullspaces[a,factor];axes[i,j].scatter(z[:,0],z[:,1],c=np.arange(36),cmap='turbo',s=22)
   for k in range(36):axes[i,j].annotate(str((shapes if factor=='shape' else lights)[k]),z[k,:2],fontsize=6)
   axes[i,j].set(title=f'{a}: {factor}',xlabel='Dimension 1 (independent axes)',ylabel='Dimension 2')
 fig.suptitle('Descriptive 2D projection; dimension evidence comes from held-view prediction')
 for ext in ['png','svg','pdf']:fig.savefig(O/f'figures/Fig3_latent_spaces.{ext}',dpi=180)
 plt.close(fig)
 save_json('summary.json',{'prediction':recon,'rsa':rs.to_dict('records'),'latent_rsa':latent,'cross_view_rsa':cross})
 methods='''# Behavioral spaces and RSA v1\n\nThis analysis replaces the earlier proposed hidden-layer route. It uses only existing behavioral outputs, separately for Human, InternVL, and Qwen3. It is an exploratory analysis, not preregistered or original SPoSE.\n\nInput: 3,888 shared images, 36 shapes × 36 illuminations × 3 viewpoints; human mean match and model mean of two matches. All prior pilot/development images retained and flagged in source. No subjects' individual human judgments are available in this analysis, so no participant noise ceiling or subject-level uncertainty is claimed.\n\nThree response spaces: raw mean profiles; residuals after physical/color nuisance adjustment; and doubly centered residual interactions (removing shape and illumination additive means per view). Shape space has 36 illumination response coordinates; illumination space has 36 shape coordinates. Units are conditions, not individual images or discovered semantic attributes. The same experimental design enables matched comparisons without fitting model coordinates to human coordinates.\n\nNuisance OLS is fitted on other views: 50-level physical c, chroma categories, hue Fourier harmonics 1 and 2. No held-view offset is estimated. Residualizing is sequential, not joint alternating least squares; correlations or prediction gains do not prove causal factors or remove unmeasured image confounds. Common physical/color controls can also induce shared residual patterns, so raw, adjusted, interaction-only, and cross-view results are all retained.\n\nLow-rank factorization: residual cell means = shape/illumination additive component + rank-k signed interaction. Candidates k=0,1,2,4,8; singular-value multipliers .25,.5,1. Inner validation trains on one remaining view and validates on the other, in both directions. Selection minimizes inner MSE. Outer test is the third view; all 1,296 shape-illumination pairs remain observed during training. Thus test evaluates view transfer, not entirely unseen shape, illumination, or cell prediction. The fixed-rank curve is diagnostic; only nested selected performance is selection-independent. Squared-error difference confidence intervals resample shape and illumination clusters on fixed outer predictions, not model refitting or independent human subjects.\n\nRSA uses mean squared Euclidean distances of full response profiles, then Spearman correlation of the 630 upper-triangle entries. Two-sided node-label permutations (4,999) preserve RDM dependencies. BH controls the family of 18 full-profile comparisons. Confidence intervals resample the 36 condition nodes (2,000), excluding repeated-node diagonal pairs; these condition-level intervals do not generalize to human participant populations. Cross-view RSA compares different views across agents; within-agent view stability is descriptive and is not a noise ceiling because images/physical parameters differ across views.\n\nCoordinates are U*S for shape and V*S for illumination; these preserve truncated row/column Euclidean geometry respectively. They are separate coordinate systems, not two factor sets to multiply directly (use U*S*V-transpose for reconstruction). SVD signs/axes are arbitrary. Full-data two-dimensional maps and truncated latent RSA are descriptive, no fitted Procrustes alignment or per-axis psychological labeling. A modal selected rank of zero means no supported interaction embedding; a 2D drawing alone does not establish dimensions.\n\nReferences: Kriegeskorte et al. (2008), https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/neuro.06.004.2008/full ; Hebart et al. (2020), https://www.nature.com/articles/s41562-020-00951-3 . SPoSE motivates behavioral latent structure; this analysis uses signed low-rank regression appropriate to scalar matched judgments.\n\nReproduction: place this script in benchmark/scripts and the frozen input at benchmark/mllm_material/qwen3_full_v1/joint_model_predictions.csv, or restore that path using inputs/joint_model_predictions.csv. Run the script with the archived requirements/environment. No model inference, network, or GPU is needed. All intermediate predictions, rank candidates, condition profiles, distance matrices, coordinates, permutation nulls and figure source data are saved.\n'''
 (O/'Methods.md').write_text(methods,encoding='utf-8')
 content='<html><head><meta charset="utf-8"><title>行为潜在空间与 RSA</title><style>body{font-family:Arial,sans-serif;max-width:1150px;margin:35px auto;line-height:1.6;padding:20px}table{border-collapse:collapse;font-size:14px}td,th{border:1px solid #ccc;padding:7px}img{width:100%}h1,h2{color:#123e59}</style></head><body><h1>人类 · InternVL · Qwen3：行为表征结构</h1><p>完成：3,888 图；36 形状 × 36 照明 × 3 视角。独立学习行为结构，不提取神经网络隐藏层，也不强制对齐坐标。</p><h2>1 留出视角预测：潜在维度是否有用？</h2>'+pd.DataFrame(recon).T.to_html()+'<img src="figures/Fig2_held_view_prediction.png"><h2>2 表征相似性：整体与扣除平均效应后的结构</h2>'+rs.round(5).to_html(index=False)+'<p>raw：原始响应；physical_color_adjusted：物理与颜色调整；interaction_only：再扣除形状、照明平均效应。p 为条件标签置换；q 为 18 项比较的 BH 校正。区间反映条件重采样，不是人群置信区间。</p><img src="figures/Fig1_shape_profiles_RDM.png"><img src="figures/Fig1_illumination_profiles_RDM.png"><h2>3 潜在空间（描述性投影）</h2><img src="figures/Fig3_latent_spaces.png">'+pd.DataFrame(latent).to_html(index=False)+'<h2>4 不同视角之间的结构对照</h2>'+pd.DataFrame(cross).round(4).to_html(index=False)+'<p>这是已见形状×照明组合的视角迁移分析，不是新形状泛化。三方共享刺激，不能排除共同图像特征；尚未给潜在轴赋予心理属性名称。</p><p><a href="Methods.md">完整方法与范围</a> · <a href="analysis_plan.json">分析设置</a> · <a href="tables/predictions.csv">全部留出预测</a> · <a href="tables/rsa_results.csv">RSA 表</a> · <a href="behavioral_spaces_reproducibility.zip">复现包</a></p></body></html>'
 (O/'report.html').write_text(content,encoding='utf-8')
 save_json('environment.json',{'python':platform.python_version(),'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__})
 req=M/'qwen3_full_v1/requirements.lock.txt'
 if req.exists():shutil.copy2(req,O/'requirements.lock.txt')
 save_json('validation.json',{'n_unique_images':3888,'complete_shape_light_view_grid':True,'outer_predictions_per_agent':3888,'all_predictions_finite':bool(np.isfinite(pd.DataFrame(predrows).select_dtypes('number')).all().all()),'rsa_comparisons':len(rs),'rsa_permutation_unit':'36 condition labels, not 630 independent distances','no_model_to_human_coordinate_fit':True})
 files=[p for p in O.rglob('*') if p.is_file() and p.suffix!='.zip' and p.name!='artifact_manifest.json'];save_json('artifact_manifest.json',[{'path':str(p.relative_to(O)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in files])
 with zipfile.ZipFile(O/'behavioral_spaces_reproducibility.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in O.rglob('*'):
   if p.is_file() and p.suffix!='.zip':z.write(p,p.relative_to(O))
 print(json.dumps(recon,indent=2),flush=True);print(rs.to_string(index=False),flush=True)

if __name__=='__main__':main()
