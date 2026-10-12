"""Readout, spatial response and intervention analyses for frozen internal representations."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import json,hashlib,shutil,zipfile,itertools
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1];O=B/'mllm_material/mechanisms_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def rho(x,y):return float(spearmanr(x,y).statistic)
def linear(X,Y,a):return np.linalg.solve(X.T@X+a*np.eye(X.shape[1]),X.T@Y)
def fitpred(x,y,tr,te,a):
 mu=x[tr].mean(0);sd=x[tr].std(0);sd[sd<1e-6]=1;ym=y[tr].mean(0);z=(x-mu)/sd
 return z[te]@linear(z[tr],y[tr]-ym,a)+ym
def nuisance(d):return np.column_stack([np.ones(len(d)),pd.get_dummies(d.physical_c).to_numpy()[:,1:],pd.get_dummies(d.chroma).to_numpy()[:,1:]]+[f(n*np.deg2rad(d.hue.to_numpy())) for n in [1,2] for f in [np.sin,np.cos]])
def main():
 for sub in ['tables','figures','code']: (O/sub).mkdir(exist_ok=True)
 shutil.copy2(__file__,O/'code'/Path(__file__).name);d=pd.read_csv(O/'stimuli.csv');plan=read(O/'plan.json');N=nuisance(d);rng=np.random.default_rng(plan['seed']);stats=[];rawpred=[];dirstats=[];ivs=[];ms=[];featuremaps={}
 for name,target in [('InternVL','internvl_c'),('Qwen3','model_c')]:
  out=O/name
  assert read(out/'status.json')['phase']=='complete',name+' incomplete'
  sample=np.load(out/'features'/f'{d.image_id.iloc[0]}.npz');keys=[k for k in sample.files if k not in ['bridge_tokens','last_token_logits']]
  matrices={k:[] for k in keys};token_scores=[];brightness=[];contrast=[];paths=[];directions=np.load(out/'directions.npz');direction=directions['candidate'];refscore=directions['reference_means']@direction;scale=float(directions['reference_scale'])
  for row in d.itertuples():
   z=np.load(out/'features'/f'{row.image_id}.npz')
   for k in keys:matrices[k].append(z[k])
   scores=z['bridge_tokens'].astype('float32')@direction;token_scores.append(scores);meta=read(out/'features'/f'{row.image_id}.json');brightness.append(meta['brightness']);contrast.append(meta['contrast'])
  save(out/'reference_direction_stats.json',dict(reference_monotonicity=rho(np.arange(52),refscore),physical_relation=rho(d.physical_c,np.array(matrices['bridge'])@direction),human_relation=rho(d.human_c,np.array(matrices['bridge'])@direction),model_relation=rho(d[target],np.array(matrices['bridge'])@direction),brightness_relation=rho(brightness,np.array(matrices['bridge'])@direction),candidate_brightness_cosine=read(out/'direction_metadata.json')['candidate_brightness_cosine']))
  dirstats.append(dict(model=name,**read(out/'reference_direction_stats.json')))
  y=np.column_stack([d.human_c,d.physical_c,d[target],d.human_c]);targetnames=['human','physical','model_behavior','human_beyond_physical_color']
  for key,values in matrices.items():
   x=np.asarray(values,dtype=float)
   # A fixed label-free projection limits probe capacity; archived for exact rerun.
   seed=plan['seed']+int(hashlib.sha256((name+key).encode()).hexdigest()[:6],16);prng=np.random.default_rng(seed);projection=prng.normal(size=(x.shape[1],128))/np.sqrt(128);np.save(out/f'projection_{key}.npy',projection.astype('float32'));x=x@projection
   for split in ['shape','illumination']:
    folds=d[split+'_fold'].to_numpy();pred=np.zeros_like(y);basepred=np.zeros(len(d));selected=[]
    for k in range(12):
     tr=folds!=k;te=~tr;other=sorted(set(folds[tr]));iv=tr&np.isin(folds,other[::4]);it=tr&~iv
     yi=y.copy();binner=np.linalg.lstsq(N[it],y[it,0],rcond=None)[0];yi[:,3]=y[:,0]-N@binner
     losses=np.stack([np.mean((fitpred(x,yi,it,iv,a)-yi[iv])**2,0) for a in [1.,10.,100.]]);best=np.argmin(losses,0)
     b=np.linalg.lstsq(N[tr],y[tr,0],rcond=None)[0];yt=y.copy();yt[:,3]=y[:,0]-N@b;basepred[te]=N[te]@b
     for j in range(4):
      alpha=[1.,10.,100.][best[j]];pred[te,j]=fitpred(x,yt[:,j:j+1],tr,te,alpha).ravel();selected.append(dict(fold=k,target=targetnames[j],alpha=alpha))
     pred[te,3]+=basepred[te]
    save(out/f'probe_settings_{key}_{split}.json',selected)
    for j,label in enumerate(targetnames):
     stats.append(dict(model=name,layer=key,split=split,target=label,n=len(d),spearman=rho(y[:,j],pred[:,j]),r2=r2_score(y[:,j],pred[:,j]),baseline_r2=r2_score(y[:,0],basepred) if j==3 else None))
     for i,id in enumerate(d.image_id):rawpred.append(dict(model=name,layer=key,split=split,target=label,image_id=id,truth=y[i,j],prediction=pred[i,j]))
   print('Probe',name,key,flush=True)
  scores=np.array([s.mean() for s in token_scores]);csv=pd.DataFrame(dict(image_id=d.image_id,projection=scores,brightness=brightness,contrast=contrast));csv.to_csv(out/'candidate_scores.csv',index=False)
  # Show spatial extrema only within the preselected intervention set, retain all scores separately.
  eligible=np.flatnonzero(d.image_id.isin(plan['intervention_images']));order=eligible[np.argsort(scores[eligible])];picked=np.r_[order[:2],order[-2:]];featuremaps[name]=(picked,token_scores,refscore)
  for p in (out/'interventions').glob('*.json'):
   r=read(p);a=r['candidate_probabilities'];r['model']=name;r['test_glossier_probability']=a[r['order']]+.5*a[2];r['logodds_test_vs_reference']=r['candidate_logits'][r['order']]-r['candidate_logits'][1-r['order']];r['unrestricted_top_is_candidate']=r['top_token_text'] in ['A','B','SAME'];ivs.append({k:v for k,v in r.items() if k not in ['rendered_prompt','paths','candidate_logits','candidate_probabilities']})
  for p in (out/'matching').glob('*.json'):
   r=read(p);ms.append(dict(model=name,image_id=r['image_id'],direction=r['direction'],dose=r['dose'],matching_c=r['matching_c'],steps=r['steps'],stop_reason=r['stop_reason']))
 st=pd.DataFrame(stats);st.to_csv(O/'tables/layer_readouts.csv',index=False);pd.DataFrame(rawpred).to_csv(O/'tables/readout_predictions.csv.gz',index=False,compression='gzip');pd.DataFrame(dirstats).to_csv(O/'tables/direction_associations.csv',index=False)
 iv=pd.DataFrame(ivs);base=iv[iv.direction=='baseline'][['model','image_id','order','test_glossier_probability','logodds_test_vs_reference']].rename(columns={'test_glossier_probability':'baseline_p','logodds_test_vs_reference':'baseline_logodds'});iv=iv.merge(base,on=['model','image_id','order']);iv['delta_p']=iv.test_glossier_probability-iv.baseline_p;iv['delta_logodds']=iv.logodds_test_vs_reference-iv.baseline_logodds;iv.to_csv(O/'tables/interventions.csv',index=False)
 summaries=[]
 for (name,direction,dose),g in iv.groupby(['model','direction','dose']):
  z=g.groupby('image_id').delta_p.mean().to_numpy();boots=np.array([rng.choice(z,len(z),replace=True).mean() for _ in range(4000)]);summaries.append(dict(model=name,direction=direction,dose=dose,n_images=len(z),mean_delta_p=z.mean(),ci_low=np.quantile(boots,.025),ci_high=np.quantile(boots,.975),mean_delta_logodds=g.delta_logodds.mean(),candidate_mass=g.candidate_mass.mean(),top_candidate_fraction=g.unrestricted_top_is_candidate.mean()))
 sm=pd.DataFrame(summaries);sm.to_csv(O/'tables/intervention_summary.csv',index=False);pd.DataFrame(ms).to_csv(O/'tables/matching_interventions.csv',index=False)
 # Paired candidate-vs-control dose effects, both image orders averaged before uncertainty.
 contrasts=[]
 for name,g in iv.groupby('model'):
  pivot=g.groupby(['image_id','direction','dose']).delta_p.mean().unstack(['direction','dose'])
  for control in ['brightness','random0','random1','random2']:
   val=(pivot[('candidate',1.)]-pivot[('candidate',-1.)])-(pivot[(control,1.)]-pivot[(control,-1.)]);boots=np.array([rng.choice(val,len(val),replace=True).mean() for _ in range(4000)]);contrasts.append(dict(model=name,control=control,difference_in_bidirectional_effect=val.mean(),ci_low=np.quantile(boots,.025),ci_high=np.quantile(boots,.975)))
 pd.DataFrame(contrasts).to_csv(O/'tables/intervention_control_contrasts.csv',index=False)
 plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
 def output(fig,name):
  for ext in ['png','pdf','svg']:fig.savefig(O/f'figures/{name}.{ext}',dpi=170,bbox_inches='tight')
  plt.close(fig)
 fig,axs=plt.subplots(2,2,figsize=(14,8),constrained_layout=True)
 for col,name in enumerate(['InternVL','Qwen3']):
  for row,split in enumerate(['shape','illumination']):
   ax=axs[row,col];g=st[(st.model==name)&(st.split==split)];keys=list(g.layer.unique())
   for target in ['human','physical','model_behavior']:
    z=g[g.target==target].set_index('layer').loc[keys];ax.plot(range(len(keys)),z.r2,'o-',label=target)
   ax.set_xticks(range(len(keys)),keys,rotation=60,ha='right',fontsize=8);ax.set(title=name+' | leave '+split+' groups out',ylabel='Held-out readout R²');ax.legend()
 output(fig,'Mechanism1_layer_information')
 fig,axs=plt.subplots(2,5,figsize=(16,7),constrained_layout=True)
 for row,name in enumerate(['InternVL','Qwen3']):
  picked,tokens,refscore=featuremaps[name];axs[row,0].plot(np.linspace(0,.1487,52),refscore);axs[row,0].set(title=name+' | reference sequence',xlabel='Reference physical c',ylabel='Candidate projection')
  for j,i in enumerate(picked):
   image=np.asarray(Image.open(B/d.image_512_path.iloc[i]).convert('RGB'));score=tokens[i];side=int(np.sqrt(len(score)));assert side*side==len(score);heat=score.reshape(side,side);heat=np.asarray(Image.fromarray(heat).resize((image.shape[1],image.shape[0]),Image.Resampling.BILINEAR));ax=axs[row,j+1];ax.imshow(image);ax.imshow(heat,cmap='magma',alpha=.55);ax.set_title(d.image_id.iloc[i]+'\nCandidate response (not attention)',fontsize=9);ax.axis('off')
 output(fig,'Mechanism2_reference_and_spatial_response')
 fig,axs=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
 for ax,name in zip(axs,['InternVL','Qwen3']):
  for direction in ['candidate','brightness','random0','random1','random2']:
   g=sm[(sm.model==name)&(sm.direction==direction)].sort_values('dose');ax.errorbar(g.dose,g.mean_delta_p,yerr=[g.mean_delta_p-g.ci_low,g.ci_high-g.mean_delta_p],fmt='o-',capsize=3,label=direction)
  ax.axhline(0,color='gray',ls=':');ax.set(title=name,xlabel='Dose (reference-range activation displacement)',ylabel='Change in P(test glossier), A/B/SAME-normalized');ax.legend(fontsize=8)
 output(fig,'Mechanism3_internal_intervention')
 fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True);mat=pd.DataFrame(ms)
 for ax,name in zip(axs,['InternVL','Qwen3']):
  z=mat[mat.model==name];base=z[z.direction=='baseline'].set_index('image_id').matching_c
  for direction in ['candidate','random0']:
   means=[]
   for dose in [-1.,1.]:
    a=z[(z.direction==direction)&(z.dose==dose)].set_index('image_id').matching_c;means.append(float((a-base).mean()))
   ax.plot([-1,1],means,'o-',label=direction)
  ax.axhline(0,color='gray',ls=':');ax.set(title=name,xlabel='Intervention dose',ylabel='Change in full matching c (12 images)');ax.legend()
 output(fig,'MechanismS1_full_matching')
 methods='''# Frozen internal-response mechanism analysis v1\n\nWeights are frozen throughout; ridge probes are separately supervised analysis tools. Inputs are the original full stimulus set and reconstructed references. Exact prompts, image hashes, layer definitions, feature arrays, full bridge tokens and next-token vocabulary logits are archived per stimulus. No human labels enter extraction or the reference-defined intervention direction.\n\nLocalization: selected vision blocks are mean-pooled; language blocks use only the final prompt token before an answer. These are different pooling choices, so a rising/descending curve is not automatically a processing hierarchy. Qwen multi-level DeepStack outputs are separately retained and reported.\n\nReadout: fixed label-free Gaussian projection to 128 features, followed by standardized ridge regression. Outer 12-group shape/illumination splits use existing folds; alpha 1/10/100 selected on a fixed group-disjoint inner validation subset. Projection matrices and all per-image predictions are saved. No layer is chosen by outer performance. Human-beyond-physical/color targets use training-only 50-level c, chroma and hue-harmonic nuisance regression; this is a residual readout, not proof of nonlinear causal specificity.\n\nDirection: normalized multivariate slope of reference mean bridge activation on c. Dose one equals the fitted full reference-range change in mean activation. The direction is added uniformly to every visual token of the TEST image only. This is neither neural deletion nor an on-manifold intervention. It may change several entangled image properties. Brightness direction is a whole-image-luminance covariance direction on non-intervention images, not ground-truth pure brightness. Three seeded random unit directions orthogonal to the candidate control intervention norm. Qwen changes its final merger output only, leaving DeepStack paths intact.\n\nChoice outcome: A, B and SAME are verified single tokens in both tokenizers. The primary outcome is their next-token normalized probability, with SAME assigned half credit. Unrestricted top tokens and total candidate probability mass are retained; normalized probabilities are not unconditional valid-answer probabilities. Full matching uses constrained argmax among these tokens with the same bisection controller, so it is a mechanistic assay, not a replacement for previously acquired free-generation behavior. Order is counterbalanced in the 72-image assay, but not in 12-image matching.\n\nSpatial maps are raw candidate-direction projections of bridge tokens, not attention or causal attribution maps. Four illustrative examples per model are lowest/highest whole-image candidate projection in the preselected intervention subset, not chosen on human agreement. Whole-image brightness and contrast are image-statistic controls, not highlighter masks.\n\nUncertainty: intervention bootstrap resamples images after averaging both orders, 4,000 replicates, conditional on fixed selected images/direction/model. It is not uncertainty over shapes, training runs or human populations. Candidate/control bidirectional contrasts are descriptive with unadjusted intervals. No automatic claim that interventions demonstrate a pure gloss circuit or shared human neural mechanism.\n'''
 (O/'Methods.md').write_text(methods,encoding='utf-8');save(O/'summary.json',dict(direction_associations=dirstats,interventions=summaries,control_contrasts=contrasts,n_extracted_per_model=3940,n_interventions_per_model=3024,n_matching_per_model=60))
 from mechanism_spatial_v1 import run as spatial_run
 spatial=spatial_run();spatial_table=spatial.groupby(['model','control']).spatial_spearman.mean().unstack(0)
 html='<html><head><meta charset="utf-8"><title>冻结模型内部响应与光泽机制</title><style>body{font-family:Arial;max-width:1200px;margin:35px auto;line-height:1.7}img{width:100%}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccc;padding:6px}</style></head><body><h1>InternVL · Qwen3：冻结模型内部响应与光泽判断</h1><p>全量响应提取；分组留出读出；参考序列方向；72 图双顺序干预；12 图完整匹配。权重不变。机制候选与亮度、随机方向对照一起报告。</p><h2>1 信息定位</h2><img src="figures/Mechanism1_layer_information.png">'+st.round(4).to_html(index=False)+'<h2>2 候选特征</h2><img src="figures/Mechanism2_reference_and_spatial_response.png">'+pd.DataFrame(dirstats).round(4).to_html(index=False)+'<h3>空间响应与图像特征、小网络原始核的关联</h3>'+spatial_table.round(4).to_html()+'<p>72 张预选图，逐图计算空间相关后平均。小网络核来自作者全量训练版本，仅比较空间滤波响应；不是原网络完整预测，也不是独立人类测试证据。<a href="Spatial_controls.md">空间分析方法</a></p><h2>3 内部干预</h2><img src="figures/Mechanism3_internal_intervention.png">'+sm.round(4).to_html(index=False)+pd.DataFrame(contrasts).round(4).to_html(index=False)+'<h2>4 完整匹配变化</h2><img src="figures/MechanismS1_full_matching.png"><p>匹配为 A/B/SAME 候选 token 的受限选择；不等同于原始自由生成采集。干预响应可提示模型利用了该方向，但其纯光泽特异性必须依据对照判断，不能仅凭响应改变下结论。</p><p><a href="Methods.md">完整方法</a> · <a href="plan.json">执行方案</a> · <a href="tables/interventions.csv">逐次干预数据</a></p></body></html>'
 (O/'report.html').write_text(html,encoding='utf-8');print('Analysis complete',flush=True)
if __name__=='__main__':main()
