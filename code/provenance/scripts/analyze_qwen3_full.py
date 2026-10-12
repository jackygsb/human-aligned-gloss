"""Automatic interim/final three-task analysis, paired to the InternVL protocols."""
from pathlib import Path
import json,html
import numpy as np,pandas as pd
from scipy.stats import spearmanr,pearsonr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1];M=B/'mllm_material';O=M/'qwen3_full_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rho(x,y):return float(spearmanr(x,y).statistic) if len(set(x))>1 and len(set(y))>1 else None
def main():
 a=pd.read_csv(B/'annotations/gloss_annotations.csv');f=pd.read_csv(M/'matching_full_v1/image_summary.csv');fold=pd.read_csv(B/'protocols/gloss_group_folds.csv')
 d=f[['image_id','human_c','physical_c','shape_id','illumination_id','viewpoint_id','cohort']].merge(a[['image_id','hue','chroma','source_image_index']],on='image_id').merge(fold[['image_id','shape_fold','illumination_fold']],on='image_id')
 single=[];pairs=[]
 for path in sorted((O/'raw').glob('T*.json')):
  r=read(path);v=json.loads(r['raw']);meta=r['meta']
  if meta['task']=='T1':single.append(dict(image_id=meta['image_ids'][0],single_score=v['glossiness']))
  else:pairs.append(dict(**meta,choice=v['choice']))
 if single:d=d.merge(pd.DataFrame(single),on='image_id',how='left')
 else:d['single_score']=np.nan
 matches=read(O/'matches.json') if (O/'matches.json').exists() else []
 if matches:
  t=pd.DataFrame(matches);t.to_csv(O/'matches.csv',index=False);z=t.pivot(index='image_id',columns='repetition',values='matching_c').dropna()
  if 1 in z and 2 in z:
   z=z.rename(columns={1:'session1_c',2:'session2_c'});z['model_c']=z.mean(axis=1);d=d.merge(z,on='image_id',how='left')
 if 'model_c' not in d:d['model_c']=np.nan
 cfg=read(O/'protocol.json');d['prior_pilot']=d.image_id.isin(cfg['pilot_image_ids']);d['validation']=d.cohort.eq('heldout_matching')&~d.prior_pilot
 d.to_csv(O/'image_summary.csv',index=False)
 stats=[]
 for cohort,mask in [('all',np.ones(len(d),bool)),('matching_heldout',d.cohort.eq('heldout_matching')),('new_validation',d.validation)]:
  for col in ['single_score','model_c']:
   z=d[mask].dropna(subset=[col])
   if len(z)<4:continue
   stats.append(dict(cohort=cohort,task=col,n=len(z),human_spearman=rho(z.human_c,z[col]),human_pearson=float(pearsonr(z.human_c,z[col]).statistic)))
 pd.DataFrame(stats).to_csv(O/'human_agreement.csv',index=False)
 pdata=[];spec=pd.read_csv(M/'manifests/pairs2000_analysis_only.csv');idx=d.set_index('source_image_index');by={}
 for r in pairs:by.setdefault(int(r['group'][4:]),[]).append(r)
 for k,rr in by.items():
  if len(rr)!=2:continue
  p=spec.iloc[k];aa=int(p.a);bb=int(p.b);choices=[.5 if r['choice']=='tie' else float(r['indices'][ord(r['choice'])-65]==aa) for r in rr];x=idx.loc[aa+1];y=idx.loc[bb+1]
  pdata.append(dict(pair_id=k,stratum=p.stratum,choice_a=np.mean(choices),swap_consistent=choices[0]==choices[1],delta_single=x.single_score-y.single_score,delta_matching=x.model_c-y.model_c,delta_human=x.human_c-y.human_c,delta_physical=x.physical_c-y.physical_c))
 p=pd.DataFrame(pdata);pm=[]
 if len(p):
  for stratum,z in [('all',p)]+list(p.groupby('stratum')):
   for col in ['delta_single','delta_matching','delta_human','delta_physical']:
    g=z.dropna(subset=[col]);direction=np.sign(g[col].to_numpy());direction[np.abs(g[col])<1e-10]=0
    if len(g):pm.append(dict(stratum=stratum,predictor=col,n=len(g),agreement=float(np.mean(.5+direction*(g.choice_a-.5))),tie_fraction=float(np.mean(direction==0))))
  p.to_csv(O/'pair_predictions.csv',index=False)
 pd.DataFrame(pm).to_csv(O/'pair_metrics.csv',index=False)
 cross=[];effects=[];h=d[d.cohort.eq('heldout_matching')].dropna(subset=['model_c','single_score'])
 # Only fit complete cross-task and condition models once collection finishes.
 if len(d.dropna(subset=['model_c']))==3888:
  for factor in ['shape','illumination']:
   for source,target in [('single_score','model_c'),('model_c','single_score')]:
    pred=np.zeros(len(h));x=h[source].to_numpy();y=h[target].to_numpy()
    for k in range(12):
     test=h[factor+'_fold'].to_numpy()==k;fit=IsotonicRegression(out_of_bounds='clip').fit(x[~test],y[~test]);pred[test]=fit.predict(x[test])
    cross.append(dict(split=factor,source=source,target=target,r2=float(1-np.mean((y-pred)**2)/np.var(y)),spearman=rho(y,pred)))
    pd.DataFrame(dict(image_id=h.image_id,target=y,prediction=pred)).to_csv(O/f'cross_{factor}_{source}.csv',index=False)
  for name,cols in [('single',['delta_single']),('matching',['delta_matching']),('joint',['delta_single','delta_matching'])]:
   xx=p[cols].to_numpy();yy=p.choice_a.to_numpy();pred=np.zeros(len(p))
   for tr,te in StratifiedKFold(5,shuffle=True,random_state=20260928).split(p,p.stratum):
    scale=xx[tr].std(0);scale[scale<1e-12]=1;v=xx[tr]/scale;fit=LogisticRegression(C=1,fit_intercept=False,max_iter=1000).fit(np.vstack([v,v]),np.r_[np.ones(len(tr)),np.zeros(len(tr))],sample_weight=np.r_[2*yy[tr],2*(1-yy[tr])]);pred[te]=fit.predict_proba(xx[te]/scale)[:,1]
   p[name+'_cv_probability']=pred
  p.to_csv(O/'pair_predictions.csv',index=False)
  parts=[np.ones((len(d),1))];names=['intercept']
  for col in ['physical_c','viewpoint_id','chroma','shape_id','illumination_id']:
   for level in sorted(d[col].unique())[1:]:parts.append((d[col].to_numpy()==level)[:,None].astype(float));names.append(f'{col}:{level}')
  for k in [1,2]:
   for trig in [np.sin,np.cos]:parts.append(trig(k*np.deg2rad(d.hue.to_numpy()))[:,None]);names.append(f'{trig.__name__}{k}')
  X=np.concatenate(parts,axis=1);Y=d[['human_c','model_c']].to_numpy();Y=(Y-Y.mean(0))/Y.std(0,ddof=1);beta=np.linalg.lstsq(X,Y,rcond=None)[0]
  rng=np.random.default_rng(20260928);pinv=np.linalg.pinv(X);residual=Y-X@beta
  for col in ['shape_id','illumination_id']:
   e=np.zeros((36,2))
   for k in range(2,37):e[k-1]=beta[names.index(f'{col}:{k}')]
   e-=e.mean(0)
   contrast=np.zeros((36,len(names)))
   for k in range(2,37):contrast[k-1,names.index(f'{col}:{k}')]=1
   contrast-=contrast.mean(0);other='illumination_id' if col=='shape_id' else 'shape_id';influence=[]
   for group in sorted(d[other].unique()):
    ix=d[other].to_numpy()==group;influence.append(contrast@pinv[:,ix]@residual[ix])
   delta=np.einsum('bg,glk->blk',rng.choice([-1.,1.],size=(4000,36)),np.stack(influence))*np.sqrt(36/35*(len(d)-1)/(len(d)-X.shape[1]));boot=e[None]+delta;ci=np.quantile(boot,[.025,.975],axis=0)
   pd.DataFrame(dict(condition_id=range(1,37),human_adjusted_sd=e[:,0],model_adjusted_sd=e[:,1],human_ci_low=ci[0,:,0],human_ci_high=ci[1,:,0],model_ci_low=ci[0,:,1],model_ci_high=ci[1,:,1])).to_csv(O/f'{col}_effects.csv',index=False)
   effects.append(dict(factor=col,spearman=rho(e[:,0],e[:,1]),cluster_ci95=np.quantile([rho(v[:,0],v[:,1]) for v in boot],[.025,.975]).tolist()))
  extra={}
  for name,z in [('matching_heldout',d[d.cohort.eq('heldout_matching')]),('new_validation',d[d.validation])]:
   xx=np.column_stack([np.ones(len(z)),z.physical_c]);yy=z[['human_c','model_c']].to_numpy();r=yy-xx@np.linalg.lstsq(xx,yy,rcond=None)[0]
   extra[name]=dict(n=len(z),partial_pearson_linear_physical=float(pearsonr(r[:,0],r[:,1]).statistic),repeat_spearman=rho(z.session1_c,z.session2_c),upper_mean_fraction=float(np.isclose(z.model_c,.1487).mean()))
  (O/'matching_extra_statistics.json').write_text(json.dumps(extra,indent=2))
  common=d.merge(f[['image_id','model_c']].rename(columns={'model_c':'internvl_c'}),on='image_id');common.to_csv(O/'joint_model_predictions.csv',index=False)
  pd.DataFrame(cross).to_csv(O/'cross_task_metrics.csv',index=False)
 summary=dict(single_images=len(single),pair_responses=len(pairs),complete_pairs=len(p),matching_trials=len(matches),paired_images=int(d.model_c.notna().sum()),human_agreement=stats,pair_metrics=pm,cross_task=cross,effects=effects,scope='Same point-estimate protocols as InternVL. Confidence intervals and publication layout require final review; pilot and new validation cohorts separated.')
 (O/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 fig,axes=plt.subplots(1,3,figsize=(12,3.7),constrained_layout=True)
 for ax,col,title in zip(axes,['single_score','model_c','single_score'],['Single vs human','Matching vs human','Single vs matching']):
  z=d.dropna(subset=[col,'model_c'] if title=='Single vs matching' else [col]);x=z.model_c if title=='Single vs matching' else z.human_c
  if len(z):ax.hexbin(x,z[col],gridsize=25,mincnt=1);ax.set_title(title+f' (n={len(z)})');ax.set_xlabel('Matching c' if title=='Single vs matching' else 'Human c');ax.set_ylabel(col)
 fig.savefig(O/'results.png',dpi=180);plt.close(fig)
 (O/'report.html').write_text('<meta charset="utf-8"><meta http-equiv="refresh" content="60"><style>body{font:18px system-ui;margin:35px}img{max-width:100%}pre{white-space:pre-wrap}td,th{padding:8px}</style><h1>Qwen3 三任务全量复现</h1><a href="progress.html">实时进度</a><p>先导样本单独标记；新验证集合用于扩展结论。采集中不对未完成部分作最终推断。</p><img src="results.png">'+pd.DataFrame(stats).round(4).to_html(index=False)+'<pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>',encoding='utf-8')
if __name__=='__main__':main()
