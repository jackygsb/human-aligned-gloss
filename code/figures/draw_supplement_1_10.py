from pathlib import Path
import json,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import os
OSFROOT=Path(__file__).resolve().parents[2]
OUT=Path(os.environ.get('OSF_REPRO_OUT',OSFROOT/'figures/regenerated'))/'supplement'
OUT.mkdir(parents=True,exist_ok=True)
S=OSFROOT/'figures/inputs/supplement';B=S/'supplement_inputs';M=B/'mllm_material';T=M/'submission_synthesis_v1/tables'
plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
C=['#7656A5','#148B80'];names=['InternVL','Qwen3'];gray='#697583'
def save(fig,key):
 for ext in ['pdf','png','svg']:fig.savefig(OUT/f'{key}.{ext}',dpi=220,bbox_inches='tight')
 plt.close(fig)
def panel(ax,s):ax.text(-.12,1.08,s,transform=ax.transAxes,weight='bold',fontsize=13)
joint=pd.read_csv(M/'qwen3_full_v1/joint_model_predictions.csv');stim=pd.read_csv(M/'mechanisms_v1/stimuli.csv');r=pd.read_csv(T/'rsa.csv')
# Supplement: transparent distributions and complete statistical families.
fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for j,col in enumerate(['internvl_c','model_c']):
 axs[j].hist(joint.human_c,bins=np.linspace(0,.151,22),histtype='step',lw=1.7,color=gray,label='Human');axs[j].hist(joint[col],bins=np.linspace(0,.151,22),histtype='step',lw=1.7,color=C[j],label=names[j]);axs[j].set(xlabel='Mean matched gloss c',ylabel='Images');axs[j].legend();panel(axs[j],chr(97+j))
save(fig,'S1')
fig,axs=plt.subplots(3,2,figsize=(10,10),layout='constrained')
for i,sp in enumerate(['raw','physical_color_adjusted','interaction_only']):
 for j,fac in enumerate(['shape','illumination']):
  a=axs[i,j];z=r[(r.space==sp)&(r.factor==fac)];x=np.arange(len(z));a.errorbar(x,z.rsa,yerr=[z.rsa-z.ci_low,z.ci_high-z.rsa],fmt='o',color=C[j],capsize=4);a.set(xticks=x,xticklabels=['Human–InternVL','Human–Qwen3','InternVL–Qwen3'],ylabel='Spearman RSA',ylim=(-.15,1),title=['Raw','Physical/color adjusted','Interaction only'][i]+[': shape',': illumination'][j]);a.axhline(0,color=gray,lw=.6);a.tick_params(axis='x',labelsize=8);panel(a,chr(97+i*2+j))
save(fig,'S2')
lat=json.loads((T/'latent_prediction.json').read_text())
fig,axs=plt.subplots(1,3,figsize=(10,4.2),sharey=True,layout='constrained')
for j,n in enumerate(['Human','InternVL','Qwen3']):
 a=axs[j];vals=[lat[n][k] for k in ['r2_nuisance','r2_additive','r2_selected']];a.plot(range(3),vals,'o-',color=([gray]+C)[j],ms=6,lw=1.2)
 for k,v in enumerate(vals):a.annotate(f'{v:.3f}',(k,v),xytext=(0,7),textcoords='offset points',ha='center',fontsize=9)
 a.set(xticks=range(3),xticklabels=['Physical/color','Additive','Low rank'],ylim=(0,.7),xlim=(-.35,2.35),title='Human' if n=='Human' else n);a.tick_params(axis='x',labelrotation=25,labelsize=9);panel(a,chr(97+j))
axs[0].set_ylabel('Held-out-view prediction R²');save(fig,'S3')
z=pd.read_csv(S/'tables/v8_agreement_contrasts.csv');fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
labs=['InternVL rating − physical','InternVL matching − physical','Qwen3 rating − physical','Qwen3 matching − physical','Qwen3 − InternVL: matching','Qwen3 − InternVL: rating']
for i,rw in z.iterrows(): ax.errorbar(rw.estimate,i,xerr=[[rw.estimate-rw.ci_low],[rw.ci_high-rw.estimate]],fmt='o',color=C[0] if i<2 else C[1],capsize=3)
ax.axvline(0,color=gray,ls=':',lw=1);ax.set(yticks=range(6),yticklabels=labs,xlabel='Paired difference in Spearman correlation');ax.invert_yaxis();save(fig,'S4')
tiny=json.loads((T/'tiny_incremental.json').read_text());df=pd.DataFrame(tiny['metrics']);fig,axs=plt.subplots(1,2,figsize=(9,4.6),layout='constrained')
for k,sp in enumerate(['shape','illumination']):
 z=df[df.split==sp].set_index('model');keys=['controls_tiny_cubic','controls_tiny_cubic_plus_internvl'];v=[z.loc[x,'r2'] for x in keys];axs[0].plot([0,1],v,'o-',color=C[k],label='Shape holdout' if sp=='shape' else 'Lighting holdout',ms=6)
 gain=next(x for x in tiny['gains'] if x['split']==sp and x['baseline']=='controls_tiny_cubic');lo,hi=gain['crossed_cluster_ci95_delta_r2'];value=gain['delta_r2'];axs[1].errorbar(value,k,xerr=[[value-lo],[hi-value]],fmt='o',capsize=3,color=C[k],ms=6)
axs[0].set(xticks=[0,1],xticklabels=['Controls + tiny CNN','+ InternVL matching'],ylabel='Out-of-fold prediction R²',ylim=(.3,.65),xlim=(-.3,1.3));axs[0].legend(fontsize=9);axs[1].axvline(0,color=gray,ls=':');axs[1].set(yticks=[0,1],yticklabels=['Shape holdout','Lighting holdout'],xlabel='Gain in R² with 95% interval',ylim=(-.6,1.6),xlim=(-.01,.18));panel(axs[0],'a');panel(axs[1],'b');save(fig,'S5')
mt=pd.read_csv(T/'mechanism_matches.csv');fig,axs=plt.subplots(1,2,figsize=(10,4.7),layout='constrained')
for j,n in enumerate(names):
 a=axs[j]
 for direction,col,lab in [('candidate',C[j],'Reference-defined'),('random0',gray,'Random direction 1')]:
  vals=[]
  for dose in [-1,0,1]:
   z=mt[(mt.model==n)&(mt.direction==('baseline' if dose==0 else direction))&(mt.dose==dose)];vals.append(z.matching_c.mean())
  
  base=mt[(mt.model==n)&(mt.direction=='baseline')].set_index('image_id').matching_c
  for ident,v0 in base.items():
   vm=mt[(mt.model==n)&(mt.image_id==ident)&(mt.direction==direction)].set_index('dose').matching_c
   a.plot([-1,0,1],[vm.loc[-1],v0,vm.loc[1]],color=col,alpha=.22,lw=.6,marker='o',ms=2)
  a.plot([-1,0,1],vals,'o-',color=col,label=lab,lw=2,ms=6)
 a.set(xlabel='Dose',ylabel='Complete match c (thin lines: images)',title=n,ylim=(0,.155));a.legend(fontsize=9)
save(fig,'S6')
# Replot archived adjusted matrices, coordinates and spatial projections in English.
for fac,key in [('shape','S7'),('illumination','S8')]:
 fig,axs=plt.subplots(2,3,figsize=(10.2,7.2),layout='constrained')
 for j,n in enumerate(['Human','InternVL','Qwen3']):
  p=M/f'behavioral_spaces_v1/matrices/physical_color_adjusted_{fac}_{n}_profiles_by_view.npy';cube=np.load(p);mat=cube.mean(axis=0)
  rdm=np.loadtxt(M/f'behavioral_spaces_v1/matrices/physical_color_adjusted_{fac}_{n}_rdm.csv',delimiter=',')
  im=axs[0,j].imshow(mat/mat.std(),cmap='RdBu_r',vmin=-3,vmax=3);axs[0,j].set(title=('Human' if n=='Human' else n)+': adjusted profiles',xlabel='Illumination' if fac=='shape' else 'Shape',ylabel='Shape' if fac=='shape' else 'Illumination')
  den=rdm[np.triu_indices(36,1)].mean();im2=axs[1,j].imshow(rdm/den,cmap='magma',vmin=0,vmax=3);axs[1,j].set(title='Relative behavioral dissimilarity',xlabel='Condition ID',ylabel='Condition ID')
 fig.colorbar(im,ax=axs[0,:].tolist(),shrink=.7,label='Response / profile s.d.');fig.colorbar(im2,ax=axs[1,:].tolist(),shrink=.7,label='Distance / mean distance');save(fig,key)
fig,axs=plt.subplots(2,3,figsize=(10,7.2),layout='constrained')
for i,fac in enumerate(['shape','illumination']):
 for j,n in enumerate(['Human','InternVL','Qwen3']):
  d=pd.read_csv(M/f'behavioral_spaces_v1/embeddings/{n}_{fac}_adjusted_profile_PCA.csv');a=axs[i,j];a.scatter(d.dim1,d.dim2,s=24,color=([gray]+C)[j]);
  for rr in d.itertuples():a.annotate(str(rr.condition_id),(rr.dim1,rr.dim2),fontsize=6,xytext=(2,2),textcoords='offset points')
  a.set(title=('Human' if n=='Human' else n)+': '+('Shape' if fac=='shape' else 'Illumination'),xlabel='Principal component 1',ylabel='Principal component 2');a.axhline(0,color=gray,lw=.4);a.axvline(0,color=gray,lw=.4)
save(fig,'S10')
plan=json.loads((M/'mechanisms_v1/plan.json').read_text('utf-8'));fig,axs=plt.subplots(2,5,figsize=(12,6.1),layout='constrained')
for row,n in enumerate(names):
 root=M/'mechanisms_v1'/n;dire=np.load(root/'directions.npz');ref=dire['reference_means']@dire['candidate'];axs[row,0].plot(np.linspace(0,.1487,52),ref,color=C[row]);axs[row,0].set(title=n+': references',xlabel='Reference c',ylabel='Candidate projection')
 scores=pd.read_csv(root/'candidate_scores.csv');z=scores[scores.image_id.isin(plan['intervention_images'])].sort_values('projection');ids=z.image_id.iloc[[0,1,-2,-1]].tolist()
 lo,hi=np.quantile(np.concatenate([np.load(root/'features'/f'{id}.npz')['bridge_tokens'].astype('float32')@dire['candidate'] for id in ids]),[.02,.98])
 for j,id in enumerate(ids):
  r=stim[stim.image_id==id].iloc[0];im=np.asarray(Image.open(B/r.image_512_path).convert('RGB'));token=np.load(root/'features'/f'{id}.npz')['bridge_tokens'].astype('float32')@dire['candidate'];side=int(np.sqrt(len(token)));heat=np.asarray(Image.fromarray(token.reshape(side,side)).resize((im.shape[1],im.shape[0]),Image.Resampling.BILINEAR));ax=axs[row,j+1];ax.imshow(im);hm=ax.imshow(heat,cmap='magma',alpha=.55,vmin=lo,vmax=hi);ax.set_title(id+'\nCandidate projection',fontsize=8);ax.axis('off')
 fig.colorbar(hm,ax=axs[row,1:].tolist(),shrink=.55,label='Projection (shared within model)')
save(fig,'S9')
