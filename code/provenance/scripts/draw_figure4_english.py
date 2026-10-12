"""Portable Figure 4: cross-task transfer of model gloss judgments."""
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from PIL import Image
HERE=Path(__file__).resolve().parent;O=HERE if (HERE/'data/continuous_predictions.csv').exists() else HERE.parent/'mllm_material/figure4_english_v1'
T=O/'data';d=pd.read_csv(T/'continuous_predictions.csv');ci=pd.read_csv(T/'v8_cross_task_ci.csv');pc=pd.read_csv(T/'v8_pair_ci.csv');curves=pd.read_csv(T/'choice_curves.csv');pairs=pd.read_csv(T/'example_pairs.csv')
C={'InternVL':'#147D92','Qwen3':'#CC7243','ink':'#1D2934','muted':'#6D7984','line':'#D7E0E5','pale':'#F5F8F9'}
plt.rcParams.update({'font.family':'Arial','font.size':7,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'xtick.major.width':.6,'ytick.major.width':.6,'xtick.color':C['muted'],'ytick.color':C['muted'],'axes.labelcolor':C['ink'],'text.color':C['ink']})
W,H=1800,2480;fig=plt.figure(figsize=(180/25.4,248/25.4),facecolor='white');base=fig.add_axes([0,0,1,1]);base.set(xlim=(0,W),ylim=(H,0));base.axis('off')
def txt(x,y,s,size=24,col=None,weight='normal',ha='left'):return base.text(x,y,s,fontsize=size*72/254,color=col or C['ink'],weight=weight,ha=ha,va='top',linespacing=1.25)
def panel(x,y,l,t):txt(x,y,l,33,weight='bold');txt(x+40,y+2,t,26,weight='bold')
def axat(x,y,w,h):return fig.add_axes([x/W,1-(y+h)/H,w/W,h/H])
def sep(y):base.plot([42,1758],[y,y],color=C['line'],lw=.6)
def fmt(a):a.tick_params(labelsize=6.3,pad=2);a.xaxis.label.set_size(6.7);a.yaxis.label.set_size(6.7)
def box(x,y,w,h,s):base.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=12',facecolor=C['pale'],edgecolor=C['line'],lw=.6));txt(x+w/2,y+19,s,24,ha='center')
def arrow(x,y,xx,yy):base.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=9,color=C['muted'],lw=.8))
panel(42,30,'a','A shared judgment tested through three behavioral tasks')
box(80,96,450,103,'Single-image rating\n“How glossy is this object?”')
box(690,96,450,103,'Reference matching\n“Which reference looks equally glossy?”')
box(1300,96,450,103,'Two-image choice\n“Which object looks glossier?”')
arrow(548,136,675,136);arrow(675,161,548,161);txt(610,211,'Predict in both directions',20,C['muted'],ha='center')
arrow(1158,148,1284,148);txt(1350,211,'Predict choices from both tasks',20,C['muted'],ha='center')
txt(80,260,'Continuous transfer: 12-fold shape / illumination holdout; calibration uses model outputs only.',22)
txt(80,295,'Choice transfer: 2,000 image pairs, each presented in both orders; no human labels used as predictors.',22)
txt(1350,270,'● InternVL',22,C['InternVL']);txt(1565,270,'▲ Qwen3',22,C['Qwen3']);sep(345)
# Detailed out-of-fold predictions for both directions; shape holdout pictured.
hb=[]
for j,(source,title) in enumerate([('single_score','Rating predicts reference matching'),('model_c','Matching predicts single-image rating')]):
 x=42+j*885;panel(x,380,'bc'[j],title)
 for k,model in enumerate(['InternVL','Qwen3']):
  xx=x+77+k*393;z=d[(d.model==model)&(d.source==source)&(d.split=='shape')];a=axat(xx,481,315,287);limit=.1487 if source=='single_score' else 100;h=a.hexbin(z.observed,z.predicted,gridsize=27,extent=(0,limit,0,limit),mincnt=1,cmap='viridis',linewidths=0);hb.append(h);a.plot([0,limit],[0,limit],':',lw=.75,color='#687B88');ticks=[0,.075,.15] if limit<1 else [0,50,100];a.set(xlim=(-limit*.015,limit*1.02),ylim=(-limit*.015,limit*1.02),xticks=ticks,yticks=ticks,xlabel='Observed '+('matched c' if limit<1 else 'rating'),ylabel='Out-of-fold prediction');fmt(a)
  if k==1:a.set_ylabel('')
  r=ci[(ci.model==model)&(ci.source==source)&(ci.split=='shape')].iloc[0];txt(xx+157,440,model,23,C[model],ha='center');a.text(.045,.94,f'R² = {r.estimate:.3f}',transform=a.transAxes,va='top',fontsize=6.5,bbox=dict(facecolor='white',edgecolor='none',alpha=.88,pad=2))
txt(120,866,'3,849 held-out predictions per model and direction; dashed lines indicate exact prediction.',21,C['muted'])
norm=LogNorm(1,max(max(h.get_array()) for h in hb))
for h in hb:h.set_norm(norm)
cax=axat(1470,867,215,13);cb=fig.colorbar(hb[0],cax=cax,orientation='horizontal');cb.ax.tick_params(labelsize=5.5,length=2,pad=1);txt(1578,917,'Images per hexagon',18,C['muted'],ha='center');sep(965)
for j,source in enumerate(['single_score','model_c']):
 x=42+j*575;panel(x,1000,'de'[j],'Rating → matching' if j==0 else 'Matching → rating');a=axat(x+130,1085,366,258)
 for k,model in enumerate(['InternVL','Qwen3']):
  for s,split in enumerate(['shape','illumination']):
   r=ci[(ci.model==model)&(ci.source==source)&(ci.split==split)].iloc[0];a.errorbar(s+(-.10 if k==0 else .10),r.estimate,yerr=[[r.estimate-r.ci_low],[r.ci_high-r.estimate]],fmt='o' if k==0 else '^',ms=4,capsize=2,color=C[model],lw=1)
 a.set(xlim=(-.5,1.5),ylim=(0,1),xticks=[0,1],xticklabels=['Held-out\nshapes','Held-out\nlighting'],yticks=[0,.5,1],ylabel='Cross-validated R²');fmt(a)
 txt(x+130,1440,'95% crossed-condition bootstrap intervals',19,C['muted'])
panel(1192,1000,'f','Task rankings predict choices');a=axat(1280,1085,410,258)
for k,model in enumerate(['InternVL','Qwen3']):
 for j,pred in enumerate(['single','matching']):
  r=pc[(pc.model==model)&(pc.predictor==pred)&pc.stratum.eq('all')].iloc[0];a.errorbar(j+(-.10 if k==0 else .10),r.estimate*100,yerr=[[100*(r.estimate-r.ci_low)],[100*(r.ci_high-r.estimate)]],fmt='o' if k==0 else '^',ms=4,capsize=2,color=C[model],lw=1);a.text(j+(-.13 if k==0 else .13),r.ci_high*100+1.7,f'{r.estimate*100:.1f}',ha='center',fontsize=6.3,color=C[model])
a.axhline(50,color=C['muted'],ls=':',lw=.7);a.set(xlim=(-.45,1.45),ylim=(45,95),xticks=[0,1],xticklabels=['Rating\nordering','Matching\nordering'],yticks=[50,70,90],ylabel='Choice agreement (%)');fmt(a);txt(1280,1440,'2,000 pairs; 95% image-node intervals',19,C['muted']);sep(1490)
panel(42,1525,'g','Larger task differences predict more directional choices')
for j,task in enumerate(['single','matching']):
 a=axat(135+j*547,1620,420,268)
 for model in ['InternVL','Qwen3']:
  z=curves[(curves.model==model)&(curves.task==task)];a.plot(z.delta_mean,z.choice_mean,color=C[model],lw=1);a.scatter(z.delta_mean,z.choice_mean,s=8+z.n.to_numpy()*.08,c=C[model],edgecolor='white',linewidth=.4,zorder=3)
 a.axhline(.5,color=C['muted'],ls=':',lw=.7);a.axvline(0,color=C['muted'],ls=':',lw=.7);a.set(xlim=(-1.02,1.02),ylim=(-.035,1.035),xticks=[-1,0,1],yticks=[0,.5,1],xlabel=('Rating difference / 100' if task=='single' else 'Matching difference / 0.1487')+' (A − B)',ylabel='Observed choice frequency for A');fmt(a)
txt(135,1990,'Ten fixed bins; symbols scale with pair count; lines connect bin means.',20,C['muted'])
panel(1192,1525,'h','Transfer across pair types');a=axat(1340,1630,350,258)
for j,(task,st) in enumerate([('single','random'),('single','discordant'),('matching','random'),('matching','discordant')]):
 for k,model in enumerate(['InternVL','Qwen3']):
  r=pc[(pc.model==model)&(pc.predictor==task)&(pc.stratum==st)].iloc[0];y=3-j+(.12 if k==0 else -.12);a.errorbar(100*r.estimate,y,xerr=[[100*(r.estimate-r.ci_low)],[100*(r.ci_high-r.estimate)]],fmt='o' if k==0 else '^',color=C[model],ms=3.5,lw=.9,capsize=2)
a.set(yticks=[3,2,1,0],yticklabels=['Rating | random','Rating | conflict','Match | random','Match | conflict'],xlim=(50,95),xticks=[50,70,90],ylim=(-.5,3.5),xlabel='Choice agreement (%)');fmt(a);txt(1240,1990,'1,000 random + 1,000 conflict pairs',19,C['muted']);sep(2040)
panel(42,2070,'i','Illustrative concordant pairs connect the three tasks')
for j,pid in enumerate(sorted(pairs.pair_id.unique())):
 x=70+j*875;z=pairs[pairs.pair_id==pid];r=z.iloc[0]
 for k,col in enumerate(['a_id','b_id']):
  a=axat(x+k*157,2140,135,135);a.imshow(Image.open(O/'images'/f'{r[col]}.png'));a.axis('off');txt(x+k*157+67,2285,'AB'[k],20,ha='center')
 txt(x+340,2135,'Model     Δ rating     Δ match c     Choice A',19,weight='bold')
 for k,model in enumerate(['InternVL','Qwen3']):
  r=z[z.model==model].iloc[0];txt(x+340,2180+k*47,f'{model:<9}  {r.delta_single:+.0f}          {r.delta_matching:+.3f}          {r.choice_a:.2f}',20,C[model])
 txt(x+340,2288,f'Random pair #{pid}; A − B differences',19,C['muted'])
txt(70,2360,'Concordant examples selected near the 25th / 75th percentile of random-pair InternVL matching differences.',20,C['muted'])
txt(70,2398,'Both models agree across all three tasks in these selected examples; population statistics above include every pair.',20,C['muted'])
for ext in ['svg','pdf','png']:fig.savefig(O/f'Figure4_Cross_Task_Transfer_EN.{ext}',dpi=450,facecolor='white')
print('Saved English Figure 4.')
