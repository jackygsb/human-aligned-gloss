"""Figure 6: reference-defined internal perturbations change gloss choice."""
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from PIL import Image
HERE=Path(__file__).resolve().parent;O=HERE if (HERE/'data/shape_dose_responses.csv').exists() else HERE.parent/'mllm_material/figure6_english_v1'
T=O/'data';dose=pd.read_csv(T/'v8_intervention_dose_ci.csv');tests=pd.read_csv(T/'v8_intervention_tests.csv');sv=pd.read_csv(T/'v8_intervention_shape_values.csv');sh=pd.read_csv(T/'shape_dose_responses.csv');ref=pd.read_csv(T/'reference_projections.csv')
C={'candidate':'#A64060','brightness':'#BD9445','random0':'#456877','random1':'#92A6AC','random2':'#8E849F','ink':'#29323B','muted':'#79838A','line':'#DEE3E6','pale':'#F6F2F2'}
plt.rcParams.update({'font.family':'Arial','font.size':7,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'xtick.major.width':.6,'ytick.major.width':.6,'xtick.color':C['muted'],'ytick.color':C['muted'],'axes.labelcolor':C['ink'],'text.color':C['ink']})
W,H=1800,2640;fig=plt.figure(figsize=(180/25.4,264/25.4),facecolor='white');base=fig.add_axes([0,0,1,1]);base.set(xlim=(0,W),ylim=(H,0));base.axis('off')
def txt(x,y,s,size=24,col=None,weight='normal',ha='left'):return base.text(x,y,s,fontsize=size*72/254,color=col or C['ink'],weight=weight,ha=ha,va='top',linespacing=1.25)
def panel(x,y,l,t):txt(x,y,l,33,weight='bold');txt(x+40,y+2,t,26,weight='bold')
def axat(x,y,w,h):return fig.add_axes([x/W,1-(y+h)/H,w/W,h/H])
def sep(y):base.plot([42,1758],[y,y],color=C['line'],lw=.6)
def fmt(a):a.tick_params(labelsize=6.5,pad=2);a.xaxis.label.set_size(6.8);a.yaxis.label.set_size(6.8)
def img(path,x,y,w):a=axat(x,y,w,w);a.imshow(Image.open(path));a.axis('off')
def arrow(x,y,xx,yy):base.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=9,color=C['muted'],lw=.8))
panel(42,30,'a','Define an internal direction from 52 references')
for j,k in enumerate([1,14,27,40,52]):
 img(O/f'images/reference_{k:02d}.png',82+j*181,104,163);c=ref[ref.reference_index==k].c.iloc[0];txt(164+j*181,281,f'c = {c:.3f}',20,C['muted'],ha='center')
arrow(100,330,940,330);txt(515,346,'Increasing reference gloss',23,C['candidate'],ha='center')
txt(82,390,'Mean bridge states → multivariate slope β → direction d = β / ‖β‖',23)
txt(82,429,'Scale s = ‖β‖ for the full reference range; no human judgments used.',21,C['muted'])
panel(1070,30,'b','Reference projection check');a=axat(1180,113,480,240)
for model,ls in [('InternVL','-'),('Qwen3','--')]:
 z=ref[ref.model==model];a.plot(z.c,z.projection_scaled,color=C['candidate'] if model=='InternVL' else C['random0'],ls=ls,lw=1.3,label=model)
a.set(xlim=(0,.1487),ylim=(-.03,1.03),xticks=[0,.075,.15],yticks=[0,.5,1],xlabel='Reference gloss c',ylabel='Scaled projection');a.legend(frameon=False,fontsize=6,loc='upper left');fmt(a);txt(1180,437,'Construction check on the defining reference set',20,C['muted']);sep(485)
panel(42,520,'c','Where does the direction respond?')
im=np.asarray(Image.open(O/'images/stimulus_0064.png').convert('RGB'));titles=['Original image','InternVL projection','Qwen3 projection']
for j,title in enumerate(titles):
 x=80+j*324;txt(x+140,575,title,22,ha='center');a=axat(x,620,280,280);a.imshow(im)
 if j:
  h=np.load(T/f'{["InternVL","Qwen3"][j-1]}_spatial_projection.npy');a.imshow(h,cmap='inferno',alpha=.48,extent=(0,im.shape[1],im.shape[0],0),interpolation='nearest',vmin=h.min(),vmax=h.max())
 a.axis('off')
txt(80,926,'Illustration #64; raw projections scaled per model.',20,C['muted'])
cax=axat(830,919,170,10);cb=fig.colorbar(plt.cm.ScalarMappable(cmap='inferno'),cax=cax,orientation='horizontal',ticks=[0,1]);cb.ax.set_xticklabels(['Low','High']);cb.ax.tick_params(labelsize=5.5,length=0,pad=2)
panel(1070,520,'d','Perturb test-image tokens only')
for y,st in [(594,'h′(p) = h(p) + α s d'),(655,'α = −1, −0.5, 0, +0.5, +1'),(730,'Frozen model + unchanged reference'),(806,'Score = P(test) + ½ P(SAME)')]:
 txt(1420,y,st,26 if y==594 else 23,ha='center',col=C['candidate'] if y==594 else None)
arrow(1420,690,1420,718);arrow(1420,764,1420,797)
txt(1130,866,'72 images · 36 shapes · two display orders',21,C['muted']);txt(1130,904,'One illumination; normalized over A / B / SAME',20,C['muted']);sep(977)
# Dose response uses shapes as the unit of uncertainty; same direction colors in both models.
for j,model in enumerate(['InternVL','Qwen3']):
 x=42+j*885;panel(x,1010,'ef'[j],model+': bidirectional control of gloss choice');a=axat(x+105,1130,676,275)
 for direction in ['brightness','random0','random1','random2','candidate']:
  z=dose[(dose.model==model)&(dose.direction==direction)].sort_values('dose');xx=np.r_[z.dose,0];yy=np.r_[z.mean_delta_p*100,0];lo=np.r_[z.ci_low*100,0];hi=np.r_[z.ci_high*100,0];ix=np.argsort(xx);a.plot(xx[ix],yy[ix],color=C[direction],lw=1.7 if direction=='candidate' else .8,marker='o',ms=3 if direction=='candidate' else 2,ls='-' if direction in ['candidate','brightness'] else ':')
  if direction=='candidate':a.fill_between(xx[ix],lo[ix],hi[ix],color=C[direction],alpha=.14,lw=0)
 a.axhline(0,lw=.6,color=C['muted']);a.set(xlim=(-1.05,1.05),ylim=(-15,15),xticks=[-1,-.5,0,.5,1],yticks=[-10,0,10],xlabel='Dose α (reference-range units)',ylabel='Change in choice score (pp)');fmt(a)
 z=dose[(dose.model==model)&dose.direction.eq('candidate')].set_index('dose');txt(x+105,1490,f'−1 → +1 swing: {100*(z.loc[1,"mean_delta_p"]-z.loc[-1,"mean_delta_p"]):.1f} percentage points',22,C['candidate'])
for j,(key,label) in enumerate([('candidate','Reference-defined'),('brightness','Brightness'),('random0','Random 1'),('random1','Random 2'),('random2','Random 3')]):txt(98+j*335,1070,'● '+label,21,C[key])
sep(1540)
rng=np.random.default_rng(6)
for j,model in enumerate(['InternVL','Qwen3']):
 x=42+j*885;panel(x,1575,'gh'[j],model+': advantage over matched-norm controls');a=axat(x+105,1670,676,286)
 for k,control in enumerate(['brightness','random0','random1','random2']):
  z=sv[(sv.model==model)&(sv.control==control)];a.scatter(k+rng.uniform(-.16,.16,len(z)),100*z.contrast,s=6,c=C['candidate'],alpha=.38,edgecolors='none');r=tests[(tests.model==model)&(tests.control==control)].iloc[0];a.errorbar(k,100*r.estimate,yerr=[[100*(r.estimate-r.ci_low)],[100*(r.ci_high-r.estimate)]],fmt='D',ms=4,color=C['ink'],capsize=3,lw=1.2);a.text(k,70,'q = 0.00001',ha='center',fontsize=6)
 a.axhline(0,color=C['muted'],lw=.6);a.set(xlim=(-.5,3.5),ylim=(-5,78),xticks=range(4),xticklabels=['Brightness','Random 1','Random 2','Random 3'],yticks=[0,20,40,60],ylabel='Extra bidirectional change (pp)');fmt(a);txt(x+105,2030,'36 shape-level contrasts; mean and paired 95% interval',20,C['muted'])
sep(2080)
panel(42,2115,'i','Dose responses across every tested shape')
cmap=LinearSegmentedColormap.from_list('signed',['#365E78','#F8F5F1','#A64060']);sub=sh[sh.direction.eq('candidate')];lim=np.ceil(abs(sub.delta_p).max()*100/10)*10
for j,model in enumerate(['InternVL','Qwen3']):
 x=140+j*885;a=axat(x,2225,650,195);z=sub[sub.model==model].pivot(index='dose',columns='shape_id',values='delta_p').reindex([-1,-.5,0,.5,1],fill_value=0)*100;z.loc[0]=0;implot=a.imshow(z,cmap=cmap,vmin=-lim,vmax=lim,aspect='auto',interpolation='nearest');a.set(xticks=[0,11,23,35],xticklabels=[1,12,24,36],yticks=range(5),yticklabels=['−1','−0.5','0','+0.5','+1'],xlabel='Shape ID',ylabel='Dose α');fmt(a);txt(x+325,2175,model,23,ha='center')
cax=axat(605,2524,580,15);cb=fig.colorbar(implot,cax=cax,orientation='horizontal');cb.ax.tick_params(labelsize=6,length=2,pad=2);txt(895,2580,'Change in choice score (percentage points)',21,C['muted'],ha='center')
for ext in ['svg','pdf','png']:fig.savefig(O/f'Figure6_Internal_Direction_Intervention_EN.{ext}',dpi=450,facecolor='white')
print('Saved Figure 6.')
