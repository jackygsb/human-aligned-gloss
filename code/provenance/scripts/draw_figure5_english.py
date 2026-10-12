"""Figure 5: frozen internal states support human gloss readout."""
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,LogNorm
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
HERE=Path(__file__).resolve().parent;O=HERE if (HERE/'data/layer_readouts.csv').exists() else HERE.parent/'mllm_material/figure5_english_v1'
T=O/'data';s=pd.read_csv(T/'layer_readouts.csv');ci=pd.read_csv(T/'v8_readout_ci.csv');d=pd.read_csv(T/'readout_predictions.csv.gz');gain=pd.read_csv(T/'final_layer_increment.csv')
C={'InternVL':'#7656A5','Qwen3':'#148B80','ink':'#24323B','muted':'#77838C','line':'#DDE3E7','vision':'#F1ECF7','bridge':'#E7F3EF','language':'#FBF2DE','gold':'#BA9140'}
layers=['vision_06','vision_12','vision_18','vision_24','bridge','language_09','language_18','language_27','language_36'];labels=['V6','V12','V18','V24','B','L9','L18','L27','L36']
plt.rcParams.update({'font.family':'Arial','font.size':7,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'xtick.major.width':.6,'ytick.major.width':.6,'xtick.color':C['muted'],'ytick.color':C['muted'],'axes.labelcolor':C['ink'],'text.color':C['ink']})
W,H=1800,2380;fig=plt.figure(figsize=(180/25.4,238/25.4),facecolor='white');base=fig.add_axes([0,0,1,1]);base.set(xlim=(0,W),ylim=(H,0));base.axis('off')
def txt(x,y,s,size=24,col=None,weight='normal',ha='left'):return base.text(x,y,s,fontsize=size*72/254,color=col or C['ink'],weight=weight,ha=ha,va='top',linespacing=1.25)
def panel(x,y,l,t):txt(x,y,l,33,weight='bold');txt(x+40,y+2,t,26,weight='bold')
def axat(x,y,w,h):return fig.add_axes([x/W,1-(y+h)/H,w/W,h/H])
def sep(y):base.plot([42,1758],[y,y],color=C['line'],lw=.6)
def fmt(a):a.tick_params(labelsize=6.4,pad=2);a.xaxis.label.set_size(6.8);a.yaxis.label.set_size(6.8)
def box(x,y,w,h,s,c):base.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=12',facecolor=c,edgecolor='none'));txt(x+w/2,y+18,s,23,ha='center')
def arrow(x,y,xx,yy):base.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=9,color=C['muted'],lw=.8))
panel(42,30,'a','Read human gloss judgments from frozen internal states')
box(80,98,410,106,'Vision blocks\nSpatially pooled visual features',C['vision']);box(590,98,360,106,'Vision–language bridge\nMean visual-token features',C['bridge']);box(1050,98,650,106,'Language blocks\nFinal prompt-token state, before answer generation',C['language']);arrow(505,150,575,150);arrow(965,150,1035,150)
txt(80,234,'Each sampled state → fixed 128-D projection → supervised ridge probe → held-out human judgment',24)
txt(80,278,'3,888 images; 12-fold shape / lighting splits; only probe parameters are fitted.',22,C['muted']);txt(1300,278,'● InternVL',22,C['InternVL']);txt(1528,278,'▲ Qwen3',22,C['Qwen3']);sep(335)
# Full layer curves, with stage backgrounds to distinguish architecture and pooling.
for row,target in enumerate(['human','human_beyond_physical_color']):
 for j,split in enumerate(['shape','illumination']):
  x=42+j*885;y=375+row*470;panel(x,y,'bcde'[row*2+j],('Human readout' if row==0 else 'Physical/color baseline + neural readout')+' | '+('shape holdout' if j==0 else 'lighting holdout'))
  a=axat(x+98,y+80,700,266)
  for l,r,col in [(-.5,3.5,'vision'),(3.5,4.5,'bridge'),(4.5,8.5,'language')]:a.axvspan(l,r,color=C[col],zorder=0)
  for model in ['InternVL','Qwen3']:
   z=ci[(ci.model==model)&(ci.split==split)&(ci.target==target)].set_index('layer').loc[layers];a.fill_between(range(9),z.ci_low,z.ci_high,color=C[model],alpha=.12,lw=0);a.plot(range(9),z.estimate,color=C[model],marker='o' if model=='InternVL' else '^',ms=3,lw=1.15)
  if row==1:
   b=gain[gain.split==split].baseline_r2.iloc[0];a.axhline(b,color=C['gold'],ls='--',lw=1);a.text(.10,b+.02,f'Physical/color baseline R² = {b:.3f}',color='#8A6827',fontsize=6.4)
  a.set(xlim=(-.3,8.3),ylim=(0,.85),xticks=range(9),xticklabels=labels,yticks=[0,.2,.4,.6,.8],ylabel='Held-out prediction R²');fmt(a)
  for cx,lab in [(x+236,'Vision'),(x+436,'Bridge'),(x+639,'Language')]:txt(cx,y+398,lab,21,C['muted'],ha='center')
sep(1292)
# All targets and model-specific DeepStack states in a compact complete score atlas.
panel(42,1325,'f','Target-specific information across internal states')
txt(82,1370,'Each cell: held-out R². S = shape holdout; I = illumination holdout.',21,C['muted'])
cmap=LinearSegmentedColormap.from_list('readout',['#FCFBF8','#D6DCE8','#959CBF','#544977','#29243F'])
targets=['human','physical','model_behavior'];rowlabels=['Human | S','Human | I','Physical | S','Physical | I','Behavior | S','Behavior | I']
for j,model in enumerate(['InternVL','Qwen3']):
 x=170+j*720;a=axat(x,1460,540,270);txt(x+270,1417,model,24,C[model],ha='center');A=np.array([[s[(s.model==model)&(s.layer==layer)&(s.target==tar)&(s.split==split)].r2.iloc[0] for layer in layers] for tar in targets for split in ['shape','illumination']]);im=a.imshow(A,cmap=cmap,vmin=0,vmax=.9,aspect='auto');a.set(xticks=range(9),xticklabels=labels,yticks=range(6),yticklabels=rowlabels);a.tick_params(labelsize=5.5,length=0,pad=3)
 for (r,c),v in np.ndenumerate(A):a.text(c,r,f'{v:.2f}',ha='center',va='center',fontsize=5.6,color='white' if v>.56 else C['ink'])
 for sp in a.spines.values():sp.set_visible(False)
 for c in [3.5,4.5]:a.axvline(c,color='white',lw=1.8)
# Preserve the Qwen-specific branches rather than hiding them on a matched-depth plot.
deep=sorted(s[s.model.eq('Qwen3')&s.layer.str.startswith('deepstack')].layer.unique());a=axat(1535,1460,190,270);A=np.array([[s[(s.model=='Qwen3')&(s.layer==layer)&(s.target==tar)&(s.split==split)].r2.iloc[0] for layer in deep] for tar in targets for split in ['shape','illumination']]);a.imshow(A,cmap=cmap,vmin=0,vmax=.9,aspect='auto');a.set(xticks=range(len(deep)),xticklabels=['D'+str(i) for i in range(len(deep))],yticks=[]);a.tick_params(labelsize=5.5,length=0,pad=3)
for (r,c),v in np.ndenumerate(A):a.text(c,r,f'{v:.2f}',ha='center',va='center',fontsize=5.6,color='white' if v>.56 else C['ink'])
for sp in a.spines.values():sp.set_visible(False)
txt(1630,1417,'DeepStack',22,C['Qwen3'],ha='center');cax=axat(590,1810,550,15);cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[0,.3,.6,.9]);cb.ax.tick_params(labelsize=6,pad=2,length=2);txt(1190,1800,'Held-out R²',21,C['muted']);sep(1880)
# End-of-network readout: point-level and added-information evidence.
panel(42,1915,'g','Final language state predicts human judgments')
hexb=[]
for j,model in enumerate(['InternVL','Qwen3']):
 x=133+j*495;a=axat(x,2015,345,232);z=d[(d.model==model)&d.layer.eq('language_36')&d.split.eq('shape')&d.target.eq('human')];h=a.hexbin(z.truth,z.prediction,gridsize=24,mincnt=1,cmap='cividis',linewidths=0);hexb.append(h);a.plot([0,.15],[0,.15],':',color='#8D8A80',lw=.7);a.set(xlim=(-.005,.155),ylim=(-.02,.175),xticks=[0,.075,.15],yticks=[0,.075,.15],xlabel='Human matched gloss c',ylabel='Predicted human c' if j==0 else '');fmt(a);r=s[(s.model==model)&s.layer.eq('language_36')&s.split.eq('shape')&s.target.eq('human')].r2.iloc[0];a.text(.045,.94,f'R² = {r:.3f}',transform=a.transAxes,va='top',fontsize=6.4,bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=1));txt(x+172,1973,model,22,C[model],ha='center')
norm=LogNorm(1,max(max(h.get_array()) for h in hexb))
for h in hexb:h.set_norm(norm)
cax=axat(1040,2035,13,160);cb=fig.colorbar(hexb[0],cax=cax);cb.ax.tick_params(labelsize=5.5,pad=1,length=2);cb.set_label('Images',fontsize=6)
txt(133,2330,'3,888 held-out predictions; final sampled block L36.',20,C['muted'])
panel(1180,1915,'h','Information beyond physical/color cues');a=axat(1350,2015,345,232)
for j,split in enumerate(['shape','illumination']):
 for k,model in enumerate(['InternVL','Qwen3']):
  r=gain[(gain.model==model)&(gain.split==split)].iloc[0];y=1-j+(.11 if k==0 else -.11);a.errorbar(r.gain_r2,y,xerr=[[r.gain_r2-r.ci_low],[r.ci_high-r.gain_r2]],fmt='o' if k==0 else '^',ms=4,color=C[model],lw=1,capsize=2);a.text(r.gain_r2,y+(.15 if k==0 else -.26),f'+{r.gain_r2:.3f}',color=C[model],ha='center',fontsize=6.1)
a.axvline(0,color=C['gold'],ls=':',lw=.8);a.set(xlim=(-.02,.52),ylim=(-.5,1.5),yticks=[1,0],yticklabels=['Shape\nholdout','Lighting\nholdout'],xticks=[0,.2,.4],xlabel='Final-layer gain in R²');fmt(a);txt(1250,2330,'Paired 95% crossed-condition intervals.',20,C['muted'])
for ext in ['svg','pdf','png']:fig.savefig(O/f'Figure5_Frozen_Internal_Readout_EN.{ext}',dpi=450,facecolor='white')
print('Saved Figure 5.')
