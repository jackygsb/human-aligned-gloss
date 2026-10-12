"""English Figure 3: shared condition effects and behavioral geometry."""
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import rankdata,spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import os
OSFROOT=Path(__file__).resolve().parents[2]
OUT=Path(os.environ.get('OSF_REPRO_OUT',OSFROOT/'figures/regenerated'))/'main'
OUT.mkdir(parents=True,exist_ok=True)
O=OSFROOT/'figures/inputs/main/Fig3'
T=O/'data';eff=pd.read_csv(T/'condition_effects.csv');tests=pd.read_csv(T/'v8_condition_tests.csv');rsa=pd.read_csv(T/'rsa_results.csv');rsa=rsa[rsa.space.eq('physical_color_adjusted')];gallery=pd.read_csv(T/'gallery.csv')
C={'Human':'#5F6773','InternVL':'#147D92','Qwen3':'#CC7243','ink':'#1D2934','muted':'#6D7984','line':'#D7E0E5','pale':'#F5F8F9'}
plt.rcParams.update({'font.family':'Arial','font.size':7,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'xtick.major.width':.6,'ytick.major.width':.6,'xtick.color':C['muted'],'ytick.color':C['muted'],'axes.labelcolor':C['ink'],'text.color':C['ink']})
W,H=1800,2370;fig=plt.figure(figsize=(180/25.4,237/25.4),facecolor='white');base=fig.add_axes([0,0,1,1]);base.set(xlim=(0,W),ylim=(H,0));base.axis('off')
def txt(x,y,s,size=24,col=None,weight='normal',ha='left'):return base.text(x,y,s,fontsize=size*72/254,color=col or C['ink'],weight=weight,ha=ha,va='top',linespacing=1.25)
def panel(x,y,l,t):txt(x,y,l,33,weight='bold');txt(x+40,y+2,t,26,weight='bold')
def axat(x,y,w,h):return fig.add_axes([x/W,1-(y+h)/H,w/W,h/H])
def sep(y):base.plot([42,1758],[y,y],color=C['line'],lw=.6)
def rdm(f,a):return np.loadtxt(T/f'physical_color_adjusted_{f}_{a}_rdm.csv',delimiter=',')
def fmt(a):a.tick_params(labelsize=6.2,pad=2);a.xaxis.label.set_size(6.7);a.yaxis.label.set_size(6.7)
for x,agent in [(80,'Human'),(295,'InternVL'),(535,'Qwen3')]:txt(x,70,'● '+agent,23,C[agent])
txt(900,70,'3,888 images  ·  36 shapes × 36 illuminations × 3 viewpoints',23,C['muted'])
# Complete condition effect rankings, ordered using human values only.
for j,factor in enumerate(['shape','illumination']):
 x=42+j*580;panel(x,130,chr(97+j),factor.title()+' effects across all conditions');z=eff[eff.factor.eq(factor)].sort_values('Human');a=axat(x+75,210,445,320)
 a.axhline(0,color=C['muted'],lw=.65);a.plot(range(1,37),z.Human,color=C['Human'],lw=1.1,zorder=2)
 for agent,m in [('InternVL','o'),('Qwen3','^')]:a.scatter(range(1,37),z[agent],c=C[agent],marker=m,s=11,linewidths=0,zorder=3)
 a.set(xlim=(0,37),ylim=(-1.35,1.15),xticks=[1,12,24,36],yticks=[-1,0,1],xlabel='Conditions ordered by human effect',ylabel='Adjusted effect (s.d.)');fmt(a)
 txt(x+75,632,'Lower perceived gloss  →  Higher perceived gloss',20,C['muted'])
panel(1220,130,'c','Agreement in condition effects');a=axat(1325,210,358,320)
for j,f in enumerate(['shape','illumination']):
 for k,agent in enumerate(['InternVL','Qwen3']):
  r=tests[(tests.factor==f+'_id')&(tests.model==agent)].iloc[0];y=1-j+(0.10 if k==0 else -.10);a.errorbar(r.estimate,y,xerr=[[r.estimate-r.ci_low],[r.ci_high-r.estimate]],fmt='o' if k==0 else '^',ms=4,color=C[agent],capsize=2,lw=1)
  a.text(r.estimate,y+.14 if k==0 else y-.24,f'{r.estimate:.3f}',ha='center',color=C[agent],fontsize=6.5)
a.set(xlim=(0,1),ylim=(-.5,1.5),yticks=[0,1],yticklabels=['Lighting','Shape'],xticks=[0,.5,1],xlabel='Spearman ρ with human effects');fmt(a)
txt(1280,632,'95% intervals; all four q < 0.002',20,C['muted']);sep(690)
# Six full distance matrices, in a shared original-ID order (no fitted alignment).
for j,factor in enumerate(['shape','illumination']):
 x=42+j*890;panel(x,724,'de'[j],factor.title()+' geometry: full response profiles')
 txt(x+40,772,'Distances across '+('36 illumination responses' if factor=='shape' else '36 shape responses'),22,C['muted'])
 for k,agent in enumerate(['Human','InternVL','Qwen3']):
  xx=x+45+k*269;txt(xx+113,823,agent,24,C[agent],ha='center');R=rdm(factor,agent);R=R/R[np.triu_indices(36,1)].mean();a=axat(xx,866,226,226);im=a.imshow(R,cmap='magma',vmin=0,vmax=3,interpolation='nearest');a.set(xticks=[0,17,35],xticklabels=[1,18,36],yticks=[0,17,35],yticklabels=[1,18,36]);a.tick_params(labelsize=5.5,length=2,pad=1)
  for sp in a.spines.values():sp.set_visible(False)
 txt(x+420,1139,'Condition ID (same order in all three matrices)',21,C['muted'],ha='center')
cax=axat(695,1200,400,15);cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[0,1,2,3]);cb.ax.set_xticklabels(['0','1','2','≥3']);cb.ax.tick_params(labelsize=6,length=2,pad=2);txt(900,1262,'Dissimilarity / mean off-diagonal distance',21,C['muted'],ha='center');sep(1310)
# Pairwise structure shown at full pair level and summarized with node-level inference.
panel(42,1345,'f','Condition-pair geometry tracks human geometry')
for j,factor in enumerate(['shape','illumination']):
 for k,agent in enumerate(['InternVL','Qwen3']):
  x=115+j*570+k*265;a=axat(x,1435,206,240);ix=np.triu_indices(36,1);xx=rankdata(rdm(factor,'Human')[ix])/630;yy=rankdata(rdm(factor,agent)[ix])/630;a.scatter(xx,yy,s=1.8,color=C[agent],alpha=.32,edgecolors='none',rasterized=False);a.plot([0,1],[0,1],':',lw=.6,color='#A7B1BB');a.set(xlim=(0,1),ylim=(0,1),xticks=[0,.5,1],yticks=[0,.5,1]);fmt(a)
  if k==0:a.set_ylabel('Model distance rank')
  r=rsa[(rsa.factor==factor)&(rsa.agent1=='Human')&(rsa.agent2==agent)].iloc[0];a.text(.06,.92,f'ρ = {r.rsa:.3f}',transform=a.transAxes,fontsize=6.2,bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1))
  txt(x+103,1400,agent,21,C[agent],ha='center')
 txt(350+j*570,1728,factor.title()+': human distance rank',21,ha='center')
txt(115,1790,'630 condition pairs per plot; ranks scaled to [0, 1].',21,C['muted'])
panel(1220,1345,'g','RSA across all three observers');a=axat(1430,1435,270,240)
for j,pair in enumerate([('Human','InternVL'),('Human','Qwen3'),('InternVL','Qwen3')]):
 for k,factor in enumerate(['shape','illumination']):
  r=rsa[(rsa.factor==factor)&(rsa.agent1==pair[0])&(rsa.agent2==pair[1])].iloc[0];y=2-j+(.13 if k==0 else -.13);a.errorbar(r.rsa,y,xerr=[[r.rsa-r.ci_low],[r.ci_high-r.rsa]],fmt='o' if k==0 else 's',color='#4B687A' if k==0 else '#A582B4',ms=3.5,capsize=2,lw=.9)
a.set(yticks=[2,1,0],yticklabels=['Human–InternVL','Human–Qwen3','InternVL–Qwen3'],xlim=(0,.85),xticks=[0,.4,.8],ylim=(-.5,2.5),xlabel='Spearman RSA');fmt(a)
txt(1270,1758,'● Shape   ■ Illumination',21,C['muted']);txt(1270,1790,'Node-bootstrap 95% intervals; all q ≤ 0.0009',19,C['muted']);sep(1850)
# Images anchor the condition labels: selected solely from human extremes.
panel(42,1885,'h','What do lower- and higher-gloss conditions look like?')
for j,(factor,direction) in enumerate([('shape','Lower'),('shape','Higher'),('illumination','Lower'),('illumination','Higher')]):
 x=65+j*435;z=gallery[(gallery.factor==factor)&(gallery.direction==direction)];cond=int(z.iloc[0].condition_id);txt(x,1945,f'{direction} human effect | {factor} {cond}',22,weight='bold')
 for k,(_,r) in enumerate(z.iterrows()):
  xx=x+k*132;a=axat(xx,1995,125,125);a.imshow(Image.open(O/'images'/f'{r.image_id}.png'));a.axis('off');txt(xx+62,2131,f'c = {r.physical_c:.3f}',18,C['muted'],ha='center')
 vals=eff[(eff.factor==factor)&(eff.condition_id==cond)].iloc[0]
 txt(x,2180,'Adjusted effects (s.d.)',20,C['muted'])
 for k,agent in enumerate(['Human','InternVL','Qwen3']):txt(x+k*132,2213,f'{agent}\n{vals[agent]:+.2f}',20,C[agent])
txt(65,2293,'Illustrations selected by human effects only; physical c and the other scene attributes vary across images.',21,C['muted'])
for ext in ['svg','pdf','png']:fig.savefig(OUT/f'Figure3_Shared_Condition_Geometry_EN.{ext}',dpi=450,facecolor='white')
print('Saved Figure 3: 8 panels; 6 RDMs; 12 stimulus examples.')
