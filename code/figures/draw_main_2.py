"""Portable nine-panel English result plate; input tables and images accompany output."""
from pathlib import Path
import json,shutil
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from PIL import Image
import os
OSFROOT=Path(__file__).resolve().parents[2]
OUT=Path(os.environ.get('OSF_REPRO_OUT',OSFROOT/'figures/regenerated'))/'main'
OUT.mkdir(parents=True,exist_ok=True)
O=OSFROOT/'figures/inputs/main/Fig2'
T=O/'data';d=pd.read_csv(T/'image_level_data.csv');ci=pd.read_csv(T/'v8_agreement_ci.csv').set_index('measure');pc=pd.read_csv(T/'v8_pair_ci.csv');within=pd.read_csv(T/'within_physical_levels.csv');res=pd.read_csv(T/'within_physical_association.csv');cells=json.loads((T/'gallery_cells.json').read_text());pair=json.loads((T/'conflict_example.json').read_text())
C={'InternVL':'#147D92','Qwen3':'#CC7243','human':'#5F6773','ink':'#1D2934','muted':'#6D7984','line':'#D7E0E5','pale':'#F5F8F9'}
plt.rcParams.update({'font.family':'Arial','font.size':7,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'xtick.major.width':.6,'ytick.major.width':.6,'xtick.color':C['muted'],'ytick.color':C['muted'],'axes.labelcolor':C['ink'],'text.color':C['ink']})
W,H=1800,2180;fig=plt.figure(figsize=(180/25.4,218/25.4),facecolor='white');base=fig.add_axes([0,0,1,1]);base.set(xlim=(0,W),ylim=(H,0));base.axis('off')
def txt(x,y,s,size=24,col=None,weight='normal',ha='left',rotation=0):return base.text(x,y,s,fontsize=size*72/254,color=col or C['ink'],weight=weight,ha=ha,va='top',rotation=rotation,linespacing=1.25)
def panel(x,y,letter,title):txt(x,y,letter,33,weight='bold');txt(x+39,y+2,title,26,weight='bold')
def axat(x,y,w,h):return fig.add_axes([x/W,1-(y+h)/H,w/W,h/H])
def line(x,y,xx,yy):base.plot([x,xx],[y,yy],lw=.55,color=C['line'])
def img(iid,x,y,w):a=axat(x,y,w,w);a.imshow(Image.open(O/'images'/f'{iid}.png'));a.axis('off')
def fmt(a):
 a.tick_params(labelsize=6.8,pad=2);a.xaxis.label.set_size(7);a.yaxis.label.set_size(7)
# a-c: identical density scales, all actual matching values, no fitted rescaling.
hexb=[]
for j,(letter,title,xkey,ykey,rkey) in enumerate([('a','Human judgments vs physical gloss','physical_c','human_c','physical'),('b','InternVL reference matching','human_c','internvl_c','InternVL_matching'),('c','Qwen3 reference matching','human_c','model_c','Qwen3_matching')]):
 x=42+j*570;panel(x,30,letter,title);a=axat(x+66,103,422,304)
 hb=a.hexbin(d[xkey],d[ykey],gridsize=32,extent=(0,.15,0,.15),mincnt=1,cmap='viridis',linewidths=0);hexb.append(hb)
 a.plot([0,.15],[0,.15],ls=':',color='#627080',lw=.7)
 a.set(xlim=(-.003,.153),ylim=(-.003,.153),xticks=[0,.05,.10,.15],yticks=[0,.05,.10,.15],xlabel='Physical gloss c' if j==0 else 'Human matched c',ylabel='Human matched c' if j==0 else 'Model matched c')
 a.text(.035,.95,f'n = {len(d):,}\nSpearman ρ = {ci.loc[rkey,"estimate"]:.3f}',transform=a.transAxes,va='top',fontsize=7,bbox={'facecolor':'white','edgecolor':'none','alpha':.88,'pad':2})
 fmt(a)
norm=LogNorm(1,max(max(h.get_array()) for h in hexb))
for h in hexb:h.set_norm(norm)
cax=axat(1689,119,12,271);cb=fig.colorbar(hexb[0],cax=cax);cb.set_label('Images (log scale)',fontsize=6);cb.ax.tick_params(labelsize=6,pad=2)
line(42,536,1758,536)
# d: compare human alignment across the two elicitation formats.
panel(42,566,'d','Human alignment across tasks')
a=axat(192,650,338,244)
for k,(task,label) in enumerate([('single','Single-image\nrating'),('matching','Reference\nmatching')]):
 for model,off in [('InternVL',.10),('Qwen3',-.10)]:
  r=ci.loc[model+'_'+task];a.errorbar(r.estimate,1-k+off,xerr=[[r.estimate-r.ci_low],[r.ci_high-r.estimate]],color=C[model],fmt='o',ms=4,capsize=2,lw=1)
r=ci.loc['physical'];a.errorbar(r.estimate,-.75,xerr=[[r.estimate-r.ci_low],[r.ci_high-r.estimate]],color=C['human'],fmt='D',ms=3.5,capsize=2,lw=.8)
a.set(yticks=[1,0,-.75],yticklabels=['Single-image\nrating','Reference\nmatching','Physical c'],ylim=(-1.12,1.35),xlim=(.35,.85),xticks=[.4,.6,.8],xlabel='Spearman ρ with human judgments');fmt(a)
txt(108,614,'● InternVL',21,C['InternVL']);txt(324,614,'● Qwen3',21,C['Qwen3'])
txt(108,1001,'95% crossed-condition bootstrap intervals',20,C['muted'])
# e: show all exact physical levels, not only a global effect.
panel(612,566,'e','Agreement at fixed physical gloss')
a=axat(684,650,425,244)
for model in ['InternVL','Qwen3']:
 z=within[within.model==model].rho.dropna().sort_values().to_numpy();a.step(z,np.arange(1,len(z)+1)/len(z),where='post',color=C[model],lw=1.3)
 med=np.median(z);a.axvline(med,color=C[model],ls=':',lw=.65);a.plot(med,.5,'o',color=C[model],ms=3)
a.set(xlim=(0,.82),ylim=(0,1.03),xticks=[0,.2,.4,.6,.8],yticks=[0,.5,1],xlabel='Within-level Spearman ρ',ylabel='Fraction of physical levels');fmt(a)
txt(684,614,'50 levels; medians: 0.496 / 0.619',22,C['muted'])
txt(684,1001,'Positive associations at every physical-c level',20,C['muted'])
# f: independent choice task, with both display orders included.
panel(1182,566,'f','Choices follow human ordering')
a=axat(1270,650,427,244)
for model,off in [('InternVL',-.12),('Qwen3',.12)]:
 for k,st in enumerate(['random','discordant']):
  r=pc[(pc.model==model)&(pc.predictor=='human')&(pc.stratum==st)].iloc[0]
  a.errorbar(k+off,100*r.estimate,yerr=[[100*(r.estimate-r.ci_low)],[100*(r.ci_high-r.estimate)]],color=C[model],fmt='o',ms=4,capsize=2,lw=1)
  a.text(k+off,100*r.ci_high+1.1,f'{100*r.estimate:.1f}',ha='center',fontsize=6.5,color=C[model])
a.axhline(50,color=C['human'],ls=':',lw=.8);a.text(-.32,51.2,'Chance',fontsize=6,color=C['muted'])
a.set(xlim=(-.42,1.42),ylim=(45,86),yticks=[50,60,70,80],xticks=[0,1],xticklabels=['Random pairs\n(n = 1,000)','Human–physical\nconflict (n = 1,000)'],ylabel='Human-order agreement (%)');fmt(a)
txt(1270,1001,'Both display orders; ties receive half credit',20,C['muted'])
line(42,1050,1758,1050)
# g: multi-image grid, physically/perceptually stratified; no model-based sample filtering.
panel(42,1080,'g','Stimulus-level dissociations between physical and perceived gloss')
txt(160,1130,'Physical gloss c',23,C['muted'])
for col,lab in enumerate(['Low: 0–0.0496','Middle: 0.0496–0.0991','High: 0.0991–0.1487']):txt(320+col*330,1170,lab,21,C['muted'],ha='center')
for cell in cells:
 r=2-cell['human_bin'];c=cell['physical_bin'];x=160+c*330;y=1215+r*246
 base.add_patch(Rectangle((x-8,y-9),317,226,facecolor=C['pale'],edgecolor=C['line'],lw=.5))
 txt(x+3,y,'n = '+str(cell['n']),21,C['muted'])
 for j,iid in enumerate(cell['image_ids']):img(iid,x+j*157,y+34,143)
 txt(x+3,y+187,f'Median ranks: I {cell["intern_rank_median"]:.0f}  |  Q {cell["qwen_rank_median"]:.0f}',21)
for y,lab in [(1290,'High'),(1536,'Middle'),(1782,'Low')]:txt(116,y,lab,23,C['muted'],ha='right')
txt(34,1610,'Human matched gloss (tertiles)',23,C['muted'],rotation=90)
txt(160,1979,'Examples near each cell center; selected without using model agreement.',22,C['muted'])
txt(160,2015,'I / Q = median InternVL / Qwen3 matching percentile rank (0–100) in each full cell.',21,C['muted'])
txt(160,2051,'Counts and median ranks use all 3,849 images; photographs are illustrative examples.',21,C['muted'])
line(1172,1090,1172,2085)
# h: pooled conditional association with exact physical level removed.
panel(1210,1080,'h','Shared deviations from physical gloss')
txt(1250,1130,'Subtract the mean within each physical-c level',21,C['muted'])
a=axat(1360,1210,342,172)
for k,model in enumerate(['InternVL','Qwen3']):
 r=res[res.model==model].iloc[0];a.errorbar(r.estimate,1-k,xerr=[[r.estimate-r.ci_low],[r.ci_high-r.estimate]],color=C[model],fmt='o',ms=5,lw=1.2,capsize=3)
 a.text(r.estimate,1-k+.19,f'r = {r.estimate:.3f}',ha='center',fontsize=7,color=C[model])
a.set(xlim=(.3,.8),ylim=(-.4,1.55),yticks=[1,0],yticklabels=['InternVL','Qwen3'],xticks=[.3,.5,.7],xlabel='Within-level residual Pearson r');fmt(a)
txt(1250,1490,'Physical-level means are refitted in each bootstrap.',20,C['muted'])
txt(1250,1522,'Crossed shape × illumination 95% intervals.',20,C['muted'])
# i: a concrete selected positive example, separate from population statistics.
panel(1210,1590,'i','A human–physical conflict example')
lookup=d.set_index('image_id');ids=[pair['a_id'],pair['b_id']]
for k,iid in enumerate(ids):img(iid,1270+k*229,1650,180);txt(1360+k*229,1840,chr(65+k),23,weight='bold',ha='center')
for yy,label,key in [(1892,'Physical c','physical_c'),(1930,'Human match','human_c')]:
 txt(1250,yy,label,23)
 for k,iid in enumerate(ids):txt(1490+k*172,yy,f'{lookup.loc[iid,key]:.3f}',23,ha='center')
winner='A' if pair['delta_human']>0 else 'B';phys='A' if pair['delta_physical']>0 else 'B'
txt(1250,1980,f'Physical ordering: {phys} > {winner}',23,C['muted'])
txt(1250,2018,f'Human ordering: {winner} > {phys}',23,weight='bold')
txt(1250,2056,f'InternVL: {winner}     Qwen3: {winner}  (both orders)',23,C['InternVL'],weight='bold')
line(42,2111,1758,2111)
txt(82,2138,'Two frozen models · common evaluation cohort · archived responses · illustrative examples distinguished from full-sample statistics',21,C['muted'])
for ext in ['svg','pdf','png']:fig.savefig(OUT/f'Figure2_HumanLike_Gloss_EN.{ext}',dpi=450,facecolor='white')
plt.close(fig)
print('Figure 2 exported:',O)
