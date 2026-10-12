"""Chinese publication figures from frozen, previously analysed source tables."""
from pathlib import Path
import json, shutil
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/'figure_inputs').exists():
 S=ROOT;B=ROOT/'figure_inputs';M=B/'mllm_material'
else:
 B=ROOT;M=B/'mllm_material';S=M/'nmi_manuscript_zh_v1/source'
T=M/'submission_synthesis_v1/tables'
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'xtick.major.width':.7,'ytick.major.width':.7,'pdf.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':False,'savefig.facecolor':'white'})
C=['#167D9A','#D47735']; names=['InternVL','Qwen3'];gray='#697583'
def save(fig,key):
 for ext in ['pdf','png','svg']:fig.savefig(S/f'figures/{key}.{ext}',dpi=240,bbox_inches='tight')
 plt.close(fig)
def panel(ax,s): ax.text(-.12,1.08,s,transform=ax.transAxes,weight='bold',fontsize=14)
def bars(ax,values,labels,ylabel,lim=(0,1)):
 x=np.arange(len(labels))
 for j in range(2):
  b=ax.bar(x+(j-.5)*.36,values[j],.33,color=C[j],label=names[j])
  for p,v in zip(b,values[j]):ax.text(p.get_x()+p.get_width()/2,v+.022,f'{v:.3f}',ha='center',fontsize=9)
 ax.set(xticks=x,xticklabels=labels,ylabel=ylabel,ylim=lim);ax.grid(axis='y',alpha=.13);ax.set_axisbelow(True)
for p in T.iterdir():
 if p.is_file():shutil.copy2(p,S/'tables'/p.name)
joint=pd.read_csv(M/'qwen3_full_v1/joint_model_predictions.csv');held=joint[joint.cohort!='development']
# Explicit held-out flag values are audited; figures use original cohort when present.
if 'matching_heldout' in joint.cohort.values:held=joint[joint.cohort=='matching_heldout']
stim=pd.read_csv(M/'mechanisms_v1/stimuli.csv')
fig=plt.figure(figsize=(10.4,8.7),layout='constrained');gs=fig.add_gridspec(3,4,height_ratios=[1.1,1,.75])
images=[(B/stim.iloc[0].image_512_path,'测试刺激实例')]+[(M/f'reference_reconstruction_v3/output/reference_{k:02d}.png',f'重建参考：第 {k} 档') for k in [1,27,52]]
for i,(p,lab) in enumerate(images):
 ax=fig.add_subplot(gs[0,i]);ax.imshow(Image.open(p));ax.set_title(lab,fontsize=10);ax.axis('off')
 if i==0:panel(ax,'a')
for i,(title,txt) in enumerate([('单图评分','0（哑光）至 100（高光泽）\n每个模型 3,888 张图'),('双图选择','比较外观光泽，交换顺序\n2,000 图对 × 2 顺序'),('非对称参考匹配','52 档；最多 7 次比较\n3,888 张图 × 2 次起始'),('内部响应与干预','回答前提取；冻结权重\n读出与干预独立于采集')]):
 ax=fig.add_subplot(gs[1,i]);ax.axis('off');ax.text(.5,.8,title,ha='center',fontsize=11,weight='bold');ax.text(.5,.43,txt,ha='center',va='center',linespacing=1.8,fontsize=9.5,bbox=dict(boxstyle='round,pad=.7',fc='#EEF4F7',ec='#D5E1E8'))
 if i==0:panel(ax,'b')
ax=fig.add_subplot(gs[2,:]);ax.axis('off');panel(ax,'c')
ax.text(.5,.88,'相同刺激与核心任务 → 两个冻结的预训练视觉语言系统',ha='center',weight='bold',fontsize=12)
ax.text(.5,.60,'人类数据：36 种形状 × 36 种照明 × 3 个视角 = 3,888 张图',ha='center')
ax.text(.5,.32,'行为一致性 → 条件组织 → 跨任务预测 → 内部信息 → 功能干预',ha='center',color=C[0],fontsize=12)
ax.text(.5,.05,'行为采集不使用目标人类评分微调；监督读出单独标记。',ha='center',color=gray)
save(fig,'Fig1')
fig,axs=plt.subplots(2,2,figsize=(10,8),layout='constrained')
for j,col in enumerate(['internvl_c','model_c']):
 ax=axs[0,j];ax.hexbin(held.human_c,held[col],gridsize=28,mincnt=1,cmap=['Blues','Oranges'][j]);ax.plot([0,.15],[0,.15],':',color=gray);ax.set(xlabel='人类平均匹配值 c',ylabel='模型平均匹配值 c',title=names[j],xlim=(0,.155),ylim=(0,.155));panel(ax,chr(97+j))
bars(axs[1,0],[[.69613245,.58714454],[.72716503,.68040256]],['单图评分','参考匹配'],'与人类的 Spearman ρ');axs[1,0].axhline(.53017542,color=gray,ls=':',label='物理 c');axs[1,0].legend(fontsize=8);panel(axs[1,0],'c')
bars(axs[1,1],[[.719,.688],[.706,.7525]],['随机图对','人类–物理冲突图对'],'跟随人类评分排序的比例');axs[1,1].axhline(.5,color=gray,ls=':');panel(axs[1,1],'d');save(fig,'Fig2')
e=pd.read_csv(T/'internvl_condition_effects.csv');r=pd.read_csv(T/'rsa.csv')
fig,axs=plt.subplots(2,2,figsize=(10,8),layout='constrained')
for j,fac in enumerate(['shape_id','illumination_id']):
 ax=axs[0,j];z=e[e.factor==fac];q=pd.read_csv(M/f'qwen3_full_v1/{fac}_effects.csv')
 ax.scatter(z.human_adjusted_sd,z.model_adjusted_sd,color=C[0],s=24,label=names[0]);ax.scatter(q.human_adjusted_sd,q.model_adjusted_sd,color=C[1],marker='^',s=26,label=names[1]);ax.axhline(0,color=gray,lw=.6);ax.axvline(0,color=gray,lw=.6);ax.set(xlabel='人类调整后条件效应（标准差）',ylabel='模型调整后条件效应（标准差）',title=['形状：36 个条件','照明：36 个条件'][j]);ax.legend(fontsize=8);panel(ax,chr(97+j))
bars(axs[1,0],[[.50399,.809009],[.67129987,.79948520]],['形状','照明'],'调整后条件效应 Spearman ρ');panel(axs[1,0],'c')
ax=axs[1,1]
for j,n in enumerate(names):
 z=r[(r.space=='physical_color_adjusted')&(r.agent1=='Human')&(r.agent2==n)].set_index('factor').loc[['shape','illumination']];x=np.arange(2)+(j-.5)*.14;ax.errorbar(x,z.rsa,yerr=[z.rsa-z.ci_low,z.ci_high-z.rsa],fmt='o',color=C[j],capsize=4,label=n)
ax.set(xticks=[0,1],xticklabels=['形状响应轮廓','照明响应轮廓'],ylabel='人类–模型 RSA（Spearman）',ylim=(0,.9));ax.legend(fontsize=8);panel(ax,'d');save(fig,'Fig3')
fig,axs=plt.subplots(2,2,figsize=(10,7.8),layout='constrained')
bars(axs[0,0],[[.652251827,.653815392],[.775781586,.777807883]],['留出形状','留出照明'],'单图 → 匹配：交叉验证 R²');panel(axs[0,0],'a');axs[0,0].legend(fontsize=8)
bars(axs[0,1],[[.689379288,.690535274],[.748970441,.750473038]],['留出形状','留出照明'],'匹配 → 单图：交叉验证 R²');panel(axs[0,1],'b')
bars(axs[1,0],[[.767,.7965],[.8035,.85075]],['单图评分排序','匹配值排序'],'预测实际双图选择的比例');panel(axs[1,0],'c')
ax=axs[1,1];ax.axis('off');panel(ax,'d');ax.text(.5,.84,'共同光泽判断的三个行为接口',ha='center',weight='bold',fontsize=12)
for y,txt in [(.62,'单图：可见表面有多光泽？'),(.40,'双图：哪一个看起来更光泽？'),(.18,'匹配：与哪一档参考同样光泽？')]:ax.text(.5,y,txt,ha='center',bbox=dict(boxstyle='round,pad=.6',fc='#EEF4F7',ec='none'))
save(fig,'Fig4')
rd=pd.read_csv(T/'mechanism_readout.csv');layers=['vision_06','vision_12','vision_18','vision_24','bridge','language_09','language_18','language_27','language_36']; actual=rd.layer.unique().tolist();print('layers',actual)
layers=[x for x in layers if x in actual]
if len(layers)<9:layers=[x for x in actual if not 'deepstack' in x.lower()]
fig,axs=plt.subplots(2,2,figsize=(11,8),layout='constrained')
for a,(split,target) in zip(axs.ravel(),[('shape','human'),('illumination','human'),('shape','human_beyond_physical_color'),('illumination','human_beyond_physical_color')]):
 for j,n in enumerate(names):
  z=rd[(rd.model==n)&(rd.split==split)&(rd.target==target)].set_index('layer').reindex(layers);a.plot(range(len(layers)),z.r2,'o-',color=C[j],label=n,ms=4)
 a.set(xticks=range(len(layers)),xticklabels=[x.replace('vision_','视').replace('language_','语').replace('bridge','映射') for x in layers],ylabel='留出预测 R²',ylim=(-.05,.85),title=('留形状' if split=='shape' else '留照明')+('：人类评分' if target=='human' else '：控制物理与颜色后的读出'))
 a.tick_params(axis='x',labelrotation=45,labelsize=8);a.legend(fontsize=8);panel(a,chr(97+list(axs.ravel()).index(a)))
save(fig,'Fig5')
iv=pd.read_csv(T/'mechanism_intervention.csv')
fig,axs=plt.subplots(2,2,figsize=(10,8.2),layout='constrained')
for j,n in enumerate(names):
 ax=axs[0,j]
 for d,col,lab in [('candidate',C[j],'参考定义方向'),('brightness','#525A64','亮度方向'),('random0','#A7B1BB','随机方向 1'),('random1','#8AA896','随机方向 2'),('random2','#B2A1BB','随机方向 3')]:
  z=iv[(iv.model==n)&(iv.direction==d)].sort_values('dose');x=np.r_[z.dose.values,0];y=np.r_[z.mean_delta_p.values,0];ii=np.argsort(x);ax.plot(x[ii],y[ii],'o-',color=col,label=lab,lw=2 if d=='candidate' else 1,ms=3)
  if d=='candidate':ax.fill_between(z.dose,z.ci_low,z.ci_high,color=col,alpha=.14)
 ax.axhline(0,color=gray,lw=.6);ax.set(xlabel='干预剂量（参考范围单位）',ylabel='测试图更光泽的条件概率变化',title=n,ylim=(-.16,.16));ax.legend(fontsize=8);panel(ax,chr(97+j))
 z=pd.read_csv(T/'mechanism_cluster_contrasts.csv');z=z[z.model==n];ax=axs[1,j];x=np.arange(len(z));ax.errorbar(x,z.effect,yerr=[z.effect-z.ci_low,z.ci_high-z.effect],fmt='o',capsize=4,color=C[j]);ax.axhline(0,color=gray,lw=.6);ax.set(xticks=x,xticklabels=['亮度','随机 1','随机 2','随机 3'],ylabel='候选相对对照的双向效应差',xlabel='对照方向',ylim=(-.01,.31),title='36 形状聚类的成对比较');panel(ax,chr(99+j))
save(fig,'Fig6')
# Supplement: transparent distributions and complete statistical families.
fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for j,col in enumerate(['internvl_c','model_c']):
 axs[j].hist(joint.human_c,bins=np.linspace(0,.151,22),histtype='step',lw=1.7,color=gray,label='人类');axs[j].hist(joint[col],bins=np.linspace(0,.151,22),histtype='step',lw=1.7,color=C[j],label=names[j]);axs[j].set(xlabel='平均匹配值 c',ylabel='图像数');axs[j].legend();panel(axs[j],chr(97+j))
save(fig,'S1')
fig,axs=plt.subplots(3,2,figsize=(10,10),layout='constrained')
for i,sp in enumerate(['raw','physical_color_adjusted','interaction_only']):
 for j,fac in enumerate(['shape','illumination']):
  a=axs[i,j];z=r[(r.space==sp)&(r.factor==fac)];x=np.arange(len(z));a.errorbar(x,z.rsa,yerr=[z.rsa-z.ci_low,z.ci_high-z.rsa],fmt='o',color=C[j],capsize=4);a.set(xticks=x,xticklabels=['人–InternVL','人–Qwen3','两模型'],ylabel='RSA 相关',ylim=(-.15,1),title=['原始','物理与颜色调整','仅交互'][i]+['：形状','：照明'][j]);a.axhline(0,color=gray,lw=.6);a.tick_params(axis='x',labelsize=8);panel(a,chr(97+i*2+j))
save(fig,'S2')
lat=json.loads((T/'latent_prediction.json').read_text())
fig,ax=plt.subplots(figsize=(8,4.6),layout='constrained');x=np.arange(3)
for j,(key,lab,col) in enumerate([('r2_nuisance','物理与颜色',gray),('r2_additive','加性形状与照明',C[0]),('r2_selected','嵌套选择低秩',C[1])]):
 ax.bar(x+(j-1)*.25,[lat[a][key] for a in ['Human','InternVL','Qwen3']],.23,label=lab,color=col)
ax.set(xticks=x,xticklabels=['人类','InternVL','Qwen3'],ylabel='留出视角预测 R²',ylim=(0,.75));ax.legend(fontsize=9);save(fig,'S3')
pilot=pd.read_csv(T/'pilot_and_tiny.csv');fig,ax=plt.subplots(figsize=(9,5),layout='constrained');labs=['GPT 先导','InternVL','Qwen3','物理 c','作者单层/照明','作者单层/形状','作者三层/照明','作者三层/形状','独立小网络/形状','独立小网络/照明'];ax.barh(labs,pilot.spearman,color=[C[0]]*3+[gray]*7);ax.invert_yaxis();ax.set(xlabel='同 12 张图：与人类的 Spearman ρ',xlim=(0,1));save(fig,'S4')
tiny=json.loads((T/'tiny_incremental.json').read_text());df=pd.DataFrame(tiny['metrics']);fig,axs=plt.subplots(1,2,figsize=(9,4.5),layout='constrained')
for a,sp in zip(axs,['shape','illumination']):
 z=df[df.split==sp].set_index('model');keys=['controls_tiny_cubic','controls_tiny_cubic_plus_internvl'];a.bar(['控制＋小网络','加入 InternVL'],[z.loc[k,'r2'] for k in keys],color=[gray,C[0]]);a.set(ylabel='留出预测 R²',ylim=(0,.7),title='留出'+('形状' if sp=='shape' else '照明'))
save(fig,'S5')
mt=pd.read_csv(T/'mechanism_matches.csv');fig,axs=plt.subplots(1,2,figsize=(10,4.7),layout='constrained')
for j,n in enumerate(names):
 a=axs[j]
 for direction,col,lab in [('candidate',C[j],'参考定义方向'),('random0',gray,'随机方向 1')]:
  vals=[]
  for dose in [-1,0,1]:
   z=mt[(mt.model==n)&(mt.direction==('baseline' if dose==0 else direction))&(mt.dose==dose)];vals.append(z.matching_c.mean())
  a.plot([-1,0,1],vals,'o-',color=col,label=lab)
 a.set(xlabel='剂量',ylabel='12 张图平均完整匹配值 c',title=n,ylim=(.09,.14));a.legend(fontsize=9)
save(fig,'S6')
# Replot saved adjusted matrices, coordinates and spatial projections in Chinese.
for fac,key in [('shape','S7'),('illumination','S8')]:
 fig,axs=plt.subplots(2,3,figsize=(10.2,7.2),layout='constrained')
 for j,n in enumerate(['Human','InternVL','Qwen3']):
  p=M/f'behavioral_spaces_v1/matrices/physical_color_adjusted_{fac}_{n}_profiles_by_view.npy';cube=np.load(p);mat=cube.mean(axis=0)
  rdm=pd.read_csv(M/f'behavioral_spaces_v1/matrices/physical_color_adjusted_{fac}_{n}_rdm.csv').to_numpy()
  im=axs[0,j].imshow(mat/mat.std(),cmap='RdBu_r',vmin=-3,vmax=3);axs[0,j].set(title=('人类' if n=='Human' else n)+'：调整后轮廓',xlabel='照明' if fac=='shape' else '形状',ylabel='形状' if fac=='shape' else '照明')
  den=rdm[np.triu_indices(36,1)].mean();im2=axs[1,j].imshow(rdm/den,cmap='magma',vmin=0,vmax=3);axs[1,j].set(title='相对行为不相似性',xlabel='条件索引',ylabel='条件索引')
 fig.colorbar(im,ax=axs[0,:].tolist(),shrink=.7,label='响应 / 轮廓标准差');fig.colorbar(im2,ax=axs[1,:].tolist(),shrink=.7,label='距离 / 平均距离');save(fig,key)
fig,axs=plt.subplots(2,3,figsize=(10,7.2),layout='constrained')
for i,fac in enumerate(['shape','illumination']):
 for j,n in enumerate(['Human','InternVL','Qwen3']):
  d=pd.read_csv(M/f'behavioral_spaces_v1/embeddings/{n}_{fac}_adjusted_profile_PCA.csv');a=axs[i,j];a.scatter(d.dim1,d.dim2,s=24,color=([gray]+C)[j]);
  for rr in d.itertuples():a.annotate(str(rr.condition_id),(rr.dim1,rr.dim2),fontsize=6,xytext=(2,2),textcoords='offset points')
  a.set(title=('人类' if n=='Human' else n)+'：'+('形状' if fac=='shape' else '照明'),xlabel='主成分 1',ylabel='主成分 2');a.axhline(0,color=gray,lw=.4);a.axvline(0,color=gray,lw=.4)
save(fig,'S10')
plan=json.loads((M/'mechanisms_v1/plan.json').read_text('utf-8'));fig,axs=plt.subplots(2,5,figsize=(12,6.1),layout='constrained')
for row,n in enumerate(names):
 root=M/'mechanisms_v1'/n;dire=np.load(root/'directions.npz');ref=dire['reference_means']@dire['candidate'];axs[row,0].plot(np.linspace(0,.1487,52),ref,color=C[row]);axs[row,0].set(title=n+'：参考序列',xlabel='参考 c',ylabel='候选方向投影')
 scores=pd.read_csv(root/'candidate_scores.csv');z=scores[scores.image_id.isin(plan['intervention_images'])].sort_values('projection');ids=z.image_id.iloc[[0,1,-2,-1]].tolist()
 for j,id in enumerate(ids):
  r=stim[stim.image_id==id].iloc[0];im=np.asarray(Image.open(B/r.image_512_path).convert('RGB'));token=np.load(root/'features'/f'{id}.npz')['bridge_tokens'].astype('float32')@dire['candidate'];side=int(np.sqrt(len(token)));heat=np.asarray(Image.fromarray(token.reshape(side,side)).resize((im.shape[1],im.shape[0]),Image.Resampling.BILINEAR));ax=axs[row,j+1];ax.imshow(im);ax.imshow(heat,cmap='magma',alpha=.55);ax.set_title(id+'\n候选响应投影',fontsize=8);ax.axis('off')
save(fig,'S9')
if Path(__file__).resolve()!=(S/'code'/Path(__file__).name).resolve():shutil.copy2(__file__,S/'code'/Path(__file__).name)
print('figures complete; cohort counts',joint.cohort.value_counts().to_dict(),'scatter n',len(held),'targets',rd.target.unique().tolist())
