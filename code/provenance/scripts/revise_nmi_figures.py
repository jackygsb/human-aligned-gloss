from pathlib import Path
B=Path(__file__).resolve().parents[1];O=B/'mllm_material/nmi_revision_v8/source'
s=(B/'scripts/plot_nmi_manuscript.py').read_text('utf-8')
a=s.index('ROOT=');b=s.index("T=M/",a)
s=s[:a]+"S=Path(__file__).resolve().parents[1];B=S/'figure_inputs';M=B/'mllm_material'\n"+s[b:]
s=s.replace("def save(fig,key):\n", "def save(fig,key):\n fig.set_size_inches(*(fig.get_size_inches()*.84))\n")
a=s.index('def bars(');b=s.index('for p in T.iterdir()',a)
s=s[:a]+'''ci=pd.read_csv(S/'tables/v8_agreement_ci.csv'); pci=pd.read_csv(S/'tables/v8_pair_ci.csv'); cci=pd.read_csv(S/'tables/v8_condition_tests.csv'); cvci=pd.read_csv(S/'tables/v8_cross_task_ci.csv')
def bars(ax,values,labels,ylabel,lim=(0,1)):
 x=np.arange(len(labels))
 for j,name in enumerate(names):
  vals=[];los=[];his=[]
  for k in range(len(labels)):
   if '与人类的 Spearman' in ylabel: z=ci[ci.measure.eq(name+'_'+['single','matching'][k])].iloc[0]
   elif '跟随人类' in ylabel:z=pci[(pci.model==name)&(pci.predictor=='human')&(pci.stratum==['random','discordant'][k])].iloc[0]
   elif '条件效应' in ylabel:z=cci[(cci.model==name)&(cci.factor==['shape_id','illumination_id'][k])].iloc[0]
   elif '交叉验证' in ylabel:z=cvci[(cvci.model==name)&(cvci.split==['shape','illumination'][k])&(cvci.source==('single_score' if '单图 →' in ylabel else 'model_c'))].iloc[0]
   elif '预测实际' in ylabel:z=pci[(pci.model==name)&(pci.predictor==['single','matching'][k])&(pci.stratum=='all')].iloc[0]
   else:raise ValueError(ylabel)
   vals.append(z.estimate);los.append(z.estimate-z.ci_low);his.append(z.ci_high-z.estimate)
  ax.errorbar(x+(j-.5)*.18,vals,yerr=[los,his],fmt=['o','s'][j],ms=6,capsize=3,elinewidth=1.3,color=C[j],label=name)
 ax.set(xticks=x,xticklabels=labels,ylabel=ylabel,ylim=lim,xlim=(-.4,len(labels)-.6));ax.grid(axis='y',alpha=.13);ax.set_axisbelow(True)
''' +s[b:]
# Add density scales; use a single perceptually sequential colormap for both panels.
s=s.replace("ax.hexbin(held.human_c,held[col],gridsize=28,mincnt=1,cmap=['Blues','Oranges'][j]);", "hb=ax.hexbin(held.human_c,held[col],gridsize=28,mincnt=1,cmap='viridis',bins='log');fig.colorbar(hb,ax=ax,shrink=.65,label='图像数（对数色阶）');")
# S3: connected estimates, showing the model sequence instead of arbitrary bar area.
s=s.replace("ax.bar(x+(j-1)*.25,[lat[a][key] for a in ['Human','InternVL','Qwen3']],.23,label=lab,color=col)","ax.plot(x,[lat[a][key] for a in ['Human','InternVL','Qwen3']],'o--',ms=6,lw=.8,label=lab,color=col)")
a=s.index('pilot=pd.read_csv');b=s.index('tiny=json.loads',a)
s=s[:a]+'''z=pd.read_csv(S/'tables/v8_agreement_contrasts.csv');fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
labs=['InternVL 单图 − 物理','InternVL 匹配 − 物理','Qwen3 单图 − 物理','Qwen3 匹配 − 物理','Qwen3 − InternVL：匹配','Qwen3 − InternVL：单图']
for i,rw in z.iterrows(): ax.errorbar(rw.estimate,i,xerr=[[rw.estimate-rw.ci_low],[rw.ci_high-rw.estimate]],fmt='o',color=C[0] if i<2 else C[1],capsize=3)
ax.axvline(0,color=gray,ls=':',lw=1);ax.set(yticks=range(6),yticklabels=labs,xlabel='同一重采样下的相关差 Δρ');ax.invert_yaxis();save(fig,'S4')
''' +s[b:]
s=s.replace("a.bar(['控制＋小网络','加入 InternVL'],[z.loc[k,'r2'] for k in keys],color=[gray,C[0]])", "a.plot([0,1],[z.loc[k,'r2'] for k in keys],'o-',color=C[0],ms=7);a.set_xticks([0,1]);a.set_xticklabels(['控制＋小网络','加入 InternVL'])")
# Show all intervention matching trajectories rather than their means alone.
s=s.replace("a.plot([-1,0,1],vals,'o-',color=col,label=lab)","\n  base=mt[(mt.model==n)&(mt.direction=='baseline')].set_index('image_id').matching_c\n  for ident,v0 in base.items():\n   vm=mt[(mt.model==n)&(mt.image_id==ident)&(mt.direction==direction)].set_index('dose').matching_c\n   a.plot([-1,0,1],[vm.loc[-1],v0,vm.loc[1]],color=col,alpha=.22,lw=.6,marker='o',ms=2)\n  a.plot([-1,0,1],vals,'o-',color=col,label=lab,lw=2,ms=6)")
s=s.replace("ylabel='12 张图平均完整匹配值 c'","ylabel='完整匹配值 c（细线：单图）'").replace("ylim=(.09,.14)","ylim=(0,.155)")
# Replace cluster contrasts with all shape-level paired effects and corrected tests.
a=s.index(" z=pd.read_csv(T/'mechanism_cluster_contrasts.csv')");b=s.index("save(fig,'Fig6')",a)
s=s[:a]+''' z=pd.read_csv(S/'tables/v8_intervention_tests.csv');z=z[z.model==n];raw=pd.read_csv(S/'tables/v8_intervention_shape_values.csv');ax=axs[1,j]
 for k,rw in enumerate(z.itertuples()):
  v=raw[(raw.model==n)&(raw.control==rw.control)].contrast.to_numpy();xx=k+np.linspace(-.12,.12,len(v));ax.scatter(xx,v,s=10,color=C[j],alpha=.3,edgecolors='none');ax.errorbar(k,rw.estimate,yerr=[[rw.estimate-rw.ci_low],[rw.ci_high-rw.estimate]],fmt='o',ms=6,capsize=3,color='#202B36');ax.text(k,.69,r'$q = 10^{-5}$',ha='center',fontsize=8)
 ax.axhline(0,color=gray,lw=.6);ax.set(xticks=range(4),xticklabels=['亮度','随机 1','随机 2','随机 3'],ylabel='候选相对对照的双向效应差',xlabel='对照方向',ylim=(-.05,.76),xlim=(-.4,3.4),title='36 个形状：配对效应与 95% 区间');panel(ax,chr(99+j))
''' +s[b:]
# Fixed per-model overlay range across all four images, with explicit colour bars.
s=s.replace("for j,id in enumerate(ids):", "lo,hi=np.quantile(np.concatenate([np.load(root/'features'/f'{id}.npz')['bridge_tokens'].astype('float32')@dire['candidate'] for id in ids]),[.02,.98])\n for j,id in enumerate(ids):")
s=s.replace("ax.imshow(heat,cmap='magma',alpha=.55)","hm=ax.imshow(heat,cmap='magma',alpha=.55,vmin=lo,vmax=hi)")
s=s.replace("save(fig,'S9')", " fig.colorbar(hm,ax=axs[row,1:].tolist(),shrink=.55,label='投影（模型内共同色阶）')\nsave(fig,'S9')")
s=s.replace("rd=pd.read_csv(T/'mechanism_readout.csv');", "rd=pd.read_csv(T/'mechanism_readout.csv');rdci=pd.read_csv(S/'tables/v8_readout_ci.csv');")
s=s.replace("iv=pd.read_csv(T/'mechanism_intervention.csv')", "iv=pd.read_csv(S/'tables/v8_intervention_dose_ci.csv')")
a=s.index("fig,ax=plt.subplots(figsize=(8,4.6)");b=s.index("save(fig,'S3')",a)+len("save(fig,'S3')")
s=s[:a]+'''fig,axs=plt.subplots(1,3,figsize=(10,4.2),sharey=True,layout='constrained')
for j,n in enumerate(['Human','InternVL','Qwen3']):
 a=axs[j];vals=[lat[n][k] for k in ['r2_nuisance','r2_additive','r2_selected']];a.plot(range(3),vals,'o-',color=([gray]+C)[j],ms=6,lw=1.2)
 for k,v in enumerate(vals):a.annotate(f'{v:.3f}',(k,v),xytext=(0,7),textcoords='offset points',ha='center',fontsize=9)
 a.set(xticks=range(3),xticklabels=['物理/颜色','加性','低秩'],ylim=(0,.7),xlim=(-.35,2.35),title='人类' if n=='Human' else n);a.tick_params(axis='x',labelrotation=25,labelsize=9);panel(a,chr(97+j))
axs[0].set_ylabel('留出视角预测 R²');save(fig,'S3')''' +s[b:]
a=s.index("tiny=json.loads");b=s.index("mt=pd.read_csv",a)
s=s[:a]+'''tiny=json.loads((T/'tiny_incremental.json').read_text());df=pd.DataFrame(tiny['metrics']);fig,axs=plt.subplots(1,2,figsize=(9,4.6),layout='constrained')
for k,sp in enumerate(['shape','illumination']):
 z=df[df.split==sp].set_index('model');keys=['controls_tiny_cubic','controls_tiny_cubic_plus_internvl'];v=[z.loc[x,'r2'] for x in keys];axs[0].plot([0,1],v,'o-',color=C[k],label='留形状' if sp=='shape' else '留照明',ms=6)
 gain=next(x for x in tiny['gains'] if x['split']==sp and x['baseline']=='controls_tiny_cubic');lo,hi=gain['crossed_cluster_ci95_delta_r2'];value=gain['delta_r2'];axs[1].errorbar(value,k,xerr=[[value-lo],[hi-value]],fmt='o',capsize=3,color=C[k],ms=6)
axs[0].set(xticks=[0,1],xticklabels=['控制＋小网络','加入 InternVL'],ylabel='折外预测 R²',ylim=(.3,.65),xlim=(-.3,1.3));axs[0].legend(fontsize=9);axs[1].axvline(0,color=gray,ls=':');axs[1].set(yticks=[0,1],yticklabels=['留形状','留照明'],xlabel='增量 ΔR² 与 95% 区间',ylim=(-.6,1.6),xlim=(-.01,.18));panel(axs[0],'a');panel(axs[1],'b');save(fig,'S5')
''' +s[b:]
s=s.replace("a.plot(range(len(layers)),z.r2,'o-',color=C[j],label=n,ms=4)","a.plot(range(len(layers)),z.r2,'o-',color=C[j],label=n,ms=4);zz=rdci[(rdci.model==n)&(rdci.split==split)&(rdci.target==target)].set_index('layer').reindex(layers);a.fill_between(range(len(layers)),zz.ci_low,zz.ci_high,color=C[j],alpha=.12)")
(O/'code/plot_nmi_v8.py').write_text(s,encoding='utf-8')
print('Wrote portable figure script')
