"""Publication-readable presentation and archive for behavioral RSA; no new fitting choices."""
from pathlib import Path
import json,hashlib,shutil,zipfile
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1];O=B/'mllm_material/behavioral_spaces_v1'
def main():
 shutil.copy2(__file__,O/'code'/Path(__file__).name)
 rs=pd.read_csv(O/'tables/rsa_results.csv');summary=json.loads((O/'prediction_summary.json').read_text());agents=['Human','InternVL','Qwen3']
 plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
 fig,axs=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
 for ax,factor in zip(axs,['shape','illumination']):
  for i,(space,label,color) in enumerate([('raw','Raw','#97b5c9'),('physical_color_adjusted','Physical/color adjusted','#287ba3'),('interaction_only','Interaction only','#d88736')]):
   z=rs[(rs.factor==factor)&(rs.space==space)];x=np.arange(3)+(i-1)*.25;val=z.rsa.to_numpy();err=np.vstack([val-z.ci_low.to_numpy(),z.ci_high.to_numpy()-val]);ax.bar(x,val,.23,color=color,label=label);ax.errorbar(x,val,yerr=err,fmt='none',ecolor='#444',capsize=3)
  ax.set_xticks(range(3),['Human–InternVL','Human–Qwen3','InternVL–Qwen3'],rotation=15);ax.set(title=factor.title()+' response geometry',ylabel='RSA (Spearman)',ylim=(-.2,.9));ax.axhline(0,color='gray',lw=.7);ax.legend(fontsize=8)
 for ext in ['png','svg','pdf']:fig.savefig(O/f'figures/Fig4_RSA_comparison.{ext}',dpi=180)
 plt.close(fig)
 fig,axs=plt.subplots(2,3,figsize=(12,8),constrained_layout=True);spectra=[]
 for i,factor in enumerate(['shape','illumination']):
  for j,agent in enumerate(agents):
   x=np.load(O/f'matrices/physical_color_adjusted_{factor}_{agent}_profiles_by_view.npy').mean(0);xc=x-x.mean(0);u,sv,vt=np.linalg.svd(xc,full_matrices=False);coords=u*sv;var=sv**2/np.sum(sv**2)
   pd.DataFrame(coords,columns=[f'dim{k+1}' for k in range(36)]).assign(condition_id=np.arange(1,37)).to_csv(O/f'embeddings/{agent}_{factor}_adjusted_profile_PCA.csv',index=False)
   for k in range(36):spectra.append(dict(agent=agent,factor=factor,dimension=k+1,variance_fraction=var[k],cumulative=var[:k+1].sum()))
   ax=axs[i,j];ax.scatter(coords[:,0],coords[:,1],c=np.arange(36),cmap='turbo',s=24)
   for k in range(36):ax.annotate(str(k+1),coords[k,:2],fontsize=6)
   ax.set(title=f'{agent}: {factor}',xlabel=f'PC1 ({var[0]:.1%})',ylabel=f'PC2 ({var[1]:.1%})')
 fig.suptitle('Adjusted response-profile spaces: independent descriptive PCA, no coordinate alignment')
 for ext in ['png','svg','pdf']:fig.savefig(O/f'figures/Fig5_adjusted_profile_spaces.{ext}',dpi=180)
 plt.close(fig);pd.DataFrame(spectra).to_csv(O/'tables/adjusted_profile_spectrum.csv',index=False)
 text='''# 行为潜在空间与 RSA：结果与论文定位\n\n已完成三方全部 3,888 张图的分析。使用现有非对称参考匹配均值，不新增采集。每种形状以它对 36 种照明的响应模式表征；每种照明以它对 36 种形状的响应模式表征。三个视角提供迁移检查。\n\n## 已得到的结构证据\n\n控制物理光泽和颜色后，形状空间 RSA：人类–InternVL = 0.327，人类–Qwen3 = 0.475，InternVL–Qwen3 = 0.450。照明空间 RSA：分别为 0.537、0.487、0.653。六项在 18 项比较的 BH 校正后均 q≤0.0009。每个空间有 36 个条件、630 个距离，但统计置换单位是 36 个条件标签，不把 630 个距离当独立样本。\n\n这是响应模式的距离结构比较，不再只是逐图评分相关。不同视角之间，调整后形状 RSA 平均为人类–InternVL 0.274、人类–Qwen3 0.428；照明分别为 0.145、0.160，低于三视角平均后的结果。不能把三视角平均的 RSA 直接称为强跨视角不变性。\n\n## 潜在维度检验告诉了什么\n\n将判断分解为物理/颜色项、形状和照明平均效应、低秩交互项。维度与收缩强度在内层验证选择，第三个视角用于外层评估。\n\n人类：加性模型 R²=0.5441；增加经选择的潜在交互后 0.5461，误差减少 0.45%。InternVL：0.4400→0.4398，误差变化约 −0.03%。Qwen3：0.5831→0.5856，误差减少 0.60%。三者误差改善的条件聚类 95% 区间均跨零。\n\n人类三个外层折都选择 1 维；InternVL 为 0、2、0；Qwen3 为 2、2、1。但这不是“人类只有一维、Qwen 有两维”的证据，因为额外维度未显示稳定预测收益。维度数只针对本方法中的剩余形状×照明交互，不是整个知觉系统的维度数。\n\n扣除形状、照明平均效应后，交互 RSA：形状人类–InternVL 0.237（q=0.0206），人类–Qwen3 0.169（q=0.1169）；照明分别为 0.194（q=0.0152）、0.174（q=0.0502）。这比完整响应结构弱，且不是两个模型都通过所有检验。\n\n## 可以写进论文的结论\n\n两种多模态模型与人类具有共同的、受形状和照明组织的光泽行为表征几何；当前最明确的共同结构包含形状与照明平均效应。现有数据和本次低秩分析尚不足以证明，两模型都拥有额外稳定的多维交互表征。\n\n建议把调整后的条件 RSA 作为已有行为证据的结构性扩展，把维度选择和交互检验放入补充材料并在正文如实限定结论。二维潜在空间图是描述性投影，不用于凭视觉相似断言空间一致。尚未给维度赋予“高光敏感”“照明补偿”等心理名称；需要独立特征或任务约束才能做这种解释。\n\n## 文件说明\n\n- Fig1：交互响应矩阵与 RDM（形状/照明各一张）。\n- Fig2：潜在交互秩与留出视角预测。\n- Fig3：交互空间二维描述投影（InternVL 选择秩可能为零，不作验证通过的空间）。\n- Fig4：原始、物理颜色调整、交互三层 RSA 及条件重采样区间。\n- Fig5：调整后完整响应模式的独立 PCA 空间（包含平均效应），各坐标轴标注解释方差。\n- matrices：所有逐视角响应模式、完整 RDM、置换零分布。\n- embeddings：三方坐标和维度极端条件编号；坐标不强制对齐。\n- tables：所有图的来源数据、逐图外层预测、内层所有候选结果、维度选择、RSA、跨视角稳定性。\n- inputs：冻结的三方输入与原有开发/试验标记。\n- code：两个完整脚本；先运行分析脚本，再运行本整理脚本。\n\n这是已有匹配任务的条件行为表征分析，不是原版 SPoSE、不是神经网络内部表征、也不是独立采集的通用材料相似性空间。没有参与者层面的独立评分用于本次拟合，不能报告人群噪声上限。本次使用全部已有数据，属于探索性分析。\n'''
 (O/'结果解读.md').write_text(text,encoding='utf-8')
 p=O/'report.html';h=p.read_text(encoding='utf-8');insert='<section style="background:#edf5fa;padding:20px"><h2>主要发现</h2><p><b>三方共享形状与照明的行为响应几何。</b>控制物理与颜色后，人类–InternVL 的形状/照明 RSA 为 0.327/0.537，人类–Qwen3 为 0.475/0.487；均通过多重比较校正。</p><p><b>更强的多维交互结论尚未成立。</b>增加经内层选择的交互维度，留出预测改善约 −0.03% 到 0.60%，区间均跨零。空间相似主要包含形状与照明平均效应；不能用二维图代替维度验证。</p><p><a href="结果解读.md">中文结果解读与论文定位</a></p><img src="figures/Fig4_RSA_comparison.png"><h2>完整响应模式的潜在空间</h2><p>下图是物理/颜色调整后响应模式的独立 PCA；包含条件平均效应。每个点是一种形状或照明。坐标未向人类空间拟合；RSA 比较完整距离矩阵，不比较二维图的朝向。</p><img src="figures/Fig5_adjusted_profile_spaces.png"></section>'
 h=h.replace('<h2>1 留出视角预测',insert+'<h2>1 留出视角预测');p.write_text(h,encoding='utf-8')
 # Explicitly supersede hidden-layer route without deleting historical plans.
 (B/'mllm_material/protocol/行为表征结构分析_v2_已执行.md').write_text('''# 行为表征结构分析 v2（已执行）\n\n按用户最新澄清，仅分析行为表征。原《表征结构研究路线_v1.md》中的路线 B（隐藏层提取）不再作为本阶段执行方案。\n\n执行日期：2026-09-29。完整设置、输入冻结哈希与结果见 ../behavioral_spaces_v1/analysis_plan.json、Methods.md、结果解读.md。\n\n方法为独立条件响应空间 + RSA + 嵌套留视角低秩交互预测，借鉴 SPoSE 的行为结构思想，没有使用原版 SPoSE 的三元相似性损失。已保存全部候选和未通过的检验，不把可视化维度当作已验证的心理维度。\n''',encoding='utf-8')
 manifest=[]
 for p in O.rglob('*'):
  if p.is_file() and p.suffix!='.zip' and p.name!='artifact_manifest.json':manifest.append({'path':str(p.relative_to(O)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 (O/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 with zipfile.ZipFile(O/'behavioral_spaces_reproducibility.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in O.rglob('*'):
   if p.is_file() and p.suffix!='.zip':z.write(p,p.relative_to(O))
 print('Finalized',len(manifest),'files; zip bytes',(O/'behavioral_spaces_reproducibility.zip').stat().st_size)
if __name__=='__main__':main()
