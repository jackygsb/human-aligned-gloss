from pathlib import Path
import shutil,json,re
import pandas as pd
B=Path(__file__).resolve().parents[1];M=B/'mllm_material';V=M/'nmi_revision_v8/source';O=B/'output/NMI_English_Manuscript_v9';D=O/'supplement_inputs'
def cp(p,relative):
 dst=D/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
for p in (V/'figure_inputs/mllm_material/submission_synthesis_v1/tables').glob('*'):cp(p,Path('mllm_material/submission_synthesis_v1/tables')/p.name)
for rel in ['qwen3_full_v1/joint_model_predictions.csv','mechanisms_v1/stimuli.csv','mechanisms_v1/plan.json'] :cp(M/rel,Path('mllm_material')/rel)
for fac in ['shape','illumination']:
 for n in ['Human','InternVL','Qwen3']:
  for name in [f'matrices/physical_color_adjusted_{fac}_{n}_profiles_by_view.npy',f'matrices/physical_color_adjusted_{fac}_{n}_rdm.csv',f'embeddings/{n}_{fac}_adjusted_profile_PCA.csv']:
   p=V/'figure_inputs/mllm_material/behavioral_spaces_v1'/name;cp(p,Path('mllm_material/behavioral_spaces_v1')/name)
stim=pd.read_csv(M/'mechanisms_v1/stimuli.csv');plan=json.loads((M/'mechanisms_v1/plan.json').read_text())
for n in ['InternVL','Qwen3']:
 root=M/'mechanisms_v1'/n
 for name in ['directions.npz','candidate_scores.csv']:cp(root/name,Path('mllm_material/mechanisms_v1')/n/name)
 scores=pd.read_csv(root/'candidate_scores.csv');z=scores[scores.image_id.isin(plan['intervention_images'])].sort_values('projection')
 for iid in z.image_id.iloc[[0,1,-2,-1]]:
  cp(root/'features'/f'{iid}.npz',Path('mllm_material/mechanisms_v1')/n/'features'/f'{iid}.npz');im=stim[stim.image_id==iid].iloc[0].image_512_path;cp(B/im,Path(im))
src=(V/'code/plot_nmi_v8.py').read_text(encoding='utf-8');tail=src[src.index('# Supplement:'):];tail=tail[:tail.index('if Path(__file__)')]
translations={'平均匹配值 c':'Mean matched gloss c','图像数':'Images','人类':'Human','人–InternVL':'Human–InternVL','人–Qwen3':'Human–Qwen3','两模型':'InternVL–Qwen3','RSA 相关':'Spearman RSA','物理与颜色调整':'Physical/color adjusted','原始':'Raw','仅交互':'Interaction only','：形状':': shape','：照明':': illumination','物理/颜色':'Physical/color','加性':'Additive','低秩':'Low rank','留出视角预测 R²':'Held-out-view prediction R²','InternVL 单图 − 物理':'InternVL rating − physical','InternVL 匹配 − 物理':'InternVL matching − physical','Qwen3 单图 − 物理':'Qwen3 rating − physical','Qwen3 匹配 − 物理':'Qwen3 matching − physical','Qwen3 − InternVL：匹配':'Qwen3 − InternVL: matching','Qwen3 − InternVL：单图':'Qwen3 − InternVL: rating','同一重采样下的相关差 Δρ':'Paired difference in Spearman correlation','留形状':'Shape holdout','留照明':'Lighting holdout','控制＋小网络':'Controls + tiny CNN','加入 InternVL':'+ InternVL matching','折外预测 R²':'Out-of-fold prediction R²','增量 ΔR² 与 95% 区间':'Gain in R² with 95% interval','参考定义方向':'Reference-defined','随机方向 1':'Random direction 1','剂量':'Dose','完整匹配值 c（细线：单图）':'Complete match c (thin lines: images)','：调整后轮廓':': adjusted profiles','照明':'Illumination','形状':'Shape','相对行为不相似性':'Relative behavioral dissimilarity','条件索引':'Condition ID','响应 / 轮廓标准差':'Response / profile s.d.','距离 / 平均距离':'Distance / mean distance','主成分 1':'Principal component 1','主成分 2':'Principal component 2','：参考序列':': references','参考 c':'Reference c','候选方向投影':'Candidate projection','候选响应投影':'Candidate projection','投影（模型内共同色阶）':'Projection (shared within model)','：':': '}
for a,b in sorted(translations.items(),key=lambda x:-len(x[0])):tail=tail.replace(a,b)
tail=tail.replace("pd.read_csv(M/f'behavioral_spaces_v1/matrices/physical_color_adjusted_{fac}_{n}_rdm.csv').to_numpy()","np.loadtxt(M/f'behavioral_spaces_v1/matrices/physical_color_adjusted_{fac}_{n}_rdm.csv',delimiter=',')")
header='''from pathlib import Path
import json,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
S=Path(__file__).resolve().parents[1];B=S/'supplement_inputs';M=B/'mllm_material';T=M/'submission_synthesis_v1/tables'
plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
C=['#7656A5','#148B80'];names=['InternVL','Qwen3'];gray='#697583'
def save(fig,key):
 for ext in ['pdf','png','svg']:fig.savefig(S/f'figures/{key}.{ext}',dpi=220,bbox_inches='tight')
 plt.close(fig)
def panel(ax,s):ax.text(-.12,1.08,s,transform=ax.transAxes,weight='bold',fontsize=13)
joint=pd.read_csv(M/'qwen3_full_v1/joint_model_predictions.csv');stim=pd.read_csv(M/'mechanisms_v1/stimuli.csv');r=pd.read_csv(T/'rsa.csv')
'''
(O/'code/draw_supplement_english.py').write_text(header+tail,encoding='utf-8');print('Prepared portable English supplementary plotting code.')
