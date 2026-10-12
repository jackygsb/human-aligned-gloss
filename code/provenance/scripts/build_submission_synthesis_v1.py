"""Assemble existing results into a bounded submission narrative; no new experiments."""
from pathlib import Path
import json,shutil,hashlib,zipfile,html
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1];M=B/'mllm_material';O=M/'submission_synthesis_v1'
def main():
 for sub in ['figures','supplement','tables','code']: (O/sub).mkdir(parents=True,exist_ok=True)
 shutil.copy2(__file__,O/'code'/Path(__file__).name)
 sources={
 'internvl_human_agreement':M/'paper_figures_v1/tables/Table2_human_agreement.csv',
 'internvl_cross_task':M/'paper_figures_v1/tables/Table3_continuous_cross_task.csv',
 'internvl_pair_cross_task':M/'paper_figures_v1/tables/Table4_pair_cross_task.csv',
 'internvl_condition_effects':M/'paper_figures_v1/tables/Table6_shared_condition_effects.csv',
 'qwen_summary':M/'qwen3_full_v1/summary.json',
 'qwen_extra':M/'qwen3_full_v1/matching_extra_statistics.json',
 'qwen_pairs':M/'qwen3_full_v1/pair_metrics.csv',
 'qwen_cross_task':M/'qwen3_full_v1/cross_task_metrics.csv',
 'rsa':M/'behavioral_spaces_v1/tables/rsa_results.csv',
 'latent_prediction':M/'behavioral_spaces_v1/prediction_summary.json',
 'mechanism_readout':M/'mechanisms_v1/tables/layer_readouts.csv',
 'mechanism_intervention':M/'mechanisms_v1/tables/intervention_summary.csv',
 'mechanism_cluster_contrasts':M/'mechanisms_v1/tables/shape_cluster_intervention_contrasts.csv',
 'mechanism_matches':M/'mechanisms_v1/tables/matching_interventions.csv',
 'mechanism_spatial':M/'mechanisms_v1/tables/spatial_feature_summary.csv',
 'pilot_and_tiny':M/'paper_figures_v1/supplement/S1_other_models_and_tiny_same12.csv',
 'tiny_incremental':M/'matching_full_v1/incremental_tiny/summary.json'}
 source_manifest=[]
 for name,p in sources.items():
  dest=O/'tables'/(name+p.suffix);shutil.copy2(p,dest);source_manifest.append(dict(name=name,source=str(p),copy=str(dest.relative_to(O)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 q=json.loads(sources['qwen_summary'].read_text());rsa=pd.read_csv(sources['rsa']);iv=pd.read_csv(sources['mechanism_intervention']);rd=pd.read_csv(sources['mechanism_readout']);ci=pd.read_csv(sources['internvl_cross_task'])
 a=pd.read_csv(sources['internvl_human_agreement']);a=a[a.cohort=='matching_heldout'].set_index('model_output');qa=pd.DataFrame(q['human_agreement']);qa=qa[qa.cohort=='matching_heldout'].set_index('task')
 evidence=[
 ['E1','跨任务人类一致性','单图评分 Spearman',3849,a.loc['single_score','human_spearman'],qa.loc['single_score','human_spearman'],'三任务没有人类任务标签微调；开发与先导过程有记录'],
 ['E2','跨任务人类一致性','参考匹配 Spearman',3849,a.loc['model_c','human_spearman'],qa.loc['model_c','human_spearman'],'参考序列为重建刺激；排序一致不等于绝对校准'],
 ['E3','人类式物理偏离','物理/人类冲突图对的跟随人类比例',1000,.688,.7525,'参照由人类平均评分推导；不是独立人类二选一数据'],
 ['E4','共同条件效应','形状效应 Spearman',36,.50399,.6712998713,'控制物理/颜色/视角及另一条件；非因果操纵效应'],
 ['E5','共同条件效应','照明效应 Spearman',36,.809009,.7994851995,'全部3888图条件分析；36个条件是相关的单位'],
 ['E6','跨任务一致','匹配顺序预测双图选择比例',2000,.7965,.85075,'两种顺序平均；平局半权；固定图对设计'],
 ['E7','行为响应几何','调整后形状 RSA',36,.3273726469,.4749744057,'包含形状平均效应；额外交互维度无稳定增益'],
 ['E8','行为响应几何','调整后照明 RSA',36,.5371907379,.4870399438,'条件标签置换检验；非独立人类多维相似性判断'],
 ['E9','内部信息','最后语言层读出人类 Spearman：留形状',3888,.751906,.813444,'监督读出器；基础模型冻结；回答前响应'],
 ['E10','内部信息','最后语言层读出人类 Spearman：留照明',3888,.772769,.818724,'不能作为零监督回答性能或人脑机制证据'],
 ['E11','功能干预','候选方向 +1 剂量概率变化',72,.0991927873,.1024856138,'36形状但仅照明1；两顺序；A/B/SAME条件概率'],
 ['E12','功能干预','候选方向 -1 剂量概率变化',72,-.0937367281,-.0986994910,'有亮度及3个等范数随机方向对照；可能离开自然激活分布']]
 ev=pd.DataFrame(evidence,columns=['id','claim','metric','n_unit','InternVL','Qwen3','scope']);ev.to_csv(O/'tables/Table1_evidence_chain.csv',index=False)
 plt.rcParams.update({'font.size':11,'svg.fonttype':'none'})
 def output(fig,name):
  for ext in ['png','svg','pdf']:fig.savefig(O/f'figures/{name}.{ext}',dpi=180,bbox_inches='tight')
  plt.close(fig)
 def cpfigure(src,dest,folder='figures'):
  for ext in ['png','svg','pdf']:
   p=Path(str(src)+'.'+ext)
   if p.exists():shutil.copy2(p,O/folder/(dest+'.'+ext))
 cpfigure(M/'paper_figures_v1/figures/Fig1_tasks_and_acquisition','Fig1_paradigm_existing_InternVL_panel')
 # Replace old single-model overview with an explicitly two-model submission overview.
 from PIL import Image
 stimuli=pd.read_csv(M/'mechanisms_v1/stimuli.csv');example=stimuli.iloc[0]
 fig=plt.figure(figsize=(13,8),constrained_layout=True);gs=fig.add_gridspec(3,4,height_ratios=[1.2,1,.6])
 ims=[(B/example.image_512_path,'Example test image: '+example.image_id)]+[(M/f'reference_reconstruction_v3/output/reference_{k:02d}.png',f'Reconstructed reference {k}/52') for k in [1,27,52]]
 for i,(p,title) in enumerate(ims):
  ax=fig.add_subplot(gs[0,i]);ax.imshow(Image.open(p));ax.set_title(title,fontsize=10);ax.axis('off')
 boxes=[('Single-image rating','Rate visible gloss, 0–100\n3,888 images / model'),('Pairwise comparison','Which object looks glossier?\n2,000 pairs × 2 orders / model'),('Reference matching','Compare test with reference\n52 levels; at most 7 decisions\n3,888 images × 2 starts / model'),('Internal mechanism','3,940 contexts / model\n72 images × 2 orders × 21 settings\n12 images × 5 full matches')]
 for i,(title,body) in enumerate(boxes):
  ax=fig.add_subplot(gs[1,i]);ax.axis('off');ax.text(.5,.85,title,ha='center',va='center',weight='bold',fontsize=12);ax.text(.5,.48,body,ha='center',va='center',fontsize=10,bbox=dict(boxstyle='round,pad=.8',facecolor='#edf4f8',edgecolor='#bed0df'))
 ax=fig.add_subplot(gs[2,:]);ax.axis('off');ax.text(.5,.8,'Frozen InternVL3.5-4B + Qwen3-VL-4B-Instruct  |  Same stimuli, core tasks and reference sequence',ha='center',weight='bold',fontsize=12);ax.text(.5,.35,'Existing human matching data: 36 shapes × 36 illuminations × 3 views = 3,888 images\nBehavior uses no target-label fine-tuning; supervised readout is a separate analysis. Interventions use illumination 1.',ha='center',fontsize=10)
 output(fig,'Fig1_paradigm_existing_InternVL_panel')
 fig,ax=plt.subplots(1,2,figsize=(11,4),constrained_layout=True);x=np.arange(2)
 for off,name,color,vals in [(-.18,'InternVL','#297da5',[ev.iloc[0].InternVL,ev.iloc[1].InternVL]),(.18,'Qwen3','#d98238',[ev.iloc[0].Qwen3,ev.iloc[1].Qwen3])]:ax[0].bar(x+off,vals,.34,label=name,color=color)
 ax[0].set(xticks=x,xticklabels=['Single-image rating','Reference matching'],ylabel='Human agreement (Spearman)',ylim=(0,1));ax[0].legend()
 for off,name,color,vals in [(-.18,'InternVL','#297da5',[.719,.688]),(.18,'Qwen3','#d98238',[.706,.7525])]:ax[1].bar(x+off,vals,.34,label=name,color=color)
 ax[1].axhline(.5,color='gray',ls=':');ax[1].set(xticks=x,xticklabels=['Random pairs','Human–physical conflict'],ylabel='Agreement with human-derived order',ylim=(0,1));fig.suptitle('Two models reproduce human gloss judgments');output(fig,'Fig2_two_model_human_alignment')
 fig,ax=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
 for off,name,color,vals in [(-.18,'InternVL','#297da5',[.50399,.809009]),(.18,'Qwen3','#d98238',[.6712998713,.7994851995])]:ax[0].bar(x+off,vals,.34,label=name,color=color)
 ax[0].set(xticks=x,xticklabels=['Shape','Illumination'],ylabel='Adjusted human–model effect correlation',ylim=(0,1));ax[0].legend()
 for off,name,color in [(-.18,'InternVL','#297da5'),(.18,'Qwen3','#d98238')]:
  z=rsa[(rsa.space=='physical_color_adjusted')&(rsa.agent1=='Human')&(rsa.agent2==name)].set_index('factor').loc[['shape','illumination']];val=z.rsa.to_numpy();ax[1].bar(x+off,val,.34,label=name,color=color);ax[1].errorbar(x+off,val,yerr=[val-z.ci_low.to_numpy(),z.ci_high.to_numpy()-val],fmt='none',color='black',capsize=3)
 ax[1].set(xticks=x,xticklabels=['Shape profiles','Illumination profiles'],ylabel='Human–model RSA',ylim=(0,1));fig.suptitle('Shared condition effects and response geometry');output(fig,'Fig3_condition_structure')
 fig,ax=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
 for off,name,color,vals in [(-.18,'InternVL','#297da5',[.652251827,.653815392]),(.18,'Qwen3','#d98238',[.775781586,.777807883])]:ax[0].bar(x+off,vals,.34,label=name,color=color)
 ax[0].set(xticks=x,xticklabels=['Held-out shape groups','Held-out illumination groups'],ylabel='Single rating → matching, CV R²',ylim=(0,1));ax[0].legend()
 for off,name,color,vals in [(-.18,'InternVL','#297da5',[.767,.7965]),(.18,'Qwen3','#d98238',[.8035,.85075])]:ax[1].bar(x+off,vals,.34,label=name,color=color)
 ax[1].set(xticks=x,xticklabels=['Single-rating order','Reference-match order'],ylabel='Predicting observed pair choices',ylim=(0,1));fig.suptitle('A consistent judgment across three tasks');output(fig,'Fig4_cross_task_structure')
 cpfigure(M/'mechanisms_v1/figures/Mechanism1_layer_information','Fig5_internal_readout')
 cpfigure(M/'mechanisms_v1/figures/Mechanism3_internal_intervention','Fig6_functional_intervention')
 supp=[('S1_pilot',M/'paper_figures_v1/supplement/FigS1_cross_model_same12'),('S2_tiny_incremental',M/'paper_figures_v1/supplement/FigS2_incremental_prediction'),('S3_RSA_all',M/'behavioral_spaces_v1/figures/Fig4_RSA_comparison'),('S4_latent_validation',M/'behavioral_spaces_v1/figures/Fig2_held_view_prediction'),('S5_candidate_spatial',M/'mechanisms_v1/figures/Mechanism2_reference_and_spatial_response'),('S6_intervention_matching',M/'mechanisms_v1/figures/MechanismS1_full_matching')]
 for name,src in supp:cpfigure(src,name,'supplement')
 figureplan=[['1','实验设计','三任务、52档参考序列、共享刺激、冻结模型','双模型总览；提示词与控制器属于方法贡献'],['2','人类式判断','两模型单图、匹配、随机/冲突双图','主表n=3849；Qwen排除12先导后的n=3837放补充；冲突图对是有意抽样'],['3','共同知觉组织','条件效应＋调整后RSA','主效应与整体结构是结论；不能表述为已验证高维交互空间'],['4','跨任务一致性','任务间留出预测、对实际双图选择的预测','跨任务拟合没有人类标签；现有双图分析不等于全新图像上的泛化'],['5','内部信息','视觉—映射—回答前语言层读出','读出器有监督；不把层深曲线当作纯时间过程，各层池化不同'],['6','功能干预','候选、亮度、随机方向的剂量曲线','固定照明1的功能证据；不是纯光泽神经元或全照明机制；完整匹配放补充']]
 pd.DataFrame(figureplan,columns=['figure','role','contents','caption_boundary']).to_csv(O/'tables/Table2_main_figure_plan.csv',index=False)
 inventory=[['InternVL','单图评分',3888,'全量'],['Qwen3','单图评分',3888,'全量'],['InternVL','双图回答',4000,'2000图对×2顺序'],['Qwen3','双图回答',4000,'2000图对×2顺序'],['InternVL','完整参考匹配',7776,'3888图×2起始'],['Qwen3','完整参考匹配',7776,'3888图×2起始；包含24次先导复用'],['两模型','主特征记录',7880,'各3888测试+52参考'],['两模型','内部干预',6048,'各72图×2顺序×21条件；照明1'],['两模型','干预完整匹配',120,'各12图×5条件'],['GPT','先导参考匹配',24,'12图×2；不构成全量第三模型验证']]
 pd.DataFrame(inventory,columns=['model','unit','count','scope']).to_csv(O/'tables/Table3_experiment_inventory.csv',index=False)
 abstract='''Surface gloss depends on both material reflectance and the interaction of shape and illumination. We examined whether pretrained multimodal language models exhibit human-like gloss judgments without fine-tuning on the target behavioral labels. InternVL and Qwen3 were evaluated using single-image ratings, pairwise comparisons and asymmetric reference matching on a shared set of 3,888 stimuli. Single-image ratings correlated with human judgments at 0.696 and 0.727, while reference matches correlated at 0.587 and 0.680. When human and physical gloss orderings conflicted, the models followed the human-derived ordering in 68.8% and 75.3% of comparisons. Shared shape and illumination effects, response-profile geometry and cross-task prediction further supported structured behavioral alignment. Supervised readouts of frozen, pre-answer representations predicted held-out human judgments. A visual direction defined solely from a reconstructed reference sequence also influenced model decisions: adding or subtracting this direction shifted gloss-choice probabilities by approximately ten percentage points, with larger bidirectional effects than tested brightness and random-direction controls. These interventions covered 36 shapes under one illumination. Together, the findings link human-like gloss behavior to manipulable visual representations in pretrained multimodal models, while leaving the generality and visual specificity of the candidate mechanism open.'''
 (O/'abstract_draft.txt').write_text(abstract,encoding='utf-8')
 captions={
 'Fig1_paradigm_existing_InternVL_panel':'双模型实验范式与复现流程。示例测试图为编号排序第一张，不按结果挑选；参考展示1、27、52档。人类数据来自原论文，参考序列为52档重建图，不是作者原始PNG。',
 'Fig2_two_model_human_alignment':'共享评估图上的行为一致性。连续评分采用匹配评估集3849图；人类顺序来自既有人类均值，不是新增人类二选一实验。图对分随机与物理/人类冲突两个预设层。柱为点估计，未新增置信区间。',
 'Fig3_condition_structure':'条件平均效应及行为响应几何。效应控制物理/颜色/视角及另一条件；RSA采用36个条件响应模式的距离矩阵，物理/颜色调整后仍包括条件平均效应。右图区间为条件节点重采样，非人群区间；左图点估计来源于已有条件分析。',
 'Fig4_cross_task_structure':'单图到参考匹配的分组留出预测，以及由两种标量输出排序预测现有双图回答。跨任务映射不使用人类标签。右图同一刺激可重复出现，不是新图像泛化实验。',
 'Fig5_internal_readout':'冻结网络内部信息的监督读出。每层独立使用固定无标签随机投影及嵌套选择正则的岭回归；人类标签仅用于读出器训练。视觉层取均值，语言层取回答前末token，池化不同，不能将曲线等同于单一加工时间过程。',
 'Fig6_functional_intervention':'候选视觉方向对判断的功能作用。候选由参考序列定义；测试图片保持不变，干预仅施加于测试图片的视觉映射token。72图两顺序、36形状、仅照明1。SAME按半权，概率在三候选内归一化；区间为图像重采样，形状聚类对照另表报告。此干预可离开自然激活分布，不证明候选方向是纯光泽或唯一机制。'}
 save={'preferred_title':'Human-like gloss judgments in pretrained multimodal language models','stronger_optional_title':'Human-like gloss judgments and their visual representations in pretrained multimodal language models','central_claim':'未经目标人类光泽标签微调的多模态模型，在多任务中表现出人类式光泽判断；这种行为有共同条件结构，并与可读出、可干预的视觉内部成分相联系。','scope_frozen':'gloss, two pretrained visual-language systems, target-label-free behavioral inference, supervised readout separated, functional intervention under one illumination','not_claimed':['general material understanding','human neural mechanism equivalence','scaling-law emergence','training-data contamination ruled out','fully independent language-model families','validated common multidimensional interaction space','state-of-the-art superiority to author tiny networks'],'date':'2026-09-30'}
 (O/'claim_scope.json').write_text(json.dumps(save,ensure_ascii=False,indent=2),encoding='utf-8');(O/'source_manifest.json').write_text(json.dumps(source_manifest,ensure_ascii=False,indent=2),encoding='utf-8')
 sections=[]
 def section(title,body):sections.append('<section><h2>'+title+'</h2>'+body+'</section>')
 section('结论：可以收束为投稿稿件，不能把录用当作已有结果','<p><b>现有工作量已足以组织一篇完整研究稿件，不必先完成更多数据集或模型。</b>最有竞争力的主线是行为→共同条件结构→跨任务稳定性→内部信息→功能干预。当前建议进入写作与投稿整理阶段，冻结新增实验范围。</p><p>对 Nature 子刊，判断是“有理由冲击，尚不能认定已达到录用门槛”。主要不确定性是：相对已有工作的新概念贡献是否足够大，以及有限机制范围能否支撑编辑期待的普遍意义。更多模型回答次数本身不能解决这个问题。</p>')
 section('标题与一句话贡献','<h3>'+save['preferred_title']+'</h3><p>'+save['central_claim']+'</p><p>建议以 human-like gloss judgments 为主标题用语。若保留 emerge naturally，必须把它限定为“预训练模型无需目标行为标签微调即可表现出该能力”，不是参数规模临界涌现或训练过程发现。“材料感知”与“人脑机制”等更宽结论不纳入当前稿件。</p>')
 section('完整逻辑故事','''<ol><li><b>科学问题：</b>光泽不是物理反射参数的简单读数。通用多模态模型是否也能作出受形状和照明影响的人类式判断？</li><li><b>行为发现：</b>两模型在同一套刺激上的单图、双图与参考匹配均显示人类一致性。人类与物理排序冲突时，仍更多跟随人类顺序。</li><li><b>组织规律：</b>这种一致性包含共同的形状、照明效应；调整后的条件响应几何也相似。强调共同组织结构，不把弱交互结果升级为高维表征发现。</li><li><b>任务稳定性：</b>不同任务的输出能相互预测，说明发现不限于某一种回答尺度；该证据本身并不证明提示词完全不敏感。</li><li><b>内部证据：</b>回答前的冻结内部响应可通过监督读出预测留出人类判断。监督分析与原零任务标签微调行为明确分开。</li><li><b>功能连接：</b>从参考序列独立定义的视觉方向，在不改变图片时能定向改变选择和匹配，作用超过所测亮度/随机对照。结论限定在所测照明和干预路径。</li><li><b>意义：</b>通用预训练视觉语言系统包含可支持人类式光泽行为的视觉信息；行为与内部功能成分建立了联系，但未完整解释光泽算法。</li></ol>''')
 section('核心数字与每项结论的边界',ev.round(4).to_html(index=False))
 section('工作量：不同单位分别计数',pd.DataFrame(inventory,columns=['模型','项目','数量','范围']).to_html(index=False)+'<p>不能将匹配的每一步比较、完成的匹配次数和独立图像数相加后称为独立样本量。共享刺激也意味着两个模型不是新增的人类观察者。</p>')
 section('六张正文图：已有图稿与来源',pd.DataFrame(figureplan,columns=['图','角色','内容','说明']).to_html(index=False)+''.join('<h3>'+k+'</h3><img loading="lazy" src="figures/'+k+'.png"><p>'+v+'</p>' for k,v in captions.items()))
 section('补充材料安排','''<ol><li><b>先导跨模型：</b>GPT12图 rho=0.867；Qwen先导12图0.694；仅作先导，不宣称完成第三模型全量复现。记录实际模型接口与设置。</li><li><b>小网络来源与对照：</b>原作者发布预测、原核、独立重实现分开。InternVL加入独立tiny与物理/颜色控制后，留形状/照明的误差减少约17.17%/18.19%；不能写成“击败作者原网络”。</li><li><b>评分与匹配校准：</b>参考图重建、绝对值偏移、上限饱和、两次匹配一致性；上限约InternVL27.6%、Qwen28.9%。</li><li><b>完整RSA与维度验证：</b>所有调整层次、交互RSA、维度选择和无稳定增益结果。正文结论必须同时承认额外交互维度未得到稳定支持。</li><li><b>机制细节：</b>参考方向、完整空间图、图像特征/小网络核关联、随机/亮度方向、双顺序结果、概率质量、完整匹配、36形状聚类区间。</li><li><b>全部探索记录：</b>旧Qwen2.5、三图任务与提示词/模型选择历史保留，标明哪些用于探索或方案选择，不纳入支持泛化的确认性证据。</li><li><b>可复现性：</b>模型版本、提示词、图像绑定、随机种子、数据划分、全部候选分析设置、原始回答、权重与数据哈希，以及代码与数据公开位置。</li></ol>'''+''.join('<h3>'+name+'</h3><img loading="lazy" src="supplement/'+name+'.png">' for name,_ in supp))
 section('当前不写进摘要的强主张','<p>不写“理解所有材料属性”“发现与人脑相同机制”“完整恢复多维材料空间”“显著优于所有小网络”“三个独立模型家族全量复现”。提示词、匹配控制器和采集管线是重要方法贡献；没有系统比较时不宣称最优提示词或全新学习算法。</p><p>研究经历多轮探索，必须披露，不将现有全数据分析改称预注册确认实验。正文可以聚焦最能支撑问题的结果，结论相关的失败/限制则在正文简述并在补充材料完整呈现。</p>')
 section('Nature 子刊定位：这是投稿判断，不是录用保证','''<p><b>首投建议：Nature Machine Intelligence。</b>最匹配现有“多模态智能行为＋可解释内部作用”的问题。应以新科学发现为中心，而不是把三个问答模板当作主要创新。其栏目范围涉及自然与人工智能等问题，见<a href="https://blogs.nature.com/blog/a-multidisciplinary-forum-for-machine-intelligence/">官方创刊范围说明</a>。这只是契合度判断，不是编辑意向。</p><p><b>Nature Communications：</b>可作为另一个选择；官方标准强调对领域专家有意义的重要进展，见<a href="https://www.nature.com/ncomms/aims">Aims &amp; Scope</a>。现有多层证据提供基础，但“gloss这一新领域＋干预证据”是否达到重要进展仍需同行评审判断，不能视为保底。</p><p><b>Nature Human Behaviour：</b>目前不是优先建议，因为当前新增发现主要关于模型，尚未对人的知觉机制提出并检验新的预测。这是对本稿证据重心的判断，不代表该刊要求必须新增人类实验。</p><p>最需要避免的是把“工作量够多”当作顶刊判据。现有数据足以开始完整写作，投稿前还需要相关工作差异梳理、统计与图注统一、方法可复现核验和来源/公开材料准备。当前范围内不把新增大规模实验列为写稿前置条件。</p>''')
 section('收束规则：从现在起优先完成稿件','''<p><b>冻结：</b>以两个模型、gloss、现有三任务和限定条件机制实验为主。不启动42120图第二数据集、不扩展GPT全量、不继续寻找更多潜在维度、不为扩大故事添加新任务。</p><p><b>剩余工作是投稿整理：</b>把全文组织为上述六段结果；统一人类评分/模型评分、图像数/回答数/匹配数的口径；完成图注、方法、讨论和补充表；核对来源许可与公开数据清单；准备给编辑的贡献说明。若声称跨照明的干预机制，则必须另有数据，但当前稿件不作该主张。</p><p>原始资料已本地保存并不等于完成公共存储库发布。正式投稿时应给出可访问的代码与数据位置、第三方刺激/权重来源及重建参考图说明。</p>''')
 section('英文摘要工作稿','<p>'+abstract+'</p><p>这是基于现有数字的摘要工作稿，尚未按具体期刊字数与最终统计口径完成编辑。</p>')
 section('文件与原报告导航','''<ul><li><a href="tables/Table1_evidence_chain.csv">主证据总表</a>、<a href="tables/Table2_main_figure_plan.csv">正文图安排</a>、<a href="tables/Table3_experiment_inventory.csv">实验清单</a></li><li><a href="source_manifest.json">全部数值来源与校验</a>、<a href="claim_scope.json">冻结的论文主张</a></li><li><a href="submission_synthesis.zip">当前故事、图表与来源表打包</a>（不重复含10.7GB原始机制数组）</li><li>原报告：<a href="http://127.0.0.1:8777/report.html">InternVL主图</a> · <a href="http://127.0.0.1:8778/report.html">Qwen全量</a> · <a href="http://127.0.0.1:8779/report.html">行为RSA</a> · <a href="http://127.0.0.1:8780/report.html">内部机制</a></li></ul>''')
 doc='<!doctype html><html lang="zh"><meta charset="utf-8"><title>光泽论文：完整证据链与投稿整理</title><style>body{font:17px/1.8 system-ui;max-width:1180px;margin:40px auto;padding:20px;color:#173348;background:#fbfcfd}h1,h2{color:#155b7e}section{background:white;border:1px solid #d9e3e9;padding:25px;margin:22px 0;border-radius:10px}img{max-width:100%}table{border-collapse:collapse;font-size:13px;width:100%}td,th{border:1px solid #cdd8df;padding:8px;vertical-align:top}a{color:#087ba6}li{margin:10px 0}@media print{section{break-inside:auto}img{max-height:650px;object-fit:contain}}</style><h1>人类式光泽判断：现有证据、正文故事与投稿定位</h1><p>版本日期：2026-09-30 · 只整理已有结果，无新增采集 · 六张正文图工作版＋补充材料索引</p>'+''.join(sections)+'</html>'
 (O/'report.html').write_text(doc,encoding='utf-8')
 files=[p for p in O.rglob('*') if p.is_file() and p.suffix!='.zip' and p.name!='artifact_manifest.json'];(O/'artifact_manifest.json').write_text(json.dumps([dict(path=str(p.relative_to(O)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in files],ensure_ascii=False,indent=2),encoding='utf-8')
 with zipfile.ZipFile(O/'submission_synthesis.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in O.rglob('*'):
   if p.is_file() and p.suffix!='.zip':z.write(p,p.relative_to(O))
 print('Created',len(files),'files; no new experiments. Figures and statistics assembled from saved results.')
if __name__=='__main__':main()
