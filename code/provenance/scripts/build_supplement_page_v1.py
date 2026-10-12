"""Readable supplement from existing results; no new inference or analysis."""
from pathlib import Path
import json,shutil,html
import pandas as pd
B=Path(__file__).resolve().parents[1];M=B/'mllm_material';O=M/'paper_figures_v1';S=O/'supplement'
comparison=M/'matching_qwen3vl4b_pilot12/comparison';increment=M/'matching_full_v1/incremental_tiny'
for src,name in [(comparison/'same12_comparison.png','FigS1_cross_model_same12.png'),(comparison/'same12_comparison.svg','FigS1_cross_model_same12.svg'),(comparison/'same12_predictions.csv','S1_same12_predictions.csv'),(comparison/'小网络对照审计.md','S2_tiny_provenance_audit.md'),(comparison/'author_fig3_reproduced_folds.csv','S2_author_fig3_folds.csv'),(comparison/'author_fig3_by_split.csv','S2_author_fig3_summary.csv'),(increment/'incremental_gain.png','FigS2_incremental_prediction.png'),(increment/'incremental_gain.svg','FigS2_incremental_prediction.svg'),(increment/'分析报告.md','S2_incremental_report.md')]:shutil.copyfile(src,S/name)
df=pd.read_csv(S/'S1_other_models_and_tiny_same12.csv');gains=json.loads((S/'S2_incremental_tiny_summary.json').read_text(encoding='utf-8'))['gains']
labels={'gpt_c':'GPT（订阅 CLI / low）','internvl_c':'InternVL3.5-4B','qwen_c':'Qwen3-VL-4B-Instruct','physical_c':'物理光泽参数','onelayer_lighting':'作者单层：照明列','onelayer_shape':'作者单层：形状列','threelayer_lighting':'作者三层：照明列','threelayer_shape':'作者三层：形状列','independent_tiny_shape':'独立重训单核：形状留出','independent_tiny_illumination':'独立重训单核：照明留出'}
def table(ids):
    z=df.set_index('model').loc[ids].reset_index();z['model']=z.model.map(labels);z=z[['model','n','spearman','pearson','mae']].rename(columns={'model':'模型／数据来源','n':'图像数','spearman':'Spearman ρ','pearson':'Pearson r','mae':'平均绝对误差 c'})
    return z.to_html(index=False,float_format=lambda v:f'{v:.3f}',border=0)
rows=[]
for r in gains:
    names={'tiny_linear':'小网络线性校准','tiny_cubic':'小网络非线性校准','controls_tiny_cubic':'非线性校准＋物理／颜色／视角控制'}
    lo,hi=r['crossed_cluster_ci95_delta_r2'];rows.append({'留出条件':'形状' if r['split']=='shape' else '照明','基线':names[r['baseline']],'加入 InternVL 后 ΔR²':f"{r['delta_r2']:.3f}",'均方误差降低':f"{r['relative_mse_reduction']:.1%}",'ΔR² 95% 区间':f'[{lo:.3f}, {hi:.3f}]'})
body='''<!doctype html><meta charset="utf-8"><title>补充材料 · 光泽判断研究</title><style>body{font:18px system-ui;line-height:1.7;color:#183b46;max-width:1150px;margin:36px auto;padding:0 24px}h1,h2{line-height:1.3}h2{margin-top:48px}table{border-collapse:collapse;width:100%;font-size:16px}th,td{padding:10px;border-bottom:1px solid #dbe4e8;text-align:left}th{background:#edf4f6}img{max-width:100%}a{color:#176b88}.note{background:#f1f5f7;padding:16px;border-left:4px solid #6f8c99}nav a{margin-right:18px}</style>
<p><a href="../report.html">← 主论文图表</a></p><h1>Supplementary evidence · 补充材料</h1><p>这里就是正文中 “Cross-model pilot results and tiny-network comparisons are reported as supplementary evidence” 所指的结果。全部来自已完成实验，本页未新增采集。</p><nav><a href="#cross">跨模型先导</a><a href="#tiny">小网络同图对照</a><a href="#gain">增量预测</a><a href="#files">数据与方法</a></nav>
<h2 id="cross">S1 · 三个模型的同图先导比较</h2><p>同样的 12 张图，每个模型两次完整参考匹配；分析每图两次的均值。InternVL 的同图结果取自全量实验。GPT 为 gpt-6-astra / low，经订阅 CLI 调用，与网页 Pro 不是同一实验条件。</p>'''
body+=table(['gpt_c','qwen_c','internvl_c','physical_c'])
body+='<img src="FigS1_cross_model_same12.png" alt="三个模型在相同12张图上的人类一致性"><p><a href="FigS1_cross_model_same12.svg">矢量图 SVG</a> · <a href="S1_same12_predictions.csv">逐图预测 CSV</a></p><div class="note">三种模型均呈正向人类一致性，但这是 12 图先导结果，不能据此确定模型排名。GPT 人类相关的探索性 bootstrap 区间约 [0.469, 1.000]，Qwen 约 [0.227, 0.949]。Qwen 有 7/12 图达到参考上限，排序相关不等于匹配数值准确。GPT 一次上下文最多四对图，两个开源模型每次一对，呈现条件并非完全相同。</div>'
body+='<h2 id="tiny">S2 · 小网络：同一批 12 张图</h2><h3>作者发布的单层／三层预测</h3>'
body+=table(['onelayer_lighting','onelayer_shape','threelayer_lighting','threelayer_shape'])
body+='<p>作者逐图预测直接取自发布 CSV，列名区分照明与形状划分。具体逐行检查点及训练折映射尚未完整核实，因此本表作为描述性对照，不能冒称已经审计的严格留出预测。没有从四列中只挑最高的一列。</p><h3>本项目独立重训的单核对照</h3>'
body+=table(['independent_tiny_shape','independent_tiny_illumination'])
body+='<p>这是依照论文单层结构独立实现的网络，不是作者原始模型的精确复现。两 seed 的分组留出预测取均值；48 份训练／测试划分已核验。小网络有行为标签监督、使用去背景输入，与多模态模型的任务和输入不同。</p><p>另外，作者 Figure 3 的 384 个逐折结果已按原 MATLAB 规则重算；它们是折内 Pearson 相关，不能与上方 12 图 Spearman 直接排名。</p><p><a href="S2_tiny_provenance_audit.md">小网络完整溯源审计</a> · <a href="S2_author_fig3_folds.csv">作者逐折结果</a> · <a href="S2_author_fig3_summary.csv">作者结果汇总</a></p>'
body+='<h2 id="gain">S3 · 全量分析：InternVL 是否增加小网络之外的预测信息？</h2><p>主要评估 3,849 张非开发图；分别留出形状、照明。比较同一测试图上的小网络预测与“小网络＋InternVL”预测，组合权重仅在训练组拟合。该小网络是上述独立单核对照。</p>'
body+=pd.DataFrame(rows).to_html(index=False,border=0)
body+='<img src="FigS2_incremental_prediction.png" alt="InternVL加入后在未见形状照明上的增量预测"><p>增加 InternVL 后，未见条件上的误差减少；在已有控制变量和非线性校准下仍有增益。区间来自固定预测的配对条件重采样，不包括重新训练的不确定性；不等于共享因果机制，也不代表超越作者所有小网络。</p><p><a href="FigS2_incremental_prediction.svg">矢量图 SVG</a> · <a href="S2_incremental_report.md">完整分析说明</a> · <a href="S2_incremental_tiny_metrics.csv">全部模型指标</a></p>'
body+='<h2 id="files">S4 · 补充数据与复现</h2><ul><li><a href="S1_other_models_and_tiny_same12.csv">跨模型与小网络同图统计表</a></li><li><a href="S3_cross_task_uncertainty.csv">跨任务结果的不确定性区间</a></li><li><a href="S0_all_existing_task_inventory.csv">所有已有任务记录清单（含不进入主文的探索任务）</a></li><li><a href="../paper_figures_reproducibility.zip">完整论文图表复现包</a></li></ul>'
(S/'index.html').write_text(body,encoding='utf-8')
page=O/'report.html';s=page.read_text(encoding='utf-8');marker='<!-- supplement-entry -->'
if marker not in s:s=s.replace('<h1>',marker+'<p style="padding:16px;background:#edf4f6"><b><a href="supplement/index.html">查看补充材料：跨模型先导结果、小网络对照与增量分析 →</a></b></p><h1>',1)
page.write_text(s,encoding='utf-8')
shutil.copyfile(Path(__file__),O/'code'/Path(__file__).name)
print('Supplement page ready: supplement/index.html')
