from pathlib import Path
import shutil,json,re,hashlib,zipfile,difflib,sys
B=Path(__file__).resolve().parents[1];M=B/'mllm_material';O=M/'nmi_revision_v8/source';V=B.parent/'tmp/pdfs/nmi_clean_compile/NMI_Chinese_Manuscript';F=O/'figure_inputs';dest=B.parent/'output/pdf';dest.mkdir(parents=True,exist_ok=True)
files=['qwen3_full_v1/pair_predictions.csv','qwen3_full_v1/joint_model_predictions.csv','qwen3_full_v1/shape_id_effects.csv','qwen3_full_v1/illumination_id_effects.csv','qwen3_full_v1/protocol.json','paper_figures_v1/source_data/image_level_all3888.csv','paper_figures_v1/source_data/pair_cross_task_predictions.csv','paper_figures_v1/source_data/continuous_cross_task_oof.csv','paper_figures_v1/tables/Table6_shared_condition_effects.csv','mechanisms_v1/tables/interventions.csv','mechanisms_v1/tables/readout_predictions.csv.gz','matching_full_v1/protocol.json','matching_full_v1/freeze_manifest.json','protocol/prompts.json']
files += [f'qwen3_full_v1/cross_{sp}_{sr}.csv' for sp in ['shape','illumination'] for sr in ['single_score','model_c']]
for rel in files:
 dst=F/'mllm_material'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(M/rel,dst)
for src in (M/'reference_reconstruction_v3/output').glob('reference_*.png'):shutil.copy2(src,F/'mllm_material/reference_reconstruction_v3/output'/src.name)
for src in [M/'reference_reconstruction_v3/render_reference.py',M/'reference_reconstruction_v3/output/reference_levels.csv']:
 target=O/'provenance'/src.name;shutil.copy2(src,target)
for source,name in [('revise_nmi_stats.py','analyze_uncertainty_v8.py'),('revise_nmi_readouts.py','analyze_readouts_v8.py')]:
 s=(B/'scripts'/source).read_text('utf-8')
 if source=='revise_nmi_stats.py':s=s.replace("B=Path(__file__).resolve().parents[1];M=B/'mllm_material';O=M/'nmi_revision_v8/source';T=O/'tables';T.mkdir(exist_ok=True)","O=Path(__file__).resolve().parents[1];B=O/'figure_inputs';M=B/'mllm_material';T=O/'tables';T.mkdir(exist_ok=True)")
 else:s=s.replace("B=Path(__file__).resolve().parents[1];M=B/'mllm_material';O=M/'nmi_revision_v8/source'","O=Path(__file__).resolve().parents[1];B=O/'figure_inputs';M=B/'mllm_material'")
 (O/'code'/name).write_text(s,encoding='utf-8')
# Assert preservation of original files and all result/discussion/method prose outside figure legends.
old=(V/'main_NMI_submission_v7.tex').read_text('utf-8');new=(O/'main_NMI_submission_v8_red.tex').read_text('utf-8')
def body_without_plates(s):
 s=s[s.index(r'\section{结果'):]
 while r'\plate{' in s:
  a=s.index(r'\plate{');p=a+len(r'\plate')
  for _ in range(3):
   assert s[p]=='{';depth=1;p+=1
   while depth:
    if s[p]=='{':depth+=1
    elif s[p]=='}':depth-=1
    p+=1
  s=s[:a]+s[p:]
 return s
assert body_without_plates(old)==body_without_plates(new),'Unexpected main prose change'
frozen=json.loads((O/'provenance/original_v7_sha256.json').read_text('utf-8'))
assert all(hashlib.sha256((V/n).read_bytes()).hexdigest()==h for n,h in frozen.items())
(O/'provenance/main_v7_to_v8.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='main_v7',tofile='main_v8_red')),encoding='utf-8')
(O/'provenance/preservation_check.json').write_text(json.dumps({'original_files_unchanged':True,'results_discussion_methods_prose_unchanged':True,'changes':'Introduction; marked figure legends; preamble support'},indent=2),encoding='utf-8')
(O/'requirements.txt').write_text('numpy\npandas\nscipy\nmatplotlib\npillow\n',encoding='utf-8')
(O/'compile.ps1').write_text('''$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
foreach ($name in @('main_NMI_submission_v8_red','supplement_NMI_submission_v8')) {
    xelatex -interaction=nonstopmode -halt-on-error "$name.tex"
    if ($LASTEXITCODE -ne 0) { throw "LaTeX failed: $name" }
    bibtex $name
    if ($LASTEXITCODE -ne 0) { throw "BibTeX failed: $name" }
    xelatex -interaction=nonstopmode -halt-on-error "$name.tex"
    xelatex -interaction=nonstopmode -halt-on-error "$name.tex"
}
''',encoding='utf-8')
(O/'README.md').write_text('''# NMI 中文修订稿 v8

## 文件
- main_NMI_submission_v8_red.tex：正文，扩展引言和配套图注修订标红。
- supplement_NMI_submission_v8.tex：重写的中文补充材料。
- references.bib：61 条书目；正文实际引用 59 条，模型卡仅在附件引用。
- figures/：六张主图、十张补充图，提供 PDF、SVG、PNG。
- tables/v8_*.csv：本次新增区间和统计检验，数值精度高于排版表格。
- figure_inputs/：重绘与本次新增统计所需固定数据。
- provenance/：原稿哈希、修改差异、文献元数据和保存核验。

## 编译
使用 XeLaTeX → BibTeX → XeLaTeX → XeLaTeX，或在已配置 TeX 环境运行 compile.ps1。
使用随附 sn-jnl.cls 与 sn-nature.bst。中文字体优先宋体/微软雅黑，缺省 Fandol。

## 重现本次分析和图
安装 requirements.txt，在本目录依次运行：
```
python code/analyze_uncertainty_v8.py
python code/analyze_readouts_v8.py
python code/plot_nmi_v8.py
```
分析不调用模型、不下载权重，不重新拟合已有预测器。前两步约一分钟，具体取决于机器。
主分析种子 20261001，读出区间种子 20261002。模型训练和原始采集流程留存于 code/original_analysis 与既有研究归档；本源包以重现文稿图表和新增统计为范围，不声称包含全量隐藏状态、全部原始刺激或全部模型权重。

## 统计口径
总体相关和连续预测：形状×照明交叉重采样；双图：图像节点重采样；条件效应：节点置换，四项 BH；内部干预：36 个形状级配对符号随机化，八项 BH；RSA：原有完整 18 项家族。95% 区间为逐项区间，固定预测器区间不包括重新训练的不确定性。

GPT 小样本可行性结果不用于本次投稿附件；历史探索文件留在原项目，未删除。本包中旧表仅作为既有分析来源，正文/SI 的新增统计以 v8 前缀表为准。
''',encoding='utf-8')
memo='''# v8 修订说明与作者审阅要点

本次保留 v7 原文件；正文结果、讨论、方法的叙述未改动。只重写并标红引言、补充必要图注统计说明，并修正原图 6 图注中不存在的 e 面板为补充图 S6。补充材料全面重写，实验记录式说明转入 README。

## 实质修改
1. 正文实际引用 59 篇，书目库 61 条。新增文献集中在材料心理物理学、形状/照明作用、视觉语言学习、行为表征对齐与机制解释。原作者署名/摘要等保持原样。
2. 移出 GPT 12 图可行性与折映射未核定的作者预测排行榜；保留全量两模型结果、监督小网络的独立实现，以及正文直接涉及的 12 图功能干预。
3. 六幅主图、十幅补充图重绘，去除结果柱形图；展示配对点、区间、完整轨迹、统一标尺及可编辑矢量图。图 5 增加固定折外预测的区间；图 6 按 36 个形状重采样。
4. 四项共同条件效应经 BH 校正均显著；八项内部候选方向相对对照的配对检验均达到蒙特卡洛分辨率 p=q=0.00001。应报告分辨率，不将其写成精确解析概率。
5. InternVL 参考匹配相对物理参数的相关差为 0.057，95% 区间 [-0.037,0.134]。其他三个模型任务相对物理的差值区间高于零；所有六项成对比较完整保留。

## 原正文中留待作者统一措辞的三处科学边界
这些不是本次新增的实验要求，也没有据此改动您保留的结果/讨论段落。
- “未经材料属性监督”宜统一为“未用目标人类光泽标签适配”；现有记录未审核通用预训练语料是否含材料相关信息。
- 回答前语言层响应是图像与提示共同作用的多模态状态。“生成答案之前”有测量依据，“排除语言贡献”或“纯视觉机制”不是同一个结论。
- 参考图是固定重建序列：实际用 doge2 庭院、程序几何和中性漫反射 0.24；名义 c 转换基准项仍为 1/16。补充方法按实际代码明确记录了这些值及原论文环境文本与示例的差别。该量尺不宜表述为逐像素重现原始人类装置或严格恢复物理 c。

## 投稿篇幅
NMI 官方 Article 指南通常建议正文不超过 3,500 英文词、摘要 150 词、最多六项主展示，参考文献推荐约 50 条。59 篇是供审阅的相关文献扩展版；中文页数不能替代最终英文词数核对。作者、单位、可用性链接等仍需由作者最终确定。本次不改写您保留的摘要以压缩字数。

官方指南：https://www.nature.com/natmachintell/content
图规范：https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/
'''
(O/'修订说明_作者审阅.md').write_text(memo,encoding='utf-8')
if '--final' in sys.argv:
 for src,name in [('main_NMI_submission_v8_red.pdf','NMI_中文正文_v8_红字审阅.pdf'),('supplement_NMI_submission_v8.pdf','NMI_中文补充材料_v8.pdf')]:shutil.copy2(O/src,dest/name)
 qa=B.parent/'tmp/pdfs/nmi_v8_qa/qa_report.json';shutil.copy2(qa,O/'provenance/qa_report.json')
 # Exclude obsolete manuscript versions and obsolete pilot presentation from the submission source archive.
 paths=[]
 for p in O.rglob('*'):
  if not p.is_file() or p.suffix in ['.aux','.log','.out','.blg'] or '__pycache__' in p.parts:continue
  if '_v7' in p.name and p.parent==O:continue
  if p.name in ['package_manifest.json','new_references.bib','pilot_and_tiny.csv','plot_nmi_manuscript.py','bib_nmi_manuscript.py']:continue
  paths.append(p)
 manifest=[{'path':p.relative_to(O).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]
 (O/'package_manifest_v8.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');paths.append(O/'package_manifest_v8.json')
 zipfile_path=O.parent/'NMI_Chinese_Manuscript_v8_Revision.zip'
 with zipfile.ZipFile(zipfile_path,'w',zipfile.ZIP_DEFLATED) as z:
  for p in paths:z.write(p,Path('NMI_Chinese_Manuscript_v8')/p.relative_to(O))
 print('Packaged',len(paths),'files',round(zipfile_path.stat().st_size/1024**2,2),'MiB')
else: print('Portable analysis inputs, audit, and README prepared')
