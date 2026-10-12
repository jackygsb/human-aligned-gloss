"""Package a source-complete typesetting bundle and portable figure inputs."""
from pathlib import Path
import json,shutil,hashlib,zipfile,re
import pandas as pd,numpy as np
B=Path(__file__).resolve().parents[1];M=B/'mllm_material';O=M/'nmi_manuscript_zh_v1';S=O/'source';F=S/'figure_inputs'
manifest=[]
def cp(p,q):
 q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);manifest.append({'source':str(p),'copy':str(q.relative_to(S)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
def asset(p):cp(p,F/p.relative_to(B))
for p in (M/'submission_synthesis_v1/tables').iterdir():
 if p.is_file():asset(p)
for name in ['joint_model_predictions.csv','shape_id_effects.csv','illumination_id_effects.csv']:asset(M/'qwen3_full_v1'/name)
asset(M/'mechanisms_v1/stimuli.csv');asset(M/'mechanisms_v1/plan.json')
for k in [1,27,52]:asset(M/f'reference_reconstruction_v3/output/reference_{k:02d}.png')
for fac in ['shape','illumination']:
 for n in ['Human','InternVL','Qwen3']:
  for suf in ['profiles_by_view.npy','rdm.csv']:asset(M/f'behavioral_spaces_v1/matrices/physical_color_adjusted_{fac}_{n}_{suf}')
  asset(M/f'behavioral_spaces_v1/embeddings/{n}_{fac}_adjusted_profile_PCA.csv')
stim=pd.read_csv(M/'mechanisms_v1/stimuli.csv');plan=json.loads((M/'mechanisms_v1/plan.json').read_text('utf-8'));ids={stim.iloc[0].image_id}
for n in ['InternVL','Qwen3']:
 root=M/'mechanisms_v1'/n
 for fn in ['directions.npz','candidate_scores.csv']:asset(root/fn)
 z=pd.read_csv(root/'candidate_scores.csv');z=z[z.image_id.isin(plan['intervention_images'])].sort_values('projection');picked=z.image_id.iloc[[0,1,-2,-1]].tolist();ids.update(picked)
 for id in picked:
  p=root/'features'/f'{id}.npz';q=F/p.relative_to(B);q.parent.mkdir(parents=True,exist_ok=True)
  with np.load(p) as x:np.savez_compressed(q,bridge_tokens=x['bridge_tokens'])
  manifest.append({'source':str(p),'copy':str(q.relative_to(S)),'transformation':'bridge_tokens only; other arrays omitted','source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'sha256':hashlib.sha256(q.read_bytes()).hexdigest()})
for id in sorted(ids):asset(B/stim[stim.image_id==id].iloc[0].image_512_path)
for stage in ['mechanisms_v1','behavioral_spaces_v1','qwen3_full_v1']:
 for p in (M/stage/'code').glob('*.py'):cp(p,S/'code/original_analysis'/stage/p.name)
 for name in ['artifact_manifest.json','environment.json','requirements.lock.txt','protocol.json','model_manifest.json','plan.json','validation.json','Methods.md']:
  p=M/stage/name
  if p.exists():cp(p,S/'provenance'/stage/name)
for name in ['analyze_shared_condition_effects.py','analyze_matching_incremental_tiny.py','build_submission_synthesis_v1.py','build_paper_figures_v1.py']:
 p=B/'scripts'/name
 if p.exists():cp(p,S/'code/original_analysis'/name)
cp(M/'protocol/prompts.json',S/'provenance/prompts.json')
cp(B/'annotations/gloss_annotations.csv',S/'tables/human_stimulus_annotations.csv')
cp(M/'submission_synthesis_v1/source_manifest.json',S/'provenance/statistics_source_manifest.json')
cp(M/'submission_synthesis_v1/claim_scope.json',S/'provenance/claim_scope.json')
cp(B/'scripts/plot_nmi_manuscript.py',S/'code/plot_nmi_manuscript.py')
cp(B/'scripts/bib_nmi_manuscript.py',S/'code/bib_nmi_manuscript.py')
wrong=S/'provenance/marlow2012.json'
if wrong.exists():wrong.unlink()
p=S/'provenance/bibliography_metadata.json';j=json.loads(p.read_text('utf-8'));j.pop('marlow2012',None);p.write_text(json.dumps(j,ensure_ascii=False,indent=2),'utf-8')
(S/'provenance/template_source.json').write_text(json.dumps({'source_page':'https://www.springernature.com/gp/authors/campaigns/latex-author-support','download':'https://cms-resources.apps.public.k8s.springernature.io/springer-cms/rest/v1/content/18782940/data/v12','download_date':'2026-09-30','template_version':'December 2024, v3.1 example','class_unmodified':hashlib.sha256((S/'sn-jnl.cls').read_bytes()).hexdigest()==hashlib.sha256((O/'official_template/sn-article-template/sn-jnl.cls').read_bytes()).hexdigest(),'journal_rules':'https://www.nature.com/natmachintell/content','adaptation':'XeLaTeX + xeCJK; Chinese fonts with Fandol fallback; A4 review layout. pdflatex class option avoids legacy breakurl branch; engine remains XeLaTeX.'},indent=2),'utf-8')
(S/'provenance/figure_input_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),'utf-8')
readme='''# 中文论文初稿：阅读与复现

题目：预训练多模态语言模型中的人类式光泽判断及其视觉表征
版本：v1.0，2026-09-30。

## 打开与编辑
- main.pdf：正文完整稿；supplement.pdf：补充材料。
- main.tex、supplement.tex：两个独立、完整的 LaTeX 文档；不需要原项目路径即可编译。
- references.bib：33 条文献；bbl 为已生成的数字编号文献表。
- figures：6 张主图、10 张补图，每张均提供 PDF、SVG、PNG。
- tables：核心统计源表；figure_inputs：重新生成这 16 张图所需的精选输入。
- provenance：来源、模型/协议和数据清单；code：绘图入口与原分析代码快照。

## 编译
使用 XeLaTeX（不是直接使用 pdfLaTeX），依次执行：
```
xelatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error supplement.tex
bibtex supplement
xelatex -interaction=nonstopmode -halt-on-error supplement.tex
xelatex -interaction=nonstopmode -halt-on-error supplement.tex
```
已在 Windows TeX Live 2017 实测；现代 TeX Live / Overleaf 选择 XeLaTeX 亦可按此流程尝试。Windows 优先宋体及微软雅黑，其他环境回退到 TeX Live 的 Fandol。字体变化可能改变分页。文档类中的 pdflatex 选项是官方类分支选择，不代表应改用 pdfLaTeX 引擎。sn-jnl.cls 与 sn-nature.bst 均来自官方模板，未修改。详见 provenance/template_source.json。

## 图表复现
Python 依赖：numpy、pandas、matplotlib、Pillow。安装后在 source 目录执行：
```
python code/plot_nmi_manuscript.py
```
脚本识别 figure_inputs，重绘所有图；无需重新调用模型或访问 10.7GB 全量特征。中文绘图默认微软雅黑，其他系统可将 font.family 替换为可用中文字体。代码不会重新拟合主统计、挑选模型或修改原始数据。空间图只打包精选示例的 bridge_tokens，清单明确记录该提取。

## 科学范围
完整主线为两个模型的三任务行为、共同物理偏离/条件效应、行为 RSA、跨任务预测、监督内部读出和固定照明下的功能干预。小网络、GPT12图先导、量尺分布及完整检验置于补充材料。基础模型冻结不等于后续探针无监督；参考序列是重建而非原作者52张PNG；两模型共享Qwen3语言骨干谱系。正文中的统计从已归档分析表引用，未新增模型采集。

## 文献核验
文献元数据主要由 Crossref DOI 记录核验；模型报告、方法会议论文及模型卡按原始论文页面补入。核验中排除了一个标题不匹配的初始候选 DOI；Marlow 等人的最终文献为 10.1016/j.cub.2012.08.009。引用表提供真实文献，不虚构作者、页码或仓库链接。英文参考文献题名及实际英文提示词为可追溯性而保留，其他叙述和图标为中文。

## 完整数据与投稿
这个包是论文排版与图表复现包，不含全部模型权重、所有原始响应或10.7GB内部特征。完整实验仍在同级研究项目的 mllm_material 各阶段目录中，provenance 包含源清单和校验；原分析快照的路径以原项目结构为准。没有新建公共仓库或DOI。

本稿按 Nature Machine Intelligence Article 的引言、结果、讨论、方法结构组织，6个正文展示项。期刊官网当前列出英文正文最多3500词、摘要最多150词、展示项最多6项、参考文献通常最多50篇；不含方法等的正文词数与中文审阅版页数不同。正式英文投稿需再精简与核对。官网：https://www.nature.com/natmachintell/content 。

作者名单、单位、经费、贡献、利益冲突及公开数据地址必须由真实作者补充确认。当前是完整中文审阅初稿，不是已经获期刊认可或可保证录用的版本。没有添加尚未完成的实验或伦理审批。第三方刺激、权重和作者核的公开分发须遵循各自许可。
'''
(S/'README_中文说明.md').write_text(readme,'utf-8')
(S/'compile.ps1').write_text("$ErrorActionPreference = 'Stop'\nPush-Location $PSScriptRoot\ntry {\n  foreach ($doc in @('main','supplement')) {\n    & xelatex -interaction=nonstopmode -halt-on-error ($doc + '.tex')\n    if ($LASTEXITCODE -ne 0) { throw 'XeLaTeX failed' }\n    & bibtex $doc\n    if ($LASTEXITCODE -ne 0) { throw 'BibTeX failed' }\n    & xelatex -interaction=nonstopmode -halt-on-error ($doc + '.tex')\n    & xelatex -interaction=nonstopmode -halt-on-error ($doc + '.tex')\n    if ($LASTEXITCODE -ne 0) { throw 'XeLaTeX failed' }\n  }\n} finally { Pop-Location }\n",'utf-8')
files=[p for p in S.rglob('*') if p.is_file() and p.suffix not in ['.aux','.log','.out','.blg'] and not p.name.startswith('test.') and p.name!='package_manifest.json']
(S/'package_manifest.json').write_text(json.dumps([{'path':str(p.relative_to(S)).replace('\\','/'),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files],ensure_ascii=False,indent=2),'utf-8')
with zipfile.ZipFile(O/'NMI_Chinese_Manuscript_LaTeX_v1.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in files+[S/'package_manifest.json']:z.write(p,'NMI_Chinese_Manuscript/'+str(p.relative_to(S)))
final=B.parent/'output/pdf';final.mkdir(parents=True,exist_ok=True)
for stem,filename in [('main','NMI_中文正文_v1.pdf'),('supplement','NMI_中文补充材料_v1.pdf')]:shutil.copy2(S/(stem+'.pdf'),final/filename)
print('Packaged',len(files)+1,'files;',round((O/'NMI_Chinese_Manuscript_LaTeX_v1.zip').stat().st_size/1024**2,2),'MiB')
