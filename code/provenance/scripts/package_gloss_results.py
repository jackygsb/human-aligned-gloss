"""Audit actual completeness and package research outputs, excluding model/runtime files."""
from pathlib import Path
import json,hashlib,zipfile
import pandas as pd
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'mllm_material'
audit=[]
for model in ['qwen25vl3b','internvl35_4b']:
    for phase,manifest in [('full','requests.jsonl'),('input_control','input_control.jsonl')]:
        required={json.loads(l)['request_id'] for l in (OUT/'manifests'/manifest).read_text().splitlines()}
        rows={r['request_id']:r for r in map(json.loads,(OUT/'responses'/model/(phase+'.jsonl')).read_text().splitlines())}
        missing=required-set(rows);audit.append(dict(model=model,phase=phase,expected=len(required),recorded=len(required&set(rows)),missing=len(missing),valid=sum(rows[k]['parse_status']=='valid' for k in required if k in rows)))
        assert not missing,(model,phase,len(missing))
for source in ['qwen25vl3b','internvl35_4b','human_rating_induced','synthetic_1d']:
    assert (OUT/'spose'/source/'selected_20seeds.csv').exists()
    assert (OUT/'spose'/source/'additional_checks/COMPLETE.json').exists()
pd.DataFrame(audit).to_csv(OUT/'reports/completeness_audit.csv',index=False)
models=pd.read_csv(OUT/'statistics/model_summary.csv');pairs=pd.read_csv(OUT/'statistics/pair_summary.csv');spose=pd.read_csv(OUT/'statistics/spose_summary.csv')
lines=['# Gloss 计算实验结果','本报告只使用已保存的真实模型回答；假说是否成立由效应量、区间及控制结果判断。','## 研究问题','通用多模态模型在未使用本项目人类评分训练的条件下，是否表现出与人类一致的光泽排序和相对物理值的系统偏差？SPoSE 进一步检验模型自身选择结构是否具有可复现的预测价值。','## 主要结果','| 模型 | 有效单图数 | 人类 Spearman ρ | 95% 区间 | 物理 Spearman ρ |','|---|---:|---:|---|---:|']
for r in models.itertuples():lines.append(f'| {r.model} | {r.n} | {r.human_spearman:.4f} | [{r.human_rho_ci_low:.4f}, {r.human_rho_ci_high:.4f}] | {r.physical_spearman:.4f} |')
lines+=['','| 模型 | 冲突对随人类方向比例 | 95% 区间 | 交换一致性 |','|---|---:|---|---:|']
for r in pairs[pairs.stratum=='discordant'].itertuples():lines.append(f'| {r.model} | {r.follow_human:.4f} | [{r.ci_low:.4f}, {r.ci_high:.4f}] | {r.swap_consistency:.4f} |')
lines+=['','## SPoSE','| 数据来源 | SPoSE 测试期望一致率 | 单图距离参照 | 平均活动维数 |','|---|---:|---:|---:|']
for r in spose.itertuples():lines.append(f'| {r.source} | {r.test_agreement_mean:.4f} | {r.scalar_test_agreement:.4f} | {r.active_dimensions_mean:.1f} |')
lines+=['','上述维数不是人类心理维数。人类评分诱导参照与合成一维校准单列；最小预测维数、自由一维模型、投影敏感性和训练量曲线见各来源 additional_checks。','## 小网络对照','作者原 MATLAB Figure 3 已运行并与整理表核验。人类目标训练单核网络均值 r=0.645488、三层64核 r=0.809816。其专项监督、灰背景输入、作者选行规则与 MLLM 零样本条件不同。不能直接据此宣布参数效率胜负。','## 证据边界与仍需工作','本轮只完成 gloss 的已配置计算。Serrano 独立外推尚未执行；原参考刺激未确认完整，未执行参考匹配 T4；作者逐图预测的折/检查点关系未充分确认，不作为严格同折的三选一基准；未将旧弱拟合网络冒称原作者模型，也未声称重新训练了288个作者网络。维度语义关联及结构 RSA 属于探索性描述，不能充当因果机制证据。','人类参照 SPoSE 的早期参数搜索提前计算了测试指标，虽未用于选择参数，也已在修订记录披露；核心 MLLM 版本将验证选择与测试计算分开。','## 数据与图表','逐图评分、偏差对、分组表现、物理/图像统计控制、提示与重复结果见 statistics；原始问答见 responses；PNG/SVG 图见 reports/figures；完整协议和代码来源见 protocol。所有失败回答保留。Nature 系列是投稿目标，不是本轮结果可以自动保证的结论。']
(OUT/'reports/RESULTS.md').write_text('\n'.join(s if s.startswith('|') else s+'\n' for s in lines),encoding='utf-8')
files=[p for folder in ['protocol','manifests','responses','statistics','baselines','spose','reports'] for p in (OUT/folder).rglob('*') if p.is_file() and p.suffix not in ['.log'] and not p.name.endswith('_pid.txt') and p.name not in ['results_file_hashes.json','pipeline_status.json']]
hashes={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};(OUT/'reports/results_file_hashes.json').write_text(json.dumps(hashes,indent=2))
archive=BASE/'gloss_mllm_results.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in files:z.write(p,str(p.relative_to(OUT)))
    z.write(OUT/'reports/results_file_hashes.json','reports/results_file_hashes.json')
    provenance=json.loads((OUT/'protocol/source_provenance.json').read_text())
    for relative in provenance['files']:
        source=BASE.parent/relative
        z.write(source,'source_code/'+relative)
print('Audited research outputs packaged:',archive)
