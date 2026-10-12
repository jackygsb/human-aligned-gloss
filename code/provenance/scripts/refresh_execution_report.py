"""Maintain a human-readable status file while the authorized batch is running."""
from pathlib import Path
import json,time,collections,datetime
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'mllm_material';REPORT=OUT/'reports'
while True:
    statepath=REPORT/'pipeline_status.json';state=json.loads(statepath.read_text()) if statepath.exists() else {}
    lines=['# Gloss 实验运行进度',f'更新时间：{datetime.datetime.now():%Y-%m-%d %H:%M:%S}（本机时间）',f'总流程状态：{state.get("status","初始化")}。只有 complete 才表示已配置批处理执行完毕。','| 模型 | 主流程已记录 / 计划 | 输入控制已记录 / 计划 | 工程检查 |','|---|---:|---:|---|']
    for model in ['qwen25vl3b','internvl35_4b']:
        folder=OUT/'responses'/model;counts=[]
        for phase in ['full','input_control']:
            p=folder/(phase+'.jsonl')
            if p.exists():
                with p.open(encoding='utf-8') as f:counts.append(sum(1 for line in f if line.endswith('\n')))
            else:counts.append(0)
        gate=folder/'engineering_gate.json';g=json.loads(gate.read_text()) if gate.exists() else {};label='288/288 通过' if g.get('passed') else '尚未通过'
        lines.append(f'| {model} | {counts[0]} / 54,400 | {counts[1]} / 864 | {label} |')
    lines+=['','主流程包含单图、提示/复测/无图条件、两图比较及三图选择。计数是已保存回答数，包含需如实报告的无效或无法评分回答。','## 已有输出','作者原始 MATLAB Figure 3 已复现；小网络逐折表已核验；人类可靠性与评分诱导 SPoSE 参照已有结果。大模型统计在每款模型完成全量单图后生成，选择与表征分析在对应采集完成后生成。','## 文件入口','[实际结果说明](RESULTS.md) · [全部输出入口](../实验输出入口.md) · [论文候选图](figures/)','本轮不下载 Serrano 大型图像。投稿层级和“人类式感知涌现”的结论必须等待全部效应及控制结果，不能由任务完成状态推定。']
    if state.get('scope'):lines+=['','当前执行范围：'+state['scope'],'当前阶段：'+state.get('stage',''),'priority_complete 仅表示 InternVL 优先任务完成，Qwen 暂停部分不计为完成。']
    if state.get('error'):lines+=['','## 运行错误',state['error'],'已完成的原始问答与拟合检查点保留，不能把错误状态当作完成。']
    tmp=REPORT/'运行进度.tmp';tmp.write_text('\n'.join(s if s.startswith('|') else s+'\n' for s in lines),encoding='utf-8');tmp.replace(REPORT/'运行进度.md')
    if state.get('status') in ['failed','complete','priority_complete','needs_method_review']:break
    time.sleep(60)
