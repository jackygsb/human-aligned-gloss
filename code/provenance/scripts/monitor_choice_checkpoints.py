"""Refresh interim choice analyses during collection, independent of full completion."""
from pathlib import Path
import subprocess,sys,time,json,datetime
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'mllm_material';REPORT=OUT/'reports'
last=-1
while True:
    p=OUT/'responses/internvl35_4b/full.jsonl'
    raw=p.read_bytes();n=raw.count(b'\n')
    try:status=json.loads((REPORT/'pipeline_status.json').read_text()).get('status')
    except (ValueError,FileNotFoundError):status='running'
    terminal=status in ['failed','priority_complete','complete','needs_method_review']
    if last<0 or n-last>=100 or terminal:
        process=subprocess.run([sys.executable,str(BASE/'scripts/analyze_live_choices.py')],capture_output=True,text=True)
        if process.returncode:
            (REPORT/'choice_monitor_error.txt').write_text(process.stderr,encoding='utf-8');break
        data=json.loads((REPORT/'live_choice_analysis.json').read_text())
        lines=['# 采集中间结果',f'更新时间：{data["timestamp"]}','不需要达到计划总次数即可分析。每新增约 100 条 InternVL 回答刷新一次；小样本结果为描述性中间结果。','| 模型 | 图像对类型 | 已完成双向图像对 | 跟随人类 | 位置交换一致性 |','|---|---|---:|---:|---:|']
        for model in ['internvl35_4b','qwen25vl3b']:
            for kind,r in data[model]['T2_complete_pairs'].items():
                lines.append(f'| {model} | {kind} | {r["n"]} | {r["follow_human"]:.1%} | {r["swap_consistency"]:.1%} |')
        lines+=['','注意：采集顺序先冲突对、后随机对，前期数值不能代表全部图像对。', '当前阶段策略：在约 200、500、1,000 对冲突图像对时查看累计趋势；没有必要等 4,000 次全收齐后才解读。较好的结果继续积累，明显退化则转入方法开发，不反复追逐显著性。']
        q=data['qwen25vl3b'];lines += ['','## Qwen 已有三图回答',str(q['T3_position']),'人类评分诱导的一维参照（不是独立人类三图行为）：'+json.dumps(q.get('T3_human_scalar_proxy',{}),ensure_ascii=False)]
        tmp=REPORT/'采集中间结果.tmp';tmp.write_text('\n\n'.join(lines[:3])+'\n\n'+'\n'.join(lines[3:]),encoding='utf-8');tmp.replace(REPORT/'采集中间结果.md');last=n
    if terminal:break
    time.sleep(30)
