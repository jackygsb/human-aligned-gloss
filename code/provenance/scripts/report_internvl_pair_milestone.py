from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parents[1]; OUT=BASE/'mllm_material'; R=OUT/'reports'
d=pd.read_csv(OUT/'statistics/internvl35_4b_live_pair_details.csv')
d=d[d.stratum=='discordant'].sort_values('pair')
assert len(d)==1000
counts=[int(((d.follow_human==1)&d.swap_consistent).sum()),int(((d.follow_human==0)&d.swap_consistent).sum()),int((~d.swap_consistent).sum())]
assert sum(counts)==1000
fig,ax=plt.subplots(1,2,figsize=(10,4.2))
ax[0].bar(['Human in\nboth orders','Physical in\nboth orders','Changes with\nimage order'],counts,color=['#15877a','#8a96a6','#dfad50'])
for i,n in enumerate(counts):ax[0].text(i,n+12,str(n),ha='center')
ax[0].set_ylim(0,700);ax[0].set_ylabel('Image pairs (n = 1,000)');ax[0].set_title('Direct comparison on discordant pairs')
x=np.arange(1,len(d)+1)
ax[1].plot(x,d.follow_human.cumsum()/x,label='Human agreement',color='#15877a')
ax[1].plot(x,d.swap_consistent.cumsum()/x,label='Same image after swap',color='#4265a8')
ax[1].axhline(.5,color='gray',linestyle=':',label='50% reference (agreement)')
ax[1].set(xlim=(100,1000),ylim=(.45,1),xlabel='Cumulative completed pairs',ylabel='Proportion',title='Descriptive accumulation; not a stopping test')
ax[1].legend(fontsize=8,loc='lower right');fig.tight_layout()
for ext in ['png','svg']:fig.savefig(R/'figures'/f'internvl_completed_discordant_pairs.{ext}',dpi=220)
plt.close(fig)
text=f'''# InternVL：完整冲突图像对子实验

快照：2026-09-27。全部 1,000 对 × 两种顺序 = 2,000 次直接选择已完成；整个双图实验还包含正在采集的 1,000 对随机图像。此文件不代表全部实验完成。

## 现在已经获得什么

- 全部 3,888 张图像的单图评分与人类 Spearman 相关为 0.698。
- 单图分数推导的冲突对人类一致计分为 74.15%，其中平分按 0.5 计分。
- 同时呈现双图的直接选择，在同一组 1,000 对冲突图像上，人类一致比例为 68.8%，位置交换一致性为 80.4%。
- {counts[0]} 对在两种顺序下都跟随人类，{counts[1]} 对在两种顺序下都跟随物理排序，{counts[2]} 对交换位置后改变所选图像。

两种任务提供相互支持的描述性证据，但共享同一组刺激与人类数据，不是两个独立数据集的复现。当前图未附置信区间；不从累积曲线推断显著性。三图相似性及模型 SPoSE 尚未开始，不能把评分一致性直接称为表征结构已得到验证。

## 可供审阅的研究表述

未经本任务人类评分训练的 InternVL，在单图评分和直接比较中均呈现人类式光泽判断；当物理参数与人类判断排序冲突时，直接选择更常跟随人类。

## 用户现在可以推进的事情

1. 审阅上面的核心表述和结果图，确认它是否清楚表达“人类式判断”，并将尚待三图验证的“表征结构”单列。
2. 明确商业模型 API 可用服务和费用上限，使跨模型验证能够并行启动。无需在聊天中发送密钥。
3. 阅读冻结的 T1/T2/T3 提问内容，关注是否忠实表达可见表面光泽与光泽相似性；措辞建议记录为后续独立版本。

## 持续执行

完成剩余随机图像对，更新完整双图统计；通过现有阶段检查后启动 1,024 次三图先导，再决定扩量。已有结果随采集自动刷新，不等总任务完成。商业 API 尚未调用。
'''
(R/'InternVL完整冲突对子实验解读.md').write_text(text,encoding='utf-8')
print(dict(counts=counts,figure=str(R/'figures/internvl_completed_discordant_pairs.png')))
