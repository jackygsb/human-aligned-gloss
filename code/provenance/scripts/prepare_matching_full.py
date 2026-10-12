from pathlib import Path
import csv,json,random,hashlib
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'mllm_material/matching_full_v1';OUT.mkdir(exist_ok=True)
assert not (OUT/'schedule.json').exists(),'Do not overwrite frozen schedule'
pilot=BASE/'mllm_material/matching_paper_pilot'
cfg=json.loads((pilot/'protocol.json').read_text())
old=json.loads((pilot/'schedule.json').read_text())
dev={x['image_id'] for x in old if 'image_id' in x}
rows=list(csv.DictReader((BASE/'annotations/gloss_annotations.csv').open(encoding='utf-8-sig')))
rng=random.Random(2026092901)
held=[r for r in rows if r['image_id'] not in dev];development=[r for r in rows if r['image_id'] in dev]
rng.shuffle(held);rng.shuffle(development);ordered=held+development
schedule=[dict(x) for x in old if x['phase']=='practice']
for block,start in enumerate(range(0,len(ordered),72),1):
    for rep in [1,2]:
        items=ordered[start:start+72].copy();rng.shuffle(items);trials=[]
        for r in items:trials.append(dict(trial_id=f"{r['image_id']}_r{rep}",phase='main',image_id=r['image_id'],image_path=r['image_512_path'],repetition=rep,start=rng.randint(2,51),block=block,cohort='development' if r['image_id'] in dev else 'heldout_matching'))
        trials.insert(rng.randrange(len(trials)+1),dict(trial_id=f'catch_b{block}_r{rep}',phase='catch',repetition=rep,block=block))
        if rep==2:trials[-1]['checkpoint']=True
        schedule.extend(trials)
freeze={p:h for p,h in cfg['freeze_sha256'].items() if 'reference_reconstruction' in p}
for r in rows:freeze[r['image_512_path']]=hashlib.sha256((BASE/r['image_512_path']).read_bytes()).hexdigest()
cfg.update(full_collection=True,n_images=len(rows),total_scheduled=len(schedule),freeze_sha256=freeze,schedule_seed=2026092901,
    sampling='All 3888 images; shuffled matching-heldout images first; 39 development/practice images flagged separately; 72-image blocks, two randomized sessions per block',
    controller='Independent uniform start 2..51, then bisection over remaining integer interval; unchanged pilot stopping and conversion rules',
    scope='Frozen-protocol model adaptation with reconstructed references; matching-heldout validation, not exact reproduction of human apparatus',
    development_image_ids=sorted(dev),heldout_images=len(held),checkpoints='Every completed 72-image block; first 360 images are an interim checkpoint; continue full schedule regardless of correlation',
    primary_analysis='Heldout image-level Spearman human vs mean of two model matching estimates; bootstrap by image and separately by shape/illumination',
    secondary_analysis=['linear physical-c partial Pearson (descriptive)','repeat consistency','comparison to existing single-image scores on identical images','shape and illumination group summaries'],
    paper_adaptations=['72 unique images per computational block, no original 12 common human-observer items','three practice trials at run start and one no-image catch per session','two ordered 448px encoded images; deterministic fresh-context comparisons','bisection rather than mouse adjustment; crossing threshold rather than explicit equal-gloss match','no human monitor adaptation or reaction-time exclusion','reconstructed v3 references; not identical author stimuli'])
(OUT/'protocol.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
(OUT/'schedule.json').write_text(json.dumps(schedule,indent=2),encoding='utf-8')
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/'protocol.json',OUT/'schedule.json',BASE/'scripts/run_matching_paper_pilot.py',BASE/'scripts/internvl_backend.py']}
(OUT/'freeze_manifest.json').write_text(json.dumps(hashes,indent=2))
(OUT/'扩大采集方案.md').write_text('''# InternVL 光泽匹配扩大采集（冻结版）

采集全部 3,888 张图，每图两次匹配，共 7,776 次主任务。39 张先导主任务或练习图标记为开发集，另 3,849 张作为匹配方案未调试过的验证集；这些图此前可能参加过单图评分，不能称模型从未见过。先导数据不混入本轮响应。

沿用先导的参考序列 v3、提示词、图像编码、确定性生成、1–52 档搜索、停止规则与 c 转换。每个 72 图块分别随机两次顺序和起点。3 次练习、每个会话 1 次无图检查。它是原论文匹配任务的模型适配，不是人类鼠标实验的严格复现。

每完成一个图块自动分析；360 张为预定中期查看点，继续全量采集，不按相关性选择停止或改提示词。预计数小时，按实测速率更新。主指标为验证集人类—模型 Spearman，另报告形状/照明聚类重采样区间、控制物理 c 的描述性残差相关、重复一致性和同图单图评分对照。全部边界响应保留并标记，不据此删掉低一致性数据。

输出包括全部逐轮回答、匹配轨迹、逐图 CSV、统计 JSON、PNG/SVG 图、分组结果及实时网页。先导为探索性结果，本轮用于检查固定方法能否推广；结果不预设为显著。参考图为重建，模型 c 与人类原参考 c 不自动视为同一标尺。
''',encoding='utf-8')
print(json.dumps(dict(images=len(rows),heldout=len(held),schedule=len(schedule))))
