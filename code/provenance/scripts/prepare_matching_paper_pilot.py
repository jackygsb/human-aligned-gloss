from pathlib import Path
import json,csv,random
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'mllm_material/matching_paper_pilot';OUT.mkdir(exist_ok=True)
source=BASE/'mllm_material/matching_pilot_v2';manifest=json.loads((source/'manifest.json').read_text());cfg=json.loads((source/'protocol.json').read_text())
rng=random.Random(2026092801);schedule=[]
labels=list(csv.DictReader((BASE/'annotations/gloss_annotations.csv').open(encoding='utf-8-sig')))
unused=[r for r in labels if r['image_id'] not in {r['image_id'] for r in manifest}]
for i,r in enumerate(rng.sample(unused,3),1):schedule.append({'trial_id':f'practice_{i}','phase':'practice','image_id':r['image_id'],'image_path':r['image_512_path'],'repetition':0,'start':rng.randint(2,51)})
for rep in [1,2]:
    block=list(manifest);rng.shuffle(block);temp=[]
    for r in block:temp.append({'trial_id':r['image_id']+f'_r{rep}','phase':'main','image_id':r['image_id'],'image_path':r['image_path'],'repetition':rep,'start':rng.randint(2,51)})
    temp.insert(rng.randrange(len(temp)+1),{'trial_id':f'catch_{rep}','phase':'catch','repetition':rep})
    schedule.extend(temp)
cfg.update({'schedule_seed':2026092801,'starts':'independent uniform integers 2..51 in both sessions','presentation_order':'independent randomized order per session','practice_trials':3,'catch_trials_per_session':1,'total_scheduled':77,'n_images':36,
 'paper_adaptations':['36-image pilot, not the original 72+12 image-set design','ordered image placeholders rather than literal screen and mouse slider','fresh-context comparisons with deterministic generation; practice checks operation, not human adaptation','no monitor, viewing distance, dark room, one-minute visual adaptation or human response-time exclusion imposed on model','algorithmic bisection controller instead of freely moving mouse','reconstructed v3 reference; no claim of exact original stimulus identity'],
 'catch_instruction':'No stimulus images are shown in this attention check. Move the reference slider to its LEFTMOST endpoint. The allowed indices are 1 to 52. Return only the selected index.'})
(OUT/'protocol.json').write_text(json.dumps(cfg,indent=2));(OUT/'schedule.json').write_text(json.dumps(schedule,indent=2))
