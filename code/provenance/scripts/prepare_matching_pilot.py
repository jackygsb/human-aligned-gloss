from pathlib import Path
import csv,json,random,hashlib
BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'mllm_material/matching_pilot_v1';OUT.mkdir(exist_ok=True)
rows=list(csv.DictReader((BASE/'annotations/gloss_annotations.csv').open(encoding='utf-8-sig')))
rng=random.Random(20260928);lights=list(range(1,37));rng.shuffle(lights)
selected=[]
for shape,light in zip(range(1,37),lights):
    candidates=[r for r in rows if int(r['shape_id'])==shape and int(r['illumination_id'])==light]
    r=rng.choice(candidates)
    selected.append({'image_id':r['image_id'],'image_path':r['image_512_path'],'starts':[rng.randint(8,18),rng.randint(35,45)]})
rng.shuffle(selected)
refs=[f'mllm_material/reference_reconstruction_v3/output/reference_{k:02d}.png' for k in range(1,53)]
prompt='''Image A is the TEST object. Image B is the adjustable REFERENCE object.
Adjust the gloss level of the reference object until it matches the apparent gloss of the test object.
Judge surface gloss (shine/lustre), not overall brightness, body colour, shape, or background similarity. Compare the visible appearance of the surfaces; do not estimate material names or physical parameters.
If the reference appears LESS glossy than the test, answer INCREASE.
If the reference appears MORE glossy than the test, answer DECREASE.
If their apparent gloss is indistinguishable, answer MATCH.
Return exactly one word: INCREASE, DECREASE, or MATCH.'''
protocol={'model':'internvl35_4b','n_images':36,'repetitions':2,'max_steps':7,'seed':20260928,
 'prompt':prompt,'system':'You are a visual observer performing a surface-gloss matching task. Follow the response format exactly.',
 'sampling':'one image per shape and per illumination using a seeded permutation; random viewpoint; no human scores used for selection',
 'presentation':'two original 512-square images in ordered placeholders, A=test and B=reference, each encoded to one 448 tile; no literal mouse slider; fresh context each decision',
 'controller':'random low/high initial reference; then midpoint of remaining integer interval; MATCH accepts current index; crossing interval returns last presented index flagged bracket_exhausted; out-of-range requests flagged boundary',
 'scope':'exploratory engineering/behavioral pilot; reference v3 is reconstructed; not an exact reproduction of human apparatus',
 'freeze_sha256':{p:hashlib.sha256((BASE/p).read_bytes()).hexdigest() for p in refs+[r['image_path'] for r in selected]}}
(OUT/'manifest.json').write_text(json.dumps(selected,indent=2))
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
print('Prepared',len(selected),'images;',len(refs),'references')
