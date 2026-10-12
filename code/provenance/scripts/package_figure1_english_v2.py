from pathlib import Path
import json
B=Path(__file__).resolve().parents[1]
O=B/'mllm_material/figure1_english_v2'
records=[]
with (B/'mllm_material/matching_full_v1/decisions.jsonl').open(encoding='utf-8') as f:
    for line in f:
        r=json.loads(line)
        if r['trial_id']=='morimoto_1889_r1':records.append(r)
        if len(records)==6:break
assert [r['reference_index'] for r in records]==[15,34,43,38,40,39]
assert [r['raw'].strip() for r in records]==['A','A','B','A','B','A']
assert [(r['low'],r['high']) for r in records]==[(1,52),(16,52),(35,52),(35,42),(39,42),(39,39)]
(O/'matching_decisions_example.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
s=(B/'scripts/package_figure1_english.py').read_text('utf-8')
s=s.replace('figure1_english_v1','figure1_english_v2').replace('draw_figure1_english.py','draw_figure1_english_v2.py')
s=s.replace('Figure1_English_Editable_Package.zip','Figure1_English_Detailed_v2_Editable_Package.zip')
s=s.replace('180 x 227 mm','180 x 434 mm (expanded editorial master)')
s=s.replace('Primary editable file: Figure1_Methods_EN.svg','Primary editable file: Figure1_Methods_EN.svg\nTask-only editable extract: Figure1_bcd_Task_Detail_EN.svg\nComplete task and system instructions: prompts_verbatim.json\nVerified original six-step decision log: matching_decisions_example.json')
exec(compile(s,str(__file__),'exec'))
