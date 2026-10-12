from pathlib import Path
import json,shutil
BASE=Path(__file__).resolve().parents[1];old=BASE/'mllm_material/matching_pilot_v1';new=BASE/'mllm_material/matching_pilot_v2';new.mkdir(exist_ok=True)
shutil.copy2(old/'manifest.json',new/'manifest.json')
cfg=json.loads((old/'protocol.json').read_text())
cfg['prompt']='Compare the apparent surface gloss of the two objects. Which object looks glossier? Judge gloss, not overall brightness. Answer exactly A if Image A looks glossier, B if Image B looks glossier, or SAME if their surface gloss looks indistinguishable.'
cfg['development_reason']='v1 adjustment wording passed 3/9 reference-only controls; this exact direct-comparison wording passed 9/9. Same images and initial levels; no selection by human correlations.'
cfg['action_mapping']={'A':'INCREASE','B':'DECREASE','SAME':'MATCH'}
(new/'protocol.json').write_text(json.dumps(cfg,indent=2))
s=(BASE/'scripts/run_matching_pilot.py').read_text();s=s.replace('matching_pilot_v1','matching_pilot_v2').replace("action=normalized if normalized in ['INCREASE','DECREASE','MATCH'] else None","action={'A':'INCREASE','B':'DECREASE','SAME':'MATCH'}.get(normalized)")
(BASE/'scripts/run_matching_pilot_v2.py').write_text(s)
s=(BASE/'scripts/analyze_matching_pilot.py').read_text(encoding='utf-8').replace('matching_pilot_v1','matching_pilot_v2')
(BASE/'scripts/analyze_matching_pilot_v2.py').write_text(s,encoding='utf-8')
