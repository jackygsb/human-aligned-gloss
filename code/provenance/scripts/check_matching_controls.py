import os,json,time
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'mllm_material/matching_pilot_v1'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['HF_MODULES_CACHE']=str(BASE/'mllm_runtime/hf_modules')
from internvl_backend import InternVLBackend
backend=InternVLBackend(str(BASE/'models/internvl35_4b'),BASE)
cfg=json.loads((OUT/'protocol.json').read_text())
direct='Compare the apparent surface gloss of the two objects. Which object looks glossier? Judge gloss, not overall brightness. Answer exactly A if Image A looks glossier, B if Image B looks glossier, or SAME if their surface gloss looks indistinguishable.'
pairs=[(1,1),(27,27),(52,52),(1,52),(52,1),(1,27),(27,1),(27,52),(52,27)]
results=[]
for variant,prompt in [('adjustment',cfg['prompt']),('direct_comparison',direct)]:
    for a,b in pairs:
        req={'image_paths':[f'mllm_material/reference_reconstruction_v3/output/reference_{k:02d}.png' for k in [a,b]],'task':'control','condition':'main','sample':False}
        raw,info=backend.run(req,{'system':cfg['system'],'control':prompt})
        got=raw.strip().upper().strip(' .!\n\t')
        expected=('MATCH' if a==b else 'INCREASE' if a>b else 'DECREASE') if variant=='adjustment' else ('SAME' if a==b else 'A' if a>b else 'B')
        results.append({'variant':variant,'test_index':a,'reference_index':b,'raw':raw,'expected':expected,'correct':got==expected,**info})
        print(variant,a,b,got,'expected',expected,flush=True)
(OUT/'reference_controls.json').write_text(json.dumps({'note':'same reconstructed geometry: equal-image and wide physical-order checks, not human-label alignment tests','results':results},indent=2))
