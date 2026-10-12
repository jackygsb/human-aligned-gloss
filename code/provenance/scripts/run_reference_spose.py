from pathlib import Path
import os,subprocess,sys,json,time
base=Path(__file__).resolve().parents[1];os.environ['SPOSE_DEVICE']='cpu'
for source in ['human_rating_induced','synthetic_1d']:
    subprocess.run([sys.executable,str(base/'scripts/fit_gloss_spose.py'),'--source',source],check=True)
(base/'mllm_material/spose/reference_jobs_complete.json').write_text(json.dumps(dict(completed_unix=time.time())))
