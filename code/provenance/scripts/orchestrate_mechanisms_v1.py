"""Sequential GPU jobs followed by CPU analysis, with persistent status and complete archive manifest."""
from pathlib import Path
import subprocess,sys,time,json,hashlib,zipfile,traceback
B=Path(__file__).resolve().parents[1];O=B/'mllm_material/mechanisms_v1'
def save(x):(O/'pipeline_status.json').write_text(json.dumps(x,indent=2),encoding='utf-8')
def main():
 jobs=[('InternVL',[str(B/'scripts/run_mechanism_v1.py'),'--model','InternVL']),('Qwen3',[str(B/'scripts/run_mechanism_v1.py'),'--model','Qwen3']),('analysis',[str(B/'scripts/analyze_mechanism_v1.py')])]
 try:
  for name,args in jobs:
   save(dict(state='running',job=name,updated=time.time()))
   with (O/f'{name}_stdout.log').open('a',encoding='utf-8') as out,(O/f'{name}_stderr.log').open('a',encoding='utf-8') as err:subprocess.run([sys.executable]+args,stdout=out,stderr=err,check=True)
  # Arrays remain in the raw archive tree. The lightweight zip links to its checksummed manifest.
  records=[]
  for p in O.rglob('*'):
   if p.is_file() and p.name not in ['artifact_manifest.json','mechanisms_reproducibility.zip','pipeline_status.json'] and not p.name.endswith('.log'):
    h=hashlib.sha256()
    with p.open('rb') as f:
     for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    records.append(dict(path=str(p.relative_to(O)),bytes=p.stat().st_size,sha256=h.hexdigest()))
  (O/'artifact_manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
  with zipfile.ZipFile(O/'mechanisms_reproducibility.zip','w',zipfile.ZIP_DEFLATED) as z:
   for p in O.rglob('*'):
    if p.is_file() and p.suffix!='.zip' and 'features' not in p.parts and not p.name.endswith('.log'):z.write(p,p.relative_to(O))
  save(dict(state='complete',job='all',updated=time.time(),files=len(records),raw_arrays='Per-model features folders preserved separately; hashes in artifact_manifest.json; not included in lightweight zip.'))
 except Exception:
  save(dict(state='failed',job=name,updated=time.time(),error=traceback.format_exc()));raise
if __name__=='__main__':main()
