from pathlib import Path
import subprocess
S=Path(__file__).resolve().parents[1]/'mllm_material/nmi_revision_v8/source'
for name in ['main_NMI_submission_v8_red','supplement_NMI_submission_v8']:
 for exe,args in [('xelatex',['-interaction=nonstopmode','-halt-on-error',name+'.tex']),('bibtex',[name]),('xelatex',['-interaction=nonstopmode','-halt-on-error',name+'.tex']),('xelatex',['-interaction=nonstopmode','-halt-on-error',name+'.tex'])]:
  r=subprocess.run(['D:/texlive/2017/bin/win32/'+exe+'.exe']+args,cwd=S,capture_output=True)
  log=r.stdout.decode('utf-8',errors='replace');print(name,exe,r.returncode,log[-1000:],flush=True)
  if r.returncode:raise SystemExit(r.returncode)
