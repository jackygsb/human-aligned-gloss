"""Execute real CPU plotting code; no published figure is copied to an output."""
from pathlib import Path
import argparse, subprocess, os, sys, json, time, hashlib, platform

ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--group',choices=['all','main','supplement','workflows','tables'],default='all')
    ap.add_argument('--output',type=Path,default=ROOT/'figures/regenerated')
    ap.add_argument('--pdf-python',default=sys.executable,help='Optional separate Python with pypdf/reportlab')
    args=ap.parse_args();out=args.output.resolve()
    if out==ROOT/'figures/published' or (ROOT/'figures/published') in out.parents:
        ap.error('Keep regenerated outputs separate from published figure masters.')
    out.mkdir(parents=True,exist_ok=True);logs=out/'logs';logs.mkdir(exist_ok=True)
    jobs=[]
    if args.group in ['all','main']:
        jobs.extend((f'draw_main_{n}.py',sys.executable) for n in range(1,7))
    if args.group in ['all','supplement']:jobs.append(('draw_supplement_1_10.py',sys.executable))
    if args.group in ['all','workflows']:
        jobs.extend([('draw_workflow_master.py',sys.executable),('derive_supplement_11_15.py',args.pdf_python)])
    if args.group in ['all','tables']:jobs.append(('build_supplement_tables.py',sys.executable))
    report={'started':time.strftime('%Y-%m-%dT%H:%M:%S'), 'python':sys.version,
            'platform':platform.platform(),'scope':'Re-execution of figures and table formatting from packaged numerical/activation/image inputs. Does not rerun model acquisition or all statistical estimation.',
            'published_figures_copied_to_output':False,'jobs':[]}
    env=os.environ.copy();env['OSF_REPRO_OUT']=str(out);env['MPLBACKEND']='Agg'
    env['OMP_NUM_THREADS']='1';env['OPENBLAS_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
    for script,py in jobs:
        start=time.time();log=logs/(Path(script).stem+'.log')
        print('Executing '+script,flush=True)
        with log.open('w',encoding='utf-8') as stream:
            p=subprocess.run([py,str(ROOT/'code/figures'/script)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        rec={'script':script,'returncode':p.returncode,'seconds':round(time.time()-start,2),'log':str(log.relative_to(out))}
        report['jobs'].append(rec)
        (out/'reproduction_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        if p.returncode:
            print(log.read_text(encoding='utf-8',errors='replace')[-6000:]);raise SystemExit(p.returncode)
    outputs=[]
    for p in sorted(out.rglob('*')):
        if p.is_file() and p.suffix in {'.pdf','.svg','.png','.tex'}:
            outputs.append({'path':str(p.relative_to(out)).replace('\\','/'),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    tablechecks=[]
    for p in (out/'tables').glob('S*.tex'):
        archived=ROOT/'figures/published/tables'/p.name
        norm=lambda x:x.replace('\r\n','\n').strip()
        expected=json.loads((ROOT/'figures/expected_outputs.json').read_text())
        actual=hashlib.sha256(norm(p.read_text(encoding='utf-8')).encode()).hexdigest()
        tablechecks.append({'table':p.stem,'matches_published_tex':actual==expected['tables'][p.stem]['normalized_tex_sha256']})
    report.update(completed=time.strftime('%Y-%m-%dT%H:%M:%S'),outputs=outputs,table_checks=tablechecks,
                  success=all(j['returncode']==0 for j in report['jobs']))
    (out/'reproduction_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'success':report['success'],'outputs':len(outputs),'tables_match':all(z['matches_published_tex'] for z in tablechecks),'report':str(out/'reproduction_report.json')}))
if __name__=='__main__':main()
