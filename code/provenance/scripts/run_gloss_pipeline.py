"""Resumable sequential GPU workflow; fail visibly, never mark partial work complete."""
from pathlib import Path
import subprocess,sys,time,json,traceback,os
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'mllm_material';LOG=OUT/'reports';LOG.mkdir(parents=True,exist_ok=True)
os.environ['MPLCONFIGDIR']=str(BASE/'mllm_runtime/matplotlib_cache')
def state(stage,status,**kw):(LOG/'pipeline_status.json').write_text(json.dumps(dict(stage=stage,status=status,updated_unix=time.time(),**kw),indent=2))
def run(script,*args):
    stage=script+' '+' '.join(args);state(stage,'running');print(stage,flush=True)
    with (LOG/'pipeline.log').open('a',encoding='utf-8') as log:
        log.write('\n'+stage+'\n');log.flush()
        result=subprocess.run([sys.executable,str(BASE/'scripts'/script),*args],stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(stage+' failed: '+str(result.returncode))
def waitfile(path,seconds=10800):
    start=time.time()
    while not path.exists():
        if time.time()-start>seconds:raise RuntimeError('Timed out waiting for '+str(path))
        state('waiting for '+str(path),'waiting');time.sleep(10)
try:
    # Initial Qwen engineering may already be running in the interactive session.
    gate=OUT/'responses/qwen25vl3b/engineering_gate.json'
    waitfile(gate,1800)
    if not json.loads(gate.read_text())['passed']:raise RuntimeError('Qwen engineering failed')
    for model in ['qwen25vl3b','internvl35_4b']:
        waitfile(BASE/'models'/model/'manifest.json')
        if model!='qwen25vl3b':run('run_mllm_responses.py','--model',model,'--phase','engineering')
        run('run_mllm_responses.py','--model',model,'--phase','full','--main-scores-only')
        run('analyze_mllm_results.py')
        run('analyze_residual_controls.py')
        run('analyze_choice_robustness.py')
    for model in ['qwen25vl3b','internvl35_4b']:
        run('run_mllm_responses.py','--model',model,'--phase','full')
        run('run_mllm_responses.py','--model',model,'--phase','input_control')
        run('analyze_mllm_results.py')
        run('analyze_residual_controls.py')
        run('fit_gloss_spose.py','--source',model)
        run('spose_additional_checks.py','--source',model)
    waitfile(OUT/'spose/reference_jobs_complete.json',86400)
    for source in ['human_rating_induced','synthetic_1d']:run('spose_additional_checks.py','--source',source)
    run('summarize_spose_results.py')
    # Completion means all scheduled jobs succeeded, not that the hypothesis was supported.
    state('scheduled computation finished','complete')
except Exception as e:
    state('pipeline interrupted','failed',error=str(e));traceback.print_exc();sys.exit(1)
