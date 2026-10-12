"""Prioritize the authorized InternVL evidence chain; preserve frozen trial content."""
from pathlib import Path
import subprocess, sys, time, json, os, traceback
BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'mllm_material'; LOG=OUT/'reports'
os.environ['MPLCONFIGDIR']=str(BASE/'mllm_runtime/matplotlib_cache')
def state(stage,status,**kwargs):
    value=dict(stage=stage,status=status,scope='InternVL priority; Qwen paused with responses preserved',updated_unix=time.time(),**kwargs)
    tmp=LOG/'pipeline_status.priority.tmp'
    tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');tmp.replace(LOG/'pipeline_status.json')
def run(script,*args):
    stage=script+' '+' '.join(args);state(stage,'running')
    with (LOG/'pipeline.log').open('a',encoding='utf-8') as log:
        log.write('\nINTERNVL PRIORITY: '+stage+'\n');log.flush()
        result=subprocess.run([sys.executable,str(BASE/'scripts'/script),*args],stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(stage+' failed: '+str(result.returncode))
def analyze():
    run('analyze_mllm_results.py')
    run('analyze_choice_robustness.py')
def checkpoint(task):
    """Prospective resource-allocation screen, not a confirmatory hypothesis test."""
    import collections
    rows=[json.loads(line) for line in (OUT/'responses/internvl35_4b/full.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['task']==task]
    valid=[r for r in rows if r['parse_status']=='valid']
    metrics=dict(n=len(rows),valid_fraction=len(valid)/max(1,len(rows)))
    if task=='T2':
        import csv
        summary=list(csv.DictReader((OUT/'statistics/pair_summary.csv').open()))
        target=next(r for r in summary if r['model']=='internvl35_4b' and r['stratum']=='discordant')
        metrics.update(follow_human=float(target['follow_human']),swap_consistency=float(target['swap_consistency']))
        passed=metrics['valid_fraction']>=.98 and metrics['follow_human']>=.55 and metrics['swap_consistency']>=.60
        thresholds=dict(valid_fraction_min=.98,follow_human_min=.55,swap_consistency_min=.60)
    else:
        counts=collections.Counter(r['parsed']['odd'] for r in valid)
        metrics['position_counts']=dict(counts)
        metrics['largest_position_fraction']=max(counts.values(),default=0)/max(1,len(valid))
        passed=metrics['valid_fraction']>=.98 and len(counts)==3 and metrics['largest_position_fraction']<=.80
        thresholds=dict(valid_fraction_min=.98,all_three_labels_required=True,largest_position_fraction_max=.80)
    result=dict(task=task,passed=passed,metrics=metrics,thresholds=thresholds,scope='exploratory resource allocation after prior model results; not an independent confirmation',updated_unix=time.time())
    (LOG/f'internvl_{task}_decision.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if not passed:
        state(task+' checkpoint: stop expansion and review method','needs_method_review',decision=result)
        sys.exit(0)
try:
    run('run_mllm_responses.py','--model','internvl35_4b','--phase','full','--tasks','T2')
    analyze()
    (LOG/'internvl_pairs_complete.json').write_text(json.dumps(dict(completed_unix=time.time(),expected_presentations=4000)),encoding='utf-8')
    checkpoint('T2')
    run('run_mllm_responses.py','--model','internvl35_4b','--phase','full','--tasks','T3','--limit','1024')
    checkpoint('T3')
    run('run_mllm_responses.py','--model','internvl35_4b','--phase','full','--tasks','T3','T3b')
    analyze()
    run('run_mllm_responses.py','--model','internvl35_4b','--phase','full')
    run('run_mllm_responses.py','--model','internvl35_4b','--phase','input_control')
    analyze()
    run('analyze_residual_controls.py')
    run('fit_gloss_spose.py','--source','internvl35_4b')
    run('spose_additional_checks.py','--source','internvl35_4b')
    run('summarize_spose_results.py')
    state('InternVL priority experiments and analyses finished; Qwen remains paused','priority_complete')
except Exception as exc:
    state('InternVL priority interrupted','failed',error=str(exc));traceback.print_exc();sys.exit(1)
