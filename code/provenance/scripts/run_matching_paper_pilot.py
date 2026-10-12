import os,json,time,hashlib,traceback,argparse,subprocess,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'mllm_material/matching_paper_pilot'
parser=argparse.ArgumentParser();parser.add_argument('--output',default=str(OUT));args=parser.parse_args();OUT=Path(args.output)
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['HF_MODULES_CACHE']=str(BASE/'mllm_runtime/hf_modules')
from internvl_backend import InternVLBackend
cfg=json.loads((OUT/'protocol.json').read_text());schedule=json.loads((OUT/'schedule.json').read_text())
def state(**kw):
    p=OUT/'status.tmp';p.write_text(json.dumps(dict(updated_unix=time.time(),**kw),indent=2))
    for _ in range(20):
        try:p.replace(OUT/'status.json');return
        except PermissionError:time.sleep(.1)
def main():
    for p,h in cfg['freeze_sha256'].items():assert hashlib.sha256((BASE/p).read_bytes()).hexdigest()==h
    backend=InternVLBackend(str(BASE/'models/internvl35_4b'),BASE)
    prompts={'system':cfg['system'],'MATCHING':cfg['prompt'],'CATCH':cfg['catch_instruction']}
    done=set()
    if (OUT/'all_trials.jsonl').exists():done={json.loads(l)['trial_id'] for l in (OUT/'all_trials.jsonl').read_text().splitlines()}
    with (OUT/'all_trials.jsonl').open('a',encoding='utf-8',buffering=1) as allf,(OUT/'trials.jsonl').open('a',encoding='utf-8',buffering=1) as mainf,(OUT/'decisions.jsonl').open('a',encoding='utf-8',buffering=1) as df:
        for trial in schedule:
            tid=trial['trial_id']
            if tid in done:continue
            if trial['phase']=='catch':
                raw,info=backend.run({'task':'CATCH','condition':'no_image','sample':False,'image_paths':[]},prompts)
                result={**trial,'raw':raw,'passed':raw.strip().strip('.')=='1'}
            else:
                low,high,k=1,52,trial['start'];trace=[];stop='max_steps';t0=time.time()
                for step in range(1,8):
                    paths=[trial['image_path'],f'mllm_material/reference_reconstruction_v3/output/reference_{k:02d}.png']
                    tic=time.time();raw,info=backend.run({'task':'MATCHING','condition':'main','sample':False,'image_paths':paths},prompts)
                    action={'A':'INCREASE','B':'DECREASE','SAME':'MATCH'}.get(raw.strip().upper().strip(' .!\n\t'))
                    record={**trial,'step':step,'reference_index':k,'low':low,'high':high,'action':action,'raw':raw,'elapsed_s':time.time()-tic,'image_paths':paths,**info};trace.append(record);df.write(json.dumps(record)+'\n')
                    if action is None:stop='invalid_response';break
                    if action=='MATCH':stop='explicit_match';break
                    if action=='INCREASE':low=k+1
                    else:high=k-1
                    if low>52 or high<1:stop='boundary';break
                    if low>high:stop='bracket_exhausted';break
                    if step<7:k=(low+high)//2
                result={**trial,'start_index':trial['start'],'final_presented_index':k,'matching_c':(k-1)*.1487/51 if stop!='invalid_response' else None,'stop_reason':stop,'valid':stop!='invalid_response','decisions':len(trace),'elapsed_s':time.time()-t0,'trace_indices':[r['reference_index'] for r in trace],'trace_actions':[r['action'] for r in trace]}
                if trial['phase']=='main':mainf.write(json.dumps(result)+'\n')
            allf.write(json.dumps(result)+'\n');done.add(tid);print(tid,result.get('final_presented_index',result.get('passed')),len(done),'/',len(schedule),flush=True)
            state(state='running',completed=len(done),total=len(schedule),last_trial=tid)
            if trial.get('checkpoint'):
                subprocess.run([sys.executable,str(BASE/'scripts/analyze_matching_full.py')],check=True)
    state(state='complete',completed=len(done),total=len(schedule))
    if cfg.get('full_collection'):subprocess.run([sys.executable,str(BASE/'scripts/analyze_matching_full.py')],check=True)
if __name__=='__main__':
    try:main()
    except Exception as e:state(state='failed',error=str(e));traceback.print_exc();raise
