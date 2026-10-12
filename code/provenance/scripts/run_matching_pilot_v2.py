import os
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['HF_MODULES_CACHE']=str(BASE/'mllm_runtime/hf_modules')
import json,time,re,hashlib,traceback
import torch
from internvl_backend import InternVLBackend
OUT=BASE/'mllm_material/matching_pilot_v2'
cfg=json.loads((OUT/'protocol.json').read_text());manifest=json.loads((OUT/'manifest.json').read_text())
def status(**kw):
    p=OUT/'status.tmp';p.write_text(json.dumps(dict(updated_unix=time.time(),**kw),indent=2))
    for _ in range(20):
        try:p.replace(OUT/'status.json');return
        except PermissionError:time.sleep(.1)
    print('Status update deferred due to reader lock; response logs retained.',flush=True)
def main():
    for path,h in cfg['freeze_sha256'].items():
        assert hashlib.sha256((BASE/path).read_bytes()).hexdigest()==h,path
    torch.manual_seed(cfg['seed'])
    status(state='loading_model',completed=0,total=72)
    backend=InternVLBackend(str(BASE/'models/internvl35_4b'),BASE)
    prompts={'system':cfg['system'],'MATCHING':cfg['prompt']}
    previous=[]
    if (OUT/'trials.jsonl').exists():previous=[json.loads(l) for l in (OUT/'trials.jsonl').read_text().splitlines()]
    done={r['trial_id'] for r in previous};completed=len(done)
    with (OUT/'decisions.jsonl').open('a',encoding='utf-8',buffering=1) as decisions, (OUT/'trials.jsonl').open('a',encoding='utf-8',buffering=1) as trials:
        for row in manifest:
            for rep,start in enumerate(row['starts'],1):
                tid=row['image_id']+f'_r{rep}'
                if tid in done:continue
                low,high,k=1,52,start;trace=[];stop='max_steps';t0=time.time()
                for step in range(1,cfg['max_steps']+1):
                    image_paths=[row['image_path'],f'mllm_material/reference_reconstruction_v3/output/reference_{k:02d}.png']
                    req={'image_paths':image_paths,'condition':'main','task':'MATCHING','sample':False}
                    tic=time.time();raw,info=backend.run(req,prompts)
                    normalized=raw.strip().upper().strip(' .!\n\t')
                    action={'A':'INCREASE','B':'DECREASE','SAME':'MATCH'}.get(normalized)
                    record={'trial_id':tid,'image_id':row['image_id'],'repetition':rep,'step':step,'reference_index':k,'low':low,'high':high,'action':action,'raw':raw,'elapsed_s':time.time()-tic,'image_paths':image_paths,**info}
                    decisions.write(json.dumps(record)+'\n');trace.append(record)
                    status(state='running',completed=completed,total=72,current_trial=tid,step=step,last_action=action)
                    if action is None:stop='invalid_response';break
                    if action=='MATCH':stop='explicit_match';break
                    if action=='INCREASE':low=k+1
                    else:high=k-1
                    if low>52 or high<1:stop='boundary';break
                    if low>high:stop='bracket_exhausted';break
                    if step<cfg['max_steps']:k=(low+high)//2
                valid=stop!='invalid_response'
                result={'trial_id':tid,'image_id':row['image_id'],'repetition':rep,'start_index':start,'final_presented_index':k,'matching_c':(k-1)*.1487/51 if valid else None,'stop_reason':stop,'valid':valid,'decisions':len(trace),'elapsed_s':time.time()-t0,'low':low,'high':high,'trace_indices':[x['reference_index'] for x in trace],'trace_actions':[x['action'] for x in trace]}
                trials.write(json.dumps(result)+'\n');completed+=1
                print(tid,k,stop,'trials',completed,'/72',flush=True)
                status(state='running',completed=completed,total=72,last_trial=result)
        status(state='complete',completed=completed,total=72)
if __name__=='__main__':
    try:main()
    except Exception as e:status(state='failed',error=str(e));traceback.print_exc();raise
