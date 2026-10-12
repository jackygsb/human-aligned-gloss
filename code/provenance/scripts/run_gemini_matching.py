"""Budget-bounded Poe second-model pilot; no human labels enter requests."""
import os,json,time,base64,hashlib,urllib.request,urllib.error
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'mllm_material/matching_gemini31pro_v1';OUT.mkdir(exist_ok=True)
SOURCE=BASE/'mllm_material/matching_full_v1'
cfg=json.loads((SOURCE/'protocol.json').read_text());schedule=json.loads((SOURCE/'schedule.json').read_text())
KEY=os.environ.pop('POE_API_KEY');MODEL='gemini-3.1-pro'
def api(endpoint,payload=None):
    req=urllib.request.Request('https://api.poe.com/'+endpoint,data=json.dumps(payload).encode() if payload is not None else None,headers={'Authorization':'Bearer '+KEY,'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=120) as r:return json.load(r)
def balance():return api('usage/current_balance')['current_point_balance']
def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
def append(name,value):
    with (OUT/name).open('a',encoding='utf-8') as f:f.write(json.dumps(value)+'\n')
def records(name):return [json.loads(s) for s in (OUT/name).read_text().splitlines()] if (OUT/name).exists() else []
def ref(k):return f'mllm_material/reference_reconstruction_v3/output/reference_{k:02d}.png'
class BudgetStop(Exception):pass
def call(paths,metadata):
    now=balance();spent=START-now
    if spent>=2200 or now<600:raise BudgetStop()
    content=[]
    for label,p in zip(['A','B'],paths):
        content.extend([{'type':'text','text':f'Image {label}:'},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode((BASE/p).read_bytes()).decode()}}])
    content.append({'type':'text','text':cfg['prompt']})
    payload=dict(model=MODEL,stream=False,temperature=0,max_tokens=1024,messages=[{'role':'system','content':cfg['system']},{'role':'user','content':content}])
    # Provider-specific thinking controls deliberately omitted: Poe may ignore
    # them via the OpenAI-compatible API. Record returned reasoning usage.
    t0=time.time();response=api('v1/chat/completions',payload)
    raw=response['choices'][0]['message'].get('content') or ''
    normalized=raw.strip().upper().strip(' .!\n\t')
    action={'A':'INCREASE','B':'DECREASE','SAME':'MATCH'}.get(normalized)
    after=balance()
    rec={**metadata,'requested_model':MODEL,'image_paths':paths,'image_sha256':[hashlib.sha256((BASE/p).read_bytes()).hexdigest() for p in paths],'raw':raw,'action':action,'elapsed_s':time.time()-t0,'balance_before':now,'balance_after':after,'observed_balance_delta':now-after,'response':response}
    append('decisions.jsonl',rec)
    save('status.json',dict(state='running',completed_requests=len(records('decisions.jsonl')),observed_points_used=START-after,balance=after,last=metadata))
    if action is None:raise ValueError('Invalid response format; stopped without tuning prompt or retrying')
    return rec
def main():
    global START
    if (OUT/'pilot_run.json').exists():raise RuntimeError('Existing pilot preserved. Review it before restarting paid requests.')
    START=balance();save('pilot_run.json',dict(started_unix=time.time(),balance_before=START,target_max_points=2500,stop_new_request_at=2200,max_main_trials=6,model=MODEL))
    # Freeze the same full image schedule for any subsequent expansion; use three
    # first heldout targets here, pairing repetitions for interpretable pilot.
    ids=list(dict.fromkeys(t['image_id'] for t in schedule if t['phase']=='main'))[:3]
    trials=[next(t for t in schedule if t.get('image_id')==i and t.get('repetition')==r) for i in ids for r in [1,2]]
    controls=[(8,45,'B'),(45,8,'A'),(15,38,'B'),(38,15,'A')]
    newcfg={**cfg,'model':MODEL,'scope':'Second-family API feasibility pilot, followed by budget-dependent frozen-schedule validation','api_settings':{'temperature':0,'max_tokens':1024,'stream':False},'presentation':'Two original 512px PNG attachments, ordered A=test B=reference; provider visual tokenization unknown','pilot_image_ids':ids,'pilot_controls':controls,'budget_target_points':2500,'caveat':'Temperature zero does not guarantee deterministic hosted inference; returned model alias and usage preserved; no human or physical labels transmitted'}
    save('protocol.json',newcfg);save('schedule.json',schedule);save('pilot_schedule.json',trials)
    state='pilot_complete'
    try:
        for i,(a,b,expected) in enumerate(controls):
            rec=call([ref(a),ref(b)],dict(phase='control',trial_id=f'control_{i}',expected=expected))
            append('controls.jsonl',dict(trial_id=f'control_{i}',expected=expected,raw=rec['raw'],passed=rec['raw'].strip().upper().strip(' .!')==expected))
        if not all(r['passed'] for r in records('controls.jsonl')):
            state='control_review_required';return
        for trial in trials:
            lo,hi,k=1,52,trial['start'];trace=[];stop='max_steps'
            for step in range(1,8):
                rec=call([trial['image_path'],ref(k)],{**trial,'step':step,'reference_index':k,'low':lo,'high':hi});trace.append(rec)
                if rec['action']=='MATCH':stop='explicit_match';break
                if rec['action']=='INCREASE':lo=k+1
                else:hi=k-1
                if lo>52 or hi<1:stop='boundary';break
                if lo>hi:stop='bracket_exhausted';break
                if step<7:k=(lo+hi)//2
            append('trials.jsonl',{**trial,'start_index':trial['start'],'final_presented_index':k,'matching_c':(k-1)*.1487/51,'stop_reason':stop,'valid':True,'decisions':len(trace),'trace_indices':[r['reference_index'] for r in trace],'trace_actions':[r['action'] for r in trace]})
    except BudgetStop:state='paused_budget'
    finally:
        after=balance();rs=records('decisions.jsonl');cost=[r['observed_balance_delta'] for r in rs];mean=sum(cost)/len(cost) if cost else None
        save('status.json',dict(state=state,balance=after,observed_points_used=START-after,completed_requests=len(rs),completed_main_trials=len(records('trials.jsonl')),control_passes=sum(r['passed'] for r in records('controls.jsonl')),control_total=len(records('controls.jsonl')),mean_observed_points_per_request=mean,estimated_full_points_at_6_to_7_decisions_per_trial=[mean*7776*6,mean*7776*7] if mean else None,finished_unix=time.time()))
if __name__=='__main__':
    try:main()
    except Exception as e:
        save('failure.json',dict(error_type=type(e).__name__,http_status=getattr(e,'code',None),note='Stopped; no automatic retry. Inspect saved responses; credential not logged.'));raise SystemExit('Pilot stopped; see failure.json')
