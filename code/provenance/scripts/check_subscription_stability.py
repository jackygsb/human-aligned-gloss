"""Three fixed multimodal inference calls, not autonomous delegated tasks."""
from pathlib import Path
import json,subprocess,shutil,os,time,statistics,datetime
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'mllm_material/matching_subscription_cli_v1/stability';OUT.mkdir(exist_ok=True)
def save(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def run():
    exe=shutil.which('codex');assert exe,'Codex executable not found'
    env=os.environ.copy()
    for key in ['OPENAI_API_KEY','CODEX_API_KEY']:env.pop(key,None)
    auth=subprocess.run([exe,'login','status'],capture_output=True,text=True,env=env,timeout=30)
    assert auth.returncode==0 and 'Logged in using ChatGPT' in auth.stdout+auth.stderr,'Subscription login required'
    original=BASE/'mllm_material/matching_chatgpt_web_pilot/batch_001'
    manifest=json.loads((original/'manifest.json').read_text(encoding='utf-8'));prompt=manifest['prompt']
    instruction='Use only the attached images. Do not use tools, inspect files, or browse. Attachments are ordered P1_A, P1_B, P2_A, P2_B, P3_A, P3_B, P4_A, P4_B.\n\n'+prompt
    schema={'type':'object','properties':{f'P{i}':{'type':'string','enum':['A','B','SAME']} for i in range(1,5)},'required':[f'P{i}' for i in range(1,5)],'additionalProperties':False}
    save('protocol.json',dict(created_at=datetime.datetime.now().isoformat(),requested_model='gpt-6-astra',effort='low',requests=3,pairs_per_request=4,prompt=instruction,stimuli=manifest['pairs'],route='Official CLI, ChatGPT subscription authentication; separate from web Pro',scope='Repeated fixed-batch operational stability, not human-alignment analysis',transport='Default official client transport; internal reconnects preserved'))
    rows=[]
    for repeat in range(1,4):
        folder=OUT/f'repeat_{repeat:02d}';folder.mkdir(exist_ok=True);result=folder/'result.json'
        if result.exists():rows.append(json.loads(result.read_text()));continue
        if (folder/'started.json').exists():raise RuntimeError('Uncertain previous invocation; not resent automatically')
        images=[]
        for i in range(1,5):
            for label in 'AB':
                f=folder/f'P{i}_{label}.png';shutil.copyfile(original/f.name,f);images.append(f)
        (folder/'schema.json').write_text(json.dumps(schema));(folder/'prompt.txt').write_text(instruction,encoding='utf-8')
        cmd=[exe,'exec','--ignore-user-config','--ephemeral','--skip-git-repo-check','--sandbox','read-only','--model','gpt-6-astra','-c','model_reasoning_effort=low','-C',str(folder),'--json','--output-schema',str(folder/'schema.json'),'-o',str(folder/'answer.json')]
        for image in images:cmd.extend(['-i',str(image)])
        cmd.append('-');(folder/'started.json').write_text(json.dumps({'time':time.time(),'command':cmd}));save('status.json',dict(state='running',completed=len(rows),total=3,current_repeat=repeat,updated=time.time()))
        tic=time.time()
        with (folder/'events.jsonl').open('w',encoding='utf-8') as stdout,(folder/'stderr.log').open('w',encoding='utf-8') as stderr:
            proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=stdout,stderr=stderr,env=env,text=True,encoding='utf-8')
            try:proc.communicate(instruction,timeout=360)
            except subprocess.TimeoutExpired:proc.kill();proc.wait();raise RuntimeError('Inference timed out; no automatic retry')
        events=[]
        for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines():
            try:events.append(json.loads(line))
            except json.JSONDecodeError:pass
        completed=[e for e in events if e.get('type')=='turn.completed'];tool_items=[e for e in events if e.get('item',{}).get('type') in ['command_execution','mcp_tool_call','web_search','file_change']]
        answer=json.loads((folder/'answer.json').read_text(encoding='utf-8')) if (folder/'answer.json').exists() else None
        valid=bool(proc.returncode==0 and completed and not tool_items and isinstance(answer,dict) and set(answer)==set(schema['required']) and all(v in ['A','B','SAME'] for v in answer.values()))
        row=dict(repeat=repeat,elapsed_seconds=time.time()-tic,exit_code=proc.returncode,answer=answer,valid=valid,tool_items=len(tool_items),usage=completed[-1].get('usage') if completed else None,transport_reconnect_mentions=(folder/'stderr.log').read_text(encoding='utf-8').count('retrying sampling request'))
        result.write_text(json.dumps(row,indent=2));rows.append(row);save('results.json',rows)
        if not valid:raise RuntimeError('Invalid result or tool use; stopped for review')
    passed=sum(r['answer']['P1']=='B' and r['answer']['P2']=='A' for r in rows)
    consistency={f'P{i}':len({r['answer'][f'P{i}'] for r in rows})==1 for i in range(1,5)}
    summary=dict(state='complete',requests=len(rows),valid_requests=sum(r['valid'] for r in rows),comparisons=len(rows)*4,reference_controls_correct=sum((r['answer']['P1']=='B')+(r['answer']['P2']=='A') for r in rows),reference_controls_total=2*len(rows),repeat_consistency=consistency,median_request_seconds=statistics.median(r['elapsed_seconds'] for r in rows),total_elapsed_seconds=sum(r['elapsed_seconds'] for r in rows),total_input_tokens=sum((r['usage'] or {}).get('input_tokens',0) for r in rows),total_cached_input_tokens=sum((r['usage'] or {}).get('cached_input_tokens',0) for r in rows),total_output_tokens=sum((r['usage'] or {}).get('output_tokens',0) for r in rows),requests_with_reconnects=sum(r['transport_reconnect_mentions']>0 for r in rows),all_answers=[r['answer'] for r in rows],scope='Only three fixed-batch calls. Does not establish long-run reliability, complete matching values, or human alignment.')
    save('summary.json',summary);save('status.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':
    try:run()
    except Exception as exc:save('status.json',dict(state='paused',reason=str(exc),updated=time.time()));raise
