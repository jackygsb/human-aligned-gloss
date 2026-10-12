"""Finite, subscription-authenticated visual inference experiment; no agent tools."""
from pathlib import Path
import csv, hashlib, html, json, msvcrt, os, shutil, subprocess, time
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'mllm_material/matching_subscription_cli_v1/matching_pilot12'
SOURCE=BASE/'mllm_material/matching_full_v1'
OUT.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,obj):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def ref(k):return BASE/f'mllm_material/reference_reconstruction_v3/output/reference_{k:02d}.png'
def init():
    if (OUT/'state.json').exists():return
    old=read(SOURCE/'schedule.json');ids=[]
    for t in old:
        if t.get('phase')=='main' and t.get('cohort')=='heldout_matching' and t['image_id'] not in ids:ids.append(t['image_id'])
        if len(ids)==12:break
    trials=[]
    for block in range(3):
        for rep in (1,2):
            for key in ids[block*4:block*4+4]:
                t=next(t for t in old if t.get('phase')=='main' and t['image_id']==key and t['repetition']==rep)
                trials.append(dict(trial_id=t['trial_id'],image_id=key,image_path=t['image_path'],repetition=rep,group=block*2+rep,low=1,high=52,current=t['start'],steps=0,complete=False))
    files={str(ref(k).relative_to(BASE)):hashlib.sha256(ref(k).read_bytes()).hexdigest() for k in range(1,53)}
    for t in trials:files[t['image_path']]=hashlib.sha256((BASE/t['image_path']).read_bytes()).hexdigest()
    save(OUT/'protocol.json',dict(model='gpt-6-astra',effort='low',route='Official CLI / ChatGPT subscription; distinct from web Pro',image_ids=ids,selection='First 12 unique heldout targets in existing frozen schedule, no outcome-based selection',repetitions=2,max_steps=7,batch_size=4,prompt=read(SOURCE/'protocol.json')['prompt'],hashes=files,created=time.time(),limitations=['Reconstructed references, not author original images','Four independent pairs in each fresh context; presentation differs from original human experiment','Small exploratory sample; repeated sessions are not human observers'],stop_rules=['Invalid response or tool use','Timeout or quota failure','STOP file','Uncertain prior invocation; no automatic resubmission']))
    save(OUT/'state.json',dict(next_batch=1,trials=trials))
def advance(t,a):
    k=t['current'];t['steps']+=1
    if a=='SAME':t.update(complete=True,stop_reason='explicit_match')
    else:
        if a=='A':t['low']=k+1
        else:t['high']=k-1
        if t['low']>52 or t['high']<1:t.update(complete=True,stop_reason='boundary')
        elif t['low']>t['high']:t.update(complete=True,stop_reason='bracket_exhausted')
        elif t['steps']>=7:t.update(complete=True,stop_reason='max_steps')
    if t['complete']:t.update(final_index=k,matching_c=(k-1)*.1487/51)
    else:t['current']=(t['low']+t['high'])//2
def table(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def rho(a,b):
    if len(set(a))<2 or len(set(b))<2:return None
    return float(spearmanr(a,b).statistic)
def report(kind,reason=''):
    s=read(OUT/'state.json');done=[t for t in s['trials'] if t['complete']]
    table(OUT/'matches.csv',[{k:t[k] for k in ['trial_id','image_id','repetition','matching_c','final_index','stop_reason','steps']} for t in done])
    labs={r['image_id']:r for r in csv.DictReader((SOURCE/'image_summary.csv').open(encoding='utf-8-sig'))}
    rows=[]
    for key in read(OUT/'protocol.json')['image_ids']:
        tt=[t for t in done if t['image_id']==key]
        if len(tt)!=2:continue
        lab=labs[key];rows.append(dict(image_id=key,human_c=float(lab['human_c']),physical_c=float(lab['physical_c']),internvl_c=float(lab['model_c']),model_c=sum(t['matching_c'] for t in tt)/2,repeat1_c=tt[0]['matching_c'],repeat2_c=tt[1]['matching_c'],boundary_runs=sum(t['stop_reason']=='boundary' for t in tt),shape_id=lab['shape_id'],illumination_id=lab['illumination_id']))
    table(OUT/'image_summary.csv',rows)
    results=list(OUT.glob('batch_*/result.json'));valid=[read(p) for p in results if read(p).get('valid')]
    summary=dict(state=kind,reason=reason,completed_matches=len(done),total_matches=24,paired_images=len(rows),requests=len(valid),comparisons=sum(len(r['answer']) for r in valid),input_tokens=sum((r.get('usage') or {}).get('input_tokens',0) for r in valid),output_tokens=sum((r.get('usage') or {}).get('output_tokens',0) for r in valid),updated=time.time(),pid=os.getpid())
    if len(rows)>=4:
        h=[r['human_c'] for r in rows];m=[r['model_c'] for r in rows];p=[r['physical_c'] for r in rows]
        summary.update(human_model_spearman=rho(h,m),human_physical_spearman=rho(h,p),human_internvl_same_images_spearman=rho(h,[r['internvl_c'] for r in rows]),repeat_spearman=rho([r['repeat1_c'] for r in rows],[r['repeat2_c'] for r in rows]))
        rng=np.random.default_rng(20260928);boot=[]
        for _ in range(2000):
            ix=rng.integers(0,len(rows),len(rows));v=rho([h[i] for i in ix],[m[i] for i in ix])
            if v is not None:boot.append(v)
        summary['exploratory_bootstrap_ci95']=np.quantile(boot,[.025,.975]).tolist() if boot else None
        fig,axes=plt.subplots(1,3,figsize=(12,3.6))
        for ax,x,y,xl,yl in [(axes[0],h,m,'Human matched c','GPT matched c'),(axes[1],p,m,'Physical c','GPT matched c'),(axes[2],[r['repeat1_c'] for r in rows],[r['repeat2_c'] for r in rows],'Repeat 1','Repeat 2')]:
            ax.scatter(x,y);ax.set(xlabel=xl,ylabel=yl);ax.grid(alpha=.2)
        fig.suptitle(f'Exploratory fixed sample: {len(rows)}/12 paired images');fig.tight_layout();fig.savefig(OUT/'results.png',dpi=160);plt.close(fig)
    save(OUT/'summary.json',summary)
    (OUT/'report.html').write_text('<meta charset="utf-8"><meta http-equiv="refresh" content="20"><title>完整匹配小样本</title><style>body{font:18px system-ui;margin:36px;line-height:1.6}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>GPT 完整光泽匹配：12 图 × 2 次</h1><p>订阅 CLI · gpt-6-astra · low。固定图片与提示词；逐步调整 52 档参考图。</p><p>状态：'+html.escape(kind)+'；完成 '+str(len(done))+'/24 次匹配，'+str(len(rows))+'/12 张图完成两次重复。</p><p>'+html.escape(reason)+'</p>'+('<img src="results.png">' if len(rows)>=4 else '<p>等待至少 4 张图完成配对重复后显示初步相关性。</p>')+'<pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre><p>探索性结果；参考图为重建版本，批次含最多四对图。与网页 Pro、InternVL 分开记录。边界结果完整保留。</p><a href="matches.csv">逐次匹配数据</a> · <a href="image_summary.csv">同图对照数据</a> · <a href="protocol.json">固定方案</a>',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)
def run():
    init()
    exe=shutil.which('codex');assert exe
    env=os.environ.copy()
    for k in ('OPENAI_API_KEY','CODEX_API_KEY'):env.pop(k,None)
    auth=subprocess.run([exe,'login','status'],capture_output=True,text=True,env=env,timeout=30)
    assert auth.returncode==0 and 'Logged in using ChatGPT' in auth.stdout+auth.stderr,'Subscription authentication unavailable'
    cfg=read(OUT/'protocol.json')
    for path,digest in cfg['hashes'].items():assert hashlib.sha256((BASE/path).read_bytes()).hexdigest()==digest,'Stimulus changed'
    while True:
        s=read(OUT/'state.json');pending=[t for t in s['trials'] if not t['complete']]
        if not pending:report('complete');return
        if (OUT/'STOP').exists():raise RuntimeError('STOP requested')
        active=[t for t in pending if t['group']==pending[0]['group']][:4]
        folder=OUT/f"batch_{s['next_batch']:03d}";folder.mkdir(exist_ok=True)
        keys=[f'P{i+1}' for i in range(len(active))]
        if (folder/'result.json').exists():result=read(folder/'result.json')
        else:
            if (folder/'started.json').exists():raise RuntimeError('Uncertain previous request; paused without resending')
            images=[];mapping={}
            for key,t in zip(keys,active):
                mapping[key]=dict(trial_id=t['trial_id'],reference_index=t['current'],step=t['steps']+1)
                for label,src in [('A',BASE/t['image_path']),('B',ref(t['current']))]:
                    dest=folder/f'{key}_{label}.png';shutil.copyfile(src,dest);images.append(dest)
            schema=dict(type='object',properties={k:dict(type='string',enum=['A','B','SAME']) for k in keys},required=keys,additionalProperties=False)
            prompt='Use only the attached images. Do not use tools, inspect files, or browse. Attachments are ordered '+', '.join(p.stem for p in images)+'. Each pair is independent. For each pair: '+cfg['prompt']+' Return only one JSON object with keys '+', '.join(keys)+' and values A, B, or SAME.'
            save(folder/'schema.json',schema);save(folder/'mapping.json',mapping);(folder/'prompt.txt').write_text(prompt,encoding='utf-8')
            cmd=[exe,'exec','--ignore-user-config','--ephemeral','--skip-git-repo-check','--sandbox','read-only','--model',cfg['model'],'-c','model_reasoning_effort=low','-C',str(folder),'--json','--output-schema',str(folder/'schema.json'),'-o',str(folder/'answer.json')]
            for image in images:cmd+=['-i',str(image)]
            cmd+=['-'];save(folder/'started.json',dict(time=time.time(),command=cmd));report('running',f"正在采集第 {s['next_batch']} 批比较")
            tic=time.time()
            with (folder/'events.jsonl').open('w',encoding='utf-8') as out,(folder/'stderr.log').open('w',encoding='utf-8') as err:
                proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=out,stderr=err,env=env,text=True,encoding='utf-8')
                try:proc.communicate(prompt,timeout=360)
                except subprocess.TimeoutExpired:proc.kill();proc.wait();raise RuntimeError('Timeout; no automatic resubmission')
            events=[]
            for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines():
                try:events.append(json.loads(line))
                except json.JSONDecodeError:pass
            completed=[e for e in events if e.get('type')=='turn.completed']
            tools=[e for e in events if e.get('item',{}).get('type') in ['command_execution','mcp_tool_call','web_search','file_change']]
            try:answer=read(folder/'answer.json')
            except (OSError,ValueError):answer=None
            valid=bool(proc.returncode==0 and completed and not tools and isinstance(answer,dict) and set(answer)==set(keys) and all(v in ['A','B','SAME'] for v in answer.values()))
            result=dict(valid=valid,answer=answer,exit_code=proc.returncode,elapsed_seconds=time.time()-tic,usage=completed[-1].get('usage') if completed else None,tool_items=len(tools))
            save(folder/'result.json',result)
        if not result['valid']:raise RuntimeError('Invalid response, quota/transport failure or tool use; inspect raw logs')
        mapping=read(folder/'mapping.json')
        for key,t in zip(keys,active):
            assert mapping[key]==dict(trial_id=t['trial_id'],reference_index=t['current'],step=t['steps']+1)
            advance(t,result['answer'][key])
        s['next_batch']+=1;save(OUT/'state.json',s);report('running')
if __name__=='__main__':
    import sys
    init()
    if '--prepare' in sys.argv:report('prepared');sys.exit(0)
    lock=(OUT/'runner.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    try:run()
    except Exception as exc:report('paused',str(exc));raise
