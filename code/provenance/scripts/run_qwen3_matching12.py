"""Fixed 12-image replication. Inference receives images/prompts, never human labels."""
from pathlib import Path
import os, json, time, hashlib, shutil, sys, platform, subprocess, msvcrt, re
BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'mllm_material/matching_qwen3vl4b_pilot12'
GPT=BASE/'mllm_material/matching_subscription_cli_v1/matching_pilot12'
MODEL=BASE/'models/qwen3vl4b'
OUT.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,o):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def prepare():
    if not (OUT/'protocol.json').exists():
        source=read(GPT/'protocol.json');schedule=read(BASE/'mllm_material/matching_full_v1/schedule.json')
        trials=[]
        for old in read(GPT/'state.json')['trials']:
            t=next(t for t in schedule if t.get('trial_id')==old['trial_id'])
            trials.append(dict(trial_id=t['trial_id'],image_id=t['image_id'],image_path=t['image_path'],repetition=t['repetition'],start=t['start']))
        cfg=dict(model='Qwen/Qwen3-VL-4B-Instruct',image_ids=source['image_ids'],trials=trials,prompt=source['prompt'],hashes=source['hashes'],n_images=12,repetitions=2,max_steps=7,decoding=dict(do_sample=False,max_new_tokens=16),seed=20260928,precision='bfloat16',attention='sdpa',quantization=None,presentation='Two images in a fresh independent context per comparison; Image A=test, Image B=reference. Matches InternVL pairwise presentation; GPT used up to four pairs per context.',selection='Exactly the previously fixed GPT pilot image IDs and original schedule starts; no label-based selection',reference='Reconstructed reference v3, 52 levels; c=(k-1)*0.1487/51',stopping='SAME, boundary, bracket exhausted, or seven steps; retain last presented k',created=time.time(),scope='Pilot feasibility and same-image concordance; insufficient to estimate 36 shape and 36 illumination effects',source_gpt_protocol_sha256=digest(GPT/'protocol.json'))
        cfg['system']=read(BASE/'mllm_material/matching_full_v1/protocol.json')['system']
        cfg['decoding']['max_new_tokens']=64
        cfg['parser']='Trim whitespace and terminal . !, uppercase, then require exactly A/B/SAME; same normalization as InternVL'
        save(OUT/'protocol.json',cfg)
    code=OUT/'code';code.mkdir(exist_ok=True)
    for name in ['run_qwen3_matching12.py','download_mllm_models.py','compare_pilot12_baselines.py','package_matching12.py']:
        src=BASE/'scripts'/name
        if src.exists():shutil.copyfile(src,code/name)
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
def status(kind,trials,reason=''):
    done=[t for t in trials if t.get('complete')];data=dict(state=kind,reason=reason,completed_matches=len(done),total_matches=24,comparisons=sum(t.get('steps',0) for t in trials),updated=time.time(),pid=os.getpid())
    save(OUT/'status.json',data);save(OUT/'state.json',dict(trials=trials))
    import html,csv
    if done:
        with (OUT/'matches.csv').open('w',newline='',encoding='utf-8-sig') as f:
            fields=['trial_id','image_id','repetition','matching_c','final_index','stop_reason','steps'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:t[k] for k in fields} for t in done)
    (OUT/'progress.html').write_text('<meta charset="utf-8"><meta http-equiv="refresh" content="15"><title>Qwen3 光泽匹配</title><style>body{font:19px system-ui;margin:40px;line-height:1.7}</style><h1>Qwen3-VL-4B-Instruct · 12 图 × 2 次</h1><p>状态：'+html.escape(kind)+'；完整匹配 '+str(len(done))+'/24；比较回答 '+str(data['comparisons'])+'</p><p>'+html.escape(reason)+'</p><p>沿用固定图像、核心提示词、参考序列与停止规则。每次独立呈现一对图，不提供人类分数。</p><p><a href="comparison/report.html">GPT / InternVL / 小网络同图对照</a> · <a href="protocol.json">固定方案</a> · <a href="matches.csv">匹配结果</a></p>',encoding='utf-8')
    print(json.dumps(data,ensure_ascii=False),flush=True)
def run():
    cfg=read(OUT/'protocol.json');trials=[dict(t,low=1,high=52,current=t['start'],steps=0,complete=False) for t in cfg['trials']]
    try:
        status('waiting_for_weights',trials,'等待官方权重下载和哈希校验完成')
        deadline=time.time()+6*3600
        while not (MODEL/'manifest.json').exists():
            if time.time()>deadline or (OUT/'STOP').exists():raise RuntimeError('Weights not ready or STOP requested')
            time.sleep(15)
        for p,h in cfg['hashes'].items():assert digest(BASE/p)==h,'Stimulus hash mismatch'
        manifest=read(MODEL/'manifest.json')
        save(OUT/'model_manifest.json',manifest)
        for name,meta in manifest['files'].items():assert digest(MODEL/name)==meta['sha256'],'Model hash mismatch'
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch,transformers
        from transformers import Qwen3VLForConditionalGeneration,AutoProcessor
        from PIL import Image
        torch.set_num_threads(4);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
        torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
        freeze=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True)
        (OUT/'requirements.lock.txt').write_text(freeze.stdout,encoding='utf-8')
        save(OUT/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),model_revision=manifest['revision'],code_sha256={p.name:digest(p) for p in (OUT/'code').glob('*.py')}))
        status('loading_model',trials)
        processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
        model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True).eval()
        save(OUT/'processor_settings.json',processor.image_processor.to_dict())
        for t in trials:
            while not t['complete']:
                if (OUT/'STOP').exists():raise RuntimeError('STOP requested')
                key=f"{t['trial_id']}_step{t['steps']+1:02d}";folder=OUT/'raw'/key;folder.mkdir(parents=True,exist_ok=True)
                reference=f"mllm_material/reference_reconstruction_v3/output/reference_{t['current']:02d}.png"
                binding=dict(request_id=key,trial_id=t['trial_id'],image_id=t['image_id'],repetition=t['repetition'],step=t['steps']+1,reference_index=t['current'],image_A=t['image_path'],image_B=reference,prompt=cfg['prompt'])
                if (folder/'result.json').exists():result=read(folder/'result.json');assert result['binding']==binding
                else:
                    if (folder/'started.json').exists():raise RuntimeError('Uncertain previous inference; not resent automatically: '+key)
                    ims=[Image.open(BASE/p).convert('RGB') for p in (t['image_path'],reference)]
                    messages=[dict(role='system',content=cfg['system']),dict(role='user',content=[dict(type='text',text='Image A'),dict(type='image',image=ims[0]),dict(type='text',text='Image B'),dict(type='image',image=ims[1]),dict(type='text',text=cfg['prompt'])])]
                    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                    (folder/'rendered_prompt.txt').write_text(text,encoding='utf-8')
                    inputs=processor(text=[text],images=ims,return_tensors='pt',padding=True).to('cuda')
                    save(folder/'started.json',dict(time=time.time(),binding=binding));tic=time.time()
                    with torch.inference_mode():output=model.generate(**inputs,**cfg['decoding'])
                    raw=processor.batch_decode(output[:,inputs.input_ids.shape[1]:],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
                    answer=raw.strip().upper().strip(' .!\n\t');valid=answer in ('A','B','SAME')
                    result=dict(binding=binding,raw=raw,answer=answer if valid else None,valid=valid,elapsed_seconds=time.time()-tic,input_tokens=int(inputs.input_ids.shape[1]),output_token_ids=output[0,inputs.input_ids.shape[1]:].cpu().tolist(),image_grid_thw=inputs['image_grid_thw'].cpu().tolist(),pixel_values_shape=list(inputs['pixel_values'].shape),seed=cfg['seed'])
                    save(folder/'result.json',result)
                if not result['valid']:raise RuntimeError('Invalid answer; raw output saved: '+key)
                advance(t,result['answer']);status('running',trials)
        status('complete',trials)
        subprocess.run([sys.executable,str(BASE/'scripts/compare_pilot12_baselines.py')],check=True)
        subprocess.run([sys.executable,str(BASE/'scripts/package_matching12.py')],check=True)
        save(OUT/'artifact_manifest.json',{str(p.relative_to(OUT)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in OUT.rglob('*') if p.is_file() and p.name not in ('artifact_manifest.json','runner.lock') and not p.name.endswith('.log')})
    except Exception as exc:status('paused',trials,str(exc));raise
if __name__=='__main__':
    prepare()
    if '--prepare' in sys.argv:status('prepared',[]);sys.exit(0)
    lock=(OUT/'runner.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    run()
