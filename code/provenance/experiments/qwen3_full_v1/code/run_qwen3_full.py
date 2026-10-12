"""Frozen three-task full replication; durable per-response files, no human labels."""
from pathlib import Path
import os,json,time,hashlib,shutil,sys,subprocess,msvcrt,html
B=Path(__file__).resolve().parents[1];M=B/'mllm_material';O=M/'qwen3_full_v1';O.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');t.replace(p)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(8388608),b''):h.update(c)
 return h.hexdigest()
def status(state,phase,completed,total,detail=''):
 save(O/'status.json',dict(state=state,phase=phase,completed=completed,total=total,detail=detail,time=time.time(),pid=os.getpid()))
 (O/'progress.html').write_text('<meta charset="utf-8"><meta http-equiv="refresh" content="20"><style>body{font:20px system-ui;margin:40px;line-height:1.7}</style><h1>Qwen3-VL-4B 三任务全量复现</h1><p>'+html.escape(state+' · '+phase)+f'：{completed} / {total}</p><p>'+html.escape(detail)+'</p><p>计划：3888 张单图评分；2000 图对 × 两种顺序；3888 张图 × 两次参考匹配。GPT 暂停。全部原始响应逐条保存，失败暂停，不按相关性筛选。</p><a href="report.html">已有结果和分析</a>',encoding='utf-8')
 print(state,phase,completed,total,detail,flush=True)
def main():
 from run_qwen3_matching12 import advance
 prompts=read(M/'protocol/prompts.json');mcfg=read(M/'matching_full_v1/protocol.json')
 requests=[json.loads(l) for l in (M/'manifests/requests.jsonl').read_text().splitlines()]
 requests=[r for r in requests if (r['task']=='T1' and r['condition']=='main') or r['task']=='T2']
 schedule=[t for t in read(M/'matching_full_v1/schedule.json') if t['phase']=='main']
 assert len(requests)==7888 and len(schedule)==7776
 cfg=dict(model='Qwen/Qwen3-VL-4B-Instruct',seed=20260928,decoding=dict(do_sample=False,max_new_tokens=64),matching_prompt=mcfg['prompt'],matching_system=mcfg['system'],prompts={k:prompts[k] for k in ['system','T1','T2']},pilot_image_ids=read(M/'matching_qwen3vl4b_pilot12/protocol.json')['image_ids'],selection='All original stimuli/pairs/starts; flag prior pilot and original matching development images; no performance-dependent stopping',matching='Same 1..52 bisection, maximum7 decisions, final presented index conversion; A=test B=reference',catch_scope='Main three task responses only; original InternVL practice/catch are not rerun',presentation='One or two images per fresh context; native Qwen preprocessing; bf16 SDPA',source_hashes={str(p.relative_to(B)):digest(p) for p in [M/'manifests/requests.jsonl',M/'matching_full_v1/schedule.json',M/'protocol/prompts.json',B/'models/qwen3vl4b/manifest.json']})
 if not (O/'protocol.json').exists():save(O/'protocol.json',cfg);save(O/'schedule.json',schedule)
 else:assert read(O/'protocol.json')==cfg,'Frozen protocol changed'
 code=O/'code';code.mkdir(exist_ok=True)
 for name in ['run_qwen3_full.py','analyze_qwen3_full.py','run_qwen3_matching12.py']:shutil.copyfile(B/'scripts'/name,code/name)
 save(O/'code_manifest.json',{p.name:digest(p) for p in code.glob('*.py')})
 status('validating','assets',0,3888)
 for p,h in mcfg['freeze_sha256'].items():assert digest(B/p)==h
 manifest=read(B/'models/qwen3vl4b/manifest.json');save(O/'model_manifest.json',manifest)
 for name,meta in manifest['files'].items():assert digest(B/'models/qwen3vl4b'/name)==meta['sha256']
 os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
 import torch,transformers
 from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
 from PIL import Image
 torch.set_num_threads(4);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed']);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
 processor=AutoProcessor.from_pretrained(B/'models/qwen3vl4b',local_files_only=True)
 status('loading','model',0,1)
 model=Qwen3VLForConditionalGeneration.from_pretrained(B/'models/qwen3vl4b',torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True).eval()
 save(O/'environment.json',dict(torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(0),processor=processor.image_processor.to_dict(),model_revision=manifest['revision']))
 (O/'requirements.lock.txt').write_text(subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True).stdout,encoding='utf-8')
 def infer(key,paths,prompt,system,meta):
  if (O/'STOP').exists():raise RuntimeError('STOP requested')
  path=O/'raw'/f'{key}.json';path.parent.mkdir(exist_ok=True)
  if path.exists():return read(path)
  intent=path.with_suffix('.pending')
  if intent.exists():raise RuntimeError('Uncertain previous invocation '+key)
  content=[]
  ims=[Image.open(B/p).convert('RGB') for p in paths]
  for j,im in enumerate(ims):
   if len(ims)>1:content.append(dict(type='text',text='Image '+chr(65+j)))
   content.append(dict(type='image',image=im))
  content.append(dict(type='text',text=prompt));messages=[dict(role='system',content=system),dict(role='user',content=content)]
  text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
  x=processor(text=[text],images=ims,return_tensors='pt',padding=True).to('cuda')
  save(intent,dict(key=key,paths=paths,time=time.time(),meta=meta));tic=time.time()
  with torch.inference_mode():out=model.generate(**x,**cfg['decoding'])
  tokens=out[0,x.input_ids.shape[1]:].cpu().tolist();raw=processor.batch_decode(out[:,x.input_ids.shape[1]:],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
  result=dict(key=key,meta=meta,image_paths=paths,rendered_prompt=text,raw=raw,output_token_ids=tokens,input_tokens=int(x.input_ids.shape[1]),image_grid_thw=x.image_grid_thw.cpu().tolist(),elapsed_seconds=time.time()-tic)
  save(path,result);return result
 for i,r in enumerate(requests):
  result=infer(r['request_id'],r['image_paths'],prompts[r['task']],prompts['system'],r)
  try:v=json.loads(result['raw'].strip());assert v['status']=='ok';assert (type(v['glossiness'])==int and 0<=v['glossiness']<=100) if r['task']=='T1' else v['choice'] in ['A','B','tie']
  except Exception:raise RuntimeError('Invalid structured answer; retained for review: '+r['request_id'])
  if (i+1)%100==0 or i+1==len(requests):status('running','single_and_pair',i+1,len(requests))
  if i+1 in [3888,7888]:subprocess.run([sys.executable,str(B/'scripts/analyze_qwen3_full.py')],check=True)
 trials=[]
 pilot=M/'matching_qwen3vl4b_pilot12'
 for i,old in enumerate(schedule):
  t=dict(old,low=1,high=52,current=old['start'],steps=0,complete=False)
  while not t['complete']:
   key=f"{t['trial_id']}_step{t['steps']+1:02d}";ref=f"mllm_material/reference_reconstruction_v3/output/reference_{t['current']:02d}.png"
   saved=pilot/'raw'/key/'result.json'
   if saved.exists():
    result=read(saved);assert result['binding']['reference_index']==t['current'] and result['binding']['image_A']==t['image_path'] and result['binding']['prompt']==mcfg['prompt']
    path=O/'raw'/f'{key}.json'
    if not path.exists():save(path,dict(**result,reused_pilot_source=str(saved.relative_to(B))))
    answer=result['answer']
   else:
    result=infer(key,[t['image_path'],ref],mcfg['prompt'],mcfg['system'],dict(trial_id=t['trial_id'],step=t['steps']+1,reference_index=t['current'],low=t['low'],high=t['high']))
    answer=result['raw'].strip().upper().strip(' .!\n\t')
   if answer not in ['A','B','SAME']:raise RuntimeError('Invalid matching answer '+key)
   advance(t,answer)
  t['prior_pilot']=t['image_id'] in cfg['pilot_image_ids'];trials.append(t)
  if (i+1)%48==0 or i+1==len(schedule):
   save(O/'matches.json',trials);status('running','matching',i+1,len(schedule),'已有先导回答按原轨迹复用并标记')
  if (i+1)%144==0 or i+1==len(schedule):subprocess.run([sys.executable,str(B/'scripts/analyze_qwen3_full.py')],check=True)
 status('complete','all_tasks',len(schedule),len(schedule));subprocess.run([sys.executable,str(B/'scripts/analyze_qwen3_full.py')],check=True)
 save(O/'artifact_manifest.json',{str(p.relative_to(O)):digest(p) for p in O.rglob('*') if p.is_file() and p.suffix in ['.json','.csv','.py','.md'] and p.name!='artifact_manifest.json'})
if __name__=='__main__':
 lock=(O/'runner.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 try:main()
 except Exception as e:status('paused','review',0,0,str(e));raise
