"""Actual image-conditioned inference; append-only records; no human labels loaded."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
from pathlib import Path
os.environ['HF_MODULES_CACHE']=str(Path(__file__).resolve().parents[1]/'mllm_runtime/hf_modules')
import argparse,json,time,hashlib,re,traceback,platform
import torch
from PIL import Image
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'mllm_material'

def parse(raw,task):
    s=raw.strip()
    if s.startswith('```'):s=re.sub(r'^```(?:json)?\s*|\s*```$','',s)
    try:
        obj=json.loads(s)
        if not isinstance(obj,dict):return None,'not_object'
        key='glossiness' if task.startswith('T1') else 'choice' if task=='T2' else 'odd'
        if obj.get('status')=='unrateable' and obj.get(key) is None:return obj,'unrateable'
        if obj.get('status')!='ok':return obj,'invalid_status'
        v=obj.get(key)
        valid=(type(v)==int and 0<=v<=100) if key=='glossiness' else v in (['A','B','tie'] if key=='choice' else ['A','B','C'])
        return obj,'valid' if valid else 'invalid_value'
    except (ValueError,TypeError):return None,'invalid_json'

class QwenBackend:
    def __init__(self,path):
        from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
        self.processor=AutoProcessor.from_pretrained(path,local_files_only=True,min_pixels=256*28*28,max_pixels=512*512)
        self.model=Qwen2_5_VLForConditionalGeneration.from_pretrained(path,torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True).eval()
    def run(self,r,prompts):
        ims=[];content=[]
        for n,path in enumerate(r['image_paths']):
            if r['condition']=='no_image':continue
            im=Image.open(BASE/path).convert('RGB');ims.append(im)
            if len(r['image_paths'])>1:content.append(dict(type='text',text='Image '+chr(65+n)))
            content.append(dict(type='image',image=im))
        content.append(dict(type='text',text=prompts[r['task']]))
        msg=[dict(role='system',content=prompts['system']),dict(role='user',content=content)]
        text=self.processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
        inputs=self.processor(text=[text],images=ims or None,padding=True,return_tensors='pt').to('cuda')
        settings=dict(max_new_tokens=64,do_sample=r['sample'])
        if r['sample']:settings.update(temperature=.2,top_p=1.)
        with torch.inference_mode():out=self.model.generate(**inputs,**settings)
        raw=self.processor.batch_decode(out[:,inputs.input_ids.shape[1]:],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
        return raw,dict(input_tokens=int(inputs.input_ids.shape[1]),output_tokens=int(out.shape[1]-inputs.input_ids.shape[1]),image_grid_thw=inputs.get('image_grid_thw',torch.empty(0)).tolist())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['qwen25vl3b','internvl35_4b'],required=True);parser.add_argument('--phase',choices=['engineering','full','input_control'],default='engineering');parser.add_argument('--limit',type=int,default=0);parser.add_argument('--main-scores-only',action='store_true');parser.add_argument('--tasks',nargs='+',choices=['T1','T1b','T1c','T2','T3','T3b']);args=parser.parse_args()
    path=BASE/'models'/args.model
    if not (path/'manifest.json').exists():raise RuntimeError('Verified model manifest missing')
    prompts=json.loads((OUT/'protocol/prompts.json').read_text())
    freeze=json.loads((OUT/'protocol/freeze.json').read_text())
    audit=json.loads((OUT/'protocol/asset_audit.json').read_text())
    if not audit['image_binding_verified']:raise RuntimeError('Image binding audit not passed')
    for name,expected in freeze['sha256'].items():
        if hashlib.sha256((OUT/name).read_bytes()).hexdigest()!=expected:raise RuntimeError('Frozen input changed: '+name)
    manifest=OUT/'manifests'/dict(engineering='engineering.jsonl',full='requests.jsonl',input_control='input_control.jsonl')[args.phase]
    records=[json.loads(l) for l in manifest.read_text().splitlines()]
    if args.main_scores_only:records=[r for r in records if r['task']=='T1' and r['condition']=='main']
    if args.tasks:records=[r for r in records if r['task'] in args.tasks]
    dest=OUT/'responses'/args.model;dest.mkdir(parents=True,exist_ok=True)
    if args.phase!='engineering':
        gate=json.loads((dest/'engineering_gate.json').read_text())
        if not gate['passed']:raise RuntimeError('Engineering gate not passed')
    outfile=dest/(args.phase+'.jsonl');done={}
    if outfile.exists():
        for line in outfile.read_text().splitlines():
            r=json.loads(line);done[r['request_id']]=r
    requested_ids={r['request_id'] for r in records}
    done={k:v for k,v in done.items() if k in requested_ids}
    torch.manual_seed(20260926);torch.cuda.manual_seed_all(20260926)
    if args.model=='qwen25vl3b':backend=QwenBackend(str(path))
    else:
        from internvl_backend import InternVLBackend
        backend=InternVLBackend(str(path),BASE)
    import transformers
    metadata=dict(model=args.model,phase=args.phase,torch=torch.__version__,transformers=transformers.__version__,python=platform.python_version(),gpu=torch.cuda.get_device_name(0),precision='bfloat16',quantization=None,model_manifest_sha256=hashlib.sha256((path/'manifest.json').read_bytes()).hexdigest(),prompt_sha256=freeze['sha256']['protocol/prompts.json'])
    (dest/(args.phase+'_environment.json')).write_text(json.dumps(metadata,indent=2))
    count=0
    with outfile.open('a',encoding='utf-8',buffering=1) as f:
        for r in records:
            if r['request_id'] in done:continue
            if args.limit and count>=args.limit:break
            # A stable per-request seed makes resume independent of interruption point.
            seed=int(hashlib.sha256(r['request_id'].encode()).hexdigest()[:8],16)
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            start=time.time()
            raw,info=backend.run(r,prompts)
            parsed,status=parse(raw,r['task'])
            result=dict(r,model=args.model,raw=raw,parsed=parsed,parse_status=status,elapsed_s=time.time()-start,seed=seed,**info)
            f.write(json.dumps(result,ensure_ascii=False)+'\n');f.flush();done[r['request_id']]=result;count+=1
            if count%10==0:
                print(args.model,args.phase,len(done),'/',len(records),'last_s',round(result['elapsed_s'],2),status,flush=True)
                progress=dict(completed=len(done),expected=len(records),last_request=r['request_id'],last_task=r['task'],last_elapsed_s=result['elapsed_s'],updated_unix=time.time(),state='running')
                (dest/(args.phase+'_status.json')).write_text(json.dumps(progress,indent=2))
    status=dict(completed=len(done),expected=len(records),valid=sum(r['parse_status']=='valid' for r in done.values()),unrateable=sum(r['parse_status']=='unrateable' for r in done.values()))
    (dest/(args.phase+'_status.json')).write_text(json.dumps(status,indent=2))
    if args.phase=='engineering' and len(done)==len(records):
        rates={task:sum(r['parse_status']=='valid' for r in done.values() if r['task']==task)/96 for task in ['T1','T2','T3']}
        status.update(task_valid_rates=rates,passed=min(rates.values())>=.98)
        (dest/'engineering_gate.json').write_text(json.dumps(status,indent=2))
    print(json.dumps(status),flush=True)
if __name__=='__main__':main()
