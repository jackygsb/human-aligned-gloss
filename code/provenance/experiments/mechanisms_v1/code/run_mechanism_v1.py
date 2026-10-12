"""Reproducible frozen feature extraction and fixed candidate-direction interventions."""
from pathlib import Path
import os,json,time,argparse,hashlib,shutil,sys,subprocess
import numpy as np,pandas as pd
from PIL import Image
from mechanism_backend_v1 import Backend
B=Path(__file__).resolve().parents[1];M=B/'mllm_material';O=M/'mechanisms_v1'
def save(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
def prepare():
 O.mkdir(exist_ok=True);(O/'code').mkdir(exist_ok=True)
 d=pd.read_csv(M/'qwen3_full_v1/joint_model_predictions.csv').merge(pd.read_csv(B/'annotations/gloss_annotations.csv')[['image_id','image_512_path']],on='image_id').sort_values('source_image_index').reset_index(drop=True)
 d.to_csv(O/'stimuli.csv',index=False)
 # Balanced intervention subset selected without behavior labels: first eligible 2 IDs per shape.
 chosen=d[d.validation].groupby('shape_id',sort=True).head(2).image_id.tolist()
 if not (O/'plan.json').exists():
  save(O/'plan.json',dict(seed=20260929,n_images=3888,models=['InternVL','Qwen3'],weights='frozen; no finetuning',features='vision 25/50/75/100 percent blocks mean; bridge token mean; language same depths last prompt token, before any answer; Qwen deepstack branch means retained',task='original single-image gloss prompt; full input context recorded',raw_retention='all selected pooled vectors; full bridge token matrices for every stimulus/reference; these are selected activation records, not all neurons of every layer',direction='OLS slope of centered reference bridge mean on reference c, normalized; no human labels. Same definition in both models.',controls='unit brightness direction from non-intervention original images; 3 seeded orthogonal random directions; same norm/dose. Qwen bridge-only intervention leaves DeepStack branches intact.',intervention_images=chosen,n_intervention_images=len(chosen),intervention='72 images x 2 presentation orders; fixed reference index 27; bridge uniform additive doses -1,-0.5,0,0.5,1 reference-range slope norm; candidate, brightness, random0/1/2 directions; baseline shared',matching='12 fixed shape-diverse intervention images; start27; same A/B/SAME bisection stop rules max7; candidate +1/-1, random0 +1/-1, baseline; constrained first-token choice, not free generation; distinguish from original acquisition',probe='ridge readouts fitted on original human/physical/model targets; shape and illumination group CV; layers shown separately, no selecting best test layer; random projection for wide representations fitted without labels',scope='exploratory frozen internal-response experiments; localization/readout is not causal evidence; additive intervention measures model sensitivity and may perturb off-manifold features; no proof of human neural mechanism'))
 for name in ['run_mechanism_v1.py','mechanism_backend_v1.py','internvl_backend.py','analyze_mechanism_v1.py']:
  src=B/'scripts'/name
  if src.exists():shutil.copy2(src,O/'code'/name)
 return d
def status(name,phase,n,total,**kw):
 save(O/name/'status.json',dict(model=name,phase=phase,completed=n,total=total,time=time.time(),pid=os.getpid(),**kw));print(name,phase,n,total,flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True,choices=['InternVL','Qwen3']);ap.add_argument('--limit',type=int,default=0);ap.add_argument('--features-only',action='store_true');args=ap.parse_args();name=args.model;d=prepare();out=O/name;out.mkdir(exist_ok=True)
 plan=read(O/'plan.json');prompts=read(M/'protocol/prompts.json');match=read(M/'matching_full_v1/protocol.json')
 status(name,'loading',0,3888)
 backend=Backend(name,B);save(out/'layer_spec.json',backend.layer_spec)
 import torch,transformers
 save(out/'environment.json',dict(torch=torch.__version__,transformers=transformers.__version__,python=sys.version,gpu=torch.cuda.get_device_name(0),layer_spec=backend.layer_spec))
 (out/'requirements.lock.txt').write_text(subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True).stdout,encoding='utf-8')
 modeldir=B/'models'/('internvl35_4b' if name=='InternVL' else 'qwen3vl4b')
 save(out/'model_files.json',[dict(name=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in modeldir.iterdir() if p.is_file() and p.suffix in ['.json','.py','.safetensors']])
 records=[dict(key=f'reference_{k:02d}',path=f'mllm_material/reference_reconstruction_v3/output/reference_{k:02d}.png',c=(k-1)*.1487/51) for k in range(1,53)]+[dict(key=r.image_id,path=r.image_512_path) for r in d.itertuples()]
 if args.limit:records=records[:args.limit]
 t0=time.time()
 for i,r in enumerate(records):
  if (O/'STOP').exists():raise RuntimeError('STOP requested')
  p=out/'features'/f"{r['key']}.npz";p.parent.mkdir(exist_ok=True)
  if not p.exists():
   prepared=backend.prepare([r['path']],prompts['T1'],prompts['system']);logits,features=backend.forward(prepared,capture=True)
   rgb=np.asarray(Image.open(B/r['path']).convert('RGB'),dtype=float)/255;lum=rgb@[.2126,.7152,.0722]
   np.savez_compressed(p,**features,last_token_logits=logits)
   save(p.with_suffix('.json'),dict(**r,image_sha256=digest(B/r['path']),rendered_prompt=prepared['text'],image_token_counts=backend.image_counts,brightness=float(lum.mean()),contrast=float(lum.std()),top_token_id=int(logits.argmax()),top_token_text=backend.tok.decode([int(logits.argmax())])))
  if (i+1)%50==0 or i==len(records)-1:status(name,'features',i+1,len(records),elapsed=time.time()-t0)
 if args.limit or args.features_only:return
 refs=np.stack([np.load(out/'features'/f'reference_{k:02d}.npz')['bridge'] for k in range(1,53)]);cc=np.linspace(0,1,52);slope=(cc-cc.mean())@(refs-refs.mean(0))/np.sum((cc-cc.mean())**2);scale=float(np.linalg.norm(slope));direction=slope/scale
 original=np.stack([np.load(out/'features'/f'{id}.npz')['bridge'] for id in d.image_id]);br=np.array([read(out/'features'/f'{id}.json')['brightness'] for id in d.image_id]);train=~d.image_id.isin(plan['intervention_images']).to_numpy();v=(br[train]-br[train].mean())@(original[train]-original[train].mean(0));v/=np.linalg.norm(v)
 dirs={'candidate':direction,'brightness':v};rng=np.random.default_rng(plan['seed'])
 for k in range(3):
  r=rng.normal(size=len(direction));r-=r@direction*direction;r/=np.linalg.norm(r);dirs['random'+str(k)]=r
 np.savez_compressed(out/'directions.npz',**dirs,reference_means=refs,reference_scale=scale,reference_c=cc*.1487)
 save(out/'direction_metadata.json',dict(scale=scale,candidate_brightness_cosine=float(direction@v),selection='reference sequence only; candidate definition unchanged between models',deepstack_intervened=False))
 specs=[('baseline',0)]+[(k,a) for k in dirs for a in [-1.,-.5,.5,1.]];selected=d[d.image_id.isin(plan['intervention_images'])];total=len(selected)*2*len(specs);done=0
 for row in selected.itertuples():
  for order in [0,1]:
   paths=[row.image_512_path,'mllm_material/reference_reconstruction_v3/output/reference_27.png'];paths=paths if order==0 else paths[::-1];prepared=backend.prepare(paths,match['prompt'],match['system'])
   for kind,dose in specs:
    path=out/'interventions'/f'{row.image_id}_order{order}_{kind}_{dose:+.1f}.json'
    if not path.exists():
     result=backend.choice(prepared,None if kind=='baseline' else (dirs[kind],dose*scale,order));save(path,dict(image_id=row.image_id,paths=paths,rendered_prompt=prepared['text'],order=order,direction=kind,dose=dose,activation_l2_per_token=abs(dose)*scale if kind!='baseline' else 0,**result))
    done+=1
   status(name,'interventions',done,total)
 # Twelve images, one from each of twelve evenly spaced shape IDs.
 ids=selected.groupby('shape_id').head(1).iloc[::3];schemes=[('baseline',0),('candidate',-1.),('candidate',1.),('random0',-1.),('random0',1.)]
 from run_qwen3_matching12 import advance
 for ii,row in enumerate(ids.itertuples()):
  for kind,dose in schemes:
   target=out/'matching'/f'{row.image_id}_{kind}_{dose:+.1f}.json'
   if target.exists():continue
   t=dict(low=1,high=52,current=27,steps=0,complete=False);trace=[]
   while not t['complete']:
    k=t['current'];paths=[row.image_512_path,f'mllm_material/reference_reconstruction_v3/output/reference_{k:02d}.png'];prepared=backend.prepare(paths,match['prompt'],match['system']);r=backend.choice(prepared,None if kind=='baseline' else (dirs[kind],dose*scale,0));trace.append(dict(reference_index=k,**r));advance(t,r['candidate_choice'])
   save(target,dict(image_id=row.image_id,direction=kind,dose=dose,trajectory=trace,**t))
  status(name,'matching',ii+1,len(ids))
 status(name,'complete',1,1)
if __name__=='__main__':main()
