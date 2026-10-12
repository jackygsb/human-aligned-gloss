"""Frozen-model forwards with explicit representations and bridge intervention hooks."""
import os,importlib
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from internvl_backend import InternVLBackend

class Backend:
 def __init__(self,name,base):
  self.name=name;self.base=base;self.features={};self.capture=False;self.change=None;self.handles=[]
  torch.set_num_threads(4);torch.manual_seed(20260929);torch.backends.cuda.matmul.allow_tf32=False
  if name=='InternVL':
   self.impl=InternVLBackend(base/'models/internvl35_4b',base);self.model=self.impl.model;self.tok=self.impl.tokenizer
   self.bridge=self.model.mlp1;vl=self.model.vision_model.encoder.layers;ll=self.model.language_model.model.layers
  else:
   self.processor=AutoProcessor.from_pretrained(base/'models/qwen3vl4b',local_files_only=True);self.tok=self.processor.tokenizer
   self.model=Qwen3VLForConditionalGeneration.from_pretrained(base/'models/qwen3vl4b',torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True).eval()
   self.bridge=self.model.model.visual.merger;vl=self.model.model.visual.blocks;ll=self.model.model.language_model.layers
   for i,mod in enumerate(self.model.model.visual.deepstack_merger_list):self.handles.append(mod.register_forward_hook(self.record('deepstack_'+str(i),'vision')))
  for p in self.model.parameters():p.requires_grad_(False)
  self.layer_spec={}
  for family,layers in [('vision',vl),('language',ll)]:
   for fraction in [.25,.5,.75,1.]:
    idx=round(len(layers)*fraction)-1;key=f'{family}_{idx+1:02d}';self.layer_spec[key]=idx
    self.handles.append(layers[idx].register_forward_hook(self.record(key,family)))
  self.handles.append(self.bridge.register_forward_hook(self.bridge_hook))

 def record(self,key,family):
  def hook(module,args,out):
   if not self.capture:return
   x=out[0] if isinstance(out,tuple) else out
   if family=='language':y=x[0,-1].float()
   else:
    y=x.reshape(-1,x.shape[-1]).float().mean(0)
   self.features[key]=y.detach().cpu().numpy()
  return hook

 def bridge_hook(self,module,args,out):
  x=out
  if self.capture:
   flat=x.reshape(-1,x.shape[-1]);self.features['bridge']=flat.float().mean(0).detach().cpu().numpy();self.features['bridge_tokens']=flat.detach().float().cpu().numpy().astype('float16')
  if self.change is not None:
   direction,amount,position=self.change
   direction=torch.as_tensor(direction,device=x.device,dtype=x.dtype)
   y=x.clone();flat=y.reshape(-1,y.shape[-1]);lo=sum(self.image_counts[:position]);hi=lo+self.image_counts[position]
   flat[lo:hi]+=direction*amount
   return y
  return out

 def prepare(self,paths,prompt,system):
  ims=[Image.open(self.base/p).convert('RGB') for p in paths]
  if self.name=='InternVL':
   mod=importlib.import_module(self.model.__class__.__module__);conv=mod.get_conv_template(self.model.template);conv.system_message=system
   question='\n'.join([(('Image '+chr(65+i)+': ') if len(ims)>1 else '')+'<image>' for i in range(len(ims))]+[prompt]);conv.append_message(conv.roles[0],question);conv.append_message(conv.roles[1],None);text=conv.get_prompt()
   for im in ims:text=text.replace('<image>','<img>'+'<IMG_CONTEXT>'*self.model.num_image_token+'</img>',1)
   tokens=self.tok(text,return_tensors='pt').to('cuda');self.model.img_context_token_id=self.tok.convert_tokens_to_ids('<IMG_CONTEXT>')
   # Match the existing native backend's explicit resize then transform.
   pixels=torch.stack([self.impl.transform(im.resize((448,448))) for im in ims]).to('cuda',dtype=torch.bfloat16)
   self.image_counts=[self.model.num_image_token]*len(ims)
   return dict(tokens=tokens,pixels=pixels,text=text)
  content=[]
  for i,im in enumerate(ims):
   if len(ims)>1:content.append(dict(type='text',text='Image '+chr(65+i)))
   content.append(dict(type='image',image=im))
  content.append(dict(type='text',text=prompt));messages=[dict(role='system',content=system),dict(role='user',content=content)]
  text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
  inputs=self.processor(text=[text],images=ims,return_tensors='pt',padding=True).to('cuda');self.image_counts=(inputs.image_grid_thw.prod(-1)//4).cpu().tolist()
  return dict(inputs=inputs,text=text)

 def forward(self,prepared,capture=False,change=None):
  self.capture=capture;self.features={};self.change=change
  with torch.inference_mode():
   if self.name=='InternVL':
    inp=prepared['tokens'];visual=self.model.extract_feature(prepared['pixels']);emb=self.model.language_model.get_input_embeddings()(inp.input_ids).clone();mask=inp.input_ids==self.model.img_context_token_id;assert int(mask.sum())==visual.shape[0]*visual.shape[1];emb[mask]=visual.reshape(-1,visual.shape[-1]);out=self.model.language_model.model(inputs_embeds=emb,attention_mask=inp.attention_mask,use_cache=False,return_dict=True);logits=self.model.language_model.lm_head(out.last_hidden_state[:,-1:]).float()[0,0]
   else:
    # Reset cached rotary offsets; every stimulus is a fresh context.
    self.model.model.rope_deltas=None
    out=self.model.model(**prepared['inputs'],use_cache=False,return_dict=True);logits=self.model.lm_head(out.last_hidden_state[:,-1:]).float()[0,0]
   result=logits.cpu().numpy()
  self.change=None;self.capture=False
  return result,self.features.copy()

 def choice(self,prepared,change=None):
  logits,_=self.forward(prepared,change=change);ids=[self.tok.encode(x,add_special_tokens=False) for x in ['A','B','SAME']]
  # Single-token candidates are verified before using a common first-token decision.
  assert all(len(x)==1 for x in ids),ids
  z=logits[[x[0] for x in ids]];p=np.exp(z-z.max());p/=p.sum();top=int(np.argmax(logits));return dict(candidate_logits=z.tolist(),candidate_probabilities=p.tolist(),top_token_id=top,top_token_text=self.tok.decode([top]),candidate_choice=['A','B','SAME'][int(z.argmax())],candidate_mass=float(np.exp(z-np.max(logits)).sum()/np.exp(logits-np.max(logits)).sum()))
