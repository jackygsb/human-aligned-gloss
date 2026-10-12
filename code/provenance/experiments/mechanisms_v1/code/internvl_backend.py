"""Use pinned upstream InternVL chat API with separate-image placeholders."""
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel,AutoTokenizer

class InternVLBackend:
    def __init__(self,path,base):
        self.base=base
        self.tokenizer=AutoTokenizer.from_pretrained(path,trust_remote_code=True,use_fast=False,local_files_only=True)
        self.model=AutoModel.from_pretrained(path,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,use_flash_attn=False,trust_remote_code=True,local_files_only=True).eval().cuda()
        self.transform=T.Compose([T.Resize((448,448),interpolation=InterpolationMode.BICUBIC),T.ToTensor(),T.Normalize(mean=(.485,.456,.406),std=(.229,.224,.225))])
    def run(self,r,prompts):
        pixels=[];parts=[]
        for n,path in enumerate(r['image_paths']):
            if r['condition']=='no_image':continue
            im=Image.open(self.base/path).convert('RGB')
            assert im.size in [(512,512),(224,224)]
            # Native dynamic-preprocess rule selects one 448 tile for 512-square inputs.
            pixels.append(self.transform(im.resize((448,448))))
            parts.append(('Image '+chr(65+n)+': ' if len(r['image_paths'])>1 else '')+'<image>')
        parts.append(prompts[r['task']]);question='\n'.join(parts)
        self.model.system_message=prompts['system']
        tensor=torch.stack(pixels).to(dtype=torch.bfloat16,device='cuda') if pixels else None
        generation=dict(max_new_tokens=64,do_sample=r['sample'],pad_token_id=self.tokenizer.eos_token_id)
        if r['sample']:generation.update(temperature=.2,top_p=1.)
        with torch.inference_mode():
            raw=self.model.chat(self.tokenizer,tensor,question,generation,history=None,num_patches_list=[1]*len(pixels),return_history=False)
        return raw,dict(image_tiles=[1]*len(pixels),tile_size=448,output_tokens=len(self.tokenizer.encode(raw,add_special_tokens=False)),attention='upstream_eager')
