"""Download fixed public model files only, using immutable repository revisions."""
from pathlib import Path
import hashlib,json,urllib.request,time
BASE=Path(__file__).resolve().parents[1]
MODELS={'clip_b32':'openai/clip-vit-base-patch32','dinov2_small':'facebook/dinov2-small'}

def get(url):
    with urllib.request.urlopen(url,timeout=60) as r:return r.read()

def main():
    for name,repo in MODELS.items():
        dest=BASE/'models'/name;dest.mkdir(parents=True,exist_ok=True)
        manifest_path=dest/'manifest.json'
        if manifest_path.exists():
            print(name,'already downloaded',flush=True);continue
        info=json.loads(get('https://huggingface.co/api/models/'+repo))
        revision=info['sha'];names={f['rfilename'] for f in info['siblings']}
        weight='model.safetensors' if 'model.safetensors' in names else 'pytorch_model.bin'
        files=['config.json','preprocessor_config.json',weight]
        manifest={'repository':repo,'revision':revision,'files':{}}
        for filename in files:
            url=f'https://huggingface.co/{repo}/resolve/{revision}/{filename}'
            target=dest/filename;tmp=dest/(filename+'.part')
            print('Downloading',repo,filename,flush=True)
            if not target.exists():
                with urllib.request.urlopen(url,timeout=120) as response,open(tmp,'wb') as out:
                    while True:
                        block=response.read(4*1024*1024)
                        if not block:break
                        out.write(block)
                tmp.replace(target)
            manifest['files'][filename]={'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'url':url}
        manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        print(name,'complete',flush=True)

if __name__=='__main__':main()
