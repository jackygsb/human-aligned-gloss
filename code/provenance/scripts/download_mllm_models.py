"""Retrieve pinned public experiment models, no behavioral data is uploaded."""
from pathlib import Path
import urllib.request,json,hashlib,time,shutil
BASE=Path(__file__).resolve().parents[1]
def get(url):
    with urllib.request.urlopen(url,timeout=90) as r:return r.read()
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def verified_download(url,p,expected_size=None,expected_hash=None):
    if p.exists() and (expected_size is None or p.stat().st_size==expected_size) and (expected_hash is None or digest(p)==expected_hash):return
    tmp=p.with_name(p.name+'.part')
    if p.exists():p.replace(tmp)
    # Request bounded ranges to survive CDN/proxy truncation on large assets.
    if expected_size is None:
        with urllib.request.urlopen(url,timeout=180) as r,tmp.open('wb') as f:shutil.copyfileobj(r,f)
    else:
        while not tmp.exists() or tmp.stat().st_size<expected_size:
            start=tmp.stat().st_size if tmp.exists() else 0
            end=min(start+64*1024*1024,expected_size)-1
            for attempt in range(5):
                try:
                    req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}','User-Agent':'pip/25.0'})
                    with urllib.request.urlopen(req,timeout=180) as r:
                        if start and r.status!=206:raise RuntimeError('Server did not honor byte range')
                        with tmp.open('ab' if start else 'wb') as f:
                            while True:
                                b=r.read(1024*1024)
                                if not b:break
                                f.write(b)
                    break
                except Exception:
                    if attempt==4:raise
                    time.sleep(2)
                    start=tmp.stat().st_size if tmp.exists() else 0
            print(p.name,tmp.stat().st_size,'/',expected_size,flush=True)
    if expected_size is not None and tmp.stat().st_size!=expected_size:raise RuntimeError('Size mismatch '+str(p))
    if expected_hash is not None and digest(tmp)!=expected_hash:raise RuntimeError('SHA256 mismatch '+str(p))
    tmp.replace(p)
def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--small-only',action='store_true');parser.add_argument('--model');args=parser.parse_args()
    for name,repo in [('qwen25vl3b','Qwen/Qwen2.5-VL-3B-Instruct'),('internvl35_4b','OpenGVLab/InternVL3_5-4B'),('qwen3vl4b','Qwen/Qwen3-VL-4B-Instruct')]:
        if args.model and args.model!=name:continue
        dest=BASE/'models'/name;dest.mkdir(parents=True,exist_ok=True)
        pin=dest/'download_plan.json'
        if pin.exists():plan=json.loads(pin.read_text())
        else:
            info=json.loads(get('https://huggingface.co/api/models/'+repo))
            names=[x['rfilename'] for x in info['siblings']]
            names=[n for n in names if '/' not in n and (n.endswith(('.json','.safetensors','.txt','.py','.model')) or n in ['LICENSE','README.md'])]
            plan=dict(repository=repo,revision=info['sha'],files=names)
            pin.write_text(json.dumps(plan,indent=2))
        manifest=dict(repository=repo,revision=plan['revision'],files={})
        tree=json.loads(get(f'https://huggingface.co/api/models/{repo}/tree/{plan["revision"]}'))
        metadata={x['path']:x for x in tree}
        for n in plan['files']:
            if args.small_only and n.endswith('.safetensors'):continue
            p=dest/n;url=f"https://huggingface.co/{repo}/resolve/{plan['revision']}/{n}"
            meta=metadata[n]
            verified_download(url,p,meta['size'],meta.get('lfs',{}).get('oid'))
            print(name,n,p.stat().st_size,'verified',flush=True)
            manifest['files'][n]=dict(bytes=p.stat().st_size,sha256=digest(p))
        if not args.small_only:(dest/'manifest.json').write_text(json.dumps(manifest,indent=2))
        print(name,'COMPLETE',flush=True)
if __name__=='__main__':main()
