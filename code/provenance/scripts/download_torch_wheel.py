from download_mllm_models import get,verified_download,BASE
import re,html,json,urllib.parse
dest=BASE/'runtime_wheels';dest.mkdir(exist_ok=True)
page=get('https://download.pytorch.org/whl/cu124/torch/').decode()
links=[html.unescape(s) for s in re.findall(r'href="([^"]+)"',page)]
link=next(s for s in links if 'torch-2.6.0%2Bcu124-cp312-cp312-win_amd64.whl' in s)
url,fragment=link.split('#');sha=fragment.split('=')[1]
url=url.replace('download.pytorch.org','download-r2.pytorch.org')
print(url,sha,flush=True)
import urllib.request
req=urllib.request.Request(url,headers={'User-Agent':'pip/25.0','Range':'bytes=0-0'})
try:
    with urllib.request.urlopen(req,timeout=120) as r:
        size=int(r.headers.get('Content-Range','/').split('/')[-1]) if r.headers.get('Content-Range') else int(r.headers['Content-Length'])
except urllib.error.HTTPError as e:
    print(e.read().decode(errors='replace')[:1000]);raise
(dest/'torch_source.json').write_text(json.dumps(dict(url=url,sha256=sha,size=size),indent=2))
verified_download(url,dest/'torch-2.6.0+cu124-cp312-cp312-win_amd64.whl',size,sha)
print('VERIFIED',flush=True)
