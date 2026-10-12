"""Local, zero-API matching queue. Browser submission remains a separate step.

init: freeze 2 test targets; next: build neutral-named attachments and prompt;
record --response response.json: validate answers, save them and update search.
No account cookies, credentials, browser endpoints, or model weights are used.
"""
from pathlib import Path
import argparse,json,shutil,hashlib,datetime
BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'mllm_material/matching_chatgpt_web_pilot/queue'
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['init','next','record']);ap.add_argument('--response',type=Path);a=ap.parse_args();OUT.mkdir(exist_ok=True)
    sf=OUT/'state.json'
    if a.command=='init':
        if sf.exists():raise SystemExit('Existing state preserved')
        s=json.loads((BASE/'mllm_material/matching_full_v1/schedule.json').read_text())
        trials=[r for r in s if r['phase']=='main'][:2]
        write(sf,dict(round=0,protocol='Batch presentation; fresh unpersonalized temporary chat per round; exploratory and separate from independent-trial data',trials=[dict(**r,low=1,high=52,current=r['start'],steps=0,complete=False) for r in trials]));return
    state=json.loads(sf.read_text(encoding='utf-8'));folder=OUT/f"round_{state['round']+1:03d}"
    if a.command=='next':
        active=[t for t in state['trials'] if not t['complete']]
        if not active:print('All trials complete');return
        folder.mkdir(exist_ok=True);manifest=[]
        for i,t in enumerate(active,1):
            paths=[t['image_path'],f"mllm_material/reference_reconstruction_v3/output/reference_{t['current']:02d}.png"]
            files=[]
            for label,p in zip('AB',paths):
                name=f'P{i}_{label}.png';shutil.copyfile(BASE/p,folder/name);files.append(dict(name=name,sha256=hashlib.sha256((BASE/p).read_bytes()).hexdigest(),source=p))
            manifest.append(dict(key=f'P{i}',trial_id=t['trial_id'],index=t['current'],files=files))
        prompt='For each pair Pi, Image A is Pi_A.png and Image B is Pi_B.png. Judge each pair separately.\n\n'+json.loads((BASE/'mllm_material/matching_full_v1/protocol.json').read_text())['prompt']+'\n\nReturn only a JSON object with keys '+', '.join(m['key'] for m in manifest)+' and values A, B, or SAME.'
        write(folder/'manifest.json',manifest);(folder/'prompt.txt').write_text(prompt,encoding='utf-8');print(folder);return
    if a.response is None:raise SystemExit('--response is required')
    manifest=json.loads((folder/'manifest.json').read_text());response=json.loads(a.response.read_text(encoding='utf-8-sig'))
    if set(response)!={m['key'] for m in manifest} or any(v not in ['A','B','SAME'] for v in response.values()):raise SystemExit('Invalid answer; state unchanged')
    if (folder/'response.json').exists():raise SystemExit('Already recorded; no duplicate advance')
    for m in manifest:
        t=next(t for t in state['trials'] if t['trial_id']==m['trial_id']);assert t['current']==m['index']
        k=t['current'];t['steps']+=1;answer=response[m['key']]
        if answer=='SAME':t['complete']=True;t['stop_reason']='explicit_match'
        else:
            if answer=='A':t['low']=k+1
            else:t['high']=k-1
            if t['low']>52 or t['high']<1:t['complete']=True;t['stop_reason']='boundary'
            elif t['low']>t['high']:t['complete']=True;t['stop_reason']='bracket_exhausted'
            elif t['steps']>=7:t['complete']=True;t['stop_reason']='max_steps'
        if t['complete']:t['matching_c']=(k-1)*.1487/51;t['final_index']=k
        else:t['current']=(t['low']+t['high'])//2
    write(folder/'response.json',dict(recorded_at=datetime.datetime.now().isoformat(),answers=response));state['round']+=1;write(sf,state);print(json.dumps(state,ensure_ascii=False))
if __name__=='__main__':main()
