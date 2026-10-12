"""Standalone, visible-UI-only ChatGPT experiment runner, explicitly requested.
No API calls, cookie export, stealth, anti-bot bypass or automatic purchase.
Uses its own persistent browser profile. Login and challenges are user-operated.
"""
import argparse,csv,datetime,hashlib,json,msvcrt,os,random,re,shutil,time
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
ROOT=BASE/'mllm_material/chatgpt_web_autonomous_v1'
PROFILE=BASE/'.local-secrets/chatgpt_experiment_browser'
PROMPT=json.loads((BASE/'mllm_material/matching_full_v1/protocol.json').read_text())['prompt']
class Pause(Exception):pass
def stamp():return datetime.datetime.now().isoformat()
def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    for attempt in range(30):
        try:tmp.replace(path);return
        except PermissionError:time.sleep(.1)
    raise RuntimeError('Cannot atomically save progress')
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def reference(k):return f'mllm_material/reference_reconstruction_v3/output/reference_{k:02d}.png'
def init():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'state.json').exists():return
    cfg=read(BASE/'mllm_material/matching_full_v1/protocol.json');old=read(BASE/'mllm_material/matching_full_v1/schedule.json')
    labels=list(csv.DictReader((BASE/'annotations/gloss_annotations.csv').open(encoding='utf-8-sig')))
    rng=random.Random(2026092819);illums=list(range(1,37));rng.shuffle(illums);ids=[]
    for offset in [0,1]:
        for shape in range(1,37):
            illum=illums[(shape-1+offset)%36]
            pool=[r for r in labels if int(r['shape_id'])==shape and int(r['illumination_id'])==illum and r['image_id'] not in cfg['development_image_ids']]
            ids.append(rng.choice(pool)['image_id'])
    assert len(set(ids))==72
    trials=[]
    for rep in [1,2]:
        order=ids.copy();rng.shuffle(order)
        for key in order:
            r=next(r for r in old if r.get('image_id')==key and r.get('phase')=='main' and r['repetition']==rep)
            trials.append(dict(**{k:v for k,v in r.items() if k!='checkpoint'},low=1,high=52,current=r['start'],steps=0,complete=False))
    write(ROOT/'protocol.json',dict(created_at=stamp(),model='ChatGPT web Latest / Pro (UI labels; backend ID unverified)',n_images=72,repetitions=2,batch_size=4,seed=2026092819,image_ids=ids,
      prompt=PROMPT,method='Fresh unpersonalized temporary chat per batch; up to four pairs simultaneously; adaptive bisection unchanged; no human scores transmitted',
      controls='Four known reference comparisons, both image orders, before main acquisition',
      limitations=['Batched presentation is distinct from original independent-pair protocol','72 images do not support the prior 135-parameter condition model; primary outcome overall human correlation, condition effects provisional until expansion'],
      stop_rules=['Login/captcha/limit warning','UI or model mode changed','Invalid/missing JSON','Uncertain prior submission','STOP flag'],
      source_protocol_sha256=hashlib.sha256((BASE/'mllm_material/matching_full_v1/protocol.json').read_bytes()).hexdigest()))
    write(ROOT/'state.json',dict(created_at=stamp(),next_batch=1,controls_complete=False,trials=trials))

def parse_answer(text,keys):
    decoder=json.JSONDecoder();found=[]
    for match in re.finditer(r'\{',text):
        try:obj,_=decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:continue
        if isinstance(obj,dict) and set(obj)==set(keys) and all(v in ['A','B','SAME'] for v in obj.values()):found.append(obj)
    if not found:return None
    if any(x!=found[0] for x in found):raise Pause('Ambiguous multiple answer objects')
    return found[0]

def advance(t,answer):
    k=t['current'];t['steps']+=1
    if answer=='SAME':t.update(complete=True,stop_reason='explicit_match')
    else:
        if answer=='A':t['low']=k+1
        else:t['high']=k-1
        if t['low']>52 or t['high']<1:t.update(complete=True,stop_reason='boundary')
        elif t['low']>t['high']:t.update(complete=True,stop_reason='bracket_exhausted')
        elif t['steps']>=7:t.update(complete=True,stop_reason='max_steps')
    if t['complete']:t.update(final_index=k,matching_c=(k-1)*.1487/51)
    else:t['current']=(t['low']+t['high'])//2

def status(kind,reason=''):
    state=read(ROOT/'state.json');done=[t for t in state['trials'] if t['complete']]
    write(ROOT/'status.json',dict(state=kind,reason=reason,updated_at=stamp(),completed_matches=len(done),total_matches=len(state['trials']),next_batch=state['next_batch'],pid=os.getpid()))
    (ROOT/'progress.html').write_text('<meta charset="utf-8"><meta http-equiv="refresh" content="10"><title>网页实验进度</title><style>body{font:20px system-ui;margin:40px;line-height:1.8}</style><h1>ChatGPT 网页自动采集</h1><p>状态：'+kind+'</p><p>'+reason+'</p><p>已完成 '+str(len(done))+' / '+str(len(state['trials']))+' 次匹配。</p><p>浏览器窗口用于采集；无需保持 Work 推理运行。登录、验证码、限额或网页变化时暂停。</p>',encoding='utf-8')
    if done:
        with (ROOT/'matches.csv').open('w',newline='',encoding='utf-8-sig') as f:
            keys=['trial_id','image_id','repetition','matching_c','final_index','stop_reason','steps'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:t[k] for k in keys} for t in done)
    # Descriptive interim correlation only; no underdetermined condition model.
    pairs={}
    for t in done:pairs.setdefault(t['image_id'],[]).append(t['matching_c'])
    pairs={k:sum(v)/2 for k,v in pairs.items() if len(v)==2}
    if len(pairs)>=10:
        from scipy.stats import spearmanr
        labs={r['image_id']:float(r['human_gloss_score']) for r in csv.DictReader((BASE/'annotations/gloss_annotations.csv').open(encoding='utf-8-sig'))}
        write(ROOT/'interim_summary.json',dict(n_paired_images=len(pairs),human_model_spearman=float(spearmanr([labs[k] for k in pairs],list(pairs.values())).statistic),scope='Exploratory 72-image batched web protocol; incomplete until all 144 matches finish'))

def check_page(page):
    if (ROOT/'STOP').exists():raise Pause('STOP file detected')
    text=page.locator('body').inner_text(timeout=15000)
    if re.search(r'达到.{0,12}(上限|限额)|已用完|limit reached|usage limit|too many requests|verify you are human|验证您是真人|验证你是人类|检查浏览器|checking your browser',text,re.I):raise Pause('Page displays a quota or human-verification notice; user action required')
    return text

def button(page,pattern):return page.get_by_role('button',name=re.compile(pattern,re.I))
def wait_login(page,seconds):
    status('waiting_for_login','请在新打开的独立实验浏览器窗口登录并确认使用 最新 / Pro。程序不会读取或填写密码。')
    deadline=time.time()+seconds
    while time.time()<deadline:
        if (ROOT/'STOP').exists():raise Pause('Stopped while waiting for login')
        if button(page,r'^(临时聊天|Temporary chat)$').count() and button(page,r'^(临时聊天|Temporary chat)$').first.is_visible():return
        time.sleep(3)
    raise Pause('Login wait expired; rerun after normal login')

def fresh_chat(page):
    page.goto('https://chatgpt.com/',wait_until='domcontentloaded',timeout=60000)
    check_page(page)
    button(page,r'^(临时聊天|Temporary chat)$').click(timeout=20000)
    # Fresh page and UI inspection precede each action. Fail closed if renamed.
    check_page(page)
    personal=button(page,r'^(个性化|Personalized)$')
    if personal.count() and personal.first.is_visible():
        personal.first.click();page.get_by_text(re.compile(r'^不个性化|^Unpersonalized|^Not personalized',re.I)).last.click(timeout=10000)
    body=check_page(page)
    if not re.search(r'不会使用记忆|won.t use memor|without memor',body,re.I):raise Pause('Cannot verify memory/plugins/custom instructions are disabled')
    mode=button(page,r'选择 ChatGPT 模型|Choose.*model|Select.*model')
    if not mode.count() or 'Pro' not in mode.first.inner_text():raise Pause('Select Pro in the experiment browser; refusing silent model/mode change')

def prepare_batch(state,folder):
    manifest=[];folder.mkdir(parents=True,exist_ok=True)
    if not state['controls_complete']:
        for i,(a,b,expected) in enumerate([(8,45,'B'),(45,8,'A'),(15,38,'B'),(38,15,'A')],1):
            manifest.append(dict(key=f'P{i}',kind='control',expected=expected,sources=[reference(a),reference(b)]))
    else:
        for i,t in enumerate([t for t in state['trials'] if not t['complete']][:4],1):
            manifest.append(dict(key=f'P{i}',kind='main',trial_id=t['trial_id'],reference_index=t['current'],sources=[t['image_path'],reference(t['current'])]))
    files=[]
    for m in manifest:
        m['sha256']=[]
        for label,src in zip('AB',m['sources']):
            p=folder/f"{m['key']}_{label}.png";shutil.copyfile(BASE/src,p);files.append(str(p));m['sha256'].append(hashlib.sha256(p.read_bytes()).hexdigest())
    prompt='For each pair Pi, Image A is Pi_A.png and Image B is Pi_B.png. Judge each pair separately.\n\n'+PROMPT+'\n\nReturn only a JSON object with keys '+', '.join(m['key'] for m in manifest)+' and values A, B, or SAME.'
    write(folder/'manifest.json',manifest);(folder/'prompt.txt').write_text(prompt,encoding='utf-8');return manifest,files,prompt

def submit(page,folder,manifest,files,prompt):
    fresh_chat(page)
    button(page,r'^(添加文件等内容|Add files and more|Add photos & files|Add files)$').click(timeout=10000)
    check_page(page)
    with page.expect_file_chooser(timeout=20000) as chooser:
        button(page,r'添加照片和文件|Add photos.*files|Upload from computer').first.click()
    chooser.value.set_files(files,timeout=120000)
    for f in files:page.get_by_role('button',name=Path(f).name,exact=True).wait_for(state='visible',timeout=120000)
    check_page(page)
    page.get_by_role('textbox',name=re.compile('询问 ChatGPT|Ask ChatGPT',re.I)).fill(prompt)
    send=button(page,r'^(发送|Send|Send prompt)$')
    send.wait_for(state='visible',timeout=60000)
    deadline=time.time()+120
    while not send.is_enabled():
        check_page(page)
        if time.time()>deadline:raise Pause('Attachments not ready')
        time.sleep(2)
    # Intent committed before clicking: never silently replay an uncertain send.
    write(folder/'submission.json',dict(phase='send_intent',time=stamp(),url=page.url))
    send.click();write(folder/'submission.json',dict(phase='sent',time=stamp(),url=page.url))
    deadline=time.time()+600;keys=[m['key'] for m in manifest]
    while time.time()<deadline:
        body=check_page(page);obj=parse_answer(body,keys)
        stop=button(page,r'^(停止|Stop|Stop generating)$')
        if obj and not (stop.count() and stop.first.is_visible()):
            (folder/'raw_page_text.txt').write_text(body,encoding='utf-8');page.screenshot(path=str(folder/'response.png'),full_page=True)
            write(folder/'response.json',dict(answers=obj,time=stamp(),url=page.url,observed_mode='Pro'));return obj
        time.sleep(3)
    raise Pause('Response timed out; saved submission must be reviewed, not automatically resent')

def apply_response(state,manifest,answer):
    if manifest[0]['kind']=='control':
        if not all(answer[m['key']]==m['expected'] for m in manifest):raise Pause('Reference controls need review; responses preserved')
        state['controls_complete']=True
    else:
        for m in manifest:
            t=next(t for t in state['trials'] if t['trial_id']==m['trial_id'])
            assert t['current']==m['reference_index'] and not t['complete'];advance(t,answer[m['key']])
    state['next_batch']+=1;write(ROOT/'state.json',state)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--browser',choices=['chrome','msedge'],default='chrome');ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--login-wait',type=int,default=3600);ap.add_argument('--max-batches',type=int,default=270);args=ap.parse_args();init()
    if args.prepare_only:status('prepared','72 张图 × 两次，等待独立浏览器运行');print(ROOT);return
    lock=(ROOT/'runner.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:raise SystemExit('Another runner already owns this queue')
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        profile=PROFILE if args.browser=='chrome' else BASE/'.local-secrets/chatgpt_experiment_edge'
        write(ROOT/'browser_run.json',dict(browser=args.browser,started_at=stamp(),pid=os.getpid()))
        context=pw.chromium.launch_persistent_context(str(profile),channel=args.browser,headless=False,viewport={'width':1280,'height':900})
        page=context.pages[0] if context.pages else context.new_page()
        try:
            page.goto('https://chatgpt.com/',wait_until='domcontentloaded',timeout=60000);wait_login(page,args.login_wait)
            for _ in range(args.max_batches):
                state=read(ROOT/'state.json')
                if state['controls_complete'] and all(t['complete'] for t in state['trials']):status('complete');return
                folder=ROOT/'batches'/f"batch_{state['next_batch']:04d}"
                if (folder/'response.json').exists():
                    apply_response(state,read(folder/'manifest.json'),read(folder/'response.json')['answers']);continue
                if (folder/'submission.json').exists():raise Pause('Uncertain prior submission: review saved batch URL before resuming; no duplicate send')
                status('running','批量比较进行中；不需要 Work 继续推理')
                manifest,files,prompt=prepare_batch(state,folder);answer=submit(page,folder,manifest,files,prompt);apply_response(state,manifest,answer);status('running');time.sleep(5)
            status('paused','本次批次数上限已到；重新启动可续跑')
        except Exception as exc:
            status('paused',str(exc)[:500]);print('Paused:',str(exc)[:500],flush=True)
            # Keep the browser open for normal user login / mode selection.
            while not (ROOT/'CLOSE_BROWSER').exists():
                try:page.wait_for_timeout(3000)
                except Exception:break
        finally:context.close()
    lock.close()
if __name__=='__main__':main()
