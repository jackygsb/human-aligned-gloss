from pathlib import Path
import re
B=Path(__file__).resolve().parents[1]
s=(B/'scripts/draw_figure1_english.py').read_text('utf-8')
s=s.replace("figure1_english_v1", "figure1_english_v2").replace('W,H=1800,2270','W,H=1800,4280').replace('227/25.4','428/25.4')
start=s.index('# b and c:');end=s.index('# f:')
middle=r'''
# b and c are aligned by task: exact inputs -> complete prompt -> real output.
import textwrap
prompts=json.loads((M/'protocol/prompts.json').read_text('utf-8'))
protocol=json.loads((M/'matching_full_v1/protocol.json').read_text('utf-8'))
sources.extend([str(M/'protocol/prompts.json'),str(M/'matching_full_v1/protocol.json')])
def wrapped(x,y,s,width=72,size=23,color=None):
    text='\n'.join(textwrap.fill(p,width=width,break_long_words=False,break_on_hyphens=False) for p in s.splitlines())
    return txt(x,y,text,size,color)
panel('b','Visual inputs',42,370,400)
panel('c','Exact instructions, responses and measurements',500,370,1258)
txt(82,424,'Archived InternVL examples',22,C['muted'])
txt(540,424,'Prompts shown in full; line breaks adjusted for readability',22,C['muted'])
for row,(y,task,key) in enumerate([(475,'01  Single-image rating','T1'),(1010,'02  Pairwise gloss choice','T2'),(1545,'03  Reference-based gloss matching','MATCHING')]):
    rect(42,y-12,1716,504,'#F7F9FA')
    txt(66,y+4,task,27,weight='bold')
    line(470,y+46,470,y+469)
    txt(512,y+50,'SYSTEM MESSAGE',20,C['I'],weight='bold')
    sy=prompts['system'] if row<2 else protocol['system']
    wrapped(512,y+80,sy,79,22)
    txt(512,y+157,'USER PROMPT',20,C['I'],weight='bold')
    p=prompts[key] if row<2 else protocol['prompt']
    wrapped(512,y+191,p,78,23)
    # Output and interpretation occupy a separate column, not part of the prompt.
    line(1341,y+49,1341,y+469)
    txt(1370,y+51,'MODEL RESPONSE',20,C['I'],weight='bold')
    if row==0:
        image_at(stim('morimoto_0001'),130,y+75,196)
        txt(228,y+284,'Image input',23,C['muted'],ha='center')
        txt(80,y+338,'One test image',25,weight='bold')
        txt(80,y+378,'No reference image',23,C['muted'])
        wrapped(1370,y+99,'{"glossiness": 20,\n "status": "ok"}',26,23,C['I'])
        arrow(1510,y+185,1510,y+224)
        txt(1370,y+246,'RECORDED MEASURE',20,C['muted'],weight='bold')
        txt(1370,y+285,'Rating = 20 / 100',25,weight='bold')
        wrapped(1370,y+339,'One apparent-gloss score for this image.',27,23)
    elif row==1:
        for j,id in enumerate(two['image_ids']):
            image_at(stim(id),76+j*188,y+94,164)
            txt(158+j*188,y+273,'Image '+chr(65+j),23,C['muted'],ha='center')
        txt(80,y+338,'Two test images',25,weight='bold')
        wrapped(80,y+378,'Repeat the pair with A/B positions reversed.',31,23,C['muted'])
        wrapped(1370,y+99,'{"choice": "B",\n "status": "ok"}',26,23,C['I'])
        arrow(1510,y+185,1510,y+224)
        txt(1370,y+246,'RECORDED MEASURE',20,C['muted'],weight='bold')
        wrapped(1370,y+285,'B appears glossier than A.',27,25)
        wrapped(1370,y+369,'Response shown without its Markdown code fence.',27,21,C['muted'])
    else:
        image_at(stim('morimoto_1889'),76,y+94,164);image_at(ref(15),264,y+94,164)
        txt(158,y+273,'A: fixed test',23,C['muted'],ha='center');txt(346,y+273,'B: reference',23,C['muted'],ha='center')
        txt(80,y+338,'Reference level k = 15',25,weight='bold')
        wrapped(80,y+378,'Keep A fixed; update B across comparisons (panel d).',31,23,C['muted'])
        txt(1370,y+99,'A',42,C['I'],weight='bold')
        wrapped(1370,y+157,'The test object appears glossier.',27,25)
        arrow(1510,y+242,1510,y+272)
        txt(1370,y+292,'CONTROLLER ACTION',20,C['muted'],weight='bold')
        wrapped(1370,y+332,'Increase reference gloss: level 15 → 34.',27,25)
        txt(512,y+377,'PRESENTATION AND CONTEXT',20,C['muted'],weight='bold')
        wrapped(512,y+410,'Image A: <test image>   Image B: <reference image>\nFresh independent context for every comparison; no human scores.',77,22)
    arrow(441,y+227,494,y+227)
    arrow(1303,y+109,1356,y+109)
txt(82,2073,'Image placeholders denote separate image attachments. Outputs and search states below are actual archived records.',22,C['muted'])
line(42,2125,1758,2125)

# d explains the full comparison-to-match controller, including non-SAME termination.
panel('d','From repeated comparisons to one complete gloss match',42,2152,1716)
txt(82,2211,'REFERENCE LIBRARY',21,C['muted'],weight='bold')
for k,x in zip([1,14,27,39,52],[82,262,442,622,802]):
    image_at(ref(k),x,2250,123)
    txt(x+61.5,2383,'k = '+str(k),22,C['muted'],ha='center')
arrow(98,2430,917,2430)
txt(82,2449,'Matte',23);txt(925,2449,'Glossy',23,ha='right')
txt(82,2490,'52 reconstructed levels; c(k) = (k − 1) × 0.1487 / 51',24)
txt(1040,2211,'RESPONSE → PROGRAM UPDATE',21,C['muted'],weight='bold')
for y,ans,meaning,rule in [(2262,'A','Test appears glossier','Raise lower bound: L = k + 1'),(2335,'B','Reference appears glossier','Lower upper bound: U = k − 1'),(2408,'SAME','Indistinguishable gloss','Accept current reference and stop')]:
    rect(1040,y-7,98,54,'#E8F2F5');txt(1089,y+4,ans,25,C['I'],weight='bold',ha='center')
    txt(1160,y-4,meaning,24,weight='bold');txt(1160,y+30,rule,22,C['muted'])
txt(1040,2490,'Next k = floor[(L + U) / 2], if the search continues',22)
txt(82,2555,'ONE RECORDED MATCH',22,C['I'],weight='bold')
txt(475,2555,'Fixed test: morimoto_1889. Initial range [L, U] = [1, 52].',23,C['muted'])
intervals=['[16, 52]','[35, 52]','[35, 42]','[39, 42]','[39, 39]','[40, 39]']
answers=['A','A','B','A','B','A']
for j,k in enumerate(trial['trace_indices']):
    x=82+j*282
    rect(x,2603,253,352,'#F2F6F7',C['line'])
    txt(x+18,2621,'STEP '+str(j+1),21,C['muted'],weight='bold')
    image_at(ref(k),x+65,2660,123)
    txt(x+126.5,2798,'Reference k = '+str(k),23,weight='bold',ha='center')
    txt(x+18,2845,'Model: '+answers[j],25,C['I'],weight='bold')
    txt(x+18,2890,'Range → '+intervals[j],22,C['muted'])
    if j<5:arrow(x+256,2740,x+280,2740)
rect(82,2980,1664,100,'#EAF3F5')
txt(105,2999,'STOP: bracket exhausted',25,C['I'],weight='bold')
txt(665,2999,'Retain last presented k = 39 → matched c = 0.1108',27,weight='bold')
txt(105,3042,'This record ended when L > U; it is a search endpoint, not an explicit SAME response.',23)
txt(82,3111,'Stopping rules: SAME, boundary, empty bracket, or 7 decisions. Invalid answers are flagged, not scored.',23,C['muted'])
txt(82,3150,'Repeat from two scheduled starts; average the two valid matching values for the image-level measure.',23,C['muted'])
line(42,3200,1758,3200)

panel('e','Acquisition scale and a replayable record for every decision',42,3227,1716)
for x,num,lab in [(82,'3,888','single-image ratings'),(516,'2,000 × 2','pairs × display orders'),(1040,'3,888 × 2','images × complete matches')]:
    txt(x,3295,num,31,C['I'],weight='bold');txt(x,3340,lab,25)
txt(82,3390,'Per model: InternVL3.5-4B and Qwen3-VL-4B-Instruct. Frozen weights; no target human ratings in prompts.',23,C['muted'])
txt(82,3430,'Save image IDs, prompt, raw response, parsed judgment, reference level, bracket, update and stop reason.',23,C['muted'])
line(42,3471,1758,3471)
'''
# Shift existing analysis panels down without altering any of their content or data.
tail=s[end:]
split=tail.index('# Preserve editable')
draw,finish=tail[:split],tail[split:]
# All positional coordinates in this block >=1428 are y coordinates.
draw=re.sub(r'(?<![\w.])(1[4-9]\d\d|2[0-2]\d\d)(?![\w.])',lambda m:str(int(m[0])+2070),draw)
# New total height = old height + 2070; retain lower panel spacing.
s=s[:start]+middle+draw+finish
s=s.replace('W,H=1800,4280','W,H=1800,4340').replace('428/25.4','434/25.4')
s=s.replace('(b) Three interfaces measure ordinal ratings, pairwise choices and comparisons against a reference. The displayed inputs and outputs are archived InternVL examples; the pairwise task includes both display orders. (c) Prompt excerpts specify the perceptual attribute and constrain the response format. Each decision uses a fresh context, without supplying target human ratings.', '(b) Aligned task rows show actual input images for single-image ratings, pairwise gloss choices and reference-based matching. (c) Each row reproduces the complete archived system message and task prompt, followed by an actual InternVL response and its interpretation. Line wrapping is editorial; wording is unchanged. Image placeholders denote separate attachments. The displayed pairwise JSON omits its Markdown code fence. Matching responses A/B/SAME are mapped by the controller, not by the model, to reference adjustments. Each decision uses a fresh context without target human ratings.')
s=s.replace('The displayed six-step trajectory is the actual first match for morimoto_1889.', 'The displayed six-step trajectory is the actual first match for morimoto_1889, with references 15, 34, 43, 38, 40 and 39, and answers A, A, B, A, B and A. The search starts with bounds [1,52]. The final A response exhausts the interval, so the last presented index 39 is retained; this is not an explicit SAME response.')
s=s.replace('plt.close(fig)', '''# Export the enlarged task explanation separately for convenient editing.
from matplotlib.transforms import Bbox
box=Bbox.from_extents(0,(H-3200)/254,W/254,(H-355)/254)
for ext in ['svg','png']:
    fig.savefig(O/f'Figure1_bcd_Task_Detail_EN.{ext}',dpi=350,facecolor='white',bbox_inches=box)
plt.close(fig)
(O/'prompts_verbatim.json').write_text(json.dumps({'single_and_pair_system':prompts['system'],'single':prompts['T1'],'pair':prompts['T2'],'matching_system':protocol['system'],'matching':protocol['prompt']},indent=2),encoding='utf-8')
''')
(B/'scripts/draw_figure1_english_v2.py').write_text(s,encoding='utf-8')
print('Expanded source written')
