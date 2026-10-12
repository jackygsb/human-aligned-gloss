"""Editable, data-backed English methods plate. No simulated results."""
from pathlib import Path
import os,json,hashlib,shutil,zipfile
os.environ['OMP_NUM_THREADS']='2'
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyArrowPatch
from PIL import Image
B=Path(__file__).resolve().parents[1];M=B/'mllm_material';V=M/'nmi_revision_v8/source';O=M/'figure1_english_v5';O.mkdir(exist_ok=True)
W,H=1800,9040
C={'ink':'#202D38','muted':'#657583','line':'#CDD7DF','pale':'#F3F6F8','human':'#626A78','I':'#147D92','Q':'#CC7243','violet':'#766599'}
plt.rcParams.update({'font.family':'Arial','font.size':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'xtick.major.width':.6,'ytick.major.width':.6,'text.color':C['ink'],'axes.labelcolor':C['ink'],'xtick.color':C['muted'],'ytick.color':C['muted']})
fig=plt.figure(figsize=(180/25.4,904/25.4),facecolor='white');a=fig.add_axes([0,0,1,1]);a.set(xlim=(0,W),ylim=(H,0));a.axis('off');sources=[]
def txt(x,y,s,size=26,color=None,weight='normal',ha='left',va='top',gid=None):
 t=a.text(x,y,s,fontsize=size*72/254,color=color or C['ink'],weight=weight,ha=ha,va=va,linespacing=1.3);t.set_gid(gid);return t
def line(x1,y1,x2,y2,color=None,lw=.65):a.plot([x1,x2],[y1,y2],color=color or C['line'],lw=lw)
def rect(x,y,w,h,fc=None,ec='none',lw=.6):a.add_patch(Rectangle((x,y),w,h,facecolor=fc or C['pale'],edgecolor=ec,linewidth=lw))
def arrow(x1,y1,x2,y2,color=None,style='-',lw=.8):
 a.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=6,lw=lw,color=color or C['muted'],linestyle=style,shrinkA=1,shrinkB=1))
def panel(letter,title,x,y,w,kicker=None):
 txt(x,y,letter,34,weight='bold',gid='panel_'+letter);txt(x+40,y+2,title,29,weight='bold');
 if kicker:txt(x+40,y+43,kicker,23,C['muted'])
def axes(x,y,w,h):return fig.add_axes([x/W,1-(y+h)/H,w/W,h/H])
def image_at(p,x,y,w,h=None):
 h=h or w;ax=axes(x,y,w,h);ax.imshow(Image.open(p));ax.set_axis_off();sources.append(str(p));return ax
def ref(k):return M/f'reference_reconstruction_v3/output/reference_{k:02d}.png'
ann=pd.read_csv(M/'mechanisms_v1/stimuli.csv').set_index('image_id');sources.append(str(M/'mechanisms_v1/stimuli.csv'))
def stim(id):return B/ann.loc[id,'image_512_path']
raw=[]
with (M/'responses/internvl35_4b/full.jsonl').open(encoding='utf-8') as f:
 for l in f:
  r=json.loads(l)
  if r.get('parse_status')=='valid' and r.get('condition')=='main' and r.get('task')=='T1' and r['image_ids']==['morimoto_0001']:one=r
  if r.get('parse_status')=='valid' and r.get('task')=='T2' and not raw:raw.append(r)
two=raw[0]
trial=json.loads((M/'matching_full_v1/trials.jsonl').read_text('utf-8').splitlines()[0]);assert trial['trace_indices']==[15,34,43,38,40,39]

# a: evidence-bearing assets and the two routes to analysis.
panel('a','A shared stimulus space for human and artificial observers',42,25,1716)
txt(82,85,'Illumination',24,C['muted'])
for row,shape in enumerate([1,13,25]):
 for col,illum in enumerate([1,13,25]):
  id=ann[(ann.shape_id==shape)&(ann.illumination_id==illum)&(ann.viewpoint_id==1)].index[0]
  image_at(stim(id),82+col*100,117+row*72,69)
a.text(42,230,'Shape',fontsize=23*72/254,color=C['muted'],rotation=90,ha='center',va='center');txt(405,120,'36 shapes × 36 illuminations',30,weight='bold')
txt(405,164,'× 3 viewpoints = 3,888 images',29,weight='bold')
txt(405,218,'Physical gloss  ·  color  ·  geometry',25,C['muted']);txt(405,253,'The same test images in both models',25,C['muted'])
arrow(960,163,1012,163)
txt(1040,102,'HUMAN REFERENCE',23,C['human'],weight='bold');txt(1040,141,'Published gloss-matching judgments',27)
txt(1040,207,'FROZEN VISION–LANGUAGE MODELS',23,C['muted'],weight='bold');txt(1040,250,'InternVL3.5-4B',29,C['I'],weight='bold');txt(1400,250,'Qwen3-VL-4B',29,C['Q'],weight='bold')
line(42,343,1758,343)


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


# e: exact acquisition counts and real examples, not decorative synthetic trials.
audit=json.loads((O/'acquisition_audit.json').read_text('utf-8'))
ex=json.loads((O/'acquisition_examples.json').read_text('utf-8'))
panel('e','Large-scale acquisition: from image sets to individual responses',42,3227,1716)
rect(82,3292,1656,104,'#EAF3F5')
txt(107,3310,f"{audit['total_primary_responses']:,}",52,C['I'],weight='bold')
txt(399,3310,'recorded model responses across the three primary tasks',28,weight='bold')
txt(399,3355,'InternVL: 54,263 responses     +     Qwen3: 54,469 responses',25,C['muted'])
starts=[82,652,1222]
for x,title,subtitle in zip(starts,['Single-image ratings','Order-balanced pairwise choices','Adaptive reference matches'],['3,888 images → 3,888 ratings','2,000 pairs × 2 orders → 4,000 choices','3,888 images × 2 starts → 7,776 matches']):
    rect(x,3430,516,808,'#F7F9FA',C['line'])
    txt(x+18,3452,title,26,weight='bold')
    txt(x+18,3500,subtitle,23)
    txt(x+18,3540,'Per model',21,C['muted'])
# A genuine cross-section of the image set, paired with its recorded score.
for i,r in enumerate(ex['single']):
    row,col=divmod(i,3);x=105+col*161;y=3590+row*145
    image_at(stim(r['image_ids'][0]),x,y,110)
    txt(x+55,y+114,'Score '+str(r['parsed']['glossiness']),21,C['I'],ha='center')
txt(101,4183,'12 actual examples from the full image set',22,C['muted'])
# Three real pairs, each represented in both original and reversed positions.
txt(677,3581,'Original order',21,C['muted']);txt(938,3581,'Reversed order',21,C['muted'])
for i,pair in enumerate(ex['pairs']):
    y=3622+i*180
    for order,r in enumerate(pair):
        x=677+order*261
        for j,iid in enumerate(r['image_ids']):
            image_at(stim(iid),x+j*101,y,88)
            txt(x+44+j*101,y+92,'Image '+chr(65+j),18,C['muted'],ha='center')
        txt(x,y+123,'Choice: '+r['parsed']['choice'],23,C['I'],weight='bold')
    arrow(877,y+44,925,y+44)
txt(670,4183,'Both orders retained for every image pair',22,C['muted'])
# Three real fixed-test searches; first three steps illustrate the expanding call count.
txt(1247,3581,'Fixed test',21,C['muted']);txt(1367,3581,'Changing references and replies',21,C['muted'])
for i,item in enumerate(ex['matching']):
    y=3622+i*180;t=item['trial'];ds=item['decisions']
    image_at(stim(t['image_id']),1247,y,87)
    txt(1290.5,y+95,t['image_id'].replace('morimoto_','#'),20,C['muted'],ha='center')
    for j,r in enumerate(ds):
        x=1362+j*117
        image_at(ref(r['reference_index']),x,y,87)
        txt(x+43.5,y+96,'k = '+str(r['reference_index']),20,C['muted'],ha='center')
        txt(x+43.5,y+125,r['raw'].strip(),23,C['I'],weight='bold',ha='center')
        if j<2:arrow(x+90,y+44,x+114,y+44)
txt(1240,4183,'First 3 decisions shown; each search continues',21,C['muted'])
# The scale hierarchy makes clear why 7,776 matches require >46,000 responses.
for x,top,num in [(82,'Rating responses per model','3,888'),(652,'Pairwise responses per model','4,000')]:
    txt(x+18,4275,top,22,C['muted']);txt(x+18,4315,num,35,C['I'],weight='bold')
txt(1240,4275,'Reference-comparison responses',22,C['muted'])
txt(1240,4315,'46,375',32,C['I'],weight='bold');txt(1421,4323,'InternVL',21,C['muted'])
txt(1240,4358,'46,581',32,C['Q'],weight='bold');txt(1421,4366,'Qwen3',21,C['muted'])
# Compact, example-backed record chain.
line(82,4410,1738,4410)
txt(82,4431,'REPLAYABLE RECORD',21,C['I'],weight='bold')
for x,w,head,body in [(82,407,'Stimulus + protocol','Image IDs · order · exact prompt'),(554,548,'Decision + search state','Raw reply · reference level · interval'),(1170,568,'Measurement + termination','Parsed value · trial ID · stop reason')]:
    rect(x,4475,w,77,'#F1F5F7');txt(x+15,4485,head,24,weight='bold');txt(x+15,4519,body,21,C['muted'])
arrow(495,4514,545,4514);arrow(1111,4514,1161,4514)
txt(82,4570,'Examples: archived InternVL records. Counts: primary tasks only; pilots, practice, catch trials and interventions excluded.',21,C['muted'])
line(42,4600,1758,4600)


# f: explicit behavioral-profile RSA and cross-task validation methods.
panel('f','Behavioral geometry: representational similarity analysis (RSA) and cross-task prediction',42,4638,1716)
txt(82,4694,'Behavioral geometry = relationships among conditions, estimated independently for humans and each model.',23,C['muted'])
prof={};rdms={}
for agent in ['Human','InternVL','Qwen3']:
    pp=M/f'behavioral_spaces_v1/matrices/physical_color_adjusted_shape_{agent}_profiles_by_view.npy'
    rr=M/f'behavioral_spaces_v1/matrices/physical_color_adjusted_shape_{agent}_rdm.csv'
    sources.extend([str(pp),str(rr)]);prof[agent]=np.load(pp).mean(0);rdms[agent]=np.loadtxt(rr,delimiter=',')
    z=prof[agent];recalc=((z[:,None,:]-z[None,:,:])**2).mean(2)
    assert z.shape==(36,36) and np.allclose(recalc,rdms[agent])
joint_path=M/'qwen3_full_v1/joint_model_predictions.csv';sources.append(str(joint_path))
joint=pd.read_csv(joint_path).set_index('image_id')
for x,w,title in [(82,477,'1  Assemble matched judgments'),(623,515,'2  Adjust physical and color effects'),(1200,538,'3  Build a condition-profile table')]:
    rect(x,4750,w,394,'#F7F9FA',C['line']);txt(x+15,4770,title,24,weight='bold')
txt(97,4815,'3,888 images: human + 2 model datasets',22,C['muted'])
for x,name,col in [(227,'Human',C['human']),(330,'InternVL',C['I']),(441,'Qwen3',C['Q'])]:txt(x,4866,name,21,col,weight='bold')
for j,iid in enumerate(['morimoto_0001','morimoto_1889','morimoto_0095']):
    yy=4905+j*70;image_at(stim(iid),97,yy,59)
    for x,key in [(240,'human_c'),(352,'internvl_c'),(463,'model_c')]:txt(x,yy+18,f'{joint.loc[iid,key]:.3f}',22,ha='center')
txt(638,4815,'Apply separately to each observer',23,C['muted'])
rect(654,4860,451,50,'#EAF3F5');txt(879,4872,'Recorded gloss judgment',25,ha='center',weight='bold')
arrow(879,4914,879,4940)
txt(654,4948,'Subtract fitted physical c + chroma + hue',22)
arrow(879,4987,879,5013)
rect(654,5020,451,48,'#EAF3F5');txt(879,5030,'Adjusted judgment (residual)',24,ha='center',weight='bold')
txt(638,5087,'Fit on 2 views; apply to the remaining view.',21,C['muted'])
txt(638,5114,'Rotate through all 3 held-out viewpoints.',21,C['muted'])
txt(1215,4815,'Average across the 3 viewpoints',23,C['muted'])
def matrix(z,x,y,size,cmap='magma',relative=False):
    zz=z.copy()
    if relative:zz=zz/zz[np.triu_indices(36,1)].mean();lo,hi=0,3
    else:lo,hi=-np.max(np.abs(zz)),np.max(np.abs(zz))
    cm=plt.get_cmap(cmap)
    for r in range(36):
        for c in range(36):rect(x+c*size/36,y+r*size/36,size/36+.04,size/36+.04,cm(np.clip((zz[r,c]-lo)/(hi-lo),0,1)))
matrix(prof['Human'],1286,4880,174,'RdBu_r')
txt(1373,5065,'36 illuminations →',20,C['muted'],ha='center')
txt(1484,4891,'Each row:',23,weight='bold');wrapped(1484,4930,'one shape, judged under 36 lights',19,22)
txt(1215,5114,'36 shapes × 36 illuminations',23,C['muted'])
arrow(565,4945,614,4945);arrow(1144,4945,1192,4945)

txt(82,5188,'4  Convert profile differences into distances',27,weight='bold')
txt(82,5236,'Example: two human shape profiles',23,C['muted'])
ax=axes(155,5270,468,110)
for i,col in [(0,C['human']),(12,C['violet'])]:ax.plot(np.arange(1,37),prof['Human'][i],color=col,lw=.8,label=f'Shape {i+1}')
ax.set(xlim=(1,36),xticks=[1,18,36],xlabel='Illumination',ylabel='Adjusted judgment');ax.tick_params(labelsize=6,pad=2);ax.xaxis.label.set_size(6.5);ax.yaxis.label.set_size(6.5);ax.legend(fontsize=6,frameon=False,loc='upper right')
arrow(659,5350,728,5350)
txt(751,5243,'Shape distance',24,weight='bold')
txt(751,5284,'D(i, j) = mean over lights of [profile(i) − profile(j)]²',24)
txt(751,5326,'Similar response patterns → small distance',23,C['muted'])
txt(751,5368,'For illumination geometry, transpose the table:',23,weight='bold')
txt(751,5410,'compare each light’s 36-shape profile using the same rule.',23)
txt(82,5485,'5  RSA: compare the same condition relationships across observers',27,weight='bold')
for j,(agent,col) in enumerate([('Human',C['human']),('InternVL',C['I']),('Qwen3',C['Q'])]):
    x=82+j*222;txt(x+86,5537,agent,23,col,weight='bold',ha='center')
    matrix(rdms[agent],x,5581,172,relative=True)
    line(x,5581,x+172,5753,'#DFE5E8',.55)
txt(82,5771,'RDM = representational dissimilarity matrix',21,C['muted'])
txt(82,5803,'36 shapes; same order. Bright = larger distance.',21,C['muted'])
arrow(721,5652,794,5652)
txt(814,5537,'Read only the upper triangle of each RDM',24,weight='bold')
txt(814,5576,'36 × 35 / 2 = 630 unique condition-pair distances',23,C['muted'])
ix=np.triu_indices(36,1)
for j,(agent,col) in enumerate([('Human',C['human']),('InternVL',C['I']),('Qwen3',C['Q'])]):
    yy=5623+j*37;txt(814,yy,agent,21,col)
    vec=rdms[agent][ix];rank=pd.Series(vec).rank().to_numpy()/630
    for k,value in enumerate(rank):rect(956+k*1.2,yy,1.24,20,plt.get_cmap('magma')(value))
txt(814,5745,'Spearman correlation between corresponding distances',23,weight='bold')
txt(814,5785,'Human ↔ InternVL     Human ↔ Qwen3     InternVL ↔ Qwen3',23)
line(82,5853,1738,5853)
txt(82,5880,'6  Quantify uncertainty at the condition level',26,weight='bold')
txt(82,5925,'Permutation test',23,weight='bold')
wrapped(82,5964,'Shuffle the 36 condition identities in one RDM; move its rows and columns together. Recompute RSA 4,999 times.',65,22)
txt(82,6060,'Correct the 18 full-profile comparisons using BH-FDR.',22,C['muted'])
txt(969,5925,'Bootstrap confidence interval',23,weight='bold')
wrapped(969,5964,'Resample the 36 shapes (or lights), rebuild their pair distances, and recompute RSA 2,000 times.',62,22)
txt(969,6060,'Resample conditions, not 630 independent matrix cells.',22,C['muted'])
txt(82,6107,'The same RSA pipeline is applied to raw, physical/color-adjusted and interaction-only profiles.',22,C['muted'])
txt(82,6140,'Interaction-only profiles additionally remove the row and column means; no human-to-model coordinate fit.',22,C['muted'])
line(42,6190,1758,6190)

# Keep the second analysis logically separate: consistency across questions, within each model.
txt(82,6220,'CROSS-TASK PREDICTION: do different questions recover consistent gloss judgments?',27,weight='bold')
# This explanatory band extends the figure below the RSA flow.
rect(82,6275,803,485,'#F7F9FA',C['line']);rect(935,6275,803,485,'#F7F9FA',C['line'])
txt(101,6294,'A  Predict ratings from matches, and vice versa',25,weight='bold')
txt(101,6337,'Split by shape OR illumination into 12 grouped folds',22,C['muted'])
for k in range(12):
    x=102+k*61;rect(x,6380,53,31,'#CC7243' if k==11 else '#A3CCD3');txt(x+26.5,6385,str(k+1),18,ha='center')
txt(101,6425,'Fit mapping on 11 folds',23,C['I'],weight='bold')
arrow(465,6439,547,6439)
txt(569,6425,'Predict 1 test fold',23,C['Q'],weight='bold')
txt(101,6478,'Single-image rating → predicted reference match',23)
txt(101,6518,'Reference match → predicted single-image rating',23)
txt(101,6565,'Use a monotonic mapping; rotate the held-out group.',22)
txt(101,6610,'Compare predictions with recorded targets: R² and Spearman ρ.',22,weight='bold')
txt(101,6697,'Run independently for InternVL and Qwen3.',22,C['muted'])
txt(954,6294,'B  Predict which object wins a pairwise choice',25,weight='bold')
for j,iid in enumerate(two['image_ids']):
    image_at(stim(iid),966+j*145,6350,105)
    txt(1018.5+j*145,6468,'Object '+chr(65+j),21,C['muted'],ha='center')
arrow(1236,6403,1290,6403)
txt(1310,6360,'Score(A) − Score(B)',24,weight='bold')
txt(1310,6400,'or Match(A) − Match(B)',23)
txt(1310,6440,'Sign predicts the choice',22,C['muted'])
txt(954,6525,'Compare with recorded A/B choices; retain both display orders.',22)
txt(954,6565,'Evaluate all 2,000 pairs and predefined subsets.',22,weight='bold')
wrapped(954,6610,'Human–physical conflict subset: human judgments and physical gloss rank the two objects in opposite directions.',66,22)
txt(954,6697,'Question: does the model follow the human ordering?',22,C['muted'])
line(42,6805,1758,6805)

# Preserve existing g/h content, with a vertical translation only.
_txt,_line,_rect,_arrow,_axes=txt,line,rect,arrow,axes
SHIFT=3340
def txt(x,y,*args,**kwargs):return _txt(x,y+SHIFT,*args,**kwargs)
def line(x1,y1,x2,y2,*args,**kwargs):return _line(x1,y1+SHIFT,x2,y2+SHIFT,*args,**kwargs)
def rect(x,y,*args,**kwargs):return _rect(x,y+SHIFT,*args,**kwargs)
def arrow(x1,y1,x2,y2,*args,**kwargs):return _arrow(x1,y1+SHIFT,x2,y2+SHIFT,*args,**kwargs)
def axes(x,y,*args,**kwargs):return _axes(x,y+SHIFT,*args,**kwargs)

# g: archived frozen activations, candidate spatial responses and separate readout fitting.
# Undo the earlier positional translation; use full-canvas coordinates for the expanded panel.
txt,line,rect,arrow,axes=_txt,_line,_rect,_arrow,_axes
panel('g','Inside the frozen models: visual features, language context and gloss readout',42,6838,1716)
txt(82,6892,'One forward pass with the original single-image gloss prompt; record internal states before the answer is generated.',23,C['muted'])
image_at(stim('morimoto_0095'),82,7021,134)
txt(149,7172,'Input image',23,C['muted'],ha='center')
arrow(226,7090,265,7090)
for x,w,head,sub in [(278,388,'VISION ENCODER','Encode the image into visual tokens'),(742,412,'BRIDGE','Convert visual features to LM tokens'),(1230,508,'LANGUAGE MODEL','Combine image tokens with the prompt')]:
    rect(x,6980,w,319,'#F7F9FA',C['line']);txt(x+16,7000,head,28,weight='bold');txt(x+16,7044,sub,21,C['muted'])
# Schematic layer stacks, with actual sampled block indices and dimensionalities.
for j,label in enumerate(['6','12','18','24']):
    rect(300+j*82,7093,68,48,'#DCEBEF',C['I']);txt(334+j*82,7105,label,23,ha='center')
txt(294,7163,'Record blocks 6, 12, 18 and 24',22)
txt(294,7210,'Token mean → 1,024-value vector',21)
txt(294,7254,'Image features; frozen visual encoder',21,C['muted'])
arrow(674,7117,733,7117)
txt(758,7090,'InternVL: MLP projector',23,weight='bold')
txt(758,7130,'Qwen3: visual token merger',23,weight='bold')
txt(758,7174,'Saved example: 256 × 2,560 values',22)
txt(758,7210,'Mean tokens → 2,560-value vector',22)
txt(758,7254,'Keep the full 16 × 16 spatial token grid',21,C['muted'])
arrow(1162,7117,1221,7117)
rect(1250,6929,468,37,'#EAF3F5');txt(1484,6937,'Text prompt → token embeddings',22,ha='center')
arrow(1484,6968,1484,6979)
for j,label in enumerate(['9','18','27','36']):
    rect(1252+j*114,7093,92,48,'#EEEAF4',C['violet']);txt(1298+j*114,7105,label,23,ha='center')
txt(1246,7163,'Record blocks 9, 18, 27 and 36',22)
txt(1246,7210,'Last prompt-token state → 2,560 values',22)
txt(1246,7254,'Image + instruction context; before output',21,C['muted'])
arrow(472,7300,472,7355,C['Q'],style='--');line(472,7355,1484,7355,C['Q']);arrow(1484,7355,1484,7300,C['Q'],style='--')
txt(994,7320,'Qwen3 DeepStack: additional multi-level visual inputs',22,C['Q'],ha='center')

txt(82,7414,'SAVED FEATURE VECTORS',24,weight='bold')
txt(82,7460,'Same input image; actual archived activations',22,C['muted'])
feature=np.load(O/'g_internal_feature_examples.npz')
txt(415,7504,'InternVL',23,C['I'],weight='bold',ha='center');txt(811,7504,'Qwen3',23,C['Q'],weight='bold',ha='center')
for j,(key,lab) in enumerate([('vision_24','Vision 24'),('bridge','Bridge mean'),('language_36','Language 36')]):
    yy=7550+j*67;txt(82,yy,lab,21)
    for x,model in [(246,'InternVL'),(642,'Qwen3')]:
        vv=feature[model+'_'+key][:32].astype(float);vv=vv/max(np.abs(vv).max(),1e-9)
        for k,val in enumerate(vv):rect(x+k*10.6,yy,10.65,35,plt.get_cmap('RdBu_r')((val+1)/2))
txt(82,7773,'Each strip shows 32 coordinates; each vector has its own scale.',21,C['muted'])
txt(82,7810,'Vision / bridge: token mean. Language: final prompt token.',22)
line(1012,7420,1012,7835)
txt(1046,7414,'GLOSS-RELATED CANDIDATE MAP',24,weight='bold')
txt(1046,7460,'52 reference images → fitted bridge direction d',22)
txt(1046,7497,'At each spatial token p: response = h(p) · d',22,C['muted'])
for j,(lab,path,col) in enumerate([('Input',stim('morimoto_0095'),C['muted']),('InternVL',stim('morimoto_0095'),C['I']),('Qwen3',stim('morimoto_0095'),C['Q'])]):
    x=1068+j*228;txt(x+83,7542,lab,23,col,ha='center');ax=image_at(path,x,7580,166)
    if j:
        model=lab;dp=M/f'mechanisms_v1/{model}/directions.npz';sources.append(str(dp));direction=np.load(dp)['candidate']
        vals=feature[model+'_bridge_tokens'].astype('float32')@direction
        ax.imshow(vals.reshape(16,16),cmap='magma',alpha=.55,extent=(-.5,511.5,511.5,-.5),interpolation='bilinear')
txt(1046,7773,'Token-direction projections; model-specific color scales.',21,C['muted'])
txt(1046,7810,'Reference-defined direction; no test-image human labels.',21,C['muted'])
line(42,7860,1758,7860)

txt(82,7890,'PREDICT HUMAN JUDGMENTS WITH A SEPARATELY FITTED LINEAR READOUT',26,weight='bold')
for x,w,title in [(82,469,'1  Prepare each layer’s vector'),(613,519,'2  Fit the readout on training images'),(1194,544,'3  Predict held-out human judgments')]:
    rect(x,7950,w,233,'#F7F9FA',C['line']);txt(x+15,7970,title,24,weight='bold')
txt(97,8020,'Fixed Gaussian projection → 128 features',22)
txt(97,8060,'Standardize with training-set mean / SD',22)
txt(97,8110,'Repeat separately for every recorded layer',21,C['muted'])
arrow(559,8065,604,8065)
txt(628,8020,'Human gloss labels supervise ridge regression',22)
txt(628,8060,'Prediction = intercept + weighted feature sum',22)
txt(628,8110,'Choose α from {1, 10, 100} within training groups',21,C['muted'])
arrow(1140,8065,1185,8065)
txt(1209,8020,'Apply the fitted map to unseen shape / light groups',21)
txt(1209,8060,'Compare predicted values with recorded human values',21)
txt(1209,8110,'Only the readout is fitted; MLLM weights stay frozen',21,C['muted'])
txt(82,8230,'GROUPED VALIDATION',23,weight='bold')
for k in range(12):
    x=82+k*48;rect(x,8275,41,33,C['Q'] if k==11 else '#A3CCD3');txt(x+20.5,8281,str(k+1),18,ha='center')
txt(82,8330,'12 folds by shape OR illumination',23)
txt(82,8370,'Fit on 11; predict 1; rotate the test fold.',22)
wrapped(82,8412,'Select ridge strength on group-disjoint validation data inside the training folds.',48,22,C['muted'])
txt(704,8230,'ONE HELD-OUT PREDICTION',23,weight='bold')
txt(704,8275,'Image morimoto_0095 · bridge readout',21,C['muted'])
pr=pd.read_csv(O/'g_readout_examples.csv').set_index('model')
for yy,lab,val,col in [(8320,'Human target',pr.loc['InternVL','truth'],C['human']),(8368,'InternVL prediction',pr.loc['InternVL','prediction'],C['I']),(8416,'Qwen3 prediction',pr.loc['Qwen3','prediction'],C['Q'])]:
    txt(704,yy,lab,22,col);txt(1138,yy,f'{val:.4f}',24,col,weight='bold',ha='right')
txt(704,8475,'Shape-held-out predictions; same target image',20,C['muted'])
txt(1230,8230,'READOUT ACROSS STAGES',23,weight='bold')
rd=pd.read_csv(V/'tables/mechanism_readout.csv');sources.append(str(V/'tables/mechanism_readout.csv'))
ax=axes(1300,8290,423,171);layers=['vision_06','vision_12','vision_18','vision_24','bridge','language_09','language_18','language_27','language_36']
for model,col in [('InternVL',C['I']),('Qwen3',C['Q'])]:
    z=rd[(rd.model==model)&(rd.split=='shape')&(rd.target=='human')].set_index('layer').loc[layers];ax.plot(range(9),z.r2,color=col,lw=1,marker='o',ms=2)
ax.set(ylim=(0,.8),xlim=(-.2,8.2),xticks=[1.5,4,6.5],xticklabels=['Vision','Bridge','Language'],yticks=[0,.4,.8],ylabel='Held-out R²');ax.tick_params(labelsize=6,pad=2);ax.yaxis.label.set_size(6.5)
txt(1230,8540,'Human target · held-out shapes · 3,888 images',20,C['muted'])
txt(82,8580,'Spatial candidate maps visualize bridge responses; readout accuracy measures how well internal vectors predict human judgments.',22,C['muted'])
line(42,8610,1758,8610)

# Keep panel h as the subsequent functional-intervention analysis.
_txt,_line,_rect,_arrow,_axes=txt,line,rect,arrow,axes
SHIFT=4700
def txt(x,y,*args,**kwargs):return _txt(x,y+SHIFT,*args,**kwargs)
def line(x1,y1,x2,y2,*args,**kwargs):return _line(x1,y1+SHIFT,x2,y2+SHIFT,*args,**kwargs)
def rect(x,y,*args,**kwargs):return _rect(x,y+SHIFT,*args,**kwargs)
def arrow(x1,y1,x2,y2,*args,**kwargs):return _arrow(x1,y1+SHIFT,x2,y2+SHIFT,*args,**kwargs)
def axes(x,y,*args,**kwargs):return _axes(x,y+SHIFT,*args,**kwargs)
# h: real direction definition, spatial responses and controlled output changes.
panel('h','Reference-defined directions and functional intervention',42,3937,3786)
txt(82,3994,'1  Define from references',26,weight='bold');txt(572,3994,'2  Project onto visual tokens',26,weight='bold');txt(1170,3994,'3  Perturb and measure choice',26,weight='bold')
ax=axes(154,4065,312,145)
for name,col in [('InternVL',C['I']),('Qwen3',C['Q'])]:
 p=M/f'mechanisms_v1/{name}/directions.npz';sources.append(str(p));dire=np.load(p);v=dire['reference_means']@dire['candidate'];v=(v-v.min())/(v.max()-v.min());ax.plot(np.linspace(0,.1487,52),v,color=col,lw=1)
ax.set(xlabel='Reference c',ylabel='Scaled projection',xticks=[0,.075,.15],yticks=[0,.5,1]);ax.tick_params(labelsize=6.5,pad=2);ax.xaxis.label.set_size(7);ax.yaxis.label.set_size(7)
txt(82,4310,'No test-image human labels',23,C['muted'])
id='morimoto_0095';image_at(stim(id),572,4065,154);txt(649,4030,'Input',23,C['muted'],ha='center')
for j,(name,col) in enumerate([('InternVL',C['I']),('Qwen3',C['Q'])]):
 p=M/f'mechanisms_v1/{name}/features/{id}.npz';sources.append(str(p));dr=np.load(M/f'mechanisms_v1/{name}/directions.npz');v=np.load(p)['bridge_tokens'].astype('float32')@dr['candidate'];side=int(np.sqrt(v.size));assert side*side==v.size
 ax=image_at(stim(id),746+j*174,4065,154);ax.imshow(v.reshape(side,side),cmap='magma',alpha=.55,extent=(-.5,511.5,511.5,-.5),interpolation='bilinear');txt(823+j*174,4030,name,23,col,ha='center')
txt(572,4241,'Reference-direction responses',24);txt(572,4276,'Same image; model-specific color scales',22,C['muted'])
iv=pd.read_csv(V/'tables/v8_intervention_dose_ci.csv');sources.append(str(V/'tables/v8_intervention_dose_ci.csv'))
ax=axes(1250,4065,470,145)
for name,col in [('InternVL',C['I']),('Qwen3',C['Q'])]:
 z=iv[(iv.model==name)&(iv.direction=='candidate')].sort_values('dose');x=np.r_[z.dose,0];y=np.r_[z.mean_delta_p,0];ii=np.argsort(x);ax.plot(x[ii],y[ii],color=col,lw=1,marker='o',ms=2)
for name in ['InternVL','Qwen3']:
 z=iv[(iv.model==name)&(iv.direction=='brightness')].sort_values('dose');ax.plot(z.dose,z.mean_delta_p,color='#8F9AA3',lw=.6,ls='--')
ax.axhline(0,color=C['line'],lw=.6);ax.set(xlabel='Intervention dose',ylabel='ΔP(test glossier)',xticks=[-1,0,1],yticks=[-.1,0,.1],ylim=(-.13,.13));ax.tick_params(labelsize=6.5,pad=2);ax.xaxis.label.set_size(7);ax.yaxis.label.set_size(7)
txt(1170,4310,'72 images · 36 shapes · illumination 1',22,C['muted'])
arrow(498,4145,549,4145);arrow(1108,4145,1150,4145)

# Preserve editable text and vectors; embed raster stimuli into the SVG.
for ext in ['svg','pdf','png']:fig.savefig(O/f'Figure1_Methods_EN.{ext}',dpi=450,facecolor='white')
# Export the enlarged task explanation separately for convenient editing.
from matplotlib.transforms import Bbox
box=Bbox.from_extents(0,(H-3200)/254,W/254,(H-355)/254)
for ext in ['svg','png']:
    fig.savefig(O/f'Figure1_bcd_Task_Detail_EN.{ext}',dpi=350,facecolor='white',bbox_inches=box)
box_e=Bbox.from_extents(0,(H-4608)/254,W/254,(H-3210)/254)
for ext in ['svg','png']:
    fig.savefig(O/f'Figure1_e_Acquisition_EN.{ext}',dpi=350,facecolor='white',bbox_inches=box_e)
box_f=Bbox.from_extents(0,(H-6810)/254,W/254,(H-4620)/254)
for ext in ['svg','png']:
    fig.savefig(O/f'Figure1_f_RSA_CrossTask_EN.{ext}',dpi=350,facecolor='white',bbox_inches=box_f)
box_g=Bbox.from_extents(0,(H-8615)/254,W/254,(H-6820)/254)
for ext in ['svg','png']:
    fig.savefig(O/f'Figure1_g_Internal_Readout_EN.{ext}',dpi=350,facecolor='white',bbox_inches=box_g)
plt.close(fig)
(O/'prompts_verbatim.json').write_text(json.dumps({'single_and_pair_system':prompts['system'],'single':prompts['T1'],'pair':prompts['T2'],'matching_system':protocol['system'],'matching':protocol['prompt']},indent=2),encoding='utf-8')

caption='''Figure 1 | A computational framework linking material judgments, behavioral geometry and internal interventions.
(a) The shared stimulus set contains 36 shapes, 36 illuminations and three viewpoints. Published human gloss matches supply the behavioral reference. InternVL3.5-4B and Qwen3-VL-4B-Instruct are evaluated with frozen base-model weights.
(b) Aligned task rows show actual input images for single-image ratings, pairwise gloss choices and reference-based matching. (c) Each row reproduces the complete archived system message and task prompt, followed by an actual InternVL response and its interpretation. Line wrapping is editorial; wording is unchanged. Image placeholders denote separate attachments. The displayed pairwise JSON omits its Markdown code fence. Matching responses A/B/SAME are mapped by the controller, not by the model, to reference adjustments. Each decision uses a fresh context without target human ratings.
(d) A reconstructed 52-level sequence provides a nominal reference scale. A bracket-search controller converts visual comparisons into a complete match, stopping on SAME, bracket exhaustion, a boundary or seven decisions. The displayed six-step trajectory is the actual first match for morimoto_1889, with references 15, 34, 43, 38, 40 and 39, and answers A, A, B, A, B and A. The search starts with bounds [1,52]. The final A response exhausts the interval, so the last presented index 39 is retained; this is not an explicit SAME response. Each image is matched from two starts. (e) Acquisition is illustrated with archived single-image scores, three actual image pairs in both orders, and the first three steps of three actual reference searches. Examples are selected by fixed metadata or manifest order, not agreement. Each model supplies 3,888 single-image ratings, 4,000 pairwise responses (2,000 pairs, both orders) and 7,776 complete reference matches. Those matches comprise 46,375 InternVL and 46,581 Qwen3 reference-comparison responses. Across the three primary tasks, the totals are 54,263 and 54,469 model responses, respectively, or 108,732 combined. Response counts count model calls, not independent stimuli or observers; pilots, practice, catch trials and interventions are excluded. Acquisition preserves raw answers and controller states. The count audit and exact example records accompany the editable figure.
(f) The expanded workflow starts from the three observers’ matched judgments on the same 3,888 images. Separate physical/color regressions are fitted on two views and applied to the held-out view, rotating through all views. Residuals are averaged into a 36-shape by 36-illumination table. Each row is a shape response profile; each column is an illumination profile. Distances are mean squared differences between the corresponding profiles. The three shown shape RDMs are computed independently and have identical condition ordering. RSA correlates the 630 upper-triangle distances with Spearman correlation. The displayed distance strips encode within-vector ranks. Two-sided condition-label permutations move matrix rows and columns together; condition-node bootstraps retain pair dependencies. Raw, adjusted and doubly centered interaction profiles enter the same RSA pipeline, yielding 18 full-profile comparisons across spaces, factors and observer pairs. Cross-task validation separately fits monotonic rating-to-match and match-to-rating mappings in 12 shape- or illumination-grouped folds, then evaluates held-out predictions. Pairwise validation uses the sign of score or match differences to predict each recorded two-image choice; both orders are retained. The human–physical conflict subset consists of pairs with opposite human and physical orderings. All 36 shape conditions are shown. Matrices are normalized by their own mean off-diagonal distance for display; the color scale saturates at three. RSA, human–physical conflict pairs and cross-task prediction evaluate complementary aspects of the judgments. Continuous cross-task mappings are tested on held-out shapes or illuminations; pair-based analyses preserve the identity of both display orders.
(g) The expanded frozen-forward diagram shows the visual encoder, visual-to-language bridge and prompt-conditioned language blocks. InternVL uses its MLP projector; Qwen3 uses the final visual merger and retains additional DeepStack paths. We record vision blocks 6/12/18/24 with token-mean pooling, the bridge token mean, and language blocks 9/18/27/36 at the final prompt position before answer generation. The displayed single-image bridge grids contain 256 tokens with 2,560 values each. Vector strips are the first 32 coordinates of archived vision-24, bridge and language-36 vectors for morimoto_0095, normalized separately for display. These strips are vector components, not spatial maps. Bridge spatial maps project each token onto a normalized direction fitted from the 52 reference-image means, without test-image human labels; map color scales are model-specific. This candidate direction is distinct from the human-supervised readout. For each layer, a fixed label-free Gaussian projection produces 128 features. Training-set standardization precedes ridge regression. Alpha is selected from 1/10/100 in a group-disjoint inner validation subset, and 12 outer folds hold out shape or illumination groups. Model weights are never updated. The worked prediction example uses saved shape-held-out bridge predictions for the same image. The curve shows archived human-target R² over all 3,888 images at the nine recorded stages. Pooling differs across stages, so the curve describes readout performance, not a direct comparison of identically pooled layers or a causal processing hierarchy.
(h) A candidate direction is defined from responses to the reference sequence, without fitting test-image human ratings. The first plot rescales each model’s reference projection to [0,1] for display. The middle images show the same test stimulus, morimoto_0095, with its token-wise projection overlaid separately for each model; their color scales are model-specific. Colored curves show actual candidate-direction choice effects, while dashed gray curves show brightness-direction controls. The full analysis also includes three norm-matched random orthogonal directions and a 12-image complete-matching intervention. The 72-image intervention set covers 36 shapes and two viewpoints under illumination 1. Teal: InternVL; orange: Qwen3. The small plots are methodological exemplars; full effect sizes, intervals and tests are provided in the corresponding results figures.
'''
(O/'Figure1_caption_EN.txt').write_text(caption,encoding='utf-8')
shutil.copy2(__file__,O/Path(__file__).name)
manifest=[]
for p in sorted(set(sources+[str(M/'matching_full_v1/trials.jsonl'),str(M/'responses/internvl35_4b/full.jsonl')])):
 q=Path(p);manifest.append({'source':str(q.relative_to(B)),'sha256':hashlib.sha256(q.read_bytes()).hexdigest()})
(O/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
(O/'example_records.json').write_text(json.dumps({'single_image':one,'pairwise':two,'matching':trial},indent=2),encoding='utf-8')
print('Figure generated:',O)
