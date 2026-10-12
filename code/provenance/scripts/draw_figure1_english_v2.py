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
B=Path(__file__).resolve().parents[1];M=B/'mllm_material';V=M/'nmi_revision_v8/source';O=M/'figure1_english_v2';O.mkdir(exist_ok=True)
W,H=1800,4340
C={'ink':'#202D38','muted':'#657583','line':'#CDD7DF','pale':'#F3F6F8','human':'#626A78','I':'#147D92','Q':'#CC7243','violet':'#766599'}
plt.rcParams.update({'font.family':'Arial','font.size':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'xtick.major.width':.6,'ytick.major.width':.6,'text.color':C['ink'],'axes.labelcolor':C['ink'],'xtick.color':C['muted'],'ytick.color':C['muted']})
fig=plt.figure(figsize=(180/25.4,434/25.4),facecolor='white');a=fig.add_axes([0,0,1,1]);a.set(xlim=(0,W),ylim=(H,0));a.axis('off');sources=[]
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

panel('e','Acquisition scale and a replayable record for every decision',42,3227,1716)
for x,num,lab in [(82,'3,888','single-image ratings'),(516,'2,000 × 2','pairs × display orders'),(1040,'3,888 × 2','images × complete matches')]:
    txt(x,3295,num,31,C['I'],weight='bold');txt(x,3340,lab,25)
txt(82,3390,'Per model: InternVL3.5-4B and Qwen3-VL-4B-Instruct. Frozen weights; no target human ratings in prompts.',23,C['muted'])
txt(82,3430,'Save image IDs, prompt, raw response, parsed judgment, reference level, bracket, update and stop reason.',23,C['muted'])
line(42,3471,1758,3471)
# f: real full RDMs as editable rectangles; cross-task relationships alongside.
panel('f','Behavioral geometry and cross-task prediction',42,3498,1080)
txt(82,3553,'Adjusted condition profiles → dissimilarity matrices → RSA',24,C['muted'])
rdms=[]
for j,(name,label) in enumerate([('Human','Human'),('InternVL','InternVL'),('Qwen3','Qwen3')]):
 p=M/f'behavioral_spaces_v1/matrices/physical_color_adjusted_shape_{name}_rdm.csv';sources.append(str(p));d=np.loadtxt(p,delimiter=',');assert d.shape==(36,36) and np.allclose(d,d.T);den=d[np.triu_indices(36,1)].mean();z=d/den;rdms.append(d)
 x=82+j*215;y=3629;size=168;txt(x+size/2,y-37,label,25,[C['human'],C['I'],C['Q']][j],weight='bold',ha='center')
 cmap=plt.get_cmap('magma')
 for row in range(36):
  for col in range(36):rect(x+col*size/36,y+row*size/36,size/36+.06,size/36+.06,cmap(np.clip(z[row,col]/3,0,1)))
txt(82,3810,'36 conditions; shape space shown',23,C['muted'])
for k in range(100):rect(82+k*3,3852,3,11,plt.get_cmap('magma')(k/99))
txt(82,3871,'0',20,C['muted']);txt(382,3871,'3',20,C['muted'],ha='right');txt(414,3847,'Distance / mean distance',21,C['muted'])
line(722,3584,722,3879)
txt(766,3585,'Single rating',25,weight='bold');txt(766,3653,'Reference match',25,weight='bold');arrow(1010,3620,1010,3650);arrow(1031,3650,1031,3620)
txt(766,3719,'Pairwise choice',25,weight='bold');arrow(900,3690,900,3713)
txt(766,3780,'Held-out shapes / lighting',22,C['muted']);txt(766,3816,'Human–physical conflicts',22,C['muted']);txt(766,3852,'Node-level resampling',22,C['muted'])
line(1130,3514,1130,3882)

# g: actual readout estimates, explicitly separate supervised readout from frozen backbone.
panel('g','Read out frozen internal states',1170,3498,588)
txt(1210,3553,'Vision → bridge → language',26,weight='bold');txt(1210,3592,'Before the answer is generated',23,C['muted'])
rd=pd.read_csv(V/'tables/mechanism_readout.csv');sources.append(str(V/'tables/mechanism_readout.csv'))
ax=axes(1280,3652,445,157);layers=['vision_06','vision_12','vision_18','vision_24','bridge','language_09','language_18','language_27','language_36']
for name,col in [('InternVL',C['I']),('Qwen3',C['Q'])]:
 z=rd[(rd.model==name)&(rd.split=='shape')&(rd.target=='human')].set_index('layer').loc[layers];ax.plot(range(9),z.r2,color=col,lw=1,marker='o',ms=2)
ax.set(ylim=(0,.8),xlim=(-.2,8.2),xticks=[1.5,4,6.5],xticklabels=['Vision','Bridge','Language'],yticks=[0,.4,.8],ylabel='Held-out R²');ax.tick_params(labelsize=6.5,pad=2);ax.yaxis.label.set_size(7)
txt(1210,3871,'Human-label ridge readout; grouped CV',22,C['muted'])
line(42,3910,3828,3910)

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
plt.close(fig)
(O/'prompts_verbatim.json').write_text(json.dumps({'single_and_pair_system':prompts['system'],'single':prompts['T1'],'pair':prompts['T2'],'matching_system':protocol['system'],'matching':protocol['prompt']},indent=2),encoding='utf-8')

caption='''Figure 1 | A computational framework linking material judgments, behavioral geometry and internal interventions.
(a) The shared stimulus set contains 36 shapes, 36 illuminations and three viewpoints. Published human gloss matches supply the behavioral reference. InternVL3.5-4B and Qwen3-VL-4B-Instruct are evaluated with frozen base-model weights.
(b) Aligned task rows show actual input images for single-image ratings, pairwise gloss choices and reference-based matching. (c) Each row reproduces the complete archived system message and task prompt, followed by an actual InternVL response and its interpretation. Line wrapping is editorial; wording is unchanged. Image placeholders denote separate attachments. The displayed pairwise JSON omits its Markdown code fence. Matching responses A/B/SAME are mapped by the controller, not by the model, to reference adjustments. Each decision uses a fresh context without target human ratings.
(d) A reconstructed 52-level sequence provides a nominal reference scale. A bracket-search controller converts visual comparisons into a complete match, stopping on SAME, bracket exhaustion, a boundary or seven decisions. The displayed six-step trajectory is the actual first match for morimoto_1889, with references 15, 34, 43, 38, 40 and 39, and answers A, A, B, A, B and A. The search starts with bounds [1,52]. The final A response exhausts the interval, so the last presented index 39 is retained; this is not an explicit SAME response. Each image is matched from two starts. (e) Acquisition preserves raw answers and controller states; counts distinguish ratings, pairs, display orders and complete matches and are reported per model.
(f) Adjusted condition profiles define behavioral dissimilarity matrices, with all 36 shape conditions shown. Matrices are normalized by their own mean off-diagonal distance for display; the color scale saturates at three. RSA, human–physical conflict pairs and cross-task prediction evaluate complementary aspects of the judgments. Continuous cross-task mappings are tested on held-out shapes or illuminations; pair-based analyses preserve the identity of both display orders.
(g) Saved responses from vision, bridge and language stages are analyzed before answer generation. Separately fitted supervised ridge readouts predict human ratings in grouped cross-validation. Curves illustrate archived human-rating R² for held-out shapes; both models are sampled at nine stages, with stage-specific pooling.
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
