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
B=Path(__file__).resolve().parents[1];M=B/'mllm_material';V=M/'nmi_revision_v8/source';O=M/'figure1_english_v1';O.mkdir(exist_ok=True)
W,H=1800,2270
C={'ink':'#202D38','muted':'#657583','line':'#CDD7DF','pale':'#F3F6F8','human':'#626A78','I':'#147D92','Q':'#CC7243','violet':'#766599'}
plt.rcParams.update({'font.family':'Arial','font.size':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'xtick.major.width':.6,'ytick.major.width':.6,'text.color':C['ink'],'axes.labelcolor':C['ink'],'xtick.color':C['muted'],'ytick.color':C['muted']})
fig=plt.figure(figsize=(180/25.4,227/25.4),facecolor='white');a=fig.add_axes([0,0,1,1]);a.set(xlim=(0,W),ylim=(H,0));a.axis('off');sources=[]
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

# b and c: actual stimuli and archived model responses.
panel('b','Three complementary behavioral measurements',42,370,1080)
starts=[82,431,780]
for x,title in zip(starts,['Single-image rating','Pairwise choice','Reference comparison']):txt(x,428,title,26,weight='bold')
image_at(stim('morimoto_0001'),82,473,146)
for j,id in enumerate(two['image_ids']):image_at(stim(id),431+j*151,473,140)
image_at(stim('morimoto_1889'),780,473,140);image_at(ref(15),931,473,140)
txt(82,649,'How glossy does it look?',24);txt(431,649,'Which object looks glossier?',24);txt(780,649,'Test versus current reference',24)
for x,lab in [(431,'A'),(582,'B'),(780,'A: test'),(931,'B: reference')]:txt(x,614,lab,22,C['muted'])
rect(82,679,305,51,'#EDF5F7');txt(96,692,'glossiness: '+str(one['parsed']['glossiness']),26,C['I'],weight='bold')
rect(431,679,305,51,'#EDF5F7');txt(445,692,'choice: '+two['parsed']['choice'],26,C['I'],weight='bold')
rect(780,679,305,51,'#EDF5F7');txt(794,692,'A → increase reference',26,C['I'],weight='bold')
txt(82,749,'Ordinal score, 0–100',23,C['muted']);txt(431,749,'Both A/B and B/A orders',23,C['muted']);txt(780,749,'A, B or SAME',23,C['muted'])
line(1130,386,1130,781)
panel('c','Perceptual prompt design',1170,370,588)
txt(1210,430,'TARGET',22,C['I'],weight='bold');txt(1210,463,'“apparent glossiness of the',26);txt(1210,498,'main object”',26)
txt(1210,552,'JUDGMENT',22,C['I'],weight='bold');txt(1210,585,'“Judge the visible appearance,',26);txt(1210,620,'not the material name”',26)
txt(1210,674,'RESPONSE',22,C['I'],weight='bold');txt(1210,707,'Fixed formats; explicit ties',26);txt(1210,747,'Fresh context; no human answers',24,C['muted'])
line(42,808,1758,808)

# d: the genuine recorded search, not an idealized synthetic path.
panel('d','Automated asymmetric reference matching',42,835,1080)
txt(82,894,'A fixed, reconstructed 52-level reference sequence',25,C['muted'])
for k,x in zip([1,14,27,39,52],[82,276,470,664,858]):
 image_at(ref(k),x,935,123);txt(x+61.5,1068,f'{(k-1)*.1487/51:.3f}',23,C['muted'],ha='center')
arrow(107,1127,991,1127);txt(82,1145,'Matte',24);txt(984,1145,'Glossy',24,ha='right');txt(537,1145,'Nominal reference c: 0–0.1487',24,C['muted'],ha='center')
txt(82,1201,'Compare → update bracket → present next reference',27,weight='bold')
txt(82,1247,'Recorded search',23,C['muted']);txt(340,1243,'15 → 34 → 43 → 38 → 40 → 39',30,C['I'],weight='bold')
txt(82,1293,'morimoto_1889 · start 15 · 6 decisions · matched c = 0.1108',24)
txt(82,1336,'Stop: SAME / bracket exhausted / boundary / 7 decisions',23,C['muted'])
line(1130,850,1130,1372)
panel('e','Structured, replayable acquisition',1170,835,588)
txt(1210,898,'PER MODEL',23,C['muted'],weight='bold')
for yy,number,lab in [(945,'3,888','single-image ratings'),(1015,'2,000 × 2','pairs × display orders'),(1085,'3,888 × 2','images × complete matches')]:
 txt(1210,yy,number,33,C['I'],weight='bold');txt(1210,yy+38,lab,24)
line(1210,1167,1752,1167)
txt(1210,1194,'Raw answer → parsed judgment',26,weight='bold')
txt(1210,1242,'Image ID · reference level · order',24,C['muted']);txt(1210,1278,'Search bracket · action · stop state',24,C['muted'])
txt(1210,1327,'Two starts → image-level mean',26,weight='bold')
line(42,1401,1758,1401)

# f: real full RDMs as editable rectangles; cross-task relationships alongside.
panel('f','Behavioral geometry and cross-task prediction',42,1428,1080)
txt(82,1483,'Adjusted condition profiles → dissimilarity matrices → RSA',24,C['muted'])
rdms=[]
for j,(name,label) in enumerate([('Human','Human'),('InternVL','InternVL'),('Qwen3','Qwen3')]):
 p=M/f'behavioral_spaces_v1/matrices/physical_color_adjusted_shape_{name}_rdm.csv';sources.append(str(p));d=np.loadtxt(p,delimiter=',');assert d.shape==(36,36) and np.allclose(d,d.T);den=d[np.triu_indices(36,1)].mean();z=d/den;rdms.append(d)
 x=82+j*215;y=1559;size=168;txt(x+size/2,y-37,label,25,[C['human'],C['I'],C['Q']][j],weight='bold',ha='center')
 cmap=plt.get_cmap('magma')
 for row in range(36):
  for col in range(36):rect(x+col*size/36,y+row*size/36,size/36+.06,size/36+.06,cmap(np.clip(z[row,col]/3,0,1)))
txt(82,1740,'36 conditions; shape space shown',23,C['muted'])
for k in range(100):rect(82+k*3,1782,3,11,plt.get_cmap('magma')(k/99))
txt(82,1801,'0',20,C['muted']);txt(382,1801,'3',20,C['muted'],ha='right');txt(414,1777,'Distance / mean distance',21,C['muted'])
line(722,1514,722,1809)
txt(766,1515,'Single rating',25,weight='bold');txt(766,1583,'Reference match',25,weight='bold');arrow(1010,1550,1010,1580);arrow(1031,1580,1031,1550)
txt(766,1649,'Pairwise choice',25,weight='bold');arrow(900,1620,900,1643)
txt(766,1710,'Held-out shapes / lighting',22,C['muted']);txt(766,1746,'Human–physical conflicts',22,C['muted']);txt(766,1782,'Node-level resampling',22,C['muted'])
line(1130,1444,1130,1812)

# g: actual readout estimates, explicitly separate supervised readout from frozen backbone.
panel('g','Read out frozen internal states',1170,1428,588)
txt(1210,1483,'Vision → bridge → language',26,weight='bold');txt(1210,1522,'Before the answer is generated',23,C['muted'])
rd=pd.read_csv(V/'tables/mechanism_readout.csv');sources.append(str(V/'tables/mechanism_readout.csv'))
ax=axes(1280,1582,445,157);layers=['vision_06','vision_12','vision_18','vision_24','bridge','language_09','language_18','language_27','language_36']
for name,col in [('InternVL',C['I']),('Qwen3',C['Q'])]:
 z=rd[(rd.model==name)&(rd.split=='shape')&(rd.target=='human')].set_index('layer').loc[layers];ax.plot(range(9),z.r2,color=col,lw=1,marker='o',ms=2)
ax.set(ylim=(0,.8),xlim=(-.2,8.2),xticks=[1.5,4,6.5],xticklabels=['Vision','Bridge','Language'],yticks=[0,.4,.8],ylabel='Held-out R²');ax.tick_params(labelsize=6.5,pad=2);ax.yaxis.label.set_size(7)
txt(1210,1801,'Human-label ridge readout; grouped CV',22,C['muted'])
line(42,1840,1758,1840)

# h: real direction definition, spatial responses and controlled output changes.
panel('h','Reference-defined directions and functional intervention',42,1867,1716)
txt(82,1924,'1  Define from references',26,weight='bold');txt(572,1924,'2  Project onto visual tokens',26,weight='bold');txt(1170,1924,'3  Perturb and measure choice',26,weight='bold')
ax=axes(154,1995,312,145)
for name,col in [('InternVL',C['I']),('Qwen3',C['Q'])]:
 p=M/f'mechanisms_v1/{name}/directions.npz';sources.append(str(p));dire=np.load(p);v=dire['reference_means']@dire['candidate'];v=(v-v.min())/(v.max()-v.min());ax.plot(np.linspace(0,.1487,52),v,color=col,lw=1)
ax.set(xlabel='Reference c',ylabel='Scaled projection',xticks=[0,.075,.15],yticks=[0,.5,1]);ax.tick_params(labelsize=6.5,pad=2);ax.xaxis.label.set_size(7);ax.yaxis.label.set_size(7)
txt(82,2240,'No test-image human labels',23,C['muted'])
id='morimoto_0095';image_at(stim(id),572,1995,154);txt(649,1960,'Input',23,C['muted'],ha='center')
for j,(name,col) in enumerate([('InternVL',C['I']),('Qwen3',C['Q'])]):
 p=M/f'mechanisms_v1/{name}/features/{id}.npz';sources.append(str(p));dr=np.load(M/f'mechanisms_v1/{name}/directions.npz');v=np.load(p)['bridge_tokens'].astype('float32')@dr['candidate'];side=int(np.sqrt(v.size));assert side*side==v.size
 ax=image_at(stim(id),746+j*174,1995,154);ax.imshow(v.reshape(side,side),cmap='magma',alpha=.55,extent=(-.5,511.5,511.5,-.5),interpolation='bilinear');txt(823+j*174,1960,name,23,col,ha='center')
txt(572,2171,'Reference-direction responses',24);txt(572,2206,'Same image; model-specific color scales',22,C['muted'])
iv=pd.read_csv(V/'tables/v8_intervention_dose_ci.csv');sources.append(str(V/'tables/v8_intervention_dose_ci.csv'))
ax=axes(1250,1995,470,145)
for name,col in [('InternVL',C['I']),('Qwen3',C['Q'])]:
 z=iv[(iv.model==name)&(iv.direction=='candidate')].sort_values('dose');x=np.r_[z.dose,0];y=np.r_[z.mean_delta_p,0];ii=np.argsort(x);ax.plot(x[ii],y[ii],color=col,lw=1,marker='o',ms=2)
for name in ['InternVL','Qwen3']:
 z=iv[(iv.model==name)&(iv.direction=='brightness')].sort_values('dose');ax.plot(z.dose,z.mean_delta_p,color='#8F9AA3',lw=.6,ls='--')
ax.axhline(0,color=C['line'],lw=.6);ax.set(xlabel='Intervention dose',ylabel='ΔP(test glossier)',xticks=[-1,0,1],yticks=[-.1,0,.1],ylim=(-.13,.13));ax.tick_params(labelsize=6.5,pad=2);ax.xaxis.label.set_size(7);ax.yaxis.label.set_size(7)
txt(1170,2240,'72 images · 36 shapes · illumination 1',22,C['muted'])
arrow(498,2075,549,2075);arrow(1108,2075,1150,2075)

# Preserve editable text and vectors; embed raster stimuli into the SVG.
for ext in ['svg','pdf','png']:fig.savefig(O/f'Figure1_Methods_EN.{ext}',dpi=450,facecolor='white')
plt.close(fig)
caption='''Figure 1 | A computational framework linking material judgments, behavioral geometry and internal interventions.
(a) The shared stimulus set contains 36 shapes, 36 illuminations and three viewpoints. Published human gloss matches supply the behavioral reference. InternVL3.5-4B and Qwen3-VL-4B-Instruct are evaluated with frozen base-model weights.
(b) Three interfaces measure ordinal ratings, pairwise choices and comparisons against a reference. The displayed inputs and outputs are archived InternVL examples; the pairwise task includes both display orders. (c) Prompt excerpts specify the perceptual attribute and constrain the response format. Each decision uses a fresh context, without supplying target human ratings.
(d) A reconstructed 52-level sequence provides a nominal reference scale. A bracket-search controller converts visual comparisons into a complete match, stopping on SAME, bracket exhaustion, a boundary or seven decisions. The displayed six-step trajectory is the actual first match for morimoto_1889. Each image is matched from two starts. (e) Acquisition preserves raw answers and controller states; counts distinguish ratings, pairs, display orders and complete matches and are reported per model.
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
