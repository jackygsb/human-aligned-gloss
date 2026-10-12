from pathlib import Path
import json,hashlib,shutil
import numpy as np,pandas as pd
B=Path(__file__).resolve().parents[1];M=B/'mllm_material';O=M/'figure1_english_v5';O.mkdir(exist_ok=True)
for name in ['acquisition_audit.json','acquisition_examples.json','acquisition_counts.csv']:
    shutil.copy2(M/'figure1_english_v4'/name,O/name)
arrays={};provenance=[]
for model in ['InternVL','Qwen3']:
    p=M/f'mechanisms_v1/{model}/features/morimoto_0095.npz';z=np.load(p)
    for key in ['vision_24','bridge','language_36','bridge_tokens']:arrays[model+'_'+key]=z[key]
    provenance.append({'path':str(p.relative_to(B)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
np.savez_compressed(O/'g_internal_feature_examples.npz',**arrays)
p=M/'mechanisms_v1/tables/readout_predictions.csv.gz'
d=pd.read_csv(p);selected=d[(d.image_id=='morimoto_0095')&(d.layer=='bridge')&(d.split=='shape')&(d.target=='human')]
assert len(selected)==2
selected.to_csv(O/'g_readout_examples.csv',index=False)
provenance.append({'path':str(p.relative_to(B)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(O/'g_method_sources.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
s=(B/'scripts/draw_figure1_english_v4.py').read_text('utf-8').replace('figure1_english_v4','figure1_english_v5')
s=s.replace('W,H=1800,7680','W,H=1800,9040').replace('768/25.4','904/25.4')
start=s.index('# g:');end=s.index('# h:',start)
block=r'''
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
'''
s=s[:start]+block+s[end:]
s=s.replace('plt.close(fig)', '''box_g=Bbox.from_extents(0,(H-8615)/254,W/254,(H-6820)/254)
for ext in ['svg','png']:
    fig.savefig(O/f'Figure1_g_Internal_Readout_EN.{ext}',dpi=350,facecolor='white',bbox_inches=box_g)
plt.close(fig)''')
s=s.replace('(g) Saved responses from vision, bridge and language stages are analyzed before answer generation. Separately fitted supervised ridge readouts predict human ratings in grouped cross-validation. Curves illustrate archived human-rating R² for held-out shapes; both models are sampled at nine stages, with stage-specific pooling.', '(g) The expanded frozen-forward diagram shows the visual encoder, visual-to-language bridge and prompt-conditioned language blocks. InternVL uses its MLP projector; Qwen3 uses the final visual merger and retains additional DeepStack paths. We record vision blocks 6/12/18/24 with token-mean pooling, the bridge token mean, and language blocks 9/18/27/36 at the final prompt position before answer generation. The displayed single-image bridge grids contain 256 tokens with 2,560 values each. Vector strips are the first 32 coordinates of archived vision-24, bridge and language-36 vectors for morimoto_0095, normalized separately for display. These strips are vector components, not spatial maps. Bridge spatial maps project each token onto a normalized direction fitted from the 52 reference-image means, without test-image human labels; map color scales are model-specific. This candidate direction is distinct from the human-supervised readout. For each layer, a fixed label-free Gaussian projection produces 128 features. Training-set standardization precedes ridge regression. Alpha is selected from 1/10/100 in a group-disjoint inner validation subset, and 12 outer folds hold out shape or illumination groups. Model weights are never updated. The worked prediction example uses saved shape-held-out bridge predictions for the same image. The curve shows archived human-target R² over all 3,888 images at the nine recorded stages. Pooling differs across stages, so the curve describes readout performance, not a direct comparison of identically pooled layers or a causal processing hierarchy.')
(B/'scripts/draw_figure1_english_v5.py').write_text(s,encoding='utf-8')
print('Expanded internal-readout source and real examples prepared')
