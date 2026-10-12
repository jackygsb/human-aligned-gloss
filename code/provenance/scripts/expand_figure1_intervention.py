from pathlib import Path
import shutil,json,hashlib
import numpy as np,pandas as pd
B=Path(__file__).resolve().parents[1];M=B/'mllm_material';O=M/'figure1_english_v6';O.mkdir(exist_ok=True)
for name in ['acquisition_audit.json','acquisition_examples.json','acquisition_counts.csv','g_internal_feature_examples.npz','g_readout_examples.csv','g_method_sources.json']:
    shutil.copy2(M/'figure1_english_v5'/name,O/name)
saved={}
for model in ['InternVL','Qwen3']:
    d=np.load(M/f'mechanisms_v1/{model}/directions.npz')
    for key in d.files:saved[model+'_'+key]=d[key]
    for iid in ['morimoto_0064','morimoto_0088']:
        saved[model+'_'+iid]=np.load(M/f'mechanisms_v1/{model}/features/{iid}.npz')['bridge_tokens']
np.savez_compressed(O/'h_direction_examples.npz',**saved)
iv=pd.read_csv(M/'nmi_revision_v8/source/tables/v8_intervention_dose_ci.csv')
ctr=pd.read_csv(M/'mechanisms_v1/tables/shape_cluster_intervention_contrasts.csv')
iv.to_csv(O/'h_dose_effects.csv',index=False);ctr.to_csv(O/'h_control_contrasts.csv',index=False)
meta={'example':'morimoto_0064','roi_xyxy':[160,65,300,305],'selection':'Rank the 72 intervention images by the smaller of the two models\' whole-image correlations with positive local brightness; visually inspect the top eight. Select morimoto_0064 (third ranked; visibly glossy curved surface) for legible highlight/response comparison. This is an illustrative selection, not a population estimate or highlight-segmentation test.','maps':'Unmodified token projections, per-model full-image min/max; full-frame overlay alpha .45, nearest-neighbor grid. Insets use the same ROI and color limits as the full frame.','dose':'a times reference_scale times candidate direction, added to every TEST image bridge token only','outcome':'Softmax normalized over A/B/SAME; test-glossier probability plus half SAME. Report changes in percentage points, not relative percentages.','scope':'72 images; 36 shapes; 2 viewpoints; illumination 1; both test/reference orders. Intervals are shape-cluster 95% bootstrap CIs.','sources':[]}
for rel in ['scripts/run_mechanism_v1.py','scripts/mechanism_backend_v1.py','mllm_material/mechanisms_v1/tables/spatial_feature_associations.csv','mllm_material/nmi_revision_v8/source/tables/v8_intervention_dose_ci.csv','mllm_material/mechanisms_v1/tables/shape_cluster_intervention_contrasts.csv']:
    p=B/rel;meta['sources'].append({'path':rel,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(O/'h_method_provenance.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
s=(B/'scripts/draw_figure1_english_v5.py').read_text('utf-8').replace('figure1_english_v5','figure1_english_v6')
s=s.replace('W,H=1800,9040','W,H=1800,11060').replace('904/25.4','1106/25.4')
start=s.index('# h:');end=s.index('# Preserve editable',start)
block=r'''
# h: define the direction, visualize real token responses, intervene, measure changes.
txt,line,rect,arrow,axes=_txt,_line,_rect,_arrow,_axes
panel('h','From reference-defined features to controlled changes in gloss judgments',42,8637,1716)
txt(82,8692,'Define a direction using the reference sequence; visualize it; then change internal activations while keeping the images fixed.',22,C['muted'])
hs=np.load(O/'h_direction_examples.npz')
txt(82,8750,'1  DEFINE A DIRECTION: what changes inside the model as reference gloss increases?',26,weight='bold')
for k,x in zip([1,27,52],[82,255,428]):
    image_at(ref(k),x,8805,135);txt(x+67.5,8952,'Level '+str(k),22,C['muted'],ha='center')
arrow(98,9002,550,9002);txt(82,9022,'Matte',23);txt(563,9022,'Glossy',23,ha='right')
txt(82,9064,'All 52 reference levels enter the fit.',22,C['muted'])
arrow(590,8890,639,8890)
txt(665,8805,'For each reference image:',23,weight='bold')
txt(665,8847,'average its bridge tokens → one vector',22)
txt(665,8899,'Fit each vector coordinate versus reference level.',22)
txt(665,8941,'Collect the slopes into one direction vector d.',22)
txt(665,8993,'+d points toward higher reference gloss.',24,C['I'],weight='bold')
txt(665,9045,'One direction per model; no human test labels.',22,C['muted'])
txt(1280,8805,'REFERENCE PROJECTION',22,weight='bold')
ax=axes(1334,8861,389,151)
for model,col in [('InternVL',C['I']),('Qwen3',C['Q'])]:
    y=hs[model+'_reference_means']@hs[model+'_candidate'];y=(y-y.min())/(y.max()-y.min())
    ax.plot(np.arange(1,53),y,color=col,lw=.9,marker='.',ms=1.8)
ax.set(xlabel='Reference level',ylabel='Scaled projection',xticks=[1,27,52],yticks=[0,.5,1]);ax.tick_params(labelsize=6,pad=2);ax.xaxis.label.set_size(6.5);ax.yaxis.label.set_size(6.5)
txt(1280,9100,'Does the direction track the reference sequence?',19,C['muted'])
line(42,9150,1758,9150)

txt(82,9180,'2  LOCALIZE THE RESPONSE: project each spatial token onto the same direction',26,weight='bold')
txt(82,9224,'Token response = h(p) · d. Bright map colors indicate stronger projection, not a measured highlight mask.',22,C['muted'])
iid='morimoto_0064';roi=(160,65,300,305)
for j,(name,col) in enumerate([('Input image',C['muted']),('InternVL',C['I']),('Qwen3',C['Q'])]):
    x=82+j*570;txt(x,9270,name,26,col,weight='bold')
    full=image_at(stim(iid),x,9320,273)
    if j:
        model=name;v=(hs[model+'_'+iid].astype('float32')@hs[model+'_candidate']).reshape(16,16)
        full.imshow(v,cmap='magma',alpha=.45,extent=(-.5,511.5,511.5,-.5),interpolation='nearest',vmin=v.min(),vmax=v.max())
    full.add_patch(Rectangle((roi[0],roi[1]),roi[2]-roi[0],roi[3]-roi[1],fill=False,edgecolor='#00D4CC',linewidth=.9))
    zoom=image_at(stim(iid),x+311,9361,174,222)
    if j:zoom.imshow(v,cmap='magma',alpha=.45,extent=(-.5,511.5,511.5,-.5),interpolation='nearest',vmin=v.min(),vmax=v.max())
    zoom.set_xlim(roi[0],roi[2]);zoom.set_ylim(roi[3],roi[1]);zoom.set_aspect('equal')
    txt(x+398,9320,'Same-region zoom',20,C['muted'],ha='center')
    arrow(x+278,9454,x+309,9454)
    if j:
        for k in range(60):rect(x+k*4.55,9630,4.6,12,plt.get_cmap('magma')(k/59))
        txt(x,9650,'Lower',19,C['muted']);txt(x+273,9650,'Higher projection',19,C['muted'],ha='right')
    else:txt(x,9630,'Visible specular streak on a curved surface',21,C['muted'])
txt(82,9700,'Example: morimoto_0064. Native 16 × 16 token grids; each model uses its own full-image color limits, also used in its inset.',21,C['muted'])
txt(82,9735,'Illustrative example selected for visible highlights and positive local-brightness association; response also extends beyond highlights.',21,C['muted'])
line(42,9780,1758,9780)

txt(82,9810,'3  INTERVENE: change the TEST tokens, then ask the identical gloss-comparison question',26,weight='bold')
image_at(stim(iid),82,9870,120);image_at(ref(27),226,9870,120)
txt(142,10004,'Fixed test',21,C['muted'],ha='center');txt(286,10004,'Fixed reference',21,C['muted'],ha='center')
arrow(365,9933,411,9933)
rect(430,9860,736,198,'#EAF3F5')
txt(450,9877,'Modify internal features, not the image pixels',24,weight='bold')
txt(450,9923,'Each TEST token: h′(p) = h(p) + a × s × d',28,C['I'],weight='bold')
txt(450,9970,'a = −1, −0.5, 0, +0.5, +1;   s = fitted full-reference-range change',21)
txt(450,10009,'Same addition to every test token; reference tokens stay unchanged.',21,C['muted'])
arrow(1180,9933,1220,9933)
txt(1240,9875,'Continue the frozen forward pass',24,weight='bold')
txt(1240,9920,'Read the A / B / SAME token probabilities',22)
txt(1240,9962,'Score = P(test glossier) + ½ P(SAME)',23)
txt(1240,10009,'Normalize over these three answer tokens.',21,C['muted'])
txt(82,10101,'72 images · 36 shapes · two viewpoints · illumination 1 · both image orders · reference level 27',23,C['muted'])
txt(82,10143,'Controls: equal-size perturbations along a brightness direction and three random directions orthogonal to d.',22,C['muted'])

txt(82,10202,'4  MEASURE THE CHANGE: intervention effect in percentage points',26,weight='bold')
effect=pd.read_csv(O/'h_dose_effects.csv')
for j,(model,col) in enumerate([('InternVL',C['I']),('Qwen3',C['Q'])]):
    x=82+j*572;txt(x,10253,model,25,col,weight='bold')
    ax=axes(x+87,10320,405,264)
    z=effect[(effect.model==model)&(effect.direction=='candidate')].sort_values('dose')
    xx=np.r_[z.dose,0];yy=np.r_[z.mean_delta_p,0]*100;lo=np.r_[z.ci_low,0]*100;hi=np.r_[z.ci_high,0]*100;ix=np.argsort(xx)
    ax.errorbar(xx[ix],yy[ix],yerr=np.array([yy-lo,hi-yy])[:,ix],color=col,lw=1.1,marker='o',ms=3,capsize=2,label='Candidate')
    for kind in ['brightness','random0','random1','random2']:
        zc=effect[(effect.model==model)&(effect.direction==kind)].sort_values('dose');xc=np.r_[zc.dose,0];yc=np.r_[zc.mean_delta_p,0]*100;ii=np.argsort(xc)
        ax.plot(xc[ii],yc[ii],color='#727D87' if kind=='brightness' else '#BDC6CD',lw=.8 if kind=='brightness' else .6,ls='--' if kind=='brightness' else ':')
    ax.axhline(0,color=C['line'],lw=.6);ax.set(xlim=(-1.08,1.08),ylim=(-16,16),xticks=[-1,0,1],yticks=[-15,0,15],xlabel='Intervention dose a',ylabel='Choice-score change (pp)');ax.tick_params(labelsize=6.5,pad=2);ax.xaxis.label.set_size(6.5);ax.yaxis.label.set_size(6.5)
    neg=z[z.dose==-1].iloc[0];pos=z[z.dose==1].iloc[0]
    txt(x,10705,f'−1 dose: {neg.mean_delta_p*100:+.2f} pp     +1 dose: {pos.mean_delta_p*100:+.2f} pp',23,col,weight='bold')
    txt(x,10749,f'Full bidirectional swing: {(pos.mean_delta_p-neg.mean_delta_p)*100:.2f} pp',23)
txt(1252,10253,'HOW TO READ THE EFFECT',23,weight='bold')
wrapped(1252,10304,'Adding the reference-defined direction raises the test-glossier score; subtracting it lowers the score.',39,23)
txt(1252,10465,'About 10 percentage points per endpoint',23,C['I'],weight='bold')
wrapped(1252,10514,'Error bars: 95% shape-cluster confidence intervals. Lines connect observed dose means; no linear trend was fitted.',40,22,C['muted'])
txt(1252,10686,'Gray dashed: brightness control',21,C['muted']);txt(1252,10722,'Gray dotted: three random controls',21,C['muted'])
txt(82,10820,'The image and prompt are fixed: the measured change follows an internal activation perturbation.',23,weight='bold')
txt(82,10863,'Choice scores are normalized over A/B/SAME; they are not unrestricted response probabilities or units of physical gloss.',22,C['muted'])
txt(82,10904,'Candidate effects exceed norm-matched brightness and random controls; Qwen3 DeepStack pathways remain unchanged.',22,C['muted'])
line(42,10960,1758,10960)
'''
s=s[:start]+block+s[end:]
s=s.replace('plt.close(fig)', '''box_h=Bbox.from_extents(0,(H-10970)/254,W/254,(H-8620)/254)
for ext in ['svg','png']:
    fig.savefig(O/f'Figure1_h_Direction_Intervention_EN.{ext}',dpi=350,facecolor='white',bbox_inches=box_h)
plt.close(fig)''')
start=s.index('(h) A candidate direction');end=s.index("\n'''",start)
s=s[:start]+'''(h) A normalized candidate direction d is the multivariate slope of mean bridge responses to 52 reconstructed references versus normalized reference gloss. The scale s is the slope norm, corresponding to the fitted full reference-range activation change. Reference projection curves are individually rescaled to [0,1] for display. The selected spatial example morimoto_0064 has visible specular structure; it was selected after ranking the 72 intervention images by the smaller cross-model spatial association with positive local brightness and inspecting the top eight. This is an illustrative selection, not a highlight localization test. Maps show unmodified native 16 x 16 token projections h(p) dot d with independent full-image min/max color scales per model, alpha 0.45 overlays and nearest-neighbor rendering. Each inset uses the identical crop and scale; cyan rectangles identify that crop, not a segmented highlight. During intervention, a*s*d is added uniformly to every TEST image bridge token, not only the high-response or highlighted region. Input images, prompt, model weights and reference tokens stay fixed. Qwen3 final-merger intervention leaves DeepStack inputs intact. The 72-image assay uses reference level 27 and both image orders. The readout is a softmax normalized over A/B/SAME, scored as test-glossier probability plus half SAME. Dose plots show changes from each image/order baseline in percentage points, with archived 95% shape-cluster confidence intervals (36 shapes). Doses connect observed means, not a fitted straight-line model. Brightness and three random orthogonal directions are norm-matched controls. Positive endpoint effects are +9.92 pp for InternVL and +10.25 pp for Qwen3; negative endpoints are -9.37 and -9.87 pp. The full bidirectional swings are 19.29 and 20.12 pp, respectively. The original archive additionally contains a 12-image complete-matching intervention, not illustrated here. These are functional sensitivity effects of the candidate direction, not a claim of pure highlight-selective neurons. Teal: InternVL; orange: Qwen3.''' +s[end:]
(B/'scripts/draw_figure1_english_v6.py').write_text(s,encoding='utf-8')
print('Expanded intervention panel source prepared')
