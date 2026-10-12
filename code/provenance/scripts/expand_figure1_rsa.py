from pathlib import Path
B=Path(__file__).resolve().parents[1]
s=(B/'scripts/draw_figure1_english_v3.py').read_text('utf-8').replace('figure1_english_v3','figure1_english_v4')
s=s.replace('W,H=1800,5480','W,H=1800,7480').replace('548/25.4','748/25.4')
start=s.index('# Shift the previously approved');end=s.index('# g:',start)
# Retain g and h, moving them below the expanded full-width f.
block=r'''
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
'''
s=s[:start]+block+s[end:]
# Extend for g/h and provide an independently editable f master.
s=s.replace('W,H=1800,7480','W,H=1800,7680').replace('748/25.4','768/25.4')
s=s.replace('plt.close(fig)', '''box_f=Bbox.from_extents(0,(H-6810)/254,W/254,(H-4620)/254)
for ext in ['svg','png']:
    fig.savefig(O/f'Figure1_f_RSA_CrossTask_EN.{ext}',dpi=350,facecolor='white',bbox_inches=box_f)
plt.close(fig)''')
s=s.replace('(f) Adjusted condition profiles define behavioral dissimilarity matrices, with all 36 shape conditions shown.', '(f) The expanded workflow starts from the three observers’ matched judgments on the same 3,888 images. Separate physical/color regressions are fitted on two views and applied to the held-out view, rotating through all views. Residuals are averaged into a 36-shape by 36-illumination table. Each row is a shape response profile; each column is an illumination profile. Distances are mean squared differences between the corresponding profiles. The three shown shape RDMs are computed independently and have identical condition ordering. RSA correlates the 630 upper-triangle distances with Spearman correlation. The displayed distance strips encode within-vector ranks. Two-sided condition-label permutations move matrix rows and columns together; condition-node bootstraps retain pair dependencies. Raw, adjusted and doubly centered interaction profiles enter the same RSA pipeline, yielding 18 full-profile comparisons across spaces, factors and observer pairs. Cross-task validation separately fits monotonic rating-to-match and match-to-rating mappings in 12 shape- or illumination-grouped folds, then evaluates held-out predictions. Pairwise validation uses the sign of score or match differences to predict each recorded two-image choice; both orders are retained. The human–physical conflict subset consists of pairs with opposite human and physical orderings. All 36 shape conditions are shown.')
(B/'scripts/draw_figure1_english_v4.py').write_text(s,encoding='utf-8')
print('Expanded RSA figure source written')
