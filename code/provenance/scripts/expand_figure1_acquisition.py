from pathlib import Path
B=Path(__file__).resolve().parents[1]
s=(B/'scripts/draw_figure1_english_v2.py').read_text('utf-8')
s=s.replace('figure1_english_v2','figure1_english_v3').replace('W,H=1800,4340','W,H=1800,5440').replace('434/25.4','544/25.4')
start=s.index("panel('e'");end=s.index('# f:',start)
block=r'''
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

# Shift the previously approved analysis panels; nested helper calls shift only once.
_txt,_line,_rect,_arrow,_axes=txt,line,rect,arrow,axes
SHIFT=1140
def txt(x,y,*args,**kwargs):return _txt(x,y+SHIFT,*args,**kwargs)
def line(x1,y1,x2,y2,*args,**kwargs):return _line(x1,y1+SHIFT,x2,y2+SHIFT,*args,**kwargs)
def rect(x,y,*args,**kwargs):return _rect(x,y+SHIFT,*args,**kwargs)
def arrow(x1,y1,x2,y2,*args,**kwargs):return _arrow(x1,y1+SHIFT,x2,y2+SHIFT,*args,**kwargs)
def axes(x,y,*args,**kwargs):return _axes(x,y+SHIFT,*args,**kwargs)
'''
s=s[:start]+block+s[end:]
s=s.replace('W,H=1800,5440','W,H=1800,5480').replace('544/25.4','548/25.4')
# Crop e as a standalone editorial panel, alongside the preserved b-d extract.
s=s.replace('plt.close(fig)', '''box_e=Bbox.from_extents(0,(H-4608)/254,W/254,(H-3210)/254)
for ext in ['svg','png']:
    fig.savefig(O/f'Figure1_e_Acquisition_EN.{ext}',dpi=350,facecolor='white',bbox_inches=box_e)
plt.close(fig)''')
s=s.replace('(e) Acquisition preserves raw answers and controller states; counts distinguish ratings, pairs, display orders and complete matches and are reported per model.', '(e) Acquisition is illustrated with archived single-image scores, three actual image pairs in both orders, and the first three steps of three actual reference searches. Examples are selected by fixed metadata or manifest order, not agreement. Each model supplies 3,888 single-image ratings, 4,000 pairwise responses (2,000 pairs, both orders) and 7,776 complete reference matches. Those matches comprise 46,375 InternVL and 46,581 Qwen3 reference-comparison responses. Across the three primary tasks, the totals are 54,263 and 54,469 model responses, respectively, or 108,732 combined. Response counts count model calls, not independent stimuli or observers; pilots, practice, catch trials and interventions are excluded. Acquisition preserves raw answers and controller states. The count audit and exact example records accompany the editable figure.')
(B/'scripts/draw_figure1_english_v3.py').write_text(s,encoding='utf-8')
print('Acquisition-panel revision written')
