from pathlib import Path
import shutil,json,hashlib
import numpy as np,pandas as pd
B=Path(__file__).resolve().parents[1];M=B/'mllm_material';O=M/'figure4_english_v1'
for x in ['data','images','provenance_scripts']:(O/x).mkdir(parents=True,exist_ok=True)
sources=[]
def copy(p,name=None,folder='data'):
 sources.append(p);shutil.copy2(p,O/folder/(name or p.name))
for name in ['v8_cross_task_ci.csv','v8_pair_ci.csv']:copy(M/'nmi_revision_v8/source/tables'/name)
copy(M/'paper_figures_v1/source_data/image_level_all3888.csv','internvl_images.csv');copy(M/'qwen3_full_v1/joint_model_predictions.csv','qwen_images.csv')
p=M/'paper_figures_v1/source_data/continuous_cross_task_oof.csv';copy(p);z=pd.read_csv(p).assign(model='InternVL');rows=[z]
for split in ['shape','illumination']:
 for source in ['single_score','model_c']:
  p=M/f'qwen3_full_v1/cross_{split}_{source}.csv';copy(p);r=pd.read_csv(p).rename(columns={'target':'observed','prediction':'predicted'});r['model']='Qwen3';r['split']=split;r['source']=source;r['target']='model_c' if source=='single_score' else 'single_score';rows.append(r)
pd.concat(rows).to_csv(O/'data/continuous_predictions.csv',index=False)
p=M/'paper_figures_v1/source_data/pair_cross_task_predictions.csv';copy(p);ip=pd.read_csv(p);p=M/'qwen3_full_v1/pair_predictions.csv';copy(p,'qwen_pair_predictions.csv');qp=pd.read_csv(p).merge(ip[['pair_id','a_id','b_id']],on='pair_id',validate='one_to_one');ip=ip.rename(columns={'t2_probability_a':'choice_a'});p=pd.concat([ip.assign(model='InternVL'),qp.assign(model='Qwen3')],ignore_index=True);p.to_csv(O/'data/all_pair_predictions.csv',index=False)
curves=[]
for (model,task),sub in [( (m,t),p[p.model.eq(m)]) for m in ['InternVL','Qwen3'] for t in ['single','matching']]:
 x=sub['delta_'+task]/(100 if task=='single' else .1487);bins=np.linspace(-1,1,11);ix=np.clip(np.digitize(x,bins)-1,0,9)
 for k in range(10):
  use=ix==k
  if use.sum():curves.append(dict(model=model,task=task,bin=k,n=int(use.sum()),delta_mean=x[use].mean(),choice_mean=sub.loc[use,'choice_a'].mean()))
pd.DataFrame(curves).to_csv(O/'data/choice_curves.csv',index=False)
# Illustrative concordant examples, explicitly selected, separate from all-pair statistics.
random=ip[ip.stratum.eq('random')].copy();selected=[]
qidx=qp.set_index('pair_id');sg=np.sign(random.delta_matching)
eligible=(np.sign(random.delta_single)==sg)&(sg!=0)&(random.choice_a==(sg+1)/2)
for col in ['delta_single','delta_matching']:eligible &= np.sign(random.pair_id.map(qidx[col]))==sg
eligible &= random.pair_id.map(qidx.choice_a)==(sg+1)/2
pool=random[eligible]
assert len(pool)>2
for quantile in [.25,.75]:
 target=random.delta_matching.quantile(quantile);row=pool.assign(dist=(pool.delta_matching-target).abs()).sort_values(['dist','pair_id']).iloc[0];selected.append(int(row.pair_id))
p[p.pair_id.isin(selected)].to_csv(O/'data/example_pairs.csv',index=False)
meta=pd.read_csv(O/'data/internvl_images.csv').set_index('image_id')
for iid in sorted(set(ip[ip.pair_id.isin(selected)].a_id)|set(ip[ip.pair_id.isin(selected)].b_id)):
 shutil.copy2(B/meta.loc[iid,'image_512_path'],O/'images'/f'{iid}.png')
for name in ['build_paper_figures_v1.py','analyze_qwen3_full.py','revise_nmi_stats.py']:copy(B/'scripts'/name,folder='provenance_scripts')
(O/'source_hashes.json').write_text(json.dumps({str(p.relative_to(B)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2))
print('Prepared Figure 4 source tables and examples.')
