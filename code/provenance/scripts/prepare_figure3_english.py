from pathlib import Path
import shutil,json,hashlib
import numpy as np,pandas as pd
from scipy.stats import spearmanr
B=Path(__file__).resolve().parents[1];M=B/'mllm_material';O=M/'figure3_english_v1'
for s in ['data','images','provenance_scripts']:(O/s).mkdir(parents=True,exist_ok=True)
sources=[]
def cp(p,sub='data'):
 sources.append(p);shutil.copy2(p,O/sub/p.name)
for p in [M/'nmi_revision_v8/source/tables/v8_condition_tests.csv',M/'behavioral_spaces_v1/tables/rsa_results.csv',M/'behavioral_spaces_v1/tables/cross_view_rsa.csv',M/'qwen3_full_v1/joint_model_predictions.csv']:cp(p)
old=pd.read_csv(M/'paper_figures_v1/tables/Table6_shared_condition_effects.csv');rows=[];gallery=[]
stim=pd.read_csv(M/'mechanisms_v1/stimuli.csv');sources.append(M/'mechanisms_v1/stimuli.csv')
for factor in ['shape','illumination']:
 key=factor+'_id';q=pd.read_csv(M/f'qwen3_full_v1/{key}_effects.csv');i=old[old.factor.eq(key)].sort_values('condition_id');q=q.sort_values('condition_id');assert np.allclose(i.human_adjusted_sd,q.human_adjusted_sd)
 t=pd.DataFrame({'factor':factor,'condition_id':q.condition_id,'Human':q.human_adjusted_sd,'InternVL':i.model_adjusted_sd.to_numpy(),'Qwen3':q.model_adjusted_sd});rows.append(t)
 sources += [M/f'qwen3_full_v1/{key}_effects.csv']
 for agent in ['Human','InternVL','Qwen3']:
  p=M/f'behavioral_spaces_v1/matrices/physical_color_adjusted_{factor}_{agent}_rdm.csv';cp(p)
  R=np.loadtxt(p,delimiter=',');assert R.shape==(36,36) and np.allclose(R,R.T) and np.allclose(np.diag(R),0)
 for direction in ['Lower','Higher']:
  cond=int(t.loc[t.Human.idxmin() if direction=='Lower' else t.Human.idxmax(),'condition_id']);s=stim[(stim[key]==cond)&(stim.viewpoint_id==1)].copy();s['dist']=(s.physical_c-stim.physical_c.median()).abs();s=s.sort_values(['dist','image_id']).head(3)
  for _,r in s.iterrows():
   shutil.copy2(B/r.image_512_path,O/'images'/f'{r.image_id}.png');gallery.append({'factor':factor,'direction':direction,'condition_id':cond,'image_id':r.image_id,'physical_c':r.physical_c,'human_c':r.human_c,'InternVL':r.internvl_c,'Qwen3':r.model_c})
pd.concat(rows).to_csv(O/'data/condition_effects.csv',index=False);pd.DataFrame(gallery).to_csv(O/'data/gallery.csv',index=False)
sources.append(M/'paper_figures_v1/tables/Table6_shared_condition_effects.csv')
for name in ['analyze_behavioral_spaces_v1.py','analyze_qwen3_full.py','revise_nmi_stats.py']:cp(B/'scripts'/name,'provenance_scripts')
(O/'source_hashes.json').write_text(json.dumps({str(p.relative_to(B)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2))
print(O)
