from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,pandas as pd,json,hashlib,shutil
B=Path(__file__).resolve().parents[1];M=B/'mllm_material';S=M/'mechanisms_v1';O=M/'figure5_english_v1'
for folder in ['data','provenance_scripts']:(O/folder).mkdir(parents=True,exist_ok=True)
sources=[]
def cp(p,folder='data',name=None):sources.append(p);shutil.copy2(p,O/folder/(name or p.name))
for name in ['layer_readouts.csv','readout_predictions.csv.gz']:cp(S/'tables'/name)
cp(M/'nmi_revision_v8/source/tables/v8_readout_ci.csv');cp(S/'stimuli.csv');cp(S/'Methods.md',name='original_mechanism_methods.md');cp(S/'plan.json');cp(M/'protocol/prompts.json')
for name in ['analyze_mechanism_v1.py','mechanism_backend_v1.py','run_mechanism_v1.py']:cp(S/'code'/name,'provenance_scripts')
for model in ['InternVL','Qwen3']:cp(S/model/'layer_spec.json',name=model+'_layer_spec.json')
d=pd.read_csv(S/'stimuli.csv');pred=pd.read_csv(S/'tables/readout_predictions.csv.gz');stats=pd.read_csv(S/'tables/layer_readouts.csv')
# Reconstruct the exact archived training-only nuisance baseline, for a paired gain interval.
N=np.column_stack([np.ones(len(d)),pd.get_dummies(d.physical_c).to_numpy()[:,1:],pd.get_dummies(d.chroma).to_numpy()[:,1:]]+[f(n*np.deg2rad(d.hue.to_numpy())) for n in [1,2] for f in [np.sin,np.cos]])
y=d.human_c.to_numpy();rng=np.random.default_rng(20261004);cs=rng.multinomial(36,np.ones(36)/36,2000);cl=rng.multinomial(36,np.ones(36)/36,2000);w=cs[:,d.shape_id.to_numpy()-1]*cl[:,d.illumination_id.to_numpy()-1];sw=w.sum(1);den=w@(y*y)-(w@y)**2/sw
rows=[];baseline=pd.DataFrame({'image_id':d.image_id,'human_c':y});draws={}
for split in ['shape','illumination']:
 folds=d[split+'_fold'].to_numpy();b=np.zeros(len(d))
 for k in range(12):
  tr=folds!=k;b[~tr]=N[~tr]@np.linalg.lstsq(N[tr],y[tr],rcond=None)[0]
 base_r2=1-np.mean((y-b)**2)/np.var(y);baseline[split+'_baseline']=b
 assert np.allclose(stats[(stats.split==split)&stats.target.eq('human_beyond_physical_color')].baseline_r2,base_r2)
 for model in ['InternVL','Qwen3']:
  z=pred[(pred.model==model)&(pred.layer=='language_36')&(pred.split==split)&pred.target.eq('human_beyond_physical_color')].set_index('image_id').loc[d.image_id];p=z.prediction.to_numpy();gain=(w@((y-b)**2-(y-p)**2))/den;point=(np.mean((y-b)**2)-np.mean((y-p)**2))/np.var(y);lo,hi=np.quantile(gain,[.025,.975]);rows.append(dict(model=model,split=split,layer='language_36',baseline_r2=base_r2,combined_r2=base_r2+point,gain_r2=point,ci_low=lo,ci_high=hi));draws[model+'_'+split]=gain
baseline.to_csv(O/'data/nuisance_baseline_predictions.csv',index=False);pd.DataFrame(rows).to_csv(O/'data/final_layer_increment.csv',index=False);np.savez_compressed(O/'data/increment_bootstrap.npz',**draws)
(O/'data/increment_analysis.json').write_text(json.dumps(dict(seed=20261004,n_bootstrap=2000,method='Paired crossed shape x illumination multinomial bootstrap of fixed OOF predictions; no model or nuisance refit',layer_selection='Final sampled language block, not best held-out block',statistic='R2 combined prediction minus R2 physical/color baseline; denominator is human variance in each draw'),indent=2))
(O/'source_hashes.json').write_text(json.dumps({str(p.relative_to(B)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2));print(pd.DataFrame(rows).to_string(index=False))
