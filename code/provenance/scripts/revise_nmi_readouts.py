from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import pandas as pd,numpy as np
B=Path(__file__).resolve().parents[1];M=B/'mllm_material';O=M/'nmi_revision_v8/source'
d=pd.read_csv(M/'mechanisms_v1/tables/readout_predictions.csv.gz');ann=pd.read_csv(M/'qwen3_full_v1/joint_model_predictions.csv')[['image_id','shape_id','illumination_id']].sort_values('image_id')
rng=np.random.default_rng(20261002);s=pd.Categorical(ann.shape_id).codes;l=pd.Categorical(ann.illumination_id).codes
w=(rng.multinomial(36,np.ones(36)/36,2000)[:,s]*rng.multinomial(36,np.ones(36)/36,2000)[:,l]).astype(float);tot=w.sum(axis=1);out=[]
for keys,z in d[d.target.isin(['human','human_beyond_physical_color'])].groupby(['model','layer','split','target']):
 z=ann[['image_id']].merge(z,on='image_id',validate='one_to_one');assert len(z)==3888
 y=z.truth.to_numpy();p=z.prediction.to_numpy();v=1-(w@((y-p)**2))/(w@(y*y)-(w@y)**2/tot);lo,hi=np.quantile(v,[.025,.975]);out.append(dict(zip(['model','layer','split','target'],keys),estimate=1-np.mean((y-p)**2)/np.var(y),ci_low=lo,ci_high=hi))
pd.DataFrame(out).to_csv(O/'tables/v8_readout_ci.csv',index=False)
print('Readout conditional intervals',len(out))
