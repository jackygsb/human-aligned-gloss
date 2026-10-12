"""Hold RGB-trained readout fixed; diagnose sensitivity, not human counterfactuals."""
from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import numpy as np,pandas as pd,json
from evaluate_pilot import ridge_predictions,metrics
from evaluate_robustness import project
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'robustness_v2/results'

def main():
    a=pd.read_csv(BASE/'annotations/gloss_annotations.csv');f=pd.read_csv(BASE/'protocols/gloss_group_folds.csv');y=a.human_gloss_score.to_numpy();records=[];frames=[]
    for model,filename in [('clip','clip_b32'),('dino','dinov2_same_fov')]:
        orig=np.load(BASE/f'pilot/features/{filename}.npy').astype(float)
        alternatives={v:np.load(BASE/f'robustness_v2/features/{model}_{v}.npy').astype(float) for v in ['nobg','gray']}
        for factor in ['shape','illumination']:
            previous=pd.read_csv(OUT/f'{factor}_{model}_rgb64.csv').set_index('image_id')
            audit=json.loads((OUT/f'{factor}_{model}_rgb64.json').read_text())
            for fold in range(12):
                tr=np.flatnonzero(f[factor+'_fold'].to_numpy()!=fold);te=np.flatnonzero(f[factor+'_fold'].to_numpy()==fold)
                xx,transformed=project(orig[tr],[orig[te]]+[alternatives[v][te] for v in ['nobg','gray']])
                alpha=next(r['alpha'] for r in audit['fold_metrics'] if r['fold']==fold)
                pp=[ridge_predictions(xx,y[tr],t,[alpha])[:,0] for t in transformed]
                assert np.allclose(pp[0],previous.loc[a.iloc[te].image_id].prediction,atol=1e-10)
                for condition,p in zip(['rgb','shift_nobg','shift_gray'],pp):
                    records.append(dict(model=model,split=factor,fold=fold,condition=condition,mean_abs_prediction_change=float(np.mean(abs(p-pp[0]))),mean_signed_prediction_change=float(np.mean(p-pp[0])),**metrics(y[te],p)))
                    frames.append(pd.DataFrame(dict(image_id=a.iloc[te].image_id,model=model,split=factor,fold=fold,condition=condition,prediction=p,original_prediction=pp[0],original_human=y[te])))
    d=pd.DataFrame(records);d.to_csv(OUT/'input_shift_fold_metrics.csv',index=False)
    d.groupby(['model','split','condition'])[['spearman','mae','mean_abs_prediction_change','mean_signed_prediction_change']].mean().reset_index().to_csv(OUT/'input_shift_summary.csv',index=False)
    pd.concat(frames).to_csv(OUT/'input_shift_predictions.csv',index=False)
    print('Fixed-readout shift diagnostics complete; original RGB predictions reproduced.')

if __name__=='__main__':main()
