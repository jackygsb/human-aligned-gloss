"""Cross-fitted residual association; regress each system separately on physical cues."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.stats import spearmanr
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'mllm_material';ann=pd.read_csv(BASE/'annotations/gloss_annotations.csv');folds=pd.read_csv(BASE/'protocols/gloss_group_folds.csv');assert ann.image_id.tolist()==folds.image_id.tolist()
c=ann.physical_pellacini_c.to_numpy()/.1487;stats=np.load(BASE/'pilot/features/image_stats.npy');alphas=np.asarray(json.loads((BASE/'protocols/pilot_v1_config.json').read_text())['ridge_alphas'])
def ridge(x,y,t):
    mu=x.mean(0);sd=x.std(0);sd[sd<1e-12]=1;z=(x-mu)/sd;zt=(t-mu)/sd;ev,u=np.linalg.eigh(z.T@z);ev=np.maximum(ev,0);coef=u.T@(z.T@(y-y.mean(0)))
    return np.stack([(zt@u)@(coef/(ev[:,None]+a))+y.mean(0) for a in alphas],axis=1)
results=[]
for path in (OUT/'statistics').glob('*_image_scores.csv'):
    d=pd.read_csv(path);model=path.name.replace('_image_scores.csv','');y=np.column_stack([ann.human_gloss_score,d.model_score]);valid=np.isfinite(y).all(1)
    for factor in ['shape','illumination']:
        group=folds[factor+'_fold'].to_numpy();identity=folds[factor+'_id'].to_numpy()
        for condition in ['quadratic','cubic','natural_spline_df4','physical_plus_image_statistics']:
            prediction=np.full_like(y,np.nan)
            for fold in range(12):
                # Fold numbering in the stored protocol is inspected, not assumed here.
                level=sorted(np.unique(group))[fold];train=(group!=level)&valid;test=(group==level)&valid
                if not test.any():continue
                if condition=='natural_spline_df4':
                    knots=np.quantile(c[train],[0,1/3,2/3,1]);x=CubicSpline(knots,np.eye(4),bc_type='natural')(c)
                else:x=np.column_stack([np.ones(len(c)),c,c*c]+([c*c*c] if condition!='quadratic' else []))
                if condition=='physical_plus_image_statistics':
                    x=np.column_stack([x[:,1:],stats]);inner=np.full_like(y,np.nan)[:,None,:].repeat(len(alphas),axis=1);groups=sorted(np.unique(identity[train]));allocation={g:k%3 for k,g in enumerate(groups)}
                    for k in range(3):
                        iv=train&np.array([allocation.get(g,-1)==k for g in identity]);it=train&~iv;inner[iv]=ridge(x[it],y[it],x[iv])
                    losses=np.nanmean((inner[train]-y[train,None,:])**2,axis=0);chosen=losses.argmin(0);p=ridge(x[train],y[train],x[test]);prediction[test]=p[:,chosen,np.arange(2)]
                else:prediction[test]=x[test]@np.linalg.lstsq(x[train],y[train],rcond=None)[0]
            residual=y-prediction;m=np.isfinite(residual).all(1);rho=float(spearmanr(residual[m,0],residual[m,1])[0]);results.append(dict(model=model,factor=factor,condition=condition,n=int(m.sum()),residual_spearman=rho,interpretation='cross-fitted auxiliary association; not a causal explanation'))
            pd.DataFrame(dict(image_id=ann.image_id,human_residual=residual[:,0],model_residual=residual[:,1])).to_csv(OUT/'statistics'/f'residual_{model}_{factor}_{condition}.csv',index=False)
pd.DataFrame(results).to_csv(OUT/'statistics/residual_control_summary.csv',index=False);print('Residual comparisons:',len(results))
