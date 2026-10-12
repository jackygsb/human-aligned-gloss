"""Nested grouped ridge evaluation; all scaling and tuning remain inside training data."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,hashlib,argparse,time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,pearsonr
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'pilot/results'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def ridge_predictions(x,y,test,alphas):
    mu=x.mean(0);std=x.std(0);std[std<1e-12]=1
    z=(x-mu)/std;zt=(test-mu)/std
    ev,u=np.linalg.eigh(z.T@z);ev=np.maximum(ev,0)
    coeff=u.T@(z.T@(y-y.mean()))
    return (zt@u)@(coeff[:,None]/(ev[:,None]+np.asarray(alphas)[None,:]))+y.mean()

def check_solver():
    rng=np.random.default_rng(12);x=rng.normal(size=(90,9));x[:,8]=1;y=rng.normal(size=90);t=rng.normal(size=(15,9));t[:,8]=1
    scaler=StandardScaler().fit(x)
    p=Ridge(alpha=10,solver='svd').fit(scaler.transform(x),y).predict(scaler.transform(t))
    assert np.allclose(p,ridge_predictions(x,y,t,[10])[:,0],atol=1e-10)

def metrics(y,p):
    constant=np.std(p)<1e-12
    return dict(spearman=np.nan if constant else float(spearmanr(y,p)[0]),pearson=np.nan if constant else float(pearsonr(y,p)[0]),
        mae=float(np.mean(abs(y-p))),rmse=float(np.sqrt(np.mean((y-p)**2))))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--baselines-only',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);check_solver()
    config=json.loads((BASE/'protocols/pilot_v1_config.json').read_text())
    a=pd.read_csv(BASE/'annotations/gloss_annotations.csv')
    folds=pd.read_csv(BASE/'protocols/gloss_group_folds.csv')
    assert a.image_id.tolist()==folds.image_id.tolist()==pd.read_csv(BASE/'pilot/features/index.csv').image_id.tolist()
    y=a.human_gloss_score.to_numpy();c=a.physical_pellacini_c.to_numpy()
    phys=np.column_stack([c,c*c,c*c*c]);stats=np.load(BASE/'pilot/features/image_stats.npy')
    low=np.column_stack([phys,stats])
    models={'train_mean':None,'physical_direct':None,'physical_calibrated':phys,'image_stats':stats,'physical_plus_stats':low}
    if not args.baselines_only:
        for model in ['clip_b32','dinov2_small','dinov2_same_fov']:
            x=np.load(BASE/'pilot/features'/(model+'.npy')).astype(float)
            assert x.shape[0]==len(a) and np.isfinite(x).all()
            models[model]=x
            suffix={'clip_b32':'clip','dinov2_small':'dino','dinov2_same_fov':'dino_same_fov'}[model]
            models['physical_stats_plus_'+suffix]=np.column_stack([low,x])
    alphas=np.array(config['ridge_alphas'])
    records=[];predictions=[];checks=[]
    for factor in ['shape','illumination']:
        groups=folds[factor+'_id'].to_numpy()
        for model,x in models.items():
            outputfile=OUT/f'{factor}_{model}.csv'
            metafile=OUT/f'{factor}_{model}_metrics.json'
            fingerprint={'target':sha(BASE/'annotations/gloss_annotations.csv'),'folds':sha(BASE/'protocols/gloss_group_folds.csv'),'config':sha(BASE/'protocols/pilot_v1_config.json'),'features':None if x is None else hashlib.sha256(x.tobytes()).hexdigest()}
            if outputfile.exists() and metafile.exists():
                saved=json.loads(metafile.read_text())
                assert saved['input_fingerprint']==fingerprint,'Inputs changed; choose a new pilot run directory.'
                records.extend(saved['fold_metrics']);predictions.append(pd.read_csv(outputfile));continue
            start=time.time();oof=np.full(len(a),np.nan);model_records=[]
            for f in range(12):
                te=np.flatnonzero(folds[factor+'_fold'].to_numpy()==f);tr=np.flatnonzero(folds[factor+'_fold'].to_numpy()!=f)
                assert set(groups[tr]).isdisjoint(groups[te])
                alpha=np.nan;inner_mse=np.nan
                if model=='train_mean':p=np.repeat(y[tr].mean(),len(te))
                elif model=='physical_direct':p=c[te]
                else:
                    mse=[]
                    for itr,iva in GroupKFold(n_splits=3).split(x[tr],y[tr],groups[tr]):
                        train=tr[itr];valid=tr[iva]
                        assert set(groups[train]).isdisjoint(groups[valid])
                        assert set(train).isdisjoint(te) and set(valid).isdisjoint(te)
                        ip=ridge_predictions(x[train],y[train],x[valid],alphas)
                        mse.append(((ip-y[valid,None])**2).mean(0))
                    errors=np.mean(mse,axis=0);idx=int(np.argmin(errors));alpha=float(alphas[idx]);inner_mse=float(errors[idx])
                    p=ridge_predictions(x[tr],y[tr],x[te],[alpha])[:,0]
                oof[te]=p
                rec=dict(split=factor,model=model,fold=f,n_train=len(tr),n_test=len(te),selected_alpha=alpha,inner_mse=inner_mse,**metrics(y[te],p))
                model_records.append(rec)
            assert np.isfinite(oof).all()
            frame=pd.DataFrame(dict(image_id=a.image_id,split=factor,model=model,fold=folds[factor+'_fold'],human=y,physical=c,prediction=oof))
            frame.to_csv(outputfile,index=False)
            payload=json.loads(json.dumps(dict(input_fingerprint=fingerprint,fold_metrics=model_records,seconds=time.time()-start)),parse_constant=lambda value:None)
            metafile.write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')
            predictions.append(frame);records.extend(model_records)
            print(factor,model,'mean fold rho',np.nanmean([r['spearman'] for r in model_records]) if model!='train_mean' else 'undefined','seconds',round(time.time()-start,1),flush=True)
    detail=pd.DataFrame(records);detail.to_csv(OUT/'fold_metrics.csv',index=False)
    allpred=pd.concat(predictions,ignore_index=True);allpred.to_csv(OUT/'all_oof_predictions.csv',index=False)
    summary=detail.groupby(['split','model'])[['spearman','pearson','mae','rmse']].mean().rename(columns=lambda x:'mean_fold_'+x).reset_index()
    for i,r in summary.iterrows():
        frame=allpred[(allpred.split==r['split'])&(allpred.model==r.model)]
        for k,v in metrics(frame.human,frame.prediction).items():summary.loc[i,'pooled_oof_'+k]=v
    summary.to_csv(OUT/'model_summary.csv',index=False)
    # Test-fold baseline residual target; no test-fitted nuisance regression.
    increments=[]
    for factor in ['shape','illumination']:
        sub=allpred[allpred.split==factor]
        base=sub[sub.model=='physical_plus_stats'].set_index('image_id')
        for model in models:
            if model=='physical_plus_stats':continue
            p=sub[sub.model==model].set_index('image_id').loc[base.index]
            increments.append(dict(split=factor,model=model,delta_mse_vs_physical_stats=float(np.mean((base.human-base.prediction)**2)-np.mean((p.human-p.prediction)**2)),
                test_residual_increment_pearson=metrics(base.human-base.prediction,p.prediction-base.prediction)['pearson']))
    pd.DataFrame(increments).to_csv(OUT/'incremental_diagnostics.csv',index=False)
    report=dict(passed=True,annotations_unchanged_sha256=sha(BASE/'annotations/gloss_annotations.csv'),outer_and_inner_group_separation=True,
        ridge_solver_verified_against_sklearn=True,each_image_tested_once_per_split_model=True,models=list(models),
        total_oof_predictions=len(allpred),config_sha256=sha(BASE/'protocols/pilot_v1_config.json'),
        caveat='Descriptive exploratory results, not causal evidence. Fold means are primary. Constant-fold correlation undefined. No p-values or human-level claim.')
    (OUT/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(summary.to_string(index=False),flush=True)

if __name__=='__main__':main()
