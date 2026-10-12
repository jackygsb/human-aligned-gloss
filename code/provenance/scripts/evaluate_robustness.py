"""Nested grouped, train-only PCA controls and paired input ablations."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import numpy as np,pandas as pd,json,time,argparse
from sklearn.model_selection import GroupKFold
from evaluate_pilot import ridge_predictions,metrics,sha
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'robustness_v2/results';OUT.mkdir(parents=True,exist_ok=True)

def project(train,others):
    mu=train.mean(0);_,_,v=np.linalg.svd(train-mu,full_matrices=False);basis=v[:64].T
    return (train-mu)@basis,[(x-mu)@basis for x in others]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base-only',action='store_true');args=parser.parse_args()
    a=pd.read_csv(BASE/'annotations/gloss_annotations.csv');folds=pd.read_csv(BASE/'protocols/gloss_group_folds.csv');y=a.human_gloss_score.to_numpy()
    assert a.image_id.tolist()==folds.image_id.tolist()
    stats=np.load(BASE/'pilot/features/image_stats.npy');names=pd.read_csv(BASE/'pilot/features/image_stats_columns.csv').feature.tolist()
    c=a.physical_pellacini_c.to_numpy();h=np.deg2rad(a.hue.to_numpy());ch=a.chroma.to_numpy()
    phys=np.column_stack([c,c*c,c**3]);color=np.column_stack([np.sin(h),np.cos(h),ch,ch*np.sin(h),ch*np.cos(h),c*ch,c*np.sin(h),c*np.cos(h)])
    specs={'physics_color_stats':(np.column_stack([phys,color,stats]),False),
        'background_stats_only':(stats[:,[i for i,n in enumerate(names) if n.startswith('background_')]],False),
        'object_stats_only':(stats[:,[i for i,n in enumerate(names) if n.startswith('object_')]],False),
        'clip_rgb64':(np.load(BASE/'pilot/features/clip_b32.npy').astype(float),True),
        'dino_rgb64':(np.load(BASE/'pilot/features/dinov2_same_fov.npy').astype(float),True)}
    if not args.base_only:
        assert a.image_id.tolist()==pd.read_csv(BASE/'robustness_v2/features/index.csv').image_id.tolist()
        for model in ['clip','dino']:
            for variant in ['nobg','gray']:specs[f'{model}_{variant}64']=(np.load(BASE/f'robustness_v2/features/{model}_{variant}.npy').astype(float),True)
    records=[];allpred=[];alphas=[.01,.1,1,10,100,1000,10000]
    for factor in ['shape','illumination']:
        group=folds[factor+'_id'].to_numpy()
        for name,(x,reduce) in specs.items():
            path=OUT/f'{factor}_{name}.csv';metapath=OUT/f'{factor}_{name}.json'
            if path.exists():
                saved=json.loads(metapath.read_text())
                expected=dict(feature_sha256=__import__('hashlib').sha256(x.tobytes()).hexdigest(),annotation_sha256=sha(BASE/'annotations/gloss_annotations.csv'),folds_sha256=sha(BASE/'protocols/gloss_group_folds.csv'),config_sha256=sha(BASE/'protocols/robustness_v2_config.json'))
                assert all(saved[k]==v for k,v in expected.items()),'Cached inputs changed; use a new run directory.'
                allpred.append(pd.read_csv(path));records.extend(saved['fold_metrics']);continue
            t=time.time();pred=np.zeros(len(a));rows=[]
            for fold in range(12):
                test=np.flatnonzero(folds[factor+'_fold'].to_numpy()==fold);train=np.flatnonzero(folds[factor+'_fold'].to_numpy()!=fold)
                assert set(group[test]).isdisjoint(group[train]);loss=[]
                for itr,iva in GroupKFold(3).split(train,groups=group[train]):
                    ti=train[itr];vi=train[iva];assert set(group[ti]).isdisjoint(group[vi])
                    if reduce:xx,(vv,)=project(x[ti],[x[vi]])
                    else:xx,vv=x[ti],x[vi]
                    pp=ridge_predictions(xx,y[ti],vv,alphas);loss.append(np.mean((pp-y[vi,None])**2,axis=0))
                alpha=alphas[int(np.argmin(np.mean(loss,axis=0)))]
                if reduce:xx,(vv,)=project(x[train],[x[test]])
                else:xx,vv=x[train],x[test]
                pp=ridge_predictions(xx,y[train],vv,[alpha])[:,0];pred[test]=pp
                rows.append(dict(split=factor,model=name,fold=fold,alpha=alpha,dimensions=xx.shape[1],**metrics(y[test],pp)))
            frame=pd.DataFrame(dict(image_id=a.image_id,split=factor,model=name,fold=folds[factor+'_fold'],human=y,prediction=pred));frame.to_csv(path,index=False)
            meta=dict(fold_metrics=rows,feature_sha256=__import__('hashlib').sha256(x.tobytes()).hexdigest(),annotation_sha256=sha(BASE/'annotations/gloss_annotations.csv'),folds_sha256=sha(BASE/'protocols/gloss_group_folds.csv'),config_sha256=sha(BASE/'protocols/robustness_v2_config.json'),seconds=time.time()-t)
            metapath.write_text(json.dumps(meta,indent=2));allpred.append(frame);records.extend(rows)
            print(factor,name,'rho',np.mean([r['spearman'] for r in rows]),'seconds',round(time.time()-t,1),flush=True)
    d=pd.DataFrame(records);d.to_csv(OUT/'fold_metrics.csv',index=False);pd.concat(allpred).to_csv(OUT/'all_predictions.csv',index=False)
    s=d.groupby(['split','model'])[['spearman','pearson','mae','rmse']].mean().reset_index();s.to_csv(OUT/'summary.csv',index=False);print(s.to_string(index=False))

if __name__=='__main__':main()
