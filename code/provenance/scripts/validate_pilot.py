"""Independent saved-output checks, without rerunning feature extraction or model fitting."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr,spearmanr
BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'pilot/results'

def main():
    a=pd.read_csv(BASE/'annotations/gloss_annotations.csv').set_index('image_id')
    folds=pd.read_csv(BASE/'protocols/gloss_group_folds.csv').set_index('image_id')
    pred=pd.read_csv(OUT/'all_oof_predictions.csv')
    detail=pd.read_csv(OUT/'fold_metrics.csv')
    summary=pd.read_csv(OUT/'model_summary.csv')
    assert len(pred)==85536 and len(summary)==22 and len(detail)==264
    assert not pred.duplicated(['split','model','image_id']).any()
    for (split,model),p in pred.groupby(['split','model']):
        assert set(p.image_id)==set(a.index)
        assert np.allclose(p.human,a.loc[p.image_id].human_gloss_score,rtol=0,atol=1e-14)
        assert np.array_equal(p.fold,folds.loc[p.image_id,split+'_fold'])
        correlations=[]
        for f,test in p.groupby('fold'):
            ids=test.image_id
            train=folds[folds[split+'_fold']!=f]
            assert set(folds.loc[ids,split+'_id']).isdisjoint(train[split+'_id'])
            record=detail[(detail.split==split)&(detail.model==model)&(detail.fold==f)].iloc[0]
            assert len(test)==324
            assert np.isclose(record.mae,np.mean(abs(test.human-test.prediction)),rtol=0,atol=1e-12)
            assert np.isclose(record.rmse,np.sqrt(np.mean((test.human-test.prediction)**2)),rtol=0,atol=1e-12)
            if model=='train_mean':
                assert np.allclose(test.prediction,a.loc[train.index].human_gloss_score.mean())
                assert pd.isna(record.spearman)
            else:
                r=spearmanr(test.human,test.prediction)[0];correlations.append(r)
                assert np.isclose(record.spearman,r,atol=1e-12)
        if correlations:
            saved=summary[(summary.split==split)&(summary.model==model)].iloc[0].mean_fold_spearman
            assert np.isclose(saved,np.mean(correlations),atol=1e-12)
    for name,dim in [('clip_b32',512),('dinov2_small',384),('dinov2_same_fov',384)]:
        p=BASE/'pilot/features'/(name+'.npy');x=np.load(p)
        assert x.shape==(3888,dim) and np.isfinite(x).all()
        assert np.allclose(np.linalg.norm(x,axis=1),1,atol=1e-5)
        m=json.loads((p.parent/(name+'_manifest.json')).read_text())
        assert hashlib.sha256(p.read_bytes()).hexdigest()==m['feature_sha256']
    validation=json.loads((OUT/'validation.json').read_text())
    validation['independent_saved_output_validation']=dict(passed=True,rows_checked=len(pred),fold_records_checked=len(detail),feature_dimensions_and_norms=True,feature_hashes=True,metrics_recomputed=True,target_and_fold_alignment=True)
    files=[BASE/'protocols/pilot_v1_config.json',BASE/'protocols/pilot_v1_preprocessing_sensitivity.json',BASE/'protocols/gloss_group_folds.csv']+[BASE/'scripts'/n for n in ['extract_pilot_features.py','evaluate_pilot.py','report_pilot.py','validate_pilot.py']]
    validation['provenance_sha256']={p.relative_to(BASE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2,allow_nan=False),encoding='utf-8')
    # Standards-compliant JSON: undefined constant-prediction correlations remain null.
    for p in OUT.glob('*_metrics.json'):
        obj=json.loads(p.read_text(),parse_constant=lambda value:None)
        p.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(validation['independent_saved_output_validation'],indent=2))

if __name__=='__main__':main()
