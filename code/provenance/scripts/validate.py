"""Independent validation of saved labels, trial aggregation and every delivered image."""
from pathlib import Path
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from PIL import Image

BASE=Path(__file__).resolve().parents[1]

def main():
    a=pd.read_csv(BASE/'annotations/gloss_annotations.csv')
    t=pd.read_csv(BASE/'annotations/gloss_trials.csv')
    v=np.load(BASE/'annotations/human_gloss_vector.npy')
    assert len(a)==3888 and a.image_id.is_unique and a.source_image_index.is_unique
    assert not a.isna().any().any()
    assert len(a[['shape_id','illumination_id','viewpoint_id']].drop_duplicates())==3888
    assert np.array_equal(a.source_image_index,np.arange(1,3889))
    assert np.allclose(v,a.human_gloss_score,rtol=0,atol=1e-14)
    assert pd.read_csv(BASE/'annotations/human_gloss_vector_index.csv').image_id.tolist()==a.image_id.tolist()
    assert a.human_gloss_score.between(0,.1487).all()
    assert np.allclose((a.physical_pellacini_c+(.5**3/2)**(1/3))**3-.5**3/2,a.physical_specular_reflectance,atol=1e-12,rtol=0)
    main=t[t.is_main_image]
    assert not t.duplicated(['participant_id','repetition','test_image_id']).any()
    assert main.groupby(['participant_id','image_id']).size().eq(2).all()
    g=main.groupby(['image_id','participant_id']).human_pellacini_c.mean().groupby('image_id')
    assert np.allclose(g.mean().reindex(a.image_id),a.human_gloss_score,rtol=0,atol=1e-12)
    assert np.allclose(g.std().reindex(a.image_id),a.human_gloss_std,rtol=0,atol=1e-12)
    assert np.array_equal(g.count().reindex(a.image_id),a.n_observers)
    assert len(t)==295*84*2 and len(main)==295*72*2
    manifest=pd.read_csv(BASE/'metadata/image_manifest.csv').set_index('image_id')
    def check(r):
        for field,size in [('image_path',512),('image_224_path',224),('image_512_path',512)]:
            p=BASE/getattr(r,field)
            with Image.open(p) as im:
                im.load(); assert im.mode=='RGB' and im.size==(size,size),str(p)
            assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest.loc[r.image_id,f'output_sha256_{size}']
        return 3
    with ThreadPoolExecutor(max_workers=8) as pool:
        checked=sum(pool.map(check,a.itertuples(index=False)))
    result={'passed':True,'annotation_rows':len(a),'images_decoded_and_hashed':checked,'missing_labels':0,'corrupt_delivered_images':0,'trial_count':len(t),'main_trial_count':len(main),'vector_matches_csv':True,'saved_trials_reproduce_mean_std_and_counts':True,'physical_transform_verified':True}
    (BASE/'metadata/validation_report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
