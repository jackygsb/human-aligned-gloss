"""Verify saved outputs against the original Excel medians and image contents."""
import json
import numpy as np
import pandas as pd
from PIL import Image
from preprocess_material import BASE, SOURCE, ATTRS, read_source, digest, write_json

def main():
    source=read_source()
    a=pd.read_csv(BASE/'annotations/material_annotations.csv',keep_default_na=False)
    expected=source[ATTRS].apply(pd.to_numeric).to_numpy(dtype=float)
    assert len(a)==42120 and a.image_id.is_unique
    assert a.source_image_filename.tolist()==source.image.tolist()
    assert a.source_excel_row.tolist()==list(range(2,42122))
    assert np.array_equal(a[ATTRS].to_numpy(dtype=float),expected)
    assert np.array_equal(np.load(BASE/'annotations/material_attribute_matrix.npy'),expected)
    idx=pd.read_csv(BASE/'annotations/material_attribute_matrix_index.csv')
    assert idx.image_id.tolist()==a.image_id.tolist()
    assert idx.image_available.tolist()==a.image_available.tolist()
    assert a.source_illumination_code.tolist()==source.illumination.tolist()
    assert a.source_geometry_code.tolist()==source.geometry.tolist()
    assert a.viewpoint_id.eq('').all()
    assert a.groupby(['material_id','shape_id','illumination_id']).size().eq(1).all()
    assert a.groupby(['shape_id','illumination_id']).size().eq(520).all()
    assert a.groupby(['material_id','shape_id']).size().eq(9).all()
    present=a[a.image_available]; absent=a[~a.image_available]
    assert present.image_status.eq('available').all() and absent.image_status.eq('missing_local').all()
    assert absent.image_path.eq('').all() and absent.local_source_image_path.eq('').all()
    subset=pd.read_csv(BASE/'annotations/material_available_images.csv',keep_default_na=False)
    pd.testing.assert_frame_equal(subset,present.reset_index(drop=True))
    missing=pd.read_csv(BASE/'metadata/material_missing_images.csv')
    assert missing.image_id.tolist()==absent.image_id.tolist()
    manifest=pd.read_csv(BASE/'metadata/material_image_manifest.csv').set_index('image_id')
    for r in present.itertuples():
        p=BASE/r.image_path
        with Image.open(p) as im:
            im.load(); assert im.mode=='RGB'
        assert digest(p)==manifest.loc[r.image_id,'sha256']==digest(BASE.parent/r.local_source_image_path)
    assert len(list((BASE/'images/material').glob('*.jpg')))==len(present)
    summary=json.loads((BASE/'metadata/material_dataset_summary.json').read_text(encoding='utf-8'))
    assert digest(SOURCE)==summary['source_sha256']
    result=dict(passed=True,rows=len(a),rating_cells_exactly_matching_source=int(expected.size),
        matrix_shape=list(expected.shape),missing_rating_cells=0,images_present=len(present),
        images_missing=len(absent),image_paths_empty_when_missing=True,local_images_decoded_and_hash_verified=len(present),
        factor_design_verified='520 materials x 9 shapes x 9 illuminations',source_workbook_unchanged=True)
    write_json(BASE/'metadata/material_validation_report.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
