"""Prepare all author-provided image medians; no downloads or raw-trial filtering."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import numpy as np
import pandas as pd
import openpyxl
from PIL import Image

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
SOURCE = ROOT / 'Serrano_SIGG2021_PerceptualRatings.xlsx'
ATTRS = ['glossiness', 'refsharp', 'contgloss', 'metallicness', 'lightness', 'anisotropy']
LABELS = ['光泽度', '反射清晰度', '反射对比度', '金属感', '明度', '各向异性']

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def read_source():
    book = openpyxl.load_workbook(SOURCE, read_only=True, data_only=True)
    sheet = book['MedianRating_per_image']
    rows = sheet.values
    frame = pd.DataFrame(list(rows), columns=None)
    frame.columns = frame.iloc[0].tolist()
    frame = frame.iloc[1:].reset_index(drop=True)
    book.close()
    assert frame.columns.tolist() == ['image', 'illumination', 'geometry'] + ATTRS
    assert len(frame) == 42120 and frame.image.is_unique
    return frame

def main():
    for sub in ['annotations', 'metadata', 'images/material']:
        (BASE / sub).mkdir(parents=True, exist_ok=True)
    source_hash = digest(SOURCE)
    raw = read_source()
    scores = raw[ATTRS].apply(pd.to_numeric, errors='raise').astype(float)
    assert np.isfinite(scores.to_numpy()).all()
    assert ((scores >= 1) & (scores <= 7)).all().all()
    assert np.equal(scores.to_numpy()*2, np.round(scores.to_numpy()*2)).all()
    # Search both supplied source repositories, excluding derived output and thumbnails.
    candidates = {}
    for folder in ['material-illumination-geometry-main', 'gloss_tinynetworks-main']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                candidates.setdefault(p.name, []).append(p)
    records, manifest = [], []
    for i, r in raw.iterrows():
        name = r['image']
        match = re.fullmatch(r'\[([^\]]+)\]\[([^\]]+)\]\[([^\]]+)\]\.jpg', name)
        assert match, name
        illumination, geometry, material = match.groups()
        image_id = f'serrano_{i+1:05d}'
        found = sorted(candidates.get(name, []))
        if len(found) > 1:
            assert len({digest(p) for p in found}) == 1, f'Ambiguous local images: {name}'
        record = dict(image_id=image_id, image_path='', dataset_source='Serrano_et_al_2021',
                      image_available=bool(found), image_status='available' if found else 'missing_local',
                      source_image_filename=name, local_source_image_path='',
                      material_id=material, object_id=geometry, shape_id=geometry,
                      illumination_id=illumination, source_illumination_code=int(r.illumination),
                      source_geometry_code=int(r.geometry), viewpoint_id='',
                      rating_aggregation='author_provided_median', rating_scale_min=1, rating_scale_max=7,
                      source_sheet='MedianRating_per_image', source_excel_row=i+2)
        record.update({attr:float(scores.loc[i, attr]) for attr in ATTRS})
        if found:
            src = found[0]
            with Image.open(src) as im:
                im.load()
                size, mode = im.size, im.mode
                assert mode == 'RGB', f'Unexpected colour mode: {src}'
            dest = BASE/'images/material'/f'{image_id}.jpg'
            shutil.copyfile(src, dest)
            record['image_path'] = dest.relative_to(BASE).as_posix()
            record['local_source_image_path'] = src.relative_to(ROOT).as_posix()
            manifest.append(dict(image_id=image_id, source_path=record['local_source_image_path'],
                                 image_path=record['image_path'], width=size[0], height=size[1],
                                 mode=mode, sha256=digest(dest), source_sha256=digest(src)))
        records.append(record)
    ann = pd.DataFrame(records)
    assert not ann.duplicated(['material_id', 'shape_id', 'illumination_id']).any()
    assert ann.material_id.nunique() == 520 and ann.shape_id.nunique() == 9 and ann.illumination_id.nunique() == 9
    assert ann.groupby('illumination_id').source_illumination_code.nunique().eq(1).all()
    assert ann.groupby('shape_id').source_geometry_code.nunique().eq(1).all()
    ann.to_csv(BASE/'annotations/material_annotations.csv', index=False, encoding='utf-8-sig')
    ann[ann.image_available].to_csv(BASE/'annotations/material_available_images.csv', index=False, encoding='utf-8-sig')
    ann.loc[~ann.image_available, ['image_id', 'source_image_filename', 'image_status', 'source_excel_row']].to_csv(BASE/'metadata/material_missing_images.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(manifest).to_csv(BASE/'metadata/material_image_manifest.csv', index=False, encoding='utf-8-sig')
    np.save(BASE/'annotations/material_attribute_matrix.npy', scores.to_numpy())
    ann[['image_id', 'source_image_filename', 'image_available']].to_csv(BASE/'annotations/material_attribute_matrix_index.csv', index=False, encoding='utf-8-sig')
    write_json(BASE/'metadata/material_attribute_schema.json', dict(matrix_shape=[42120,6], dtype='float64',
        row_order='material_annotations.csv; source worksheet order', columns=[dict(index=i,name=a,label_zh=label,minimum=1,maximum=7,aggregation='median') for i,(a,label) in enumerate(zip(ATTRS,LABELS))],
        missing_image_policy='Keep ratings; image_path empty; image_available false; image_status missing_local',
        missing_rating_policy='No imputation; fail on nonnumeric or missing rating',
        text_number_policy='Convert numeric strings to float without rounding; preserve half-integer medians',
        viewpoint_policy='Not supplied; blank. Shape name is used for object_id, not an independently identified object.'))
    for name,cols in [('illumination',['source_illumination_code','illumination_id']),('geometry',['source_geometry_code','shape_id'])]:
        ann[cols].drop_duplicates().sort_values(cols[0]).to_csv(BASE/f'metadata/material_{name}_mapping.csv',index=False,encoding='utf-8-sig')
    distribution=[]
    for attr in ATTRS:
        for value,count in scores[attr].value_counts().sort_index().items():
            distribution.append(dict(attribute=attr,median_rating=value,image_count=int(count)))
    pd.DataFrame(distribution).to_csv(BASE/'metadata/material_rating_distribution.csv',index=False)
    summary=dict(dataset='Serrano et al. 2021', status='summary_ratings_complete_images_partial',
        annotation_rows=len(ann), material_count=520, geometry_count=9, illumination_count=9,
        attribute_count=6, attributes=ATTRS, local_images_available=int(ann.image_available.sum()),
        local_images_missing=int((~ann.image_available).sum()), missing_rating_cells=int(scores.isna().sum().sum()),
        numeric_text_cells_converted={a:int(raw[a].map(lambda x:isinstance(x,str)).sum()) for a in ATTRS},
        fractional_median_cells={a:int((scores[a]%1!=0).sum()) for a in ATTRS},
        rating_statistics=scores.describe().to_dict(), source_workbook=SOURCE.name,
        source_sheet='MedianRating_per_image', source_sha256=source_hash,
        image_match_policy='Exact filename match across both supplied repositories; decode matched images; copy bytes without resize',
        raw_all_data_processed=False, download_attempted=False, normalized=False,
        author_filtered_trial_count_inferred=False)
    write_json(BASE/'metadata/material_dataset_summary.json',summary)
    combined=json.loads((BASE/'metadata/dataset_summary.json').read_text(encoding='utf-8'))
    combined['scope']='Dataset A complete; Dataset B complete author-provided summary ratings, 7 local images, no image download'
    combined['material_dataset_summary']='material_dataset_summary.json'
    write_json(BASE/'metadata/dataset_summary.json',combined)
    assert digest(SOURCE)==source_hash
    print(json.dumps({k:v for k,v in summary.items() if k!='rating_statistics'},ensure_ascii=False,indent=2))

if __name__ == '__main__': main()
