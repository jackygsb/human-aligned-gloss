"""Rebuild Dataset A from untouched local sources. No network or model training."""
from pathlib import Path
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import loadmat
from scipy.stats import pearsonr, spearmanr
from audit_raw import read_raw, SRC, ROOT

OUT = ROOT/'Human_Material_Perception_Benchmark'

def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    for name in ['annotations','metadata','figures','images/gloss','images_processed/224x224/gloss','images_processed/512x512/gloss']:
        (OUT/name).mkdir(parents=True,exist_ok=True)
    audit, accepted = read_raw()
    assert len(accepted)==295 and len(audit)==467
    audit.to_csv(OUT/'metadata/participant_exclusions.csv',index=False)
    gt = pd.read_csv(SRC/'data/onlineExp_condition_summary.csv')
    assert len(gt)==3888 and not gt.duplicated(['groupN','testimgN']).any()
    official = loadmat(str(SRC/'data/humanANDgt_Pellacini_c.mat'),squeeze_me=True,struct_as_record=False)
    author_obs = loadmat(str(SRC/'data/onlineData.mat'),squeeze_me=True,struct_as_record=False)['data']
    unmatched = list(author_obs)
    for _,g,r,_ in accepted:
        hits=[i for i,y in enumerate(unmatched) if g==y.groupN and np.array_equal(r,y.response_rawN)]
        assert hits, 'Raw record not matched to author-cleaned record'
        unmatched.pop(hits[0])
    assert not unmatched
    c = np.linspace(0,.1487,52)
    bygroup = {g:np.stack([c[x[2].astype(int)-1].mean(axis=1) for x in accepted if x[1]==g],axis=1) for g in range(1,55)}
    rows=[]
    for idx,r in enumerate(gt.itertuples(),1):
        vals=bygroup[r.groupN][r.testimgN-1]
        rows.append(dict(image_id=f'morimoto_{idx:04d}',image_path=f'images/gloss/img{idx}.png',dataset_source='Morimoto_et_al_2026',human_gloss_score=vals.mean(),physical_specular_reflectance=r.specularity,object_id=r.objN,shape_id=r.objN,illumination_id=r.lightprobeN,viewpoint_id=r.viewPointN,physical_pellacini_c=r.Pellacini_c,hue=r.hue,chroma=r.chroma,human_gloss_std=vals.std(ddof=1),human_gloss_sem=vals.std(ddof=1)/np.sqrt(len(vals)),n_observers=len(vals),n_trials=2*len(vals),group_id=r.groupN,test_image_id=r.testimgN,source_image_index=idx,source_condition_filename=r.filename_png,image_224_path=f'images_processed/224x224/gloss/img{idx}.png',image_512_path=f'images_processed/512x512/gloss/img{idx}.png'))
    ann=pd.DataFrame(rows)
    err=float(np.max(abs(ann.human_gloss_score-official['human_Pellacini_c'].mean)))
    errstd=float(np.max(abs(ann.human_gloss_std-official['human_Pellacini_c'].std)))
    assert err<1e-12 and errstd<1e-12
    assert np.allclose(ann.physical_pellacini_c,official['gt_Pellacini_c'],atol=1e-12,rtol=0)
    for f,col in [('humanlabel.csv','humanresponse'),('groundtruthlabel.csv','groundtruth')]:
        t=pd.read_csv(SRC/'data'/f)
        assert np.array_equal(t.imgN,np.arange(1,3889))
        target=ann.human_gloss_score if col=='humanresponse' else ann.physical_pellacini_c
        assert np.allclose(t[col],target,atol=1e-12,rtol=0)
    ann.to_csv(OUT/'annotations/gloss_annotations.csv',index=False)
    np.save(OUT/'annotations/human_gloss_vector.npy',ann.human_gloss_score.to_numpy())
    ann[['image_id']].to_csv(OUT/'annotations/human_gloss_vector_index.csv',index=False)
    lookup={(r.group_id,r.test_image_id):r.image_id for r in ann.itertuples()}
    trials=[]
    correlations=[]
    prev=loadmat(str(SRC/'data/groundtruth_prev_exp.mat'),squeeze_me=True,struct_as_record=False)['GroundTruth']
    for file,g,raw,blocks in accepted:
        assert np.isfinite(raw).all() and (raw==raw.astype(int)).all() and raw.min()>=1 and raw.max()<=52
        pid='p_'+hashlib.sha256(file.encode()).hexdigest()[:12]
        response=c[raw.astype(int)-1]
        phys=np.r_[gt[gt.groupN==g].sort_values('testimgN').Pellacini_c.to_numpy(),prev.Pellacini_c[2:38:3]]
        own=response.mean(axis=1)
        others=(bygroup[g].sum(axis=1)-own)/(bygroup[g].shape[1]-1)
        correlations.append(dict(participant_id=pid,group_id=g,human_vs_physical=pearsonr(own,phys)[0],within_observer=pearsonr(response[:,0],response[:,1])[0],leave_one_out_human=pearsonr(own,others)[0]))
        for rep,b in enumerate(blocks,1):
            for j,(_,t) in enumerate(b.iloc[1:].iterrows(),1):
                trials.append(dict(participant_id=pid,group_id=g,repetition=rep,test_image_id=j,image_id=lookup.get((g,j),f'common_{j-72:02d}'),is_main_image=j<=72,slider_index=raw[j-1,rep-1],human_pellacini_c=response[j-1,rep-1],human_matched_specularity=(response[j-1,rep-1]+(.5**3/2)**(1/3))**3-.5**3/2,response_time_s=t['key_resp.rt'],slider_was_missing=pd.isna(t['slider.response'])))
    pd.DataFrame(trials).to_csv(OUT/'annotations/gloss_trials.csv',index=False)
    cr=pd.DataFrame(correlations)
    cr.to_csv(OUT/'metadata/observer_correlations.csv',index=False)
    print('Labels and raw trials verified; validating and copying images.',flush=True)
    def process_image(i):
        record={'image_id':f'morimoto_{i:04d}'}
        for size,folder in [(224,'imgs3888_bg_png'),(512,'imgs3888_bg_512_png')]:
            src=SRC/'data/imgs'/folder/f'img{i}.png'
            dest=OUT/f'images_processed/{size}x{size}/gloss'/src.name
            with Image.open(src) as im:
                im.load()
                assert im.size==(size,size),str(src)
                if im.mode=='RGB':
                    shutil.copyfile(src,dest)
                else:
                    im.convert('RGB').save(dest)
                record[f'source_mode_{size}']=im.mode
            record[f'source_sha256_{size}']=sha(src)
            record[f'output_sha256_{size}']=sha(dest)
            if size==512:
                shutil.copyfile(dest,OUT/'images/gloss'/src.name)
        return record
    with ThreadPoolExecutor(max_workers=8) as pool:
        manifest=list(pool.map(process_image,range(1,3889)))
    pd.DataFrame(manifest).to_csv(OUT/'metadata/image_manifest.csv',index=False)
    provenance=[]
    for p in sorted(list((SRC/'onlineData').glob('ID*.csv'))+list((SRC/'data').glob('*.csv'))+list((SRC/'data').glob('*.mat'))+[SRC/'process_onlinedata.m',SRC/'fig2_online_exp_results.m',SRC/'README.md',SRC/'LICENSE']):
        provenance.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    pd.DataFrame(provenance).to_csv(OUT/'metadata/source_manifest.csv',index=False)
    shutil.copyfile(SRC/'LICENSE',OUT/'metadata/Morimoto_SOURCE_LICENSE.txt')
    summary=dict(dataset='Morimoto et al. Human gloss perception reproduced by tiny neural networks',doi='10.1038/s41562-026-02445-0',source_repository='https://github.com/takuma929/gloss_tinynetworks',scope='Phase I Dataset A complete; Dataset B inventory only, awaiting user decision',image_count=len(ann),raw_records=len(audit),accepted_records=len(accepted),excluded_records=int((~audit.accepted).sum()),excluded_attention=int((~(audit.catch_sum<=4)).sum()),excluded_missing_catch=int(audit.catch_sum.isna().sum()),excluded_catch_above_threshold=int((audit.catch_sum>4).sum()),excluded_rt_after_attention=int(((audit.catch_sum<=4)&~audit.accepted).sum()),accepted_trials=len(trials),main_image_trials=int(sum(t['is_main_image'] for t in trials)),imputed_slider_trials=int(sum(t['slider_was_missing'] for t in trials)),shapes=int(ann.shape_id.nunique()),illuminations=int(ann.illumination_id.nunique()),viewpoints=int(ann.viewpoint_id.nunique()),observers_per_image={'min':int(ann.n_observers.min()),'max':int(ann.n_observers.max())},score_scale='Pellacini c; 52 linear settings from 0 to 0.1487; mean of repetitions then observers',score_statistics=ann.human_gloss_score.describe().to_dict(),label_max_abs_error=err,label_std_max_abs_error=errstd,pearson_human_physical_c=float(pearsonr(ann.human_gloss_score,ann.physical_pellacini_c)[0]),spearman_human_physical_c=float(spearmanr(ann.human_gloss_score,ann.physical_pellacini_c)[0]),observer_correlation_medians=cr.iloc[:,2:].median().to_dict(),resolutions=[224,512],image_policy='Use both author-provided background variants at native size; no artificial upsampling; images/gloss is native 512 RGB',outlier_policy='No extra image or label exclusions; flag Tukey outliers descriptively only',label_tukey_outliers=int(((ann.human_gloss_score<ann.human_gloss_score.quantile(.25)-1.5*(ann.human_gloss_score.quantile(.75)-ann.human_gloss_score.quantile(.25)))|(ann.human_gloss_score>ann.human_gloss_score.quantile(.75)+1.5*(ann.human_gloss_score.quantile(.75)-ann.human_gloss_score.quantile(.25)))).sum()))
    save_json(OUT/'metadata/dataset_summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    main()
