"""Fold-safe incremental prediction analysis using existing tiny checkpoints."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='4'
import json,time,hashlib,csv,html,sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,pearsonr
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'mllm_material/matching_full_v1/incremental_tiny';OUT.mkdir(exist_ok=True)
TINY=BASE/'robustness_v2/tiny_paper_lr'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def progress(kind,detail):
    save(OUT/'status.json',dict(state=kind,detail=detail,updated=time.time(),pid=os.getpid()))
    (OUT/'progress.html').write_text('<meta charset="utf-8"><meta http-equiv="refresh" content="15"><style>body{font:20px system-ui;margin:40px}</style><h1>InternVL 对小网络的增量解释检验</h1><p>'+html.escape(kind)+'：'+html.escape(detail)+'</p><a href="report.html">完成后的分析报告</a>',encoding='utf-8')
    print(kind,detail,flush=True)
ann=pd.read_csv(BASE/'annotations/gloss_annotations.csv')
folds=pd.read_csv(BASE/'protocols/gloss_group_folds.csv')
full=pd.read_csv(OUT.parent/'image_summary.csv')
d=ann.merge(folds[['image_id','shape_fold','illumination_fold']],on='image_id',validate='one_to_one').merge(full[['image_id','model_c','human_c','cohort']],on='image_id',validate='one_to_one')
assert len(d)==3888 and np.allclose(d.human_c,d.human_gloss_score)
seeds=[20260926,20260927]
def prepare():
    path=OUT/'protocol.json'
    if not path.exists():
        save(path,dict(created=time.time(),primary_evaluation='3849 matching-heldout images; 39 development images excluded from head fitting and scoring',outer_splits='Existing 12 shape folds and 12 illumination folds; each outer tiny checkpoint only trained on other 33 groups',tiny='Independent single-kernel control, lr .01, terminal epoch30, two-seed mean; not exact author network',calibration='Fixed ordinary least squares with intercept, predictors standardized on outer training rows. Tiny training predictions from the SAME outer checkpoint. They are in-sample to the tiny learner; no nested stacking claim. Outer test labels never used for base learner or calibration.',comparisons=['tiny linear vs tiny linear + InternVL','tiny cubic vs tiny cubic + InternVL','physical/color/viewpoint controls + tiny cubic vs same + InternVL','InternVL alone calibrated on same outer train'],controls='50 physical c categories, categorical viewpoint and chroma, sin/cos hue harmonics 1 and 2; no shape or illumination identity predictors',primary_metrics=['out-of-fold R2','MSE','MAE','Spearman'],uncertainty='4000 paired condition-cluster bootstrap replicates for loss reduction, separate shape and illumination clustering plus two-way crossed weighting. Fixed predictions; not full retraining uncertainty.',author_supplement='Released author predictions used only for descriptive residual correlations, unknown exact fold membership.',inference='No new InternVL collection, no new tiny training, only frozen tiny checkpoint inference',source_hashes={str(p.relative_to(BASE)):sha(p) for p in [BASE/'scripts/analyze_incremental_internvl.py',BASE/'scripts/train_tiny_control.py',BASE/'protocols/gloss_group_folds.csv',TINY/'all_predictions.csv',OUT.parent/'image_summary.csv']}))
def generate_features():
    paths=[OUT/f'features_{factor}_{fold}.npz' for factor in ['shape','illumination'] for fold in range(12)]
    if all(p.exists() for p in paths):return
    import torch
    from PIL import Image
    from concurrent.futures import ThreadPoolExecutor
    from train_tiny_control import Tiny
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    def imread(idx):
        p=BASE.parent/f'gloss_tinynetworks-main/data/imgs/imgs3888_nobg_png/img{idx}.png'
        with Image.open(p) as im:return np.array(im.convert('RGB'))
    progress('running','载入作者去背景图，用已有小网络检查点生成各折预测')
    with ThreadPoolExecutor(max_workers=8) as pool:arr=np.stack(list(pool.map(imread,d.source_image_index)))
    images=torch.from_numpy(arr).permute(0,3,1,2).contiguous().cuda()
    old=pd.read_csv(TINY/'all_predictions.csv');index=d.set_index('image_id')
    for factor in ['shape','illumination']:
        for fold in range(12):
            dest=OUT/f'features_{factor}_{fold}.npz'
            if dest.exists():continue
            allpred=[];auditrows=[]
            for seed in seeds:
                ckpt=TINY/f'{factor}_seed{seed}_fold{fold}.pt';ap=ckpt.with_name(ckpt.stem+'_audit.json');a=read(ap)
                tr=set(a['train_image_ids']);te=set(a['test_image_ids']);expected=set(d.loc[d[factor+'_fold']==fold,'image_id'])
                assert te==expected and tr.isdisjoint(te) and len(tr)==3564
                assert set(index.loc[list(tr),factor+'_id']).isdisjoint(set(index.loc[list(te),factor+'_id']))
                model=Tiny().cuda();model.load_state_dict(torch.load(ckpt,map_location='cuda',weights_only=True));model.eval();pred=[]
                with torch.inference_mode():
                    for start in range(0,len(d),128):pred.extend((model(images[start:start+128].float()/255)*.1487).cpu().tolist())
                pred=np.asarray(pred);prior=old[(old.split==factor)&(old.seed==seed)&(old.fold==fold)].set_index('image_id')
                actual=pd.Series(pred,index=d.image_id).loc[prior.index].to_numpy();maxerr=float(np.max(abs(actual-prior.prediction.to_numpy())))
                assert maxerr<1e-5,f'Checkpoint prediction mismatch {maxerr}'
                auditrows.append(dict(seed=seed,checkpoint_sha256=sha(ckpt),split_audit_sha256=sha(ap),max_difference_existing_oof=maxerr,train_n=len(tr),test_n=len(te)))
                allpred.append(pred);del model
            np.savez_compressed(dest,image_ids=d.image_id.to_numpy(dtype=str),prediction=np.mean(allpred,axis=0))
            save(dest.with_suffix('.json'),auditrows);progress('running',f'{factor} 折 {fold+1}/12：测试图像和分组不重叠，预测与已有结果一致')
    save(OUT/'inference_environment.json',dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0),precision='float32',batch_size=128,tf32=False))
def design_controls():
    parts=[]
    for col in ['physical_pellacini_c','viewpoint_id','chroma']:
        values=sorted(d[col].unique())
        for v in values[1:]:parts.append((d[col].to_numpy()==v).astype(float))
    angle=np.deg2rad(d.hue.to_numpy())
    for k in [1,2]:parts.extend([np.sin(k*angle),np.cos(k*angle)])
    return np.column_stack(parts)
def fit_predict(x,y,train):
    mean=x[train].mean(axis=0);std=x[train].std(axis=0);std[std<1e-12]=1
    x=np.column_stack([np.ones(len(x)),(x-mean)/std]);beta=np.linalg.lstsq(x[train],y[train],rcond=None)[0]
    return x@beta
def metrics(y,p):
    e=y-p
    return dict(r2=float(1-np.mean(e**2)/np.var(y)),mse=float(np.mean(e**2)),mae=float(np.mean(abs(e))),pearson=float(pearsonr(y,p).statistic),spearman=float(spearmanr(y,p).statistic))
def analyze():
    controls=design_controls();y=d.human_c.to_numpy();m=d.model_c.to_numpy();mask=d.cohort.eq('heldout_matching').to_numpy()
    predictions=[];foldrecords=[];headrecords=[]
    for factor in ['shape','illumination']:
        for fold in range(12):
            f=np.load(OUT/f'features_{factor}_{fold}.npz');assert np.array_equal(f['image_ids'],d.image_id)
            s=f['prediction'];train=(d[factor+'_fold'].to_numpy()!=fold)&mask;test=(d[factor+'_fold'].to_numpy()==fold)&mask
            cubic=np.column_stack([s,s*s,s*s*s])
            designs=dict(tiny_linear=s[:,None],tiny_linear_plus_internvl=np.column_stack([s,m]),tiny_cubic=cubic,tiny_cubic_plus_internvl=np.column_stack([cubic,m]),controls_tiny_cubic=np.column_stack([controls,cubic]),controls_tiny_cubic_plus_internvl=np.column_stack([controls,cubic,m]),internvl_only=m[:,None])
            for name,x in designs.items():
                p=fit_predict(x,y,train)
                for j in np.flatnonzero(test):predictions.append(dict(image_id=d.image_id.iloc[j],split=factor,fold=fold,model=name,human=y[j],prediction=p[j],shape_id=int(d.shape_id.iloc[j]),illumination_id=int(d.illumination_id.iloc[j])))
                foldrecords.append(dict(split=factor,fold=fold,model=name,n=int(test.sum()),**metrics(y[test],p[test])))
            headrecords.append(dict(split=factor,fold=fold,train_image_ids=d.image_id[train].tolist(),test_image_ids=d.image_id[test].tolist()))
    out=pd.DataFrame(predictions);out.to_csv(OUT/'oof_predictions.csv',index=False);pd.DataFrame(foldrecords).to_csv(OUT/'fold_metrics.csv',index=False);save(OUT/'head_split_audit.json',headrecords)
    scores=[];gains=[];rng=np.random.default_rng(20260928)
    for factor in ['shape','illumination']:
        wide=out[out.split==factor].pivot(index='image_id',columns='model',values='prediction')
        labels=d.set_index('image_id').loc[wide.index];yy=labels.human_c.to_numpy();var=np.var(yy)
        for name in wide:scores.append(dict(split=factor,model=name,n=len(wide),**metrics(yy,wide[name].to_numpy())))
        for base in ['tiny_linear','tiny_cubic','controls_tiny_cubic']:
            aug=base+'_plus_internvl';a=wide[base].to_numpy();b=wide[aug].to_numpy();gain=(yy-a)**2-(yy-b)**2
            record=dict(split=factor,baseline=base,delta_r2=float(gain.mean()/var),relative_mse_reduction=float(gain.mean()/np.mean((yy-a)**2)),mae_reduction=float(np.mean(abs(yy-a)-abs(yy-b))))
            for cluster in ['shape','illumination']:
                tmp=pd.DataFrame(dict(group=labels[cluster+'_id'].to_numpy(),gain=gain)).groupby('group').gain.agg(['sum','count'])
                ix=rng.integers(0,len(tmp),size=(4000,len(tmp)));boot=tmp['sum'].to_numpy()[ix].sum(axis=1)/tmp['count'].to_numpy()[ix].sum(axis=1)/var
                record[cluster+'_cluster_ci95_delta_r2']=np.quantile(boot,[.025,.975]).tolist()
            # Crossed resampling retains the shape-by-illumination repeated structure.
            shape=labels.shape_id.to_numpy()-1;illum=labels.illumination_id.to_numpy()-1
            ng=np.zeros((36,36));gs=np.zeros((36,36));np.add.at(ng,(shape,illum),1);np.add.at(gs,(shape,illum),gain)
            ws=rng.multinomial(36,np.ones(36)/36,size=4000);wi=rng.multinomial(36,np.ones(36)/36,size=4000)
            boot=np.einsum('bi,ij,bj->b',ws,gs,wi)/np.einsum('bi,ij,bj->b',ws,ng,wi)/var
            record['crossed_cluster_ci95_delta_r2']=np.quantile(boot,[.025,.975]).tolist();gains.append(record)
    pd.DataFrame(scores).to_csv(OUT/'overall_metrics.csv',index=False)
    # Author outputs: descriptive partial association only, never claim outer-fold validation.
    pred=pd.read_csv(BASE.parent/'gloss_tinynetworks-main/data/pred_validation/prediction_3888imgs.csv');supp=[]
    sourceix=d.source_image_index.to_numpy()-1
    for name in pred:
        s=pred[name].to_numpy()[sourceix][mask];a=np.column_stack([np.ones(mask.sum()),s,s*s,s*s*s])
        ry=y[mask]-a@np.linalg.lstsq(a,y[mask],rcond=None)[0];rm=m[mask]-a@np.linalg.lstsq(a,m[mask],rcond=None)[0]
        supp.append(dict(author_column=name,n=int(mask.sum()),residual_pearson=float(pearsonr(ry,rm).statistic),status='Descriptive all-sample residual association; exact author fold mapping unknown, NOT predictive validation'))
    pd.DataFrame(supp).to_csv(OUT/'author_descriptive_residuals.csv',index=False)
    save(OUT/'summary.json',dict(n_evaluated=int(mask.sum()),independent_tiny_not_exact_author=True,metrics=scores,gains=gains,author_supplement=supp,uncertainty='Fixed-prediction paired cluster bootstrap; conditional on fitted models; overlaps in training folds not treated as independent replications'))
    return scores,gains
def report(scores,gains):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),constrained_layout=True)
    for ax,factor in zip(axes,['shape','illumination']):
        rows=[r for r in gains if r['split']==factor]
        for i,r in enumerate(rows):
            lo,hi=r['crossed_cluster_ci95_delta_r2'];v=r['delta_r2'];ax.errorbar(v,i,xerr=[[max(0,v-lo)],[max(0,hi-v)]],fmt='o',color='#176b88')
        ax.set(yticks=range(3),yticklabels=['Tiny linear','Tiny cubic','Controls + tiny cubic'],xlabel='Held-out R2 gain from adding InternVL',title=f'Unseen {factor} groups');ax.axvline(0,color='gray',lw=1);ax.grid(alpha=.2)
    fig.savefig(OUT/'incremental_gain.png',dpi=180);fig.savefig(OUT/'incremental_gain.svg');plt.close(fig)
    lines=['# InternVL 对小网络的增量解释：分组留出检验','', '不重新采集 InternVL，不重训小网络。使用既有 lr=0.01 单核小网络 48 个检查点、两 seed 平均。主要评估 3849 张 matching-heldout 图，39 张开发图不用于线性组合拟合或评分。','',
    '方法：每个外层测试折使用其对应的小网络检查点重新预测全部图；仅用该外层训练组拟合固定线性校准/组合系数，再预测未见测试组。禁止直接把不同外层模型的 OOF 行拼起来训练组合头，以免测试标签经其他模型间接进入训练特征。小网络在训练组上的预测是其训练内预测，故这是可审计的外层留出评估，不声称完整嵌套 stacking。','',
    '| 留出条件 | 基线 | 加入 InternVL 后 ΔR² | MSE 降低 | 交叉条件 bootstrap 95% CI |','|---|---|---:|---:|---|']
    for r in gains:lines.append(f"| {r['split']} | {r['baseline']} | {r['delta_r2']:.4f} | {r['relative_mse_reduction']:.1%} | [{r['crossed_cluster_ci95_delta_r2'][0]:.4f}, {r['crossed_cluster_ci95_delta_r2'][1]:.4f}] |")
    lines+=['','ΔR² 为同一测试图上组合模型 R² 减基线 R²；正值表示加入 InternVL 降低了未见条件上的预测误差。不是人类知觉方差的因果解释比例。置信区间对固定预测作配对条件重采样，没有覆盖重新训练模型的不确定性。','',
    '三条分析均预先固定并完整报告：线性小网络校准、三次多项式校准、控制物理光泽类别/视角/色度/色相后的三次校准。三次校准用于检查新增信息是否仅补偿小网络输出的非线性尺度；控制变量分析用于检查是否仅补充物理和颜色参数。','',
    '主分析小网络是本项目独立的单核实现，并非作者全部小网络。作者四列单层/三层发布预测另作描述性残差相关，保存在 author_descriptive_residuals.csv；由于缺少逐行折和检查点映射，不将其作为严格留出增益证据。','',
    '潜在额外信息可能来自物体内部图像特征、背景、预训练经验或提示任务差异；本分析不能单独确定这些机制。只有 InternVL 进行了全量检验，不推广成 GPT/Qwen 均已获得同样增益。','',
    '复现：从项目目录运行 mllm_runtime/Scripts/python.exe scripts/analyze_incremental_internvl.py。每折 features*.npz 保存固定预测，配套 JSON 保存检查点哈希及与已有测试预测的最大差；head_split_audit.json 保存组合拟合/测试成员；oof_predictions.csv 保存每图留出预测。']
    note='\n'.join(lines);(OUT/'分析报告.md').write_text(note,encoding='utf-8')
    (OUT/'report.html').write_text('<meta charset="utf-8"><title>InternVL 增量解释</title><style>body{font:18px system-ui;margin:35px;line-height:1.7}img{max-width:100%}pre{white-space:pre-wrap}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ddd}</style><h1>InternVL 能否解释小网络未解释的人类判断？</h1><img src="incremental_gain.png">'+pd.DataFrame(scores).round(5).to_html(index=False)+'<pre>'+html.escape(note)+'</pre><a href="oof_predictions.csv">逐图留出预测</a> · <a href="summary.json">完整统计</a> · <a href="分析报告.md">分析报告</a>',encoding='utf-8')
    save(OUT/'artifact_manifest.json',{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name not in ['artifact_manifest.json','status.json','progress.html'] and not p.name.endswith('.log')})
if __name__=='__main__':
    prepare()
    try:
        generate_features();progress('analyzing','固定外层训练/测试划分，计算组合预测与条件重采样区间');s,g=analyze();report(s,g);progress('complete','全量留出分析完成，表格、图与复现文件已保存');print(json.dumps(g,indent=2),flush=True)
    except Exception as e:progress('failed',str(e));raise
