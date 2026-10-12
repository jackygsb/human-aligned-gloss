"""Validate completed controls and publish descriptive summaries; no test selection."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate_pilot import metrics,sha,check_solver
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'robustness_v2';RES=OUT/'results'

def main():
    check_solver()
    a=pd.read_csv(BASE/'annotations/gloss_annotations.csv');f=pd.read_csv(BASE/'protocols/gloss_group_folds.csv')
    assert a.image_id.tolist()==f.image_id.tolist()
    truth=a.set_index('image_id').human_gloss_score
    checks=[];tiny=[];losses=[]
    for folder,lr in [('tiny',.001),('tiny_paper_lr',.01)]:
        d=pd.read_csv(OUT/folder/'all_predictions.csv');assert len(d)==15552
        for (split,seed,fold),g in d.groupby(['split','seed','fold']):
            stem=f'{split}_seed{seed}_fold{fold}'
            meta=json.loads((OUT/folder/(stem+'_audit.json')).read_text())
            te=f.loc[f[split+'_fold']==fold,'image_id'].tolist();tr=f.loc[f[split+'_fold']!=fold,'image_id'].tolist()
            assert meta['test_image_ids']==te and meta['train_image_ids']==tr
            assert set(tr).isdisjoint(te) and len(tr)==3564 and len(te)==324
            assert g.image_id.tolist()==te and g.image_id.is_unique
            assert np.isfinite(g.prediction).all() and np.allclose(g.human,truth.loc[g.image_id])
            assert len(meta['train_loss_history'])==30 and meta['selected_epoch']==30 and meta['parameters']==677
            assert meta['config_sha256']==sha(BASE/'protocols/robustness_v2_config.json')
            assert meta.get('learning_rate',lr)==lr
            assert (OUT/folder/(stem+'.pt')).stat().st_size>0
            tiny.append(dict(split=split,seed=seed,fold=fold,learning_rate=lr,**metrics(g.human,g.prediction)))
            for epoch,loss in enumerate(meta['train_loss_history'],1):losses.append(dict(split=split,seed=seed,fold=fold,learning_rate=lr,epoch=epoch,normalized_augmented_train_mae=loss))
        assert d.groupby(['split','seed']).image_id.nunique().eq(3888).all()
        checks.append(f'{folder}: 48 isolated folds, two seeds, 15552 finite predictions')
    td=pd.DataFrame(tiny);td.to_csv(RES/'tiny_fold_metrics.csv',index=False)
    ts=td.groupby(['split','learning_rate','seed'])[['spearman','pearson','mae','rmse']].mean().reset_index();ts.to_csv(RES/'tiny_summary.csv',index=False)
    ld=pd.DataFrame(losses);ld.to_csv(RES/'tiny_train_loss.csv',index=False)
    d=pd.read_csv(RES/'all_predictions.csv');fm=pd.read_csv(RES/'fold_metrics.csv');assert len(d)==69984 and len(fm)==216
    for (split,model),g in d.groupby(['split','model']):
        assert g.image_id.tolist()==a.image_id.tolist() and np.allclose(g.human,truth.loc[g.image_id])
        assert np.isfinite(g.prediction).all() and g.fold.tolist()==f[split+'_fold'].tolist()
        meta=json.loads((RES/f'{split}_{model}.json').read_text())
        for key,path in [('annotation_sha256','annotations/gloss_annotations.csv'),('folds_sha256','protocols/gloss_group_folds.csv'),('config_sha256','protocols/robustness_v2_config.json')]:assert meta[key]==sha(BASE/path)
        for fold,gg in g.groupby('fold'):
            saved=fm[(fm.split==split)&(fm.model==model)&(fm.fold==fold)].iloc[0]
            for key,value in metrics(gg.human,gg.prediction).items():assert np.isclose(saved[key],value,atol=1e-10)
    checks.append('18 model/split cells: 69984 OOF predictions; all 216 fold metrics independently recomputed')
    for name in ['clip_nobg','clip_gray','dino_nobg','dino_gray']:
        path=OUT/'features'/f'{name}.npy';x=np.load(path);m=json.loads((OUT/'features'/f'{name}_manifest.json').read_text())
        assert sha(path)==m['sha256'] and x.shape[0]==3888 and np.isfinite(x).all()
        assert np.allclose(np.linalg.norm(x,axis=1),1,atol=1e-5)
    checks.append('Four feature matrices: file hashes, dimensions, finiteness and unit norms verified')
    shift=pd.read_csv(RES/'input_shift_predictions.csv');assert len(shift)==46656
    assert np.isfinite(shift.prediction).all()
    for (model,split,condition),g in shift.groupby(['model','split','condition']):
        ref=d[(d.model==model+'_rgb64')&(d.split==split)].set_index('image_id')
        assert g.image_id.is_unique and len(g)==3888
        assert np.allclose(g.original_prediction,ref.loc[g.image_id].prediction)
        assert np.allclose(g.original_human,truth.loc[g.image_id])
        if condition=='rgb':assert np.allclose(g.prediction,g.original_prediction)
    checks.append('46656 input-shift records: complete coverage and original readout predictions reproduced')
    pairs=[]
    for split in ['shape','illumination']:
        table=fm[fm.split==split].pivot(index='fold',columns='model',values='spearman')
        for model in ['clip','dino']:
            for condition in ['nobg','gray']:
                change=table[f'{model}_{condition}64']-table[f'{model}_rgb64']
                pairs.append(dict(split=split,model=model,condition=condition,mean_delta=change.mean(),min_delta=change.min(),max_delta=change.max(),positive_folds=int((change>0).sum()),folds=12))
    pd.DataFrame(pairs).to_csv(RES/'paired_condition_changes.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,4.4),sharey=True)
    for ax,split in zip(axes,['shape','illumination']):
        for model,color in [('clip','#2878a8'),('dino','#d57a28')]:
            groups=[fm[(fm.split==split)&(fm.model==model+'_'+v+'64')].spearman.to_numpy() for v in ['rgb','nobg','gray']]
            ax.plot(range(3),[g.mean() for g in groups],marker='o',label=model.upper(),color=color)
            for j,g in enumerate(groups):ax.scatter(np.full(len(g),j),g,color=color,alpha=.22,s=16)
        ax.set(xticks=range(3),xticklabels=['Original RGB','No background','Grayscale'],title='Unseen '+split,ylim=(0,1),ylabel='Held-out Spearman correlation')
        ax.grid(axis='y',alpha=.2);ax.legend()
    fig.suptitle('Same 64-dimensional readout; refit separately for each input')
    fig.text(.5,.015,'Dots: 12 held-out groups. Human labels always from original images. Descriptive results.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.95]);fig.savefig(OUT/'controlled_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,lr in zip(axes,[.001,.01]):
        sub=ld[ld.learning_rate==lr]
        for _,g in sub.groupby(['split','seed','fold']):ax.plot(g.epoch,g.normalized_augmented_train_mae,alpha=.15,color='#2878a8')
        avg=sub.groupby('epoch').normalized_augmented_train_mae.mean();ax.plot(avg.index,avg,color='black',lw=2)
        ax.set(title=f'Adam learning rate {lr}',xlabel='Epoch',ylabel='Augmented training MAE (c / 0.1487)',yscale='log');ax.grid(alpha=.2)
        ax.set_yticks([.15,.2,.3,.5,1.,1.5]);ax.set_yticklabels(['0.15','0.20','0.30','0.50','1.00','1.50'])
    fig.suptitle('Tiny control: all 48 training histories per learning rate (not test loss)')
    fig.tight_layout();fig.savefig(OUT/'tiny_optimization.png',dpi=180);plt.close(fig)
    manifest={str(p.relative_to(BASE)):sha(p) for p in list((BASE/'scripts').glob('*robustness*.py'))+[BASE/'scripts/train_tiny_control.py',BASE/'scripts/evaluate_input_shift.py',BASE/'protocols/robustness_v2_config.json',BASE/'protocols/tiny_optimizer_sensitivity.json',BASE/'annotations/gloss_annotations.csv',BASE/'protocols/gloss_group_folds.csv']}
    (OUT/'validation.json').write_text(json.dumps(dict(status='passed',checks=checks,artifact_sha256=manifest,limitations=['No independent author-exact tiny replication','PCA64 equates output dimension, not information or training history','Input-shift metrics do not measure new human responses','Fold summaries are descriptive; folds not treated as iid']),indent=2))
    print(ts.to_string(index=False));print('\n'.join(checks))

if __name__=='__main__':main()
