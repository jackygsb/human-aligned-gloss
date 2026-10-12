from pathlib import Path
import json,itertools
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'mllm_material';FIG=OUT/'reports/figures';FIG.mkdir(parents=True,exist_ok=True)
panel=pd.read_csv(OUT/'manifests/panel432.csv');rows=[];scalarrows=[];structures={};dimension_rows=[]
for folder in sorted((OUT/'spose').iterdir()):
    if not folder.is_dir() or not (folder/'selected_20seeds.csv').exists():continue
    source=folder.name;d=pd.read_csv(folder/'selected_20seeds.csv');data=np.load(folder/'fitting_data.npz');tri=data['triples'];q=data['targets'];split=data['split']
    similarities=[]
    for fit in d.itertuples():
        w=np.load(folder/f'd{int(fit.dim)}_l{float(fit.lam)}_s{int(fit.seed)}.npy');similarities.append(w@w.T)
        for j in np.flatnonzero(w.max(0)>.1):
            dimension_rows.append(dict(source=source,seed=int(fit.seed),dimension=int(j),human_gloss_rho=float(spearmanr(w[:,j],panel.human_gloss_score)[0]),physical_gloss_rho=float(spearmanr(w[:,j],panel.physical_pellacini_c)[0])))
    structures[source]=np.mean(similarities,axis=0)
    if source=='human_rating_induced':values=panel.human_gloss_score.to_numpy()
    elif source=='synthetic_1d':values=np.random.RandomState(905).uniform(0,1,432)
    else:
        scores=pd.read_csv(OUT/'statistics'/f'{source}_image_scores.csv').set_index('image_id').model_score
        values=scores.reindex(panel.image_id).to_numpy()/100
    values=(values-np.nanmean(values))/np.nanstd(values)
    v=values[tri];similarity=-np.column_stack([abs(v[:,1]-v[:,2]),abs(v[:,0]-v[:,2]),abs(v[:,0]-v[:,1])]);valid=np.isfinite(similarity).all(1)
    candidates=[]
    for temp in np.logspace(-3,2,101):
        m=valid&(split=='validation');p=softmax(similarity[m]/temp,axis=1);ce=float(-(q[m]*np.log(p.clip(1e-12))).sum(1).mean());candidates.append((ce,temp))
    _,temp=min(candidates);m=valid&(split=='test');p=softmax(similarity[m]/temp,axis=1)
    scalar=dict(source=source,test_n=int(m.sum()),temperature_selected_on_validation=temp,test_ce=float(-(q[m]*np.log(p.clip(1e-12))).sum(1).mean()),test_expected_agreement=float(q[m][np.arange(m.sum()),p.argmax(1)].mean()))
    scalarrows.append(scalar)
    rows.append(dict(source=source,seeds=len(d),active_dimensions_mean=d.active_dimensions.mean(),active_dimensions_min=d.active_dimensions.min(),active_dimensions_max=d.active_dimensions.max(),test_ce_mean=d.test_ce.mean(),test_agreement_mean=d.test_expected_agreement.mean(),scalar_test_ce=scalar['test_ce'],scalar_test_agreement=scalar['test_expected_agreement']))
pd.DataFrame(rows).to_csv(OUT/'statistics/spose_summary.csv',index=False);pd.DataFrame(scalarrows).to_csv(OUT/'statistics/scalar_distance_baselines.csv',index=False)
pd.DataFrame(dimension_rows).to_csv(OUT/'statistics/spose_dimension_associations.csv',index=False)
if structures:
    structures['human_scalar_distance']=-abs(panel.human_gloss_score.to_numpy()[:,None]-panel.human_gloss_score.to_numpy()[None,:])
    structures['physical_scalar_distance']=-abs(panel.physical_pellacini_c.to_numpy()[:,None]-panel.physical_pellacini_c.to_numpy()[None,:])
    ids=list(structures);upper=np.triu_indices(432,1);rsa=np.array([[spearmanr(structures[a][upper],structures[b][upper])[0] for b in ids] for a in ids])
    pd.DataFrame(rsa,index=ids,columns=ids).to_csv(OUT/'statistics/spose_structure_rsa_descriptive.csv')
    fig,ax=plt.subplots(figsize=(8,7));im=ax.imshow(rsa,vmin=-1,vmax=1,cmap='coolwarm');ax.set_xticks(range(len(ids)));ax.set_yticks(range(len(ids)));ax.set_xticklabels(ids,rotation=45,ha='right',fontsize=8);ax.set_yticklabels(ids,fontsize=8)
    for i,j in itertools.product(range(len(ids)),repeat=2):ax.text(j,i,f'{rsa[i,j]:.2f}',ha='center',va='center',fontsize=8)
    ax.set_title('Representational similarity (descriptive)\nHuman-induced structure is not independent choice data');fig.colorbar(im,ax=ax,shrink=.7);fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(FIG/f'spose_structure_rsa.{ext}',dpi=250)
    plt.close(fig)
if rows:
    fig,axs=plt.subplots(1,2,figsize=(11,4));x=np.arange(len(rows));axs[0].bar(x-.18,[r['test_agreement_mean'] for r in rows],width=.35,label='SPoSE');axs[0].bar(x+.18,[r['scalar_test_agreement'] for r in rows],width=.35,label='Scalar distance');axs[0].axhline(1/3,color='gray',ls='--');axs[0].set(ylabel='Held-out expected agreement',ylim=(0,1));axs[0].legend()
    axs[1].bar(x,[r['active_dimensions_mean'] for r in rows],color='#cc6677');axs[1].set(ylabel='Active SPoSE dimensions',title='Synthetic 1D is a dimensionality calibration')
    for ax in axs:ax.set_xticks(x);ax.set_xticklabels([r['source'] for r in rows],rotation=15,ha='right',fontsize=8)
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(FIG/f'spose_prediction_and_calibration.{ext}',dpi=250)
    plt.close(fig)
print('Completed SPoSE sources summarized:',len(rows))
if len(rows)==4 and all((OUT/'spose'/s/'additional_checks/COMPLETE.json').exists() for s in ['qwen25vl3b','internvl35_4b','human_rating_induced','synthetic_1d']):
    import subprocess,sys
    subprocess.run([sys.executable,str(BASE/'scripts/analyze_choice_robustness.py')],check=True)
    subprocess.run([sys.executable,str(BASE/'scripts/render_spose_dimensions.py')],check=True)
    subprocess.run([sys.executable,str(BASE/'scripts/record_experiment_provenance.py')],check=True)
    subprocess.run([sys.executable,str(BASE/'scripts/package_gloss_results.py')],check=True)
