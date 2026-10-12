"""Phase I descriptive analyses; no trained vision models or inferential p-values."""
from pathlib import Path
import hashlib
import json
import itertools
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'analysis'
ATTRS=['glossiness','refsharp','contgloss','metallicness','lightness','anisotropy']
SEED=20260926

def pca(x):
    mu=x.mean(0); sd=x.std(0,ddof=0); z=(x-mu)/sd
    u,s,vt=np.linalg.svd(z,full_matrices=False)
    for j in range(6):
        if vt[j,np.argmax(abs(vt[j]))]<0:
            vt[j]*=-1; u[:,j]*=-1
    return u*s,vt,s*s/(s*s).sum(),mu,sd,z

def main():
    OUT.mkdir(exist_ok=True)
    a=pd.read_csv(BASE/'annotations/material_annotations.csv',low_memory=False)
    x=a[ATTRS].to_numpy(float)
    score,axes,var,mu,sd,z=pca(x)
    ranked=np.column_stack([rankdata(x[:,j],method='average') for j in range(6)])
    rs,ra,rv,_,_,_=pca(ranked)
    assert np.allclose(score@axes,z,atol=1e-10)
    assert np.allclose(axes@axes.T,np.eye(6),atol=1e-10)
    assert np.isclose(var.sum(),1)
    coords=a[['image_id','material_id','shape_id','illumination_id','image_available']].copy()
    for i in range(6): coords[f'PC{i+1}']=score[:,i]
    coords.to_csv(OUT/'material_pca_scores.csv',index=False)
    pd.DataFrame(axes.T,index=ATTRS,columns=[f'PC{i+1}' for i in range(6)]).to_csv(OUT/'material_pca_components.csv',index_label='attribute')
    pd.DataFrame({'component':np.arange(1,7),'standardized_explained_ratio':var,'rank_standardized_explained_ratio':rv}).to_csv(OUT/'material_pca_explained_variance.csv',index=False)
    corr=pd.DataFrame(x,columns=ATTRS).corr(method='spearman')
    corr.to_csv(OUT/'material_attribute_spearman.csv',index_label='attribute')
    # Balanced complete factorial; deterministic decomposition of observed cell medians.
    factors=['material_id','shape_id','illumination_id']
    levels=[sorted(a[f].unique()) for f in factors]
    table=a.set_index(factors).reindex(pd.MultiIndex.from_product(levels,names=factors))
    assert not table[ATTRS].isna().any().any()
    tensor=table[ATTRS].to_numpy().reshape(520,9,9,6)
    grand=tensor.mean(axis=(0,1,2),keepdims=True)
    effects={():grand}
    decomposition=[]
    total=((tensor-grand)**2).sum(axis=(0,1,2))
    for k in [1,2,3]:
        for subset in itertools.combinations(range(3),k):
            avg=tensor.mean(axis=tuple(i for i in range(3) if i not in subset),keepdims=True) if k<3 else tensor.copy()
            effect=avg.copy()
            for old,term in effects.items():
                if set(old)<set(subset): effect=effect-term
            effects[subset]=effect
            ss=(np.broadcast_to(effect,tensor.shape)**2).sum(axis=(0,1,2))
            for j,attr in enumerate(ATTRS):
                decomposition.append(dict(attribute=attr,component=' x '.join(factors[i] for i in subset),sum_squares=ss[j],fraction=ss[j]/total[j]))
    dec=pd.DataFrame(decomposition)
    assert np.allclose(sum(effects.values()),tensor,atol=1e-10)
    assert np.allclose(dec.groupby('attribute').fraction.sum(),1,atol=1e-10)
    dec.to_csv(OUT/'material_factorial_variation.csv',index=False)
    ranges=a.groupby('material_id')[ATTRS].agg(lambda s:s.max()-s.min())
    ranges.to_csv(OUT/'material_context_ranges.csv')
    # Stratified random observer splits: preserve each observer's complete 72-image response vector.
    g=pd.read_csv(BASE/'annotations/gloss_annotations.csv')
    trials=pd.read_csv(BASE/'annotations/gloss_trials.csv')
    trials=trials[trials.is_main_image]
    observer=trials.groupby(['group_id','participant_id','image_id']).human_pellacini_c.mean().reset_index()
    groups=[]
    for group,t in observer.groupby('group_id'):
        p=t.pivot(index='image_id',columns='participant_id',values='human_pellacini_c')
        assert p.shape[0]==72 and p.shape[1]>=3 and not p.isna().any().any()
        groups.append(p.to_numpy())
    rng=np.random.default_rng(SEED)
    split=[]
    for repeat in range(1000):
        left=[];right=[]
        for matrix in groups:
            perm=rng.permutation(matrix.shape[1]); cut=len(perm)//2
            left.extend(matrix[:,perm[:cut]].mean(1));right.extend(matrix[:,perm[cut:]].mean(1))
        split.append(np.corrcoef(left,right)[0,1])
    pd.DataFrame({'iteration':np.arange(1000),'pearson_r':split}).to_csv(OUT/'gloss_observer_split_reliability.csv',index=False)
    # Draft OOD folds use IDs only; these are not the original authors' fold files.
    folds=g[['image_id','shape_id','illumination_id','viewpoint_id']].copy()
    for name,col in [('shape','shape_id'),('illumination','illumination_id')]:
        perm=np.random.default_rng(SEED+(name=='illumination')).permutation(sorted(g[col].unique()))
        mapping={value:i//3 for i,value in enumerate(perm)}
        folds[f'{name}_fold']=folds[col].map(mapping)
        assert folds.groupby(f'{name}_fold').size().eq(324).all()
    folds.to_csv(BASE/'protocols/gloss_group_folds.csv',index=False)
    hashes={name:hashlib.sha256((BASE/'annotations'/name).read_bytes()).hexdigest() for name in ['material_annotations.csv','gloss_annotations.csv','gloss_trials.csv']}
    result=dict(seed=SEED,material_images=len(a),attributes=ATTRS,pca_explained_ratio=var.tolist(),
        all_six_at_floor_fraction=float((x==1).all(axis=1).mean()),
        pca_first_two_fraction=float(var[:2].sum()),rank_pca_first_two_fraction=float(rv[:2].sum()),
        pca_mean=mu.tolist(),pca_population_std=sd.tolist(),
        material_main_effect_fraction=dec[dec.component=='material_id'].set_index('attribute').fraction.to_dict(),
        context_range_at_least_two_fraction=(ranges>=2).mean().to_dict(),
        gloss_observer_split=dict(iterations=1000,median=float(np.median(split)),split_randomization_percentiles_2_5_97_5=np.percentile(split,[2.5,97.5]).tolist(),
            interpretation='Random-split stability on these fixed participants and stimuli; NOT a population confidence interval or an RSA noise ceiling. Odd groups have unequal halves; no Spearman-Brown correction.'),
        ordinal_caution='PCA and sum-of-squares decomposition are descriptive interval-scale approximations to medians; rank PCA is a sensitivity analysis.',
        factorial_caution='No replicate error estimate: three-way term includes measurement noise; fractions are not causal effects or statistical significance.',
        validated=dict(pca_reconstruction=True,orthonormal_components=True,factorial_reconstruction=True,variance_fractions_sum_to_one=True,complete_factorial=True,fold_sizes=True),input_sha256=hashes)
    (OUT/'perception_analysis_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(2,3,figsize=(13,7))
    for j,axis in enumerate(ax.flat):
        vals,counts=np.unique(x[:,j],return_counts=True)
        axis.bar(vals,counts,width=.38,color='#2f8198');axis.set(title=ATTRS[j],xlabel='Image median rating',ylabel='Images',xticks=range(1,8))
    fig.suptitle('Serrano: distributions of all 42,120 image-level medians')
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(BASE/'figures/04_material_distributions.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(1,3,figsize=(15,4.6))
    cloud=ax[0].hexbin(score[:,0],score[:,1],gridsize=45,mincnt=1,bins='log',cmap='viridis');fig.colorbar(cloud,ax=ax[0],label='Image count (log colour scale)')
    ax[0].set(xlabel=f'PC1 ({var[0]:.1%})',ylabel=f'PC2 ({var[1]:.1%})',title='Standardized rating space')
    ax[1].plot(range(1,7),var,'o-',label='Raw-score z-standardized');ax[1].plot(range(1,7),rv,'s--',label='Rank z-standardized')
    ax[1].set(xlabel='Principal component',ylabel='Explained variance fraction',title='Ordinal-scale sensitivity');ax[1].legend(fontsize=8)
    im=ax[2].imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r');ax[2].set(xticks=range(6),yticks=range(6),xticklabels=ATTRS,yticklabels=ATTRS,title='Attribute Spearman correlations')
    ax[2].tick_params(axis='x',rotation=65);fig.colorbar(im,ax=ax[2],shrink=.75)
    fig.tight_layout();fig.savefig(BASE/'figures/05_material_perceptual_space.png',dpi=160,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(13,5))
    pivot=dec.pivot(index='attribute',columns='component',values='fraction').reindex(ATTRS)
    pivot.plot.bar(stacked=True,ax=ax[0],colormap='tab20',rot=35);ax[0].set(ylabel='Fraction of observed squared variation',xlabel='',title='Descriptive factorial decomposition');ax[0].legend(fontsize=7,loc='upper left',bbox_to_anchor=(0,-.3),ncol=2)
    ax[1].hist(split,bins=30,color='#2f8198');ax[1].set(xlabel='Pearson r between disjoint observer halves',ylabel='Random splits',title='Morimoto: 1,000 stratified observer splits')
    fig.tight_layout();fig.savefig(BASE/'figures/06_context_and_reliability.png',dpi=160,bbox_inches='tight');plt.close(fig)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
