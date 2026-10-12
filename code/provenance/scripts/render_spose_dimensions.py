"""Descriptive image exemplars of fitted dimensions, ordered without human-label selection."""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'mllm_material';panel=pd.read_csv(OUT/'manifests/panel432.csv');dest=OUT/'reports/figures/dimension_examples';dest.mkdir(parents=True,exist_ok=True)
for folder in sorted((OUT/'spose').iterdir()):
    if not folder.is_dir() or not (folder/'selected_20seeds.csv').exists() or folder.name=='synthetic_1d':continue
    fit=pd.read_csv(folder/'selected_20seeds.csv').iloc[0];w=np.load(folder/f'd{int(fit["dim"])}_l{float(fit["lam"])}_s{int(fit["seed"])}.npy');active=np.flatnonzero(w.max(0)>.1);order=active[np.argsort(-np.maximum(w[:,active],0).sum(0))]
    for page,start in enumerate(range(0,len(order),4)):
        dims=order[start:start+4];fig,axs=plt.subplots(len(dims),8,figsize=(14,2*len(dims)),squeeze=False)
        for row,dim in enumerate(dims):
            ix=np.argsort(w[:,dim]);chosen=np.r_[ix[-4:][::-1],ix[:4]]
            for col,i in enumerate(chosen):
                ax=axs[row,col]
                with Image.open(BASE/panel.iloc[i].image_512_path) as im:ax.imshow(im.resize((160,160)))
                ax.set_xticks([]);ax.set_yticks([]);ax.set_title(f'{"High" if col<4 else "Low"} {w[i,dim]:.3f}\n{panel.iloc[i].image_id}',fontsize=7)
                if col==0:ax.set_ylabel(f'Dimension {dim+1}',fontsize=9)
        fig.suptitle(folder.name+' — seed 0; dimensions ordered by weight mass',fontsize=11);fig.tight_layout(rect=(0,0,1,.96))
        for ext in ['png','svg']:fig.savefig(dest/f'{folder.name}_page{page+1}.{ext}',dpi=200)
        plt.close(fig)
print('Descriptive dimension exemplars exported.')
for folder in sorted((OUT/'spose').iterdir()):
    extra=folder/'additional_checks'
    if not (extra/'COMPLETE.json').exists():continue
    curve=pd.read_csv(extra/'dimension_truncation_curve.csv');learning=pd.read_csv(extra/'learning_curves.csv');stability=pd.read_csv(folder/'dimension_stability.csv')
    fig,axs=plt.subplots(1,3,figsize=(12,3.8));axs[0].plot(curve.dimensions,curve.validation_ce);axs[0].set(xlabel='Dimensions retained',ylabel='Validation NLL',title='Dimension truncation')
    grouped=learning.groupby('training_limit').test_ce.agg(['mean','std']);axs[1].errorbar(grouped.index,grouped['mean'],yerr=grouped['std'],fmt='o-');axs[1].set(xlabel='Training triplets',ylabel='Test NLL',title='Learning curve (SD across seeds)')
    if 'matched_dimension_r' in stability:axs[2].hist(stability.matched_dimension_r,bins=20,color='#4477aa')
    axs[2].set(xlabel='Mean matched dimension Spearman rho',ylabel='Seed pairs',title='Initialization stability (dependent pairs)');fig.suptitle(folder.name,fontsize=11);fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(OUT/'reports/figures'/f'{folder.name}_spose_diagnostics.{ext}',dpi=220)
    plt.close(fig)
