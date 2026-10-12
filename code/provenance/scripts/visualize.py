"""Scientific plots; the single human attribute is shown without fabricated embeddings."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

BASE=Path(__file__).resolve().parents[1]

def main():
    a=pd.read_csv(BASE/'annotations/gloss_annotations.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,6,figsize=(15,8))
    # One example for each of 18 distinct shapes, with varied illumination.
    for k,ax in enumerate(axes.flat):
        shape=sorted(a.shape_id.unique())[k*2]
        subset=a[a.shape_id==shape].sort_values(['illumination_id','viewpoint_id'])
        r=subset.iloc[(k*7)%len(subset)]
        with Image.open(BASE/r.image_224_path) as im:
            ax.imshow(im)
        ax.axis('off')
        ax.set_title(f'{r.image_id} | c={r.human_gloss_score:.3f}\nShape {r.shape_id}, light {r.illumination_id}, view {r.viewpoint_id}',fontsize=9)
    fig.suptitle('Morimoto dataset: 3,888 images | 36 shapes | 36 illuminations | 3 views',fontsize=16)
    fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(BASE/'figures/01_dataset_overview.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    axes[0].hist(a.human_gloss_score,bins=35,color='#307e94'); axes[0].set(xlabel='Mean human gloss (Pellacini c)',ylabel='Images',title='Human gloss distribution')
    axes[1].scatter(a.physical_pellacini_c,a.human_gloss_score,s=4,alpha=.2,color='#307e94'); axes[1].plot([0,.1487],[0,.1487],color='gray',lw=1)
    r=np.corrcoef(a.physical_pellacini_c,a.human_gloss_score)[0,1]
    axes[1].set(xlabel='Physical Pellacini c',ylabel='Mean human Pellacini c',title=f'Human vs physical gloss | r = {r:.4f}')
    axes[2].hist(a.n_observers,bins=np.arange(a.n_observers.min()-.5,a.n_observers.max()+1.5),color='#ce8840'); axes[2].set(xlabel='Observers per image',ylabel='Images',title='Annotation coverage')
    fig.tight_layout(); fig.savefig(BASE/'figures/02_annotation_distribution.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,7),gridspec_kw={'height_ratios':[2,1]})
    s=a.sort_values('human_gloss_score'); rank=np.arange(len(s))
    axes[0].plot(rank,s.human_gloss_score,color='#307e94',label='Mean human gloss')
    axes[0].fill_between(rank,(s.human_gloss_score-s.human_gloss_sem).to_numpy(),(s.human_gloss_score+s.human_gloss_sem).to_numpy(),alpha=.25,color='#307e94',label='+/- 1 SEM across observers')
    axes[0].set(xlabel='Image rank by mean human gloss',ylabel='Pellacini c',title='One-dimensional human perceptual space'); axes[0].legend()
    axes[1].scatter(a.human_gloss_score,a.shape_id,c=a.illumination_id,cmap='viridis',s=4,alpha=.4)
    axes[1].set(xlabel='Mean human gloss (Pellacini c)',ylabel='Shape ID',title='Gloss coverage across shapes (colour: illumination ID)')
    fig.tight_layout(); fig.savefig(BASE/'figures/03_human_perceptual_space.png',dpi=160); plt.close(fig)

if __name__=='__main__': main()
