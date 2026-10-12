"""Reliability and split-observer robustness of existing matching data."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'mllm_material';FIG=OUT/'reports/figures'
ann=pd.read_csv(BASE/'annotations/gloss_annotations.csv');trials=pd.read_csv(BASE/'annotations/gloss_trials.csv');trials=trials[trials.is_main_image.astype(str).str.lower()=='true'];pairs=pd.read_csv(OUT/'manifests/pairs2000_analysis_only.csv')
lookup={v:i for i,v in enumerate(ann.image_id)};groups=[]
for _,t in trials.groupby('group_id'):
    p=t.pivot_table(index='participant_id',columns='image_id',values='human_pellacini_c',aggfunc='mean');groups.append((np.array([lookup[v] for v in p.columns]),p.to_numpy()))
rng=np.random.RandomState(18721);reliability=[];direction=[];a=pairs.a.to_numpy();b=pairs.b.to_numpy();sign=pairs.human_direction.to_numpy()
for rep in range(2000):
    halves=np.empty((2,len(ann)))
    for ix,v in groups:
        order=rng.permutation(len(v));mid=len(v)//2;halves[0,ix]=v[order[:mid]].mean(0);halves[1,ix]=v[order[mid:]].mean(0)
    rho=float(spearmanr(*halves)[0]);reliability.append(dict(repetition=rep,split_half_spearman=rho,spearman_brown=2*rho/(1+rho)))
    direction.append((halves[:,a]-halves[:,b])*sign>0)
direction=np.stack(direction).mean(0)
pairs['half1_agrees_with_full_direction']=direction[0];pairs['half2_agrees_with_full_direction']=direction[1];pairs.to_csv(OUT/'statistics/human_pair_split_robustness.csv',index=False)
pd.DataFrame(reliability).to_csv(OUT/'statistics/human_split_half_reliability.csv',index=False)
rho=float(spearmanr(ann.human_gloss_score,ann.physical_pellacini_c)[0]);rel=np.array([r['spearman_brown'] for r in reliability])
summary=dict(n_images=len(ann),n_participants=trials.participant_id.nunique(),human_physical_spearman=rho,split_half_spearman_brown_median=float(np.median(rel)),split_half_spearman_brown_95_interval=np.percentile(rel,[2.5,97.5]).tolist(),interpretation='Split-half distribution is a reliability reference, not a strict maximum attainable model correlation. Pair selection used full human data; split checks are robustness, not independent confirmation.')
(OUT/'statistics/human_reference_summary.json').write_text(json.dumps(summary,indent=2))
fig,axs=plt.subplots(1,3,figsize=(12,3.7));axs[0].hexbin(ann.physical_pellacini_c,ann.human_gloss_score,gridsize=30,mincnt=1,cmap='viridis');axs[0].set(xlabel='Physical gloss c',ylabel='Human matched gloss c',title=f'Existing behavior: rho={rho:.3f}')
axs[1].hist(rel,bins=25,color='#4477aa');axs[1].set(xlabel='Spearman–Brown corrected split-half rho',ylabel='Observer splits',title='Human reliability reference')
d=pairs[pairs.stratum=='discordant'];axs[2].hist(d[['half1_agrees_with_full_direction','half2_agrees_with_full_direction']].mean(axis=1),bins=20,color='#cc6677');axs[2].set(xlabel='Agreement across observer half-splits',ylabel='Discordant pairs',title='Pair-selection robustness\nNot independent validation')
axs[1].set_xlabel('Corrected split-half Spearman rho')
for ax in axs:ax.title.set_fontsize(10);ax.xaxis.label.set_fontsize(10)
fig.tight_layout()
for ext in ['png','svg']:fig.savefig(FIG/f'human_data_and_reliability.{ext}',dpi=250)
print(json.dumps(summary,indent=2))
