"""Numerical null calibration only: no model responses or actual scores used."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import spearmanr
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'mllm_material/protocol'
rng=np.random.RandomState(48911);shape=np.repeat(np.arange(36),108);light=np.tile(np.repeat(np.arange(36),3),36);n=len(shape);results=[]
for trial in range(100):
    a=rng.normal(size=36)[shape]+rng.normal(size=36)[light]+rng.normal(size=n)*.5
    b=rng.normal(size=36)[shape]+rng.normal(size=36)[light]+rng.normal(size=n)*.5
    estimates=[]
    for rep in range(200):
        sw=rng.multinomial(36,np.ones(36)/36);lw=rng.multinomial(36,np.ones(36)/36);ix=np.repeat(np.arange(n),sw[shape]*lw[light]);estimates.append(float(spearmanr(a[ix],b[ix])[0]))
    lo,hi=np.percentile(estimates,[2.5,97.5]);results.append(dict(trial=trial,rho=float(spearmanr(a,b)[0]),low=float(lo),high=float(hi),covers_zero=bool(lo<=0<=hi)))
coverage=float(np.mean([r['covers_zero'] for r in results]));report=dict(kind='synthetic numerical null validation, not experimental observations',simulations=100,bootstrap_replicates_per_simulation=200,formal_analysis_bootstraps=2000,true_superpopulation_correlation=0.,crossed_shape_illumination_null_coverage=coverage,review_flag=coverage<.9,results=results)
(OUT/'structured_bootstrap_null_check.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k!='results'})
