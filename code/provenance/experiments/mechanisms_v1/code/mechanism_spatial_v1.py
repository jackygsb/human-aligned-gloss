"""Descriptive spatial controls: local image statistics and the released author kernel.
This is NOT a reproduced author network predictor: no fitted head, common background-inclusive input.
"""
from pathlib import Path
import json,shutil,hashlib
import numpy as np,pandas as pd
from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import uniform_filter,sobel
from scipy.signal import correlate
from scipy.stats import spearmanr
B=Path(__file__).resolve().parents[1];O=B/'mllm_material/mechanisms_v1'
def controls(image,side,kernel):
 rgb=np.asarray(Image.open(image).convert('RGB').resize((224,224)),dtype='float32')/255;lum=rgb@[.2126,.7152,.0722];mean=uniform_filter(lum,size=15);contrast=np.sqrt(np.maximum(uniform_filter(lum**2,size=15)-mean**2,0));grad=np.hypot(sobel(lum,axis=0),sobel(lum,axis=1));response=sum(correlate(np.pad(rgb[:,:,c],7,mode='reflect'),kernel[:,:,c],mode='valid',method='fft') for c in range(3));features={'luminance':lum,'local_contrast':contrast,'gradient_magnitude':grad,'positive_local_brightness':np.maximum(lum-mean,0),'yellow_minus_blue':(rgb[:,:,0]+rgb[:,:,1])/2-rgb[:,:,2],'author_kernel_signed':response,'author_kernel_positive':np.maximum(response,0)}
 return {k:np.asarray(Image.fromarray(v.astype('float32')).resize((side,side),Image.Resampling.BOX)).ravel() for k,v in features.items()}
def run():
 source=B.parent/'gloss_tinynetworks-main/data/networks/human_onelayer_kernelN1_trainedby3888imgs/kernel.mat';kernel=loadmat(source)['kernel'];shutil.copy2(source,O/'author_kernel_full_fit.mat');shutil.copy2(__file__,O/'code'/Path(__file__).name);rows=[]
 d=pd.read_csv(O/'stimuli.csv');chosen=set(json.loads((O/'plan.json').read_text())['intervention_images'])
 for name in ['InternVL','Qwen3']:
  direction=np.load(O/name/'directions.npz')['candidate']
  for r in d[d.image_id.isin(chosen)].itertuples():
   tok=np.load(O/name/'features'/f'{r.image_id}.npz')['bridge_tokens'].astype('float32');scores=tok@direction;side=int(np.sqrt(len(scores)));assert side*side==len(scores);f=controls(B/r.image_512_path,side,kernel)
   np.savez_compressed(O/name/'features'/f'{r.image_id}_spatial_controls.npz',candidate=scores,**f)
   for key,z in f.items():rows.append(dict(model=name,image_id=r.image_id,control=key,spatial_spearman=float(spearmanr(scores,z).statistic)))
 result=pd.DataFrame(rows);result.to_csv(O/'tables/spatial_feature_associations.csv',index=False);result.groupby(['model','control']).spatial_spearman.agg(['mean','median','std','count']).to_csv(O/'tables/spatial_feature_summary.csv')
 (O/'Spatial_controls.md').write_text('''# Spatial feature controls\n\nDescriptive within-image spatial Spearman correlations on the 72 predetermined intervention images. Local maps computed at 224 square, then BOX averaged to the native bridge token grid. Whole image and background included; no highlight segmentation or human region annotation. Kernel response is cross-correlation using the author-released full-data-trained 15x15x3 kernel, with reflected boundaries on RGB in [0,1]. The author regression head, exact original preprocessing and no-background input are NOT reproduced here; this is only a spatial filter-response comparison. Since this kernel was supervised on all original human data, no out-of-sample human prediction is claimed from these control maps. Each image contributes one correlation per feature; patches are not treated as independent experimental samples.\n''',encoding='utf-8')
 return result
if __name__=='__main__':run()
