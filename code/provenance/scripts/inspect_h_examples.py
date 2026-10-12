from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
B=Path(__file__).resolve().parents[1];M=B/'mllm_material';O=M/'figure1_english_v6';O.mkdir(exist_ok=True)
t=pd.read_csv(M/'mechanisms_v1/tables/spatial_feature_associations.csv')
t=t[t.control=='positive_local_brightness'].pivot(index='image_id',columns='model',values='spatial_spearman')
t['min_both']=t.min(axis=1);t=t.sort_values('min_both',ascending=False);t.to_csv(O/'h_example_selection.csv')
ann=pd.read_csv(M/'mechanisms_v1/stimuli.csv').set_index('image_id')
fig,ax=plt.subplots(4,6,figsize=(15,10))
for j,iid in enumerate(t.head(8).index):
    r,c=divmod(j,2);c*=3
    im=Image.open(B/ann.loc[iid,'image_512_path'])
    for k,name in enumerate(['Input','InternVL','Qwen3']):
        a=ax[r,c+k];a.imshow(im);a.axis('off')
        if k:
            z=np.load(M/f'mechanisms_v1/{name}/features/{iid}_spatial_controls.npz')['candidate'];a.imshow(z.reshape(16,16),cmap='magma',alpha=.55,extent=(0,512,512,0));a.set_title(f'{name} r={t.loc[iid,name]:.2f}',fontsize=10)
        else:a.set_title(iid,fontsize=10)
fig.tight_layout();fig.savefig(O/'selection_contact.png',dpi=130);plt.close(fig)
print(t.head(8).to_string())
