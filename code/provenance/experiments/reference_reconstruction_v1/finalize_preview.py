from pathlib import Path
import json,csv,shutil
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_erosion

p=Path(__file__).resolve().parent/'output'
d=np.load(p/'diffuse_with_background_linear.npy');s=np.load(p/'unit_specular_linear.npy')
obj=d[:,:,3]>.99
c=np.linspace(0,.1487,52);ps=(c+(0.5**3/2)**(1/3))**3-0.5**3/2
stack=np.stack([d[:,:,:3]+v*s[:,:,:3] for v in ps])
original=float(np.percentile(stack,97))
safe=float(max(original,np.percentile(stack[-1][obj],99.9)/.98))
baseline=p/'baseline_97percentile';baseline.mkdir(exist_ok=True)
for f in p.glob('reference_*.png'):shutil.copy2(f,baseline/f.name)
shutil.copy2(p/'gloss_preview.png',baseline/'gloss_preview.png')
frames=[];records=[]
for k,im in enumerate(stack,1):
    linear=np.clip(im/safe,0,1)
    enc=np.where(linear<=.0031308,12.92*linear,1.055*linear**(1/2.4)-.055)
    frame=Image.fromarray(np.round(enc*255).astype('uint8'));frame.save(p/f'reference_{k:02d}.png');frames.append(frame)
    records.append({'index':k,'pellacini_c':c[k-1],'ward_specularity':ps[k-1],
      'png':f'reference_{k:02d}.png','clipped_object_pixel_fraction':float(np.mean(np.any(im[obj]/safe>1,axis=1))),
      'shared_exposure_divisor':safe})
with (p/'reference_levels.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
preview=Image.new('RGB',(1280,302),'white');draw=ImageDraw.Draw(preview)
for j,k in enumerate([0,13,26,38,51]):
    preview.paste(frames[k].resize((256,256)),(j*256,0));draw.text((j*256+12,265),f'k={k+1:02d}   c={c[k]:.5f}',fill='black')
preview.save(p/'gloss_preview.png')
sheet=Image.new('RGB',(1152,1190),'white');draw=ImageDraw.Draw(sheet)
for j,im in enumerate(frames):
    x=j%8*144;y=j//8*170;sheet.paste(im.resize((144,144)),(x,y));draw.text((x+5,y+147),f'{j+1:02d} c={c[j]:.5f}',fill='black')
sheet.save(p/'all_52_levels.png')
metadata=json.loads((p/'render_metadata.json').read_text())
metadata.update({'final_exposure_divisor':safe,'exposure_adjustment_reason':'Reduce high-gloss object clipping for visual review; uniform across all 52 levels.',
  'original_97percentile_outputs':'baseline_97percentile/','final_exposure_rule':'max(global97, max-gloss-object channel percentile99.9 / .98)',
  'status':'visual_review_only_not_data_collection'})
(p/'render_metadata.json').write_text(json.dumps(metadata,indent=2))
imgs=np.stack([np.asarray(f) for f in frames]);bg=binary_erosion(~obj,iterations=2)
validation={'count':len(imgs),'unique_image_count':len(set(a.tobytes() for a in imgs)),
 'dimensions':[512,512],'background_max_byte_change_excluding_2px_edge':int(np.abs(imgs.astype('int16')-imgs[0].astype('int16'))[:,bg].max()),
 'object_clipped_pixel_fraction_last':records[-1]['clipped_object_pixel_fraction'],
 'encoded_monotonic':bool(np.all(np.diff(imgs.astype('int16'),axis=0)>=0)),
 'c_range':[float(c.min()),float(c.max())],'specular_range':[float(ps.min()),float(ps.max())]}
(p/'validation.json').write_text(json.dumps(validation,indent=2));print(validation)
