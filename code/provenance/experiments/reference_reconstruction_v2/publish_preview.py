from pathlib import Path
import shutil,json
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_erosion
p=Path(__file__).resolve().parent;out=p/'output'
source=Path('C:/Users/NCRC-3~1/AppData/Local/Temp/codex-clipboard-0986a021-da60-464e-9a7b-b3810347a1eb.png')
shutil.copy2(source,out/'paper_reference_screenshot.png')
html=(p.parent/'reference_reconstruction_v1/output/preview.html').read_text(encoding='utf-8')
html=html.replace('参考光泽序列 · 重建预览','参考光泽序列 v2 · 与论文图对照')
html=html.replace('这是依据论文方法重建的刺激，凹凸球为新建几何；尚未用于模型采集。','按论文示意图对齐背景、平滑球面并提高底色明度；尚未用于模型采集。注意：论文文字写Uffizi，图示外观却与Doge庭院相符，此处采用图示对齐版本。')
html=html.replace('<section><img id="ref" src="reference_27.png" alt="当前光泽参考球">','<section><div class="comparison"><figure><img src="paper_reference_screenshot.png" alt="用户提供的论文参考图"><figcaption>论文示例（此图具体c未知）</figcaption></figure><figure><img id="ref" src="reference_27.png" alt="当前光泽参考球"><figcaption>v2 重建（可调节档位）</figcaption></figure></div>')
html=html.replace('</style>','.comparison{display:grid;grid-template-columns:1fr 1fr;gap:16px}.comparison figure{margin:0}.comparison img{width:100%;aspect-ratio:1;object-fit:contain}.comparison figcaption{text-align:center;margin:12px 0;color:#59616c}@media(max-width:600px){.comparison{grid-template-columns:1fr}}</style>')
html=html.replace('Uffizi HDR环境','Doge庭院HDR环境（按图示重建）').replace('中性底色是本次重建设定。','中性漫反射率0.30；相机方向315°；共同97百分位曝光。均为本次重建设定，非作者原始参数恢复。')
(out/'preview.html').write_text(html,encoding='utf-8')
a=Image.open(source).convert('RGB').resize((512,512));b=Image.open(out/'reference_27.png')
canvas=Image.new('RGB',(1040,552),'white');canvas.paste(a,(0,0));canvas.paste(b,(528,0))
draw=ImageDraw.Draw(canvas);draw.text((15,525),'Paper example (c unknown)',fill='black');draw.text((543,525),'Reconstruction v2 | k=27 | c=0.07581',fill='black');canvas.save(out/'paper_vs_v2.png')
d=np.load(out/'diffuse_with_background_linear.npy');s=np.load(out/'unit_specular_linear.npy');m=d[:,:,3]>.99; bg=binary_erosion(~m,iterations=2)
imgs=np.stack([np.array(Image.open(out/f'reference_{i:02d}.png')) for i in range(1,53)])
validation={'count':len(imgs),'unique_images':len(set(im.tobytes() for im in imgs)),
'background_max_byte_change_excluding_boundary':int(np.abs(imgs.astype('int16')-imgs[0].astype('int16'))[:,bg].max()),
'object_saturated_pixel_fraction_last':float(np.any(imgs[-1][m]>=255,axis=1).mean()),
'monotonic_encoded_channels':bool(np.all(np.diff(imgs.astype('int16'),axis=0)>=0)),
'geometry_status':'new smoother geometry, not recovered author mesh','environment_status':'figure-matched Doge, conflicts with text Uffizi; not resolved',
'human_labels_used_for_tuning':False,'model_collection_started':False}
(out/'validation.json').write_text(json.dumps(validation,indent=2));print(validation)
