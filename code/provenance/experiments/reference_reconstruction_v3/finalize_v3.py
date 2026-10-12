from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import json
p=Path(__file__).resolve().parent
source=Path('C:/Users/NCRC-3~1/AppData/Local/Temp/codex-clipboard-8cc64efe-0268-40c6-97c2-e690f79ddab4.png')
paper=Image.open(source).convert('RGB')
tile=paper.crop((18,33,143,160)).resize((512,512))
a=np.asarray(tile)/255;b=np.asarray(Image.open(p/'output/reference_01.png'))/255
def linear(x):return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
ratio=float(linear(b)[465:502,35:150].mean()/linear(a)[465:502,35:150].mean())
s=(p.parent/'reference_reconstruction_v1/finalize_preview.py').read_text(encoding='utf-8')
s=s.replace('safe=float(max(original,np.percentile(stack[-1][obj],99.9)/.98))',f'safe=original*{ratio!r}')
s=s.replace('Reduce high-gloss object clipping for visual review; uniform across all 52 levels.','Uniform exposure adjustment using the published screenshot floor patch only, not human ratings.')
s=s.replace('max(global97, max-gloss-object channel percentile99.9 / .98)',f'global97 * {ratio:.10f}; ratio from fixed floor patch in screenshot')
exec(compile(s,str(p/'generated_finalize.py'),'exec'),{'__file__':str(p/'generated_finalize.py')})
s=(p.parent/'reference_reconstruction_v2/publish_preview.py').read_text(encoding='utf-8')
s=s.replace('v2','v3').replace('reflectance .30','reflectance .24').replace('漫反射率0.30','漫反射率0.24').replace('共同97百分位曝光','共同97百分位基础曝光，再按论文截图地面区域统一校准')
exec(compile(s,str(p/'generated_publish.py'),'exec'),{'__file__':str(p/'generated_publish.py')})
paper.save(p/'output/paper_five_examples.png')
contact=Image.new('RGB',(1280,650),'white');draw=ImageDraw.Draw(contact)
draw.text((15,8),'Paper examples: intermediate c values are not specified',fill='black')
contact.paste(paper.crop((18,33,663,160)).resize((1280,252)),(0,34))
draw.text((15,302),'Reconstruction v3: c = 0, 0.03790, 0.07581, 0.11080, 0.14870',fill='black')
contact.paste(Image.open(p/'output/gloss_preview.png'),(0,332));contact.save(p/'output/paper_five_vs_v3.png')
h=p/'output/preview.html';t=h.read_text(encoding='utf-8').replace('<h2>五档对比</h2>','<h2>论文五图与新版对比</h2><p>上排为论文截图，只有两端范围已标注；下排中间档位为重建采样，并非已确认的原图标签。</p><img style="width:100%" src="paper_five_vs_v3.png" alt="论文五图与重建五图"><h2>新版五档</h2>');h.write_text(t,encoding='utf-8')
(p/'output/screenshot_exposure_calibration.json').write_text(json.dumps({'ratio':ratio,'screenshot':source.name,'tile_crop':[18,33,143,160],'resized_patch_yx':[465,502,35,150],'interpretation':'display appearance matching only, not recovery of physical exposure'},indent=2))

embedded=p/'audit/pdf_figure_assets/0_Im0.jpg'
if embedded.exists():
    (p/'output/paper_embedded_five.jpg').write_bytes(embedded.read_bytes())
    im=Image.open(embedded)
    for i in range(5):
        im.crop((i*101,0,(i+1)*101,100)).save(p/'output'/f'paper_example_{i+1}.png')
    t=h.read_text(encoding='utf-8').replace('<h2>论文五图与新版对比</h2>','<h2>从论文PDF提取的原始五图</h2><img style="width:100%" src="paper_embedded_five.jpg"><p>PDF内嵌JPEG：505×100像素。中间三图档位未标注。</p><h2>论文五图与新版对比</h2>')
    h.write_text(t,encoding='utf-8')
