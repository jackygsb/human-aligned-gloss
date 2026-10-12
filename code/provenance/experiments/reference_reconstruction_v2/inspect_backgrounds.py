from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
p=Path(__file__).resolve().parent
out=p/'output';out.mkdir(exist_ok=True)
sheet=Image.new('RGB',(1024,3*276),'white');draw=ImageDraw.Draw(sheet)
for i,angle in enumerate(range(0,360,30)):
    t=np.deg2rad(angle)
    sc=mi.load_dict({'type':'scene','integrator':{'type':'path'},
      'env':{'type':'envmap','filename':str(p/'assets/doge2.hdr')},
      'sensor':{'type':'perspective','fov':55,'to_world':mi.ScalarTransform4f().look_at(origin=[4*np.sin(t),0,4*np.cos(t)],target=[0,0,0],up=[0,1,0]),
      'film':{'type':'hdrfilm','width':256,'height':256,'rfilter':{'type':'box'}},'sampler':{'type':'independent','sample_count':8}}})
    a=np.array(mi.render(sc,spp=8));a/=np.percentile(a,99)
    a=np.clip(a,0,1);a=np.where(a<=.0031308,a*12.92,1.055*a**(1/2.4)-.055)
    im=Image.fromarray(np.round(a*255).astype('uint8'));sheet.paste(im,(i%4*256,i//4*276));draw.text((i%4*256+5,i//4*276+256),str(angle),fill='black')
sheet.save(out/'doge_background_angles.png')
