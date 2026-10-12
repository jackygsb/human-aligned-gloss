"""Reconstructed reference, not the author's original mesh or experiment assets.
Balanced Ward equations follow Mitsuba 0.6 ward.cpp (Wenzel Jakob et al., GPLv3).
This implementation and derivative plugin are distributed under GPLv3.
"""
from pathlib import Path
import json, csv, math, hashlib
import numpy as np
from PIL import Image, ImageDraw
import mitsuba as mi
import drjit as dr

mi.set_variant('cuda_ad_rgb')
ROOT = Path(__file__).resolve().parent
ASSETS = ROOT/'assets'
OUT = ROOT/'output'
OUT.mkdir(exist_ok=True)

class BalancedWard(mi.BSDF):
    def __init__(self, props):
        mi.BSDF.__init__(self, props)
        self.alpha = .05
        self.m_flags = mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.FrontSide
        self.m_components = [self.m_flags]

    def eval(self, ctx, si, wo, active=True):
        h = si.wi + wo
        hz2 = dr.maximum(h.z*h.z, 1e-20)
        e = dr.exp(-(h.x*h.x+h.y*h.y)/(self.alpha**2*hz2))
        f = dr.squared_norm(h)*e/(dr.pi*self.alpha**2*hz2*hz2)
        valid = active & (si.wi.z>0) & (wo.z>0)
        return mi.Color3f(dr.select(valid, f*wo.z, 0))

    def pdf(self, ctx, si, wo, active=True):
        h = dr.normalize(si.wi+wo)
        hz2 = dr.maximum(h.z*h.z,1e-20)
        e = dr.exp(-(h.x*h.x+h.y*h.y)/(self.alpha**2*hz2))
        p = e/dr.maximum(4*dr.pi*self.alpha**2*dr.dot(h,si.wi)*h.z**3,1e-20)
        return dr.select(active & (si.wi.z>0) & (wo.z>0), p, 0)

    def sample(self, ctx, si, s1, s2, active=True):
        t2 = -self.alpha**2*dr.log(dr.maximum(1-s2.x,1e-8))
        ct = dr.rsqrt(1+t2)
        st = dr.sqrt(dr.maximum(0,1-ct*ct))
        phi = 2*dr.pi*s2.y
        h = mi.Vector3f(st*dr.cos(phi),st*dr.sin(phi),ct)
        bs = mi.BSDFSample3f()
        bs.wo = 2*dr.dot(si.wi,h)*h-si.wi
        bs.pdf = self.pdf(ctx,si,bs.wo,active)
        bs.eta = 1
        bs.sampled_component = 0
        bs.sampled_type = mi.UInt32(+mi.BSDFFlags.GlossyReflection)
        value = self.eval(ctx,si,bs.wo,active)/dr.maximum(bs.pdf,1e-20)
        return bs, dr.select(bs.pdf>0,value,mi.Color3f(0))

    def eval_pdf(self,ctx,si,wo,active=True):
        return self.eval(ctx,si,wo,active), self.pdf(ctx,si,wo,active)

    def to_string(self):
        return 'BalancedWard[alpha=0.05, unit_specular]'

mi.register_bsdf('balanced_ward_reconstructed',lambda p:BalancedWard(p))

def make_mesh():
    rng=np.random.default_rng(20260927)
    # Distributed smooth bumps on the sphere, explicit reproducible replacement mesh.
    count=125
    idx=np.arange(count)
    y=1-2*(idx+.5)/count
    phi=idx*np.pi*(3-np.sqrt(5))
    centers=np.stack([np.sqrt(1-y*y)*np.cos(phi),y,np.sqrt(1-y*y)*np.sin(phi)],1)
    centers += rng.normal(0,.055,centers.shape)
    centers /= np.linalg.norm(centers,axis=1,keepdims=True)
    widths=rng.uniform(.075,.12,count)
    amps=rng.uniform(.08,.16,count)
    nt,np_=192,384
    th=np.linspace(1e-5,np.pi-1e-5,nt)
    ph=np.arange(np_)*2*np.pi/np_
    tt,pp=np.meshgrid(th,ph,indexing='ij')
    u=np.stack([np.sin(tt)*np.cos(pp),np.cos(tt),np.sin(tt)*np.sin(pp)],-1).reshape(-1,3)
    r=np.ones(len(u)); g=np.zeros_like(u)
    for c,w,a in zip(centers,widths,amps):
        dot=u@c; b=a*np.exp((dot-1)/(w*w)); r+=b
        g+=(b/(w*w))[:,None]*(c-dot[:,None]*u)
    r-=.07
    v=r[:,None]*u
    normal=u-g/r[:,None]; normal/=np.linalg.norm(normal,axis=1,keepdims=True)
    faces=[]
    for j in range(nt-1):
        for k in range(np_):
            a=j*np_+k;b=j*np_+(k+1)%np_;c=(j+1)*np_+k;d=(j+1)*np_+(k+1)%np_
            faces.extend([(a,b,c),(b,d,c)])
    f=np.array(faces)
    if np.dot(np.cross(v[f[100,1]]-v[f[100,0]],v[f[100,2]]-v[f[100,0]]),v[f[100,0]])<0:
        f=f[:,[0,2,1]]
    path=ASSETS/'reconstructed_bumpy_sphere.obj'
    with path.open('w') as out:
        for row in v: out.write('v '+' '.join(map(str,row))+'\n')
        for row in normal: out.write('vn '+' '.join(map(str,row))+'\n')
        for row in f+1: out.write('f '+' '.join(f'{q}//{q}' for q in row)+'\n')
    return path

def scene(bsdf, angle, res=512, sphere=False, hide=False):
    t=np.deg2rad(angle)
    shape={'type':'sphere','bsdf':bsdf} if sphere else {'type':'obj','filename':str(ASSETS/'reconstructed_bumpy_sphere.obj'),'bsdf':bsdf}
    return mi.load_dict({'type':'scene',
        'integrator':{'type':'path','max_depth':2,'hide_emitters':hide},
        'environment':{'type':'envmap','filename':str(ASSETS/'uffizi-large.hdr')},
        'object':shape,
        'sensor':{'type':'perspective','fov':45,
            'to_world':mi.ScalarTransform4f().look_at(origin=[4.0*np.sin(t),0,4.0*np.cos(t)],target=[0,0,0],up=[0,1,0]),
            'sampler':{'type':'independent','sample_count':64},
            'film':{'type':'hdrfilm','width':res,'height':res,'pixel_format':'rgba','rfilter':{'type':'box'}}}})

def srgb(a):
    a=np.clip(a,0,1)
    return np.where(a<=.0031308,12.92*a,1.055*a**(1/2.4)-.055)

def render(s,spp,seed=42):
    return np.array(mi.render(s,spp=spp,seed=seed))

def geometry_mask(s, res):
    yy,xx=np.mgrid[:res,:res]
    uv=mi.Point2f(mi.Float(((xx+.5)/res).ravel()),mi.Float(((yy+.5)/res).ravel()))
    ray,_=s.sensors()[0].sample_ray(0,0,uv,mi.Point2f(.5,.5))
    return np.array(s.ray_intersect(ray).is_valid()).reshape(res,res)

def main():
    make_mesh()
    print('Mesh ready',flush=True)
    scan=[]
    for a in range(0,360,30):
        sc=scene({'type':'conductor'},a,96,True)
        im=render(sc,16)
        m=geometry_mask(sc,96)
        score=float(np.mean(im[:,:,:3][m]@np.array([.2126,.7152,.0722])))
        scan.append({'angle':a,'mirror_mean_luminance':score})
    angle=max(scan,key=lambda x:x['mirror_mean_luminance'])['angle']
    (OUT/'camera_scan.json').write_text(json.dumps(scan,indent=2))
    print('Selected angle',angle,flush=True)
    sc=scene({'type':'diffuse','reflectance':.1841865185},angle)
    d=render(sc,1024)
    d[:,:,3]=geometry_mask(sc,512)
    np.save(OUT/'diffuse_with_background_linear.npy',d)
    print('Diffuse ready',flush=True)
    s=render(scene({'type':'balanced_ward_reconstructed'},angle,hide=True),1024)
    np.save(OUT/'unit_specular_linear.npy',s)
    print('Specular ready',flush=True)
    c=np.linspace(0,.1487,52)
    ps=(c+(0.5**3/2)**(1/3))**3-0.5**3/2
    # Uniform exposure across the whole series; no per-image normalization.
    stack=np.stack([d[:,:,:3]+x*s[:,:,:3] for x in ps])
    exposure=float(np.percentile(stack,97))
    frames=[]; records=[]
    for k,(ck,pk,im) in enumerate(zip(c,ps,stack),1):
        encoded=np.round(srgb(im/exposure)*255).astype(np.uint8)
        frame=Image.fromarray(encoded); frames.append(frame)
        name=f'reference_{k:02d}.png'; frame.save(OUT/name)
        records.append({'index':k,'pellacini_c':ck,'ward_specularity':pk,'png':name,
          'clipped_channel_fraction':float(np.mean(im/exposure>1)),
          'object_mean_linear_luminance':float(np.mean((im@np.array([.2126,.7152,.0722]))[d[:,:,3]>.99]))})
    with (OUT/'reference_levels.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
    preview=Image.new('RGB',(1280,302),'white'); draw=ImageDraw.Draw(preview)
    for j,k in enumerate([0,13,26,38,51]):
        preview.paste(frames[k].resize((256,256)),(j*256,0))
        draw.text((j*256+12,265),f'k={k+1:02d}   c={c[k]:.5f}',fill='black')
    preview.save(OUT/'gloss_preview.png')
    sheet=Image.new('RGB',(8*144,7*170),'white');draw=ImageDraw.Draw(sheet)
    for j,im in enumerate(frames):
        x=j%8*144;y=j//8*170;sheet.paste(im.resize((144,144)),(x,y));draw.text((x+5,y+147),f'{j+1:02d} c={c[j]:.5f}',fill='black')
    sheet.save(OUT/'all_52_levels.png')
    metadata={'renderer':mi.MI_VERSION,'variant':mi.variant(),'material':'balanced Ward, alpha=.05',
      'source_hdr':'https://vgl.ict.usc.edu/Data/HighResProbes/probes/uffizi-large_HDR.rar',
      'hdr_sha256':hashlib.sha256((ASSETS/'uffizi-large.hdr').read_bytes()).hexdigest(),
      'geometry':'procedural replacement, seed 20260927, not author original',
      'diffuse_reflectance':.1841865185,'color_assumption':'neutral L*=50 under D65 (chosen, not recovered)',
      'camera_angle':angle,'camera_distance':4.0,'fov':45,'resolution':512,'spp_per_basis':1024,
      'exposure_97_percentile_all_levels':exposure,'transport':'direct with visibility, max_depth=2; no interreflection',
      'spectral_limitation':'RGB preview, not original 31-band rendering',
      'formula_source':'gloss_tinynetworks-main/process_onlinedata.m',
      'ward_source':'https://raw.githubusercontent.com/mitsuba-renderer/mitsuba/master/src/bsdfs/ward.cpp',
      'count':len(frames),'finite':bool(np.isfinite(stack).all()),'ps_min':float(ps.min()),'ps_max':float(ps.max()),
      'linear_monotonic':bool(np.all(np.diff(stack,axis=0)>=-1e-6))}
    (OUT/'render_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(metadata,indent=2),flush=True)

if __name__=='__main__':main()
