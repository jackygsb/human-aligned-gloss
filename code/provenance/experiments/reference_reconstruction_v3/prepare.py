from pathlib import Path
import shutil
p=Path(__file__).resolve().parent
(p/'assets').mkdir(exist_ok=True)
shutil.copy2(p.parent/'reference_reconstruction_v2/assets/doge2.hdr',p/'assets/doge2.hdr')
s=(p.parent/'reference_reconstruction_v2/render_reference.py').read_text(encoding='utf-8')
s=s.replace('count=110','count=140').replace('rng.uniform(.09,.135,count)','rng.uniform(.075,.115,count)').replace('rng.uniform(.055,.105,count)','rng.uniform(.065,.125,count)')
s=s.replace('3.0*np.sin(t)','3.3*np.sin(t)').replace('3.0*np.cos(t)','3.3*np.cos(t)').replace("'camera_distance':3.0","'camera_distance':3.3")
s=s.replace("'reflectance':.30","'reflectance':.24").replace("'diffuse_reflectance':.30","'diffuse_reflectance':.24").replace('reflectance .30','reflectance .24').replace('v2 smoother procedural replacement, 110 bumps','v3 more detailed procedural replacement, 140 bumps')
(p/'render_reference.py').write_text(s,encoding='utf-8')
