from pathlib import Path
p=Path(__file__).resolve().parent
source=(p.parent/'reference_reconstruction_v1/render_reference.py').read_text(encoding='utf-8')
source=source.replace('count=125','count=110').replace('rng.uniform(.075,.12,count)','rng.uniform(.09,.135,count)').replace('rng.uniform(.08,.16,count)','rng.uniform(.055,.105,count)')
source=source.replace('uffizi-large.hdr','doge2.hdr').replace('4.0*np.sin(t)','3.0*np.sin(t)').replace('4.0*np.cos(t)','3.0*np.cos(t)')
source=source.replace("'fov':45","'fov':50").replace("'camera_distance':4.0","'camera_distance':3.0")
source=source.replace("angle=max(scan,key=lambda x:x['mirror_mean_luminance'])['angle']","angle=315  # visual registration to the published courtyard, not max-brightness selection")
source=source.replace('.1841865185','.30')
source=source.replace("'neutral L*=50 under D65 (chosen, not recovered)'","'neutral reflectance .30 (chosen for figure appearance; not recovered)' ")
source=source.replace('https://vgl.ict.usc.edu/Data/HighResProbes/probes/uffizi-large_HDR.rar','https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Environments/low_resolution_hdrs/doge2.hdr')
source=source.replace("'geometry':'procedural replacement, seed 20260927, not author original'","'geometry':'v2 smoother procedural replacement, 110 bumps, seed 20260927, not author original', 'environment_note':'Doge courtyard matches figure visually; paper text says Uffizi. Unresolved source discrepancy.', 'camera_selection':'315 degrees selected against figure background; mirror scan retained for transparency'")
(p/'render_reference.py').write_text(source,encoding='utf-8')

