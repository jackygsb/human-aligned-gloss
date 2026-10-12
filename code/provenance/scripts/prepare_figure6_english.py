from pathlib import Path
import numpy as np,pandas as pd,json,hashlib,shutil
B=Path(__file__).resolve().parents[1];M=B/'mllm_material';S=M/'mechanisms_v1';O=M/'figure6_english_v1'
for sub in ['data','images','provenance_scripts']:(O/sub).mkdir(parents=True,exist_ok=True)
sources=[]
def cp(p,sub='data',name=None):sources.append(p);shutil.copy2(p,O/sub/(name or p.name))
for name in ['v8_intervention_tests.csv','v8_intervention_dose_ci.csv','v8_intervention_shape_values.csv']:cp(M/'nmi_revision_v8/source/tables'/name)
for name in ['interventions.csv','matching_interventions.csv','shape_cluster_intervention_contrasts.csv']:cp(S/'tables'/name)
for name in ['stimuli.csv','Methods.md','plan.json']:cp(S/name)
for name in ['h_direction_examples.npz','h_method_provenance.json','h_example_selection.csv']:cp(M/'figure1_english_v6'/name)
for name in ['analyze_mechanism_v1.py','mechanism_backend_v1.py','run_mechanism_v1.py']:cp(S/'code'/name,'provenance_scripts')
d=pd.read_csv(S/'stimuli.csv').set_index('image_id');iv=pd.read_csv(S/'tables/interventions.csv');iv=iv.merge(d[['shape_id','illumination_id','viewpoint_id']],left_on='image_id',right_index=True,validate='many_to_one')
assert iv.image_id.nunique()==72 and iv.shape_id.nunique()==36 and iv.illumination_id.nunique()==1
iv.groupby(['model','direction','dose','shape_id']).delta_p.mean().reset_index().to_csv(O/'data/shape_dose_responses.csv',index=False)
cp(B/d.loc['morimoto_0064','image_512_path'],'images','stimulus_0064.png')
for k in [1,14,27,40,52]:cp(M/f'reference_reconstruction_v3/output/reference_{k:02d}.png','images')
raw=np.load(O/'data/h_direction_examples.npz');reference=[]
for model in ['InternVL','Qwen3']:
 means=raw[model+'_reference_means'];direction=raw[model+'_candidate'];v=means@direction;x=raw[model+'_reference_c'];v=(v-v.min())/(v.max()-v.min());reference.extend([dict(model=model,reference_index=i+1,c=float(c),projection_scaled=float(q)) for i,(c,q) in enumerate(zip(x,v))])
 np.save(O/f'data/{model}_spatial_projection.npy',(raw[model+'_morimoto_0064'].astype(float)@direction).reshape(16,16))
pd.DataFrame(reference).to_csv(O/'data/reference_projections.csv',index=False)
(O/'source_hashes.json').write_text(json.dumps({str(p.relative_to(B)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2));print('Figure 6 sources prepared.')
