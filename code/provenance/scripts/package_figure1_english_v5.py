from pathlib import Path
import shutil,json
import numpy as np,pandas as pd
B=Path(__file__).resolve().parents[1];M=B/'mllm_material';O=M/'figure1_english_v5'
P=O/'provenance_scripts';P.mkdir(exist_ok=True)
for name in ['mechanism_backend_v1.py','run_mechanism_v1.py','analyze_mechanism_v1.py']:
    shutil.copy2(B/'scripts'/name,P/name)
z=np.load(O/'g_internal_feature_examples.npz');checks=[]
for model in ['InternVL','Qwen3']:
    tok=z[model+'_bridge_tokens'];mean=z[model+'_bridge'];assert tok.shape==(256,2560)
    # Stored tokens were compressed to float16, whereas the pooled vector was float32.
    delta=float(np.abs(tok.astype('float32').mean(0)-mean).max())
    checks.append({'model':model,'vision_values':len(z[model+'_vision_24']),'bridge_values':len(mean),'language_values':len(z[model+'_language_36']),'bridge_grid':[16,16],'bridge_tokens_shape':list(tok.shape),'pooled_float32_vs_float16_token_mean_max_difference':delta})
p=pd.read_csv(O/'g_readout_examples.csv');assert len(p)==2 and p.truth.nunique()==1
(O/'g_example_validation.json').write_text(json.dumps({'feature_checks':checks,'predictions':p.to_dict('records')},indent=2),encoding='utf-8')
(O/'PANEL_G_METHODS_EN.txt').write_text('''PANEL G: FROZEN INTERNAL STATES AND HUMAN-JUDGMENT READOUT

Editable standalone panel: Figure1_g_Internal_Readout_EN.svg.

Forward computation: a test image and the original single-image gloss prompt are processed with frozen weights. Selected visual encoder blocks yield visual-token features; the bridge converts visual features into language-model-width image tokens. InternVL uses mlp1 after its native visual preprocessing; Qwen3 uses visual.merger and also injects DeepStack features. Language blocks process the visual tokens jointly with text. States are recorded before the first answer token is generated.

Extraction: vision blocks 6,12,18,24 are mean-pooled over tokens (1,024 coordinates); the bridge is mean-pooled (2,560 coordinates); language blocks 9,18,27,36 contribute the final prompt-token vector (2,560 coordinates). The depicted single-image bridge arrays contain 256 spatial tokens, each of width 2,560, arranged as a 16 x 16 grid. This is the observed native grid for the displayed archived example, not a universal token count for all image settings. Qwen3 DeepStack mean vectors are separately retained in the original archive and are not shown in the nine-stage mini-curve.

Examples: morimoto_0095 is the same input for both models. Each colored vector strip displays its first 32 coordinates, scaled by that displayed strip's maximum absolute value; positive/negative values use red/blue. These are vector components, not spatial activation maps. Existing records do not retain full spatial arrays at every vision or language layer; no such maps have been reconstructed or fabricated. Full spatial arrays are available at the bridge.

Candidate map: fit the multivariate slope of the 52 reference-image bridge means against reference c and normalize it to a direction d. Project each bridge token h(p) onto d. The overlaid spatial responses use separate model-specific color scales. This direction uses reference c, not human test-image ratings; it is a gloss-related candidate direction, not a learned ridge readout coefficient, attention map, filter kernel or proof of a selective gloss neuron. The functional intervention using this direction remains in panel h.

Readout: for each layer separately, use the archived label-free seeded Gaussian projection to 128 dimensions, then training-only feature standardization and ridge regression with human matching judgments as targets. Ridge strength is chosen from 1,10,100 on a fixed group-disjoint inner validation subset. Outer 12-fold splits hold out shape or illumination groups. The MLLM parameters are unchanged. This is a supervised analysis of fixed features; it is distinct from the original free-generation behavioral responses.

The numerical example shows saved bridge-readout predictions for morimoto_0095 with its shape group held out. Human target: 0.0755648693; InternVL prediction: 0.0715023949; Qwen3 prediction: 0.0575502667. The small stage curve plots archived human-target held-out-shape R2 over 3,888 images. Vision/bridge mean pooling differs from language final-token extraction, so stage differences should be described as readout performance under these extraction choices.

Rebuild: the portable plotting script reads g_internal_feature_examples.npz and g_readout_examples.csv together with bundled assets. g_method_sources.json identifies the original source-file hashes. Original extraction/readout scripts are retained under provenance_scripts; rerunning the original experiments requires the complete project and model environment.
''',encoding='utf-8')
s=(B/'scripts/package_figure1_english_v4.py').read_text('utf-8')
s=s.replace('figure1_english_v4','figure1_english_v5').replace('Detailed_v4','Detailed_v5').replace('180 x 768','180 x 904')
exec(compile(s,str(__file__),'exec'))
