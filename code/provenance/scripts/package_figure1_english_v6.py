from pathlib import Path
import json,shutil,hashlib
import numpy as np,pandas as pd
B=Path(__file__).resolve().parents[1];M=B/'mllm_material';O=M/'figure1_english_v6'
z=np.load(O/'h_direction_examples.npz');checks=[]
for model in ['InternVL','Qwen3']:
    cc=np.linspace(0,1,52);r=z[model+'_reference_means'];slope=(cc-cc.mean())@(r-r.mean(0))/np.sum((cc-cc.mean())**2)
    expected=z[model+'_reference_scale']*z[model+'_candidate'];assert np.allclose(slope,expected,rtol=1e-5,atol=1e-5)
    checks.append({'model':model,'unit_direction_norm':float(np.linalg.norm(z[model+'_candidate'])),'reference_scale':float(z[model+'_reference_scale']),'direction_reconstruction_max_error':float(np.abs(slope-expected).max())})
e=pd.read_csv(O/'h_dose_effects.csv');vals=[]
for model in ['InternVL','Qwen3']:
    d=e[(e.model==model)&(e.direction=='candidate')].set_index('dose');vals.append({'model':model,'negative_pp':float(d.loc[-1,'mean_delta_p']*100),'positive_pp':float(d.loc[1,'mean_delta_p']*100),'swing_pp':float((d.loc[1,'mean_delta_p']-d.loc[-1,'mean_delta_p'])*100),'positive_ci_pp':[float(d.loc[1,c]*100) for c in ['ci_low','ci_high']]})
(O/'h_validation.json').write_text(json.dumps({'directions':checks,'effects':vals},indent=2),encoding='utf-8')
meta=json.loads((O/'h_method_provenance.json').read_text('utf-8'))
for model in ['InternVL','Qwen3']:
    for name in ['directions.npz','features/morimoto_0064.npz','features/morimoto_0088.npz']:
        p=M/'mechanisms_v1'/model/name;meta['sources'].append({'path':str(p.relative_to(B)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(O/'h_method_provenance.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
P=O/'provenance_scripts';P.mkdir(exist_ok=True)
shutil.copy2(B/'scripts/inspect_h_examples.py',P/'inspect_h_examples.py')
(O/'PANEL_H_METHODS_EN.txt').write_text('''PANEL H: REFERENCE-DEFINED DIRECTION AND FUNCTIONAL INTERVENTION

Standalone editable panel: Figure1_h_Direction_Intervention_EN.svg.

1. Direction construction. Pass the 52 reconstructed reference images through each frozen model under the single-image gloss prompt. Average bridge tokens for each image. Regress every coordinate against a reference level normalized to [0,1]. The vector of slopes is beta; s = ||beta|| and d = beta / s. The positive direction is defined by increasing reference gloss. No human test-image judgments define d. The plotted projection curves are rescaled separately to [0,1] for illustration.

2. Spatial projection. Map h(p) dot d at the native 16 x 16 bridge grid for test image morimoto_0064. The three columns use the identical original image and crop [160,65,300,305]. Model overlays are raw projections with independent full-image min/max limits, alpha .45 and nearest-neighbor rendering. Insets keep the full-image limits. Cyan boxes specify the shared visual crop, not a highlight mask. The visible specular streak makes this illustrative example easier to interpret; responses are not restricted to highlights. Example selection is documented in h_method_provenance.json and h_example_selection.csv. The selection is not a test of population-level highlight selectivity. No feature-response values were edited.

3. Intervention. At each TEST image bridge token, add a*s*d for a in {-1,-.5,0,.5,1}. The identical vector is added to all test tokens; this is not localized editing of the highlighted region. Pixels, text prompt, model parameters and reference tokens remain unchanged. Qwen3 intervention modifies the final merger only; DeepStack remains intact. Compare 72 test images (36 shapes, two viewpoints, illumination 1) against reference level 27 in both presentation orders.

4. Outcome. Normalize next-token logits over A, B and SAME. Map the image order to test/reference roles, then score P(test glossier) + .5 P(SAME). Subtract the baseline score for the same image/order. Report percentage-point changes, not relative percentage changes. The main curves plot empirical dose means with archived 95% shape-cluster bootstrap CIs (36 shape clusters). Gray dashed lines are equal-norm brightness perturbations; gray dotted lines are three equal-norm random directions orthogonal to d. Line segments connect measured dose means; no linear trend fit is used.

Effect size: InternVL +1 = +9.9193 pp, -1 = -9.3737 pp, swing = 19.2930 pp. Qwen3 +1 = +10.2486 pp, -1 = -9.8699 pp, swing = 20.1185 pp. Positive endpoint 95% CIs: InternVL [8.4361,11.3560] pp; Qwen3 [6.9193,13.6394] pp. These are shifts in the normalized three-answer score, not increases in behavioral accuracy or physical-gloss units. Candidate-vs-brightness bidirectional differences and the random-control differences are retained in h_control_contrasts.csv. Functional sensitivity does not establish that the direction is an exclusive gloss circuit.

The complete-matching intervention on 12 images is retained in the original archive but is not plotted in this methods panel. h_validation.json checks the direction construction and reported effect arithmetic. Raw illustration arrays, summary tables, source hashes and example selection are included in this package. No new model inference was run for the figure.
''',encoding='utf-8')
s=(B/'scripts/package_figure1_english_v5.py').read_text('utf-8')
s=s.replace('figure1_english_v5','figure1_english_v6').replace('Detailed_v5','Detailed_v6').replace('180 x 904','180 x 1106')
exec(compile(s,str(__file__),'exec'))
