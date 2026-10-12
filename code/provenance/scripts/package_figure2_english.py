from pathlib import Path
import json,hashlib,zipfile,shutil,xml.etree.ElementTree as ET
import pandas as pd,numpy as np
B=Path(__file__).resolve().parents[1];M=B/'mllm_material';O=M/'figure2_english_v1'
caption='''Figure 2 | Frozen vision-language models recover human gloss ordering beyond nominal physical gloss.

(a-c) Density plots relate physical gloss to human mean matches (a) and human matches to InternVL (b) and Qwen3 (c) reference matches. The common cohort comprises 3,849 images after excluding 39 original matching-development stimuli. Model matches are means across the two reference-search starts. All panels share a logarithmic image-count color scale. Dotted lines mark numerical equality; no calibration is fitted to align the plotted scales. The reference sequence used for models is reconstructed. Upper-bound model responses are retained.

(d) Spearman associations with human judgments for single-image ratings and reference matching, with physical c as a benchmark. Points and 95% intervals are taken from the archived analysis using 2,000 crossed shape-by-illumination bootstrap draws. Correlation with human ratings is not interchangeable with numerical-scale agreement. Intervals are conditional on the sampled stimulus conditions and saved model responses, not human-participant sampling.

(e) Empirical cumulative distributions of the model-human Spearman correlations calculated separately within each of the 50 exact physical-c levels. Every level is retained and contributes one value; the median correlations are 0.496 for InternVL and 0.619 for Qwen3. Both models have positive correlations at every level. These are descriptive distributions across physical levels, not 50 independent significance tests. Exact per-level image counts and correlations are provided in the source data.

(f) Agreement of model pairwise choices with the ordering derived from human mean matches, for 1,000 random pairs and 1,000 predefined human-physical conflict pairs. Both display orders are retained; ties receive half credit. Human pair orderings are derived from the published continuous judgments, not separately measured binary human responses. The dotted reference is 50%. Intervals are archived 95% image-node bootstrap intervals (2,000 draws), retaining both display orders together. Random/conflict agreement is 71.9%/68.8% for InternVL and 70.6%/75.25% for Qwen3.

(g) Stimulus examples organized by human-match empirical tertiles (rows) and equal-width physical-c intervals (columns). Rows run from high to low human matching values; columns run from low to high physical values. Counts are all cohort images in the cell. Two illustrative images nearest the cell center in human/physical rank coordinates are shown, preferring distinct shapes, without selection on model agreement. I and Q indicate median InternVL and Qwen3 matching percentile ranks, computed relative to the whole common cohort, over all images in each cell. Thus cell summaries and illustrations are distinct: the displayed photographs do not alone determine the reported medians. Tied human values are retained in their quantile-defined group, so row counts need not be exactly equal.

(h) Pearson correlation between model and human residuals after subtracting each variable's mean within its exact physical-c level. Associations remain positive: InternVL r = 0.522 (95% CI 0.444-0.584), Qwen3 r = 0.613 (0.524-0.683). The new post-hoc analysis uses 2,000 crossed shape-by-illumination bootstrap draws (seed 20261004); weighted physical-level means are refitted in every draw. This evaluates within-physical-level covariation and does not by itself identify its image-level cause or adjust for color/shape/illumination effects.

(i) An illustrative conflict pair: object A has higher physical c, whereas humans assign a higher gloss match to object B. Both models choose B in both presentation orders. The example is the first archived conflict pair, by pair ID, satisfying agreement in both models/orders and membership of both images in the common cohort. It is explicitly a selected positive illustration; the full conflict-pair evidence is in panel f. Exact values, image IDs and raw-derived pair-level scores are supplied.

Teal, InternVL3.5-4B; orange, Qwen3-VL-4B-Instruct; gray, the physical benchmark where applicable. Behavioral outputs were collected with frozen model weights. This figure combines previously archived a-d/f statistics with newly documented descriptive e/g and conditional h analyses; no new model responses were collected. The original common cohort retains prior Qwen pilot images and should not be described as a wholly new confirmatory test set. The source data and analysis plan record all image/pair selections. Source stimuli and human judgments originate from the Morimoto et al. dataset; model responses and reconstructed-reference measurements are from the current study.
'''
(O/'Figure2_caption_EN.txt').write_text(caption,encoding='utf-8')
(O/'README_EN.txt').write_text('''ENGLISH FIGURE 2 | MULTI-PANEL RESULT FIGURE

Main editable file: Figure2_HumanLike_Gloss_EN.svg
Vector print file: Figure2_HumanLike_Gloss_EN.pdf
Raster preview: Figure2_HumanLike_Gloss_EN.png (450 dpi)
Canvas: 180 x 218 mm; Arial. All figure text is English. Text, axes, density hexagons, points, curves and interval bars are editable vectors. The 20 displayed stimulus images are embedded; the SVG can be moved independently.

The figure builds one result across nine panels: a physical/human benchmark, two model/human density plots, two-task concordance, within-physical-level distributions, an independently elicited pairwise test, a stimulus gallery, pooled conditional associations and an illustrative conflict pair. It does not import RSA, internal interventions or condition-effect results intended for subsequent figures.

To redraw: install numpy, pandas, matplotlib and pillow; run python draw_figure2_english.py in this directory. The data/ and images/ directories contain the complete inputs for the figure. No model calls are required. Figure2_caption_EN.txt provides the detailed caption, analysis units, example selection and intervals. analysis_plan.json was saved before the new within-level calculation, as a post-hoc plan, not a preregistration.

The data directory includes all 3,849 common-cohort image records, pair records, all 50 within-physical-level correlations per model, new residual-association bootstrap draws and intervals, archived agreement intervals and paired contrasts, gallery cell counts/medians/selected IDs, and the selected conflict pair. Gallery selection does not depend on model responses; panel i deliberately illustrates a positive conflict example, while panel f retains all pairs.

provenance_scripts/ retains original project analysis code, which expects the complete benchmark folder. validate_source_data.py independently checks the portable source tables and residual-association point estimates. Original input hashes are in source_manifest.json; package_manifest.json hashes the actual included files. Source stimuli and human judgments remain attributable to Morimoto et al.; consult the original dataset terms for publication reuse.
''',encoding='utf-8')
P=O/'provenance_scripts';P.mkdir(exist_ok=True)
for name in ['prepare_figure2_english.py','revise_nmi_stats.py']:
    shutil.copy2(B/'scripts'/name,P/name)
validation='''from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr
T=Path(__file__).resolve().parent/'data'
d=pd.read_csv(T/'image_level_data.csv');assert len(d)==3849 and d.image_id.is_unique
stats=pd.read_csv(T/'v8_agreement_ci.csv').set_index('measure')
for name,col in [('InternVL_matching','internvl_c'),('Qwen3_matching','model_c'),('InternVL_single','intern_single'),('Qwen3_single','single_score'),('physical','physical_c')]:
    assert np.isclose(spearmanr(d.human_c,d[col]).statistic,stats.loc[name,'estimate'])
res=pd.read_csv(T/'within_physical_association.csv').set_index('model')
for name,col in [('InternVL','internvl_c'),('Qwen3','model_c')]:
    z=d[['human_c',col]]-d.groupby('physical_c')[['human_c',col]].transform('mean')
    assert np.isclose(z.corr().iloc[0,1],res.loc[name,'estimate'])
within=pd.read_csv(T/'within_physical_levels.csv');assert len(within)==100 and within.rho.gt(0).all()
print('All source-data checks passed.')
'''
(O/'validate_source_data.py').write_text(validation,encoding='utf-8')
ns={'s':'http://www.w3.org/2000/svg'};root=ET.parse(O/'Figure2_HumanLike_Gloss_EN.svg').getroot();texts=root.findall('.//s:text',ns);imgs=root.findall('.//s:image',ns)
assert not any('\u4e00'<=c<='\u9fff' for t in texts for c in ''.join(t.itertext()))
assert len(imgs)>=20 and all('base64,' in q.attrib.get('{http://www.w3.org/1999/xlink}href','') for q in imgs)
(O/'figure_validation.json').write_text(json.dumps({'live_text_elements':len(texts),'embedded_images':len(imgs),'all_text_english':True,'n_images':3849,'n_panels':9},indent=2),encoding='utf-8')
files=[p for p in O.rglob('*') if p.is_file() and p.name!='package_manifest.json']
(O/'package_manifest.json').write_text(json.dumps([{'path':p.relative_to(O).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files],indent=2),encoding='utf-8')
with zipfile.ZipFile(M/'Figure2_English_Editable_Package.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in O.rglob('*'):
        if p.is_file():z.write(p,Path('Figure2_English')/p.relative_to(O))
print('Figure 2 package ready; text elements:',len(texts),'images:',len(imgs))
