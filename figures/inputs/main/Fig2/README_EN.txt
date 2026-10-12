ENGLISH FIGURE 2 | MULTI-PANEL RESULT FIGURE

Main editable file: Figure2_HumanLike_Gloss_EN.svg
Vector print file: Figure2_HumanLike_Gloss_EN.pdf
Raster preview: Figure2_HumanLike_Gloss_EN.png (450 dpi)
Canvas: 180 x 218 mm; Arial. All figure text is English. Text, axes, density hexagons, points, curves and interval bars are editable vectors. The 20 displayed stimulus images are embedded; the SVG can be moved independently.

The figure builds one result across nine panels: a physical/human benchmark, two model/human density plots, two-task concordance, within-physical-level distributions, an independently elicited pairwise test, a stimulus gallery, pooled conditional associations and an illustrative conflict pair. It does not import RSA, internal interventions or condition-effect results intended for subsequent figures.

To redraw: install numpy, pandas, matplotlib and pillow; run python draw_figure2_english.py in this directory. The data/ and images/ directories contain the complete inputs for the figure. No model calls are required. Figure2_caption_EN.txt provides the detailed caption, analysis units, example selection and intervals. analysis_plan.json was saved before the new within-level calculation, as a post-hoc plan, not a preregistration.

The data directory includes all 3,849 common-cohort image records, pair records, all 50 within-physical-level correlations per model, new residual-association bootstrap draws and intervals, archived agreement intervals and paired contrasts, gallery cell counts/medians/selected IDs, and the selected conflict pair. Gallery selection does not depend on model responses; panel i deliberately illustrates a positive conflict example, while panel f retains all pairs.

provenance_scripts/ retains original project analysis code, which expects the complete benchmark folder. validate_source_data.py independently checks the portable source tables and residual-association point estimates. Original input hashes are in source_manifest.json; package_manifest.json hashes the actual included files. Source stimuli and human judgments remain attributable to Morimoto et al.; consult the original dataset terms for publication reuse.
