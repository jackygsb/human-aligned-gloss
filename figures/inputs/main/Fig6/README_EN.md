# Figure 6 — internal direction and functional intervention

## Story and design
This figure connects a reference-defined internal direction to spatial response and controlled changes in gloss choice. Panels a/b define and check the direction; c illustrates token projections; d specifies the actual intervention; e/f show dose responses; g/h compare equal-norm controls; i shows all tested shapes.

Wine consistently identifies the candidate direction across models; gold and blue-grey variants identify controls. This palette encodes intervention type rather than model identity. Models are separated into paired panels. The figure is fully English and preserves editable text and vector plots; photographs and spatial maps are embedded raster assets.

## Delivered outputs
- `Figure6_Internal_Direction_Intervention_EN.svg`: editable publication plate.
- `.pdf`: vector figure at 180 × 264 mm.
- `.png`: 450 dpi export; `preview.png`: PDF-rendered inspection image.
- `Figure6_caption_EN.md`: English caption specifying inference, selection and units.
- `data/`: complete assay table, all shape-level effects and contrasts, archived confidence intervals and tests, reference arrays and spatial projection arrays.
- `images/`: five reconstructed reference levels and the unmodified test image used in c.
- `provenance_scripts/`: original extraction/intervention analysis code.
- `source_hashes.json`, `package_hashes.json`: source and delivery provenance.

## Reproduction
Run `python draw_figure6_english.py` inside this folder with numpy, pandas, matplotlib and Pillow. Run `python validate_source_data.py` to check all displayed means and contrasts against the underlying assay, verify image/shape/illumination coverage, and recompute the reference/spatial projections. `prepare_figure6_english.py` documents extraction from the existing project archive. No new model calls or new significance tests were used in this redraw. Original inference scripts retain their project-path dependencies.

## Data distinctions
- `v8_intervention_dose_ci.csv`: archived shape-cluster uncertainty for the dose curves; original image-bootstrap estimates are not substituted.
- `v8_intervention_shape_values.csv`, `v8_intervention_tests.csv`: all 36 paired shape contrasts per control, with archived corrected inference.
- `shape_dose_responses.csv`: means from the raw assay for every shape, direction and nonzero dose; zero is added explicitly as the baseline in i.
- `interventions.csv`: all raw summary rows, including normalized scores, candidate probability mass, unrestricted top tokens and presentation orders.
- `h_direction_examples.npz`: original direction vectors, reference states and illustrated bridge tokens.
- `h_method_provenance.json`, `h_example_selection.csv`: previously documented, selected spatial illustration; it is not an unbiased sample of highlight alignment.
- `matching_interventions.csv`: the distinct 12-image constrained-matching pilot, retained for provenance but not displayed as the primary assay.

The normalized A/B/SAME score is a mechanistic assay measure, not a free-generation answer rate. Reference images are reconstructions, not asserted to be the authors' original PNGs. Perturbations are applied to all test tokens and can affect entangled properties. The scope is 36 shapes under one illumination, with two viewpoints and counterbalanced image order; the final caption states that scope explicitly.
