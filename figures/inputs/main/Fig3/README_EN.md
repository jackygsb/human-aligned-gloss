# Figure 3 — shared condition effects and behavioral geometry

## Main discovery and reading order
Humans, InternVL and Qwen3 share shape- and illumination-dependent gloss judgments after adjustment for physical gloss and color. The figure proceeds from complete condition effects (a–c), to full behavioral geometry (d–g), to concrete stimulus examples (h). It complements Figure 2's stimulus-level human agreement rather than repeating it.

This is a reorganization of existing results; no new model responses were collected. All 3,888 images are included, including original development/pilot images. This is distinct from Figure 2's 3,849-image common evaluation subset. No SPoSE embedding or neural representation is plotted here.

## Editable outputs
- `Figure3_Shared_Condition_Geometry_EN.svg`: English text remains editable; data points, axes and labels are vectors. Stimulus photographs are embedded bitmap images.
- `.pdf`: vector publication figure, 180 × 237 mm.
- `.png`: 450 dpi raster export.
- `preview.png`: PDF-rendered preview used for visual inspection.
- `Figure3_caption_EN.md`: full English caption with methods and inference units.

## Reproduction
Run `python draw_figure3_english.py` from this folder. Requires Python, numpy, pandas, scipy, matplotlib and Pillow; Arial is used when available. Run `python validate_source_data.py` to verify the plotted coefficients and full distance matrices. The portable drawing script reads only bundled source tables and images.

`prepare_figure3_english.py` records the original project extraction and deterministic illustration selection. `provenance_scripts/` contains original statistical-analysis scripts; these use the original project's paths and full source files. `source_hashes.json` identifies original inputs. `package_hashes.json` records every delivered file except itself.

## Source data and inferential distinctions
- `condition_effects.csv`: all 72 condition effects for all three observers/models.
- `v8_condition_tests.csv`: four effect correlations, confidence intervals and corrected permutation tests.
- Six `physical_color_adjusted_*_rdm.csv` files: complete 36 × 36 matrices, **without CSV headers**, original condition-ID order.
- `rsa_results.csv`: the complete original 18-comparison family, including raw and interaction-only results. The figure displays the six physical/color-adjusted full-profile comparisons; these retain additive effects and should not be described as isolated interactions.
- `cross_view_rsa.csv`: original cross-view diagnostic results, retained for context; not plotted.
- `joint_model_predictions.csv`: complete image-level responses and stimulus metadata.
- `gallery.csv`: selected examples, physical and matched values. Selection uses human effects and physical values, never model agreement.

The effects in a–c use joint additive regression on standardized responses. The RDMs in d–g use cross-view nuisance residuals on the original response scale. These complementary analyses are not the same residualization procedure. Display normalization of RDMs does not alter RSA. The 630 distance pairs are dependent; inference resamples/permutates the 36 condition nodes.
