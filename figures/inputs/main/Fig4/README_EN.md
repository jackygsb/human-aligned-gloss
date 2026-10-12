# Figure 4: cross-task transfer of gloss judgments

## Evidence organization
The central finding is that each model's gloss judgments transfer across three elicitation tasks. The figure moves from prediction logic (a), to complete image-level out-of-fold results (b,c), to two grouping-based generalization tests (d,e), to choice agreement and continuous choice curves (f–h), and finally selected concrete examples (i).

All model responses and inferential intervals are from archived analyses. The new descriptive bin summaries use ten fixed equal-width bins and are supplied as source data. No new model experiment or human-response fitting was performed.

## Files
- `Figure4_Cross_Task_Transfer_EN.svg`: editable English vector text, axes and data; embedded stimulus photographs.
- `Figure4_Cross_Task_Transfer_EN.pdf`: vector figure, 180 × 248 mm.
- `Figure4_Cross_Task_Transfer_EN.png`: 450 dpi export.
- `preview.png`: rendering from the final PDF, visually checked.
- `Figure4_caption_EN.md`: full submission-style English caption with statistical details.
- `data/`: complete image-level predictions, pair-level outputs, existing confidence intervals, descriptive bin means and selected-example manifest.
- `provenance_scripts/`: original analyses for the two models and confidence intervals.
- `source_hashes.json`: original input provenance; `package_hashes.json`: delivered-file hashes.

## Reproduce and verify
Run `python draw_figure4_english.py` within this directory. Requires numpy, pandas, matplotlib and Pillow. Arial is used if installed. Run `python validate_source_data.py` to verify sample sizes, all eight R² estimates, all pair-agreement estimates, and illustration selection. `prepare_figure4_english.py` records extraction from the original project tree; the portable drawing script reads only bundled data.

## Interpretation and scope
Continuous transfer uses 3,849 images in the original matching-heldout cohort. The existing Qwen analysis uses this same cohort for cross-task calibration; it is not restricted to the separate new-validation flag. Isotonic mappings use only model outputs and held-out condition groups. Multimodal model weights are frozen. The confidence intervals resample fixed saved predictions rather than refitting each mapping.

Pair agreement uses all 2,000 pairs in two display orders, without learned directional calibration. Existing logistic out-of-fold probabilities are preserved in the source pair tables but are not plotted or mislabeled as these directional agreement results. Figure 4's pair evidence is model cross-task consistency, whereas Figure 2 compared model choices with human-derived ordering.

Examples in panel i are deliberately selected to illustrate concordance and are labeled as such. No pair is removed from the quantitative analyses. Full counts for the choice-curve bins are available in `data/choice_curves.csv`. Human data help define the conflict-pair stratum, but are not used in the model cross-task predictors. No added significance stars or uncomputed P values are shown; the available appropriate resampling intervals accompany the effect estimates.
