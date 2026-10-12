# Figure 5 — frozen internal-state readout

## Central result
Frozen internal states support out-of-group prediction of human gloss judgments, including information beyond the physical/color baseline. This is supervised readout of frozen representations, not unsupervised prediction or fine-tuning of the multimodal models.

The figure uses purple (InternVL), jade (Qwen3), and gold (physical/color baseline), with architectural stage backgrounds. The visual style deliberately shifts from the preceding behavioral figures while retaining editable English typography and a white publication canvas.

## Files and reproduction
- `Figure5_Frozen_Internal_Readout_EN.svg`: editable vector text, data and axes.
- `.pdf`: vector publication figure, 180 × 238 mm.
- `.png`: 450 dpi export; `preview.png`: verified PDF rendering.
- `Figure5_caption_EN.md`: detailed English caption.
- `data/layer_readouts.csv`: all 168 model/layer/split/target summaries, including Qwen DeepStack.
- `data/readout_predictions.csv.gz`: all 653,184 archived per-image readout predictions.
- `data/v8_readout_ci.csv`: archived human and combined-readout confidence intervals.
- `data/final_layer_increment.csv`, `increment_bootstrap.npz`, `nuisance_baseline_predictions.csv`, `increment_analysis.json`: new paired gain summary, all draws, reconstructed baselines and exact specification.
- `data/stimuli.csv`, `prompts.json`, layer specifications and original methods/plan: experiment provenance.
- `provenance_scripts/`: original extraction/readout/CI analysis scripts. These retain project-relative dependencies; they are provenance, not standalone GPU inference bundles.

Run `python draw_figure5_english.py` within this folder to recreate the figure from bundled tables (numpy, pandas, matplotlib). Run `python validate_source_data.py` to verify all score summaries, baseline and gain estimates, intervals and English SVG. `prepare_figure5_english.py` records extraction and the new paired bootstrap, using the original project layout. Model weights and full hidden-state tensors remain in the original project archive and are not duplicated in this figure package.

## Statistical and visual decisions
- The figure retains every common sampled layer, and includes additional Qwen DeepStack states in f.
- The final-block display in g/h is defined by architecture, not by selecting the maximum test score.
- Human-beyond-physical/color scores in the original table evaluate the nuisance prediction plus predicted residual against human judgments. Panels d/e label this combined predictor explicitly, and h reports its paired gain over the nuisance-only baseline.
- Existing readout confidence intervals are reused. Only the final-layer paired gain intervals are newly calculated, from saved predictions and reconstructed nuisance-only predictions; no new model responses are collected.
- No layerwise significance stars are added. Intervals are pointwise and conditional on fixed fitted predictions, with no bootstrap refitting.
- The heatmap compares distinct supervised prediction targets descriptively. It does not claim a formal neural selectivity test.
- Spatial maps and intervention results are not used here: they answer a different question from decodability and belong in the subsequent mechanism figure.

Original sources and delivered files are identified by SHA-256 hashes in `source_hashes.json` and `package_hashes.json`.
