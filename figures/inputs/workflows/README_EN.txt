FIGURE 1 | ENGLISH EDITABLE METHODS PLATE

Primary editable file: Figure1_Methods_EN.svg
Task-only editable extract: Figure1_bcd_Task_Detail_EN.svg
Acquisition panel: Figure1_e_Acquisition_EN.svg
Count definitions: ACQUISITION_COUNTS_EN.txt
Complete task and system instructions: prompts_verbatim.json
Verified original six-step decision log: matching_decisions_example.json
All labels are live English text. Lines, axes, markers and the 36 x 36 dissimilarity matrices are vector objects. Image stimuli and token-projection overlays are embedded raster images. SVG image assets are self-contained; moving the SVG alone will not break the images.

Canvas: 180 x 1106 mm (expanded editorial master). Font: Arial. Install Arial before editing to preserve the layout. Open the SVG with Inkscape or Adobe Illustrator; ungroup the objects when editing. Panel heading groups are named panel_a ... panel_h. PDF is vector output with embedded fonts. PNG is a 450 dpi preview. No claim of compliance with every journal-specific final production dimension is implied; the editable master can be resized during submission formatting.

Figure1_caption_EN.txt supplies the full English caption and the scope of each methodological exemplar. The displayed outputs and trajectory are archived InternVL records. Both models contribute to the actual RDMs, readout curves and intervention illustrations. Colors are consistent: teal = InternVL, orange = Qwen3; dashed gray in the intervention plot denotes brightness controls. Norm-matched random controls are described in the caption and fully shown in the existing results figure.

Rebuild: install numpy, pandas, matplotlib and pillow, then run python draw_figure1_english_v6.py from this directory. The assets folder contains the exact images, selected reference arrays, token arrays and numeric tables needed for this figure. Raw response extracts contain only the displayed example records. No model calls are needed.

Source provenance: source_manifest.json identifies original project inputs and their SHA-256 hashes. Bundled arrays have been reduced to fields used by the figure. package_manifest.json hashes the actual bundled files. Human reference data and original stimulus assets remain attributable to Morimoto et al.; reconstruction assets and derived analyses are identified separately in their paths. Consult original dataset licenses for manuscript reuse.

Display details: RDM CSVs have no header and are read as complete 36 x 36 symmetric matrices. RDM values are divided by each matrix's own mean off-diagonal distance. Spatial response overlays show the same stimulus in both models, with independent model-specific color scales; they are not attention maps. Reference projection curves are rescaled separately to [0,1] for visual comparison. No new experimental responses were generated.
