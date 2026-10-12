# Code

Run the verified CPU figure pipeline from the release root:

```text
python -m pip install -r environment/requirements-cpu.txt
python code/reproduce_figures.py
python code/validate_figure_outputs.py --render
```

The optional final rendering check needs Poppler's `pdftoppm` on PATH. It checks
all 22 final PDF assets against the canonical text/dimension manifest and makes contact sheets.
Fig. S12 uses two assets, hence 22 assets for 6 main and 15 supplementary figures.
The long workflow master is a further intermediate PDF.

- `figures/`: portable drawing and table-generation code.
- `configs/`: exact prompts, plans, historical environment and layer records.
- `provenance/scripts/`: unchanged historical first-party project scripts.
- `provenance/experiments/`: source snapshots stored with particular experiments,
  including reference renderers. These may differ from later project scripts.
- `run_archived.py`: executes a historical script against a separate working
  copy of the original relative-path data tree.

The canonical final figure captions are in
`docs/figure_legends_submission_v11.tex`. Older captions inside
`figures/inputs/` are historical provenance and may contain wording superseded
by the submission audit; they are not the submission legends.

See `docs/figure_reproduction.md` for the per-figure map, validated scope, all
upstream analysis entry points, and acquisition requirements. No commands in
the verified CPU figure pipeline make network requests or run model inference.
The original experiments' model weights and private service credentials are
intentionally absent. Archived analysis code is supplied, but no claim is made
that all statistical fits and every original inference were rerun during
packaging.
