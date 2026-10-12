# Figure, table and analysis reproduction

## What the verified CPU command does

From the root of this deposit, create a Python 3.12 environment and run:

```text
python -m pip install -r environment/requirements-cpu.txt
python code/reproduce_figures.py
```

It executes the plotting calculations and assembles the figures anew from the
packaged numerical tables, activation examples, and stimulus/reference images.
Outputs go to `figures/regenerated/`, never to `figures/published/`. The report
records each executed script, exit status, duration, output SHA-256 and table
checks. Published PDFs are not copied as a substitute for re-execution.

For installations in which the scientific and PDF libraries use separate Python
environments, add `--pdf-python PATH_TO_PYTHON_WITH_PYPDF_AND_REPORTLAB`.
The recorded local verification used this option. Both environments used Python
3.12.14; package versions and script logs are preserved. No GPU or network call
is required for these figure and table commands.

The command regenerates six main figures, ten statistical supplementary figures,
the detailed workflow master, the six PDF panels that constitute supplementary
Figs. S11–S15 (S12 has two parts), and nine supplementary LaTeX tables. Select a
subset with `--group main`, `supplement`, `workflows`, or `tables`. The workflow
master is a long vector plate; its actual freshly generated PDF is cropped and
relabelled by `derive_supplement_11_15.py`. Cropping a workflow diagram is not
described as recomputing a scientific result.

## Reproduction levels

1. **CPU figure reproduction (executed and logged):** redraws every final
   evidence figure from the packaged figure inputs and creates the table TeX.
2. **Statistical reanalysis:** historical first-party analysis code and its
   archived input tables, folds, response records and arrays are preserved.
   The current packaging check does not rerun every model fit, bootstrap or
   permutation calculation. The figures' confidence intervals and inferential
   summaries are read from preserved numerical outputs. See the map below and
   `environment/seeds_and_folds.json` for the exact implementing code.
3. **New model acquisition:** requires separately downloaded, revision-pinned
   model weights, source stimuli and suitable hardware. It was not rerun while
   assembling this deposit. The saved responses are the primary records of the
   original runs. Hosted-model pilot acquisition additionally depends on a
   user's authenticated service and cannot be reproduced offline.

Thus a successful figure report proves that the deposited figure materials run;
it is not a claim that new GPU responses are bitwise deterministic or that an
independent team has rerun all experiments.

## Per-figure map

| Output | Executable code | Preserved input | Upstream analysis |
|---|---|---|---|
| Fig. 1 | `code/figures/draw_main_1.py` | Real project thumbnails embedded in the source, with source paths | Conceptual overview of the implemented protocol; no simulated outcomes |
| Fig. 2 | `draw_main_2.py` | `figures/inputs/main/Fig2/data`, `images` | `prepare_figure2_english.py`; `revise_nmi_stats.py` |
| Fig. 3 | `draw_main_3.py` | `figures/inputs/main/Fig3/data`, `images` | `analyze_behavioral_spaces_v1.py`; `analyze_shared_condition_effects.py`; `prepare_figure3_english.py` |
| Fig. 4 | `draw_main_4.py` | `figures/inputs/main/Fig4/data`, `images` | `build_paper_figures_v1.py`; `analyze_qwen3_full.py`; `prepare_figure4_english.py`; `revise_nmi_stats.py` |
| Fig. 5 | `draw_main_5.py` | `figures/inputs/main/Fig5/data` | `analyze_mechanism_v1.py`; `prepare_figure5_english.py`; `revise_nmi_readouts.py` |
| Fig. 6 | `draw_main_6.py` | `figures/inputs/main/Fig6/data`, `images` | `analyze_mechanism_v1.py`; `revise_nmi_stats.py`; `prepare_figure6_english.py` |
| Figs. S1–S10 | `draw_supplement_1_10.py` | `figures/inputs/supplement/{tables,supplement_inputs}` | Saved behavioral-space, tiny-control, cross-task and intervention analyses |
| Figs. S11–S15 | `draw_workflow_master.py` then `derive_supplement_11_15.py` | `figures/inputs/workflows/` | Exact prompts, controller traces, response examples and recorded activation examples |
| Tables S1–S9 | `build_supplement_tables.py` | `figures/inputs/supplement/tables` | Numerical summaries shown in the final supplement; S1/S2 are protocol descriptions |

The table lists short plotting filenames; all are under `code/figures/`.
Historical upstream scripts are under `code/provenance/scripts/` and retain their
original filenames. The file manifest records the hashes of the
copied source and inputs. Experimental snapshots are additionally preserved
under `code/provenance/experiments/` where available.

Final submission captions are `docs/figure_legends_submission_v11.tex`.
Historical caption files alongside the figure inputs remain provenance only;
the submission audit corrected, for example, the description of Figure 4's
identity lines and clarified that its descriptive choice curves have no
confidence bands.

## Running historical analyses safely

The following section requires the separate full data release; its archives and
extractor are not duplicated in this lightweight code repository. The existing
[Zenodo draft 23309081](https://zenodo.org/deposit/23309081) is still undergoing
file uploads and verification. Its DOI `10.5281/zenodo.23309081` is reserved
only, not registered or published. It provides no public download or released
citation at this stage. Add the verified versioned data link after publication.


Historical scripts use a project-root directory containing `mllm_material/`,
`annotations/`, `protocols/`, and other original folders. The data snapshot retains
this original-relative layout under `data/derived/project_tree/`. Create a
separate working copy first; do not edit the deposit's original record. On a
machine with sufficient disk space:

```text
python data/extract_data_archives.py --all
python -c "import shutil; shutil.copytree('data/derived/project_tree','work/analysis_project')"
python code/run_archived.py --project-root work/analysis_project analyze_qwen3_full.py
```

The extraction command restores the lossless Qwen3 raw-response archive and
both models' saved-feature archives to their original relative paths. It checks
the recorded file hashes. These large archives are unnecessary for the verified
CPU figure pipeline, whose selected inputs are already expanded in `figures/inputs/`.

`run_archived.py` stages the preserved first-party scripts in the working copy,
which supplies the relative paths expected by the original scripts. It does not
rewrite analysis calculations. Individual historical scripts may also require
particular preceding outputs, original image files or the recorded GPU stack;
read their imports, CLI arguments and input paths before running. Display/report
and packaging scripts may retain workstation-specific paths and are archived
for provenance, not advertised as a portable all-in-one pipeline.

Primary statistical entry points are `analyze_matching_full.py`,
`analyze_qwen3_full.py`, `analyze_shared_condition_effects.py`,
`analyze_behavioral_spaces_v1.py`, `analyze_mechanism_v1.py`,
`analyze_incremental_internvl.py`, `revise_nmi_stats.py`,
`revise_nmi_readouts.py`, and the `prepare_figure*_english.py` scripts.
Run only in a working copy because these scripts intentionally write derived
tables, fits and reports.

## Acquisition and reference rendering

The primary acquisition implementations are:

- Single-image ratings and pairwise choices: `run_mllm_responses.py`,
  `internvl_backend.py`, `prepare_mllm_protocol.py`.
- Full InternVL reference matching: `prepare_matching_full.py`, followed by
  `run_matching_paper_pilot.py --output WORKING_PROJECT/mllm_material/matching_full_v1`.
  The historical filename is retained although the output argument runs the
  full schedule. `protocol.json` and `schedule.json` freeze the actual run.
- Full Qwen3 three-task acquisition: `run_qwen3_full.py`, which uses the archived
  common schedules and the reference matching implementation.
- Frozen internal extraction and interventions: `run_mechanism_v1.py`,
  `mechanism_backend_v1.py`, and `mechanism_spatial_v1.py`.
- Reference rendering: the archived `reference_reconstruction_v3/render_reference.py`.
  The generated references and renderer are included; the external Doge2 HDR
  environment must be acquired separately with its original licensing. See the
  dataset provenance for its identifier and checksum.

For a new run, obtain the original stimulus images according to their source
license and the exact public model revisions in `environment/model_revisions.json`.
No credentials, browser login state, runtime binaries or model weights are in
this package. Browser/API pilot code is historical exploratory provenance;
running the figure pipeline neither authenticates nor makes external requests.

## Interpretation and display

The unit of resampling differs by analysis: crossed shape/illumination weights
for image-level associations and fixed predictions; image-node weighting for
pairs; condition-node permutation/bootstrap for RSA; paired shape-level
intervention contrasts. Exact seeds, folds and multiplicity families are in the
archived code/configurations. Do not count model calls as independent human
participants. Shape/lighting adjustment and held-out prediction procedures are
specified in the manuscript and supplement.

Editable SVG masters keep text and vector elements editable; raster stimulus
images remain embedded. The original font was Arial. Font fallback and PDF
metadata timestamps can change file hashes without changing plotted results.
The existing figures' colours and point selection are retained; this deposit
does not retrospectively replace lower-performing model responses.
