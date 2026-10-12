# Human-aligned gloss judgments in frozen vision–language models

Code and compact figure inputs for the manuscript by **Shaobing Gao, Minjie Tan,
Yuanpeng Li, Xilin Tan, Changyu Li, Jianjun Wei and Shihao Liu**.
Shaobing Gao and Minjie Tan contributed equally. Correspondence:
**Shaobing Gao**, gaoshaobing@scu.edu.cn.

Repository: **[jackygsb/human-aligned-gloss](https://github.com/jackygsb/human-aligned-gloss)**.
This repository is being prepared privately for author review. It is not yet a
public release or an accepted publication. No Zenodo deposit or DOI has been
created. The scope and limits of the completed checks are recorded in
[VALIDATION.md](docs/VALIDATION.md).

## What this repository contains

- Portable CPU code to regenerate all six main figures, supplementary Figs.
  S1–S15, and Tables S1–S9.
- Numerical plot inputs, selected activation examples, stimulus thumbnails,
  reconstructed references, exact prompts and controller examples.
- Original first-party acquisition and analysis scripts, historical experimental
  source snapshots, model revisions, runtime records, seeds and fold information.
- Selected original author MATLAB code for the Morimoto tiny-network paper,
  kept separately with its original MIT license and provenance.

The full response archives, complete activation arrays, complete stimulus set,
and manuscript PDFs are separate release components. Model weights, private
credentials, browser profiles and third-party runtime binaries are not included.
Selected small activation arrays in `figures/inputs/` are necessary figure inputs;
they are not the full internal-response archive.

| Directory | Contents |
|---|---|
| `code/figures/` | Main and supplementary figure generation |
| `code/provenance/` | Preserved acquisition, analysis and experimental source records |
| `code/configs/` | Recorded protocols, prompts and analysis configuration |
| `code/third_party/` | Selected upstream code with original attribution |
| `figures/inputs/` | Compact numerical inputs and image examples used by the figures |
| `environment/` | Dependencies, model revisions, seeds and fold information |
| `docs/` | Reproduction guide, figure legends and validation records |

See [Data availability](docs/DATA_AVAILABILITY.md) for the distinction between
the data in this Git repository and the full archive prepared for Zenodo.

## Quick start

Python 3.12 is recommended. From the repository root:

```text
python -m venv .venv
```

Activate this environment using the command appropriate to your shell, then:

```text
python code/verify_release_manifest.py
python -m pip install -r environment/requirements-cpu.txt
python code/reproduce_figures.py
python code/validate_figure_outputs.py
```

Outputs are written to `figures/regenerated/`. The pipeline really executes the
drawing calculations from the compact inputs; it does not copy published figure
files. A successful run creates 66 output artifacts, including 22 final PDF
assets (Fig. S12 has two parts), an intermediate workflow master, SVG/PNG figures,
and nine LaTeX tables. The optional command
`python code/validate_figure_outputs.py --render` uses Poppler's `pdftoppm` to
produce contact sheets.

`figures/expected_outputs.json` contains the canonical PDF text/dimension and
table hashes, allowing verification without duplicating all manuscript figure
masters in this repository. The original typography used Arial; font fallback
can change PDF layout and text extraction without changing numerical values.

The pipeline was verified on CPU. This checks figure recreation and table
formatting, not a fresh run of every bootstrap, model fit or GPU experiment.
Historical analysis code and full data are supplied separately for that purpose.
See [the reproduction guide](docs/figure_reproduction.md) for explicit scope,
per-figure inputs and upstream scripts. The final submission captions are in
[the current figure legends](docs/figure_legends_submission_v11.tex).

## Model and analysis provenance

The principal frozen models are InternVL3.5-4B and Qwen3-VL-4B-Instruct. Their exact
public repository revisions are recorded in `environment/model_revisions.json`.
An earlier Qwen2.5-VL-3B experiment is retained as exploratory provenance.
Behavioral acquisition supplies no target human ratings; separate supervised
readout analyses use human labels within their training folds. The reference
direction used for intervention is derived without test-image human judgments.

## Full data and citation

The full data release is prepared for a subsequent Zenodo deposit. Its DOI is
**not yet assigned**. Repository access currently requires permission from the
owner. After the archive is deposited, its verified version-specific DOI will be
added to this README and the citation metadata. `CITATION.cff` records author
order and affiliations. `.zenodo.json` is intended deposit metadata; its presence
does not mean a Zenodo record exists.

## License and third-party materials

The authors' original code is MIT-licensed; see [code/LICENSE](code/LICENSE).
This grant does **not** relicense third-party code, stimulus images, human data,
environment maps, model weights, model outputs or source-paper material.
Original notices and scope distinctions are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). In particular, the external
Doge2 HDR environment is not redistributed; references derived using it retain
that provenance. Preserve all original notices when redistributing assets.

## Authors and affiliations

1. Shaobing Gao — School of Artificial Intelligence, Sichuan University, Chengdu,
   China. Equal first author and corresponding author.
2. Minjie Tan — College of Computer Science and Artificial Intelligence,
   Southwest Minzu University, Chengdu, China. Equal first author.
3. Yuanpeng Li; Xilin Tan; Changyu Li; Jianjun Wei; Shihao Liu — College of
   Computer Science, Sichuan University, Chengdu, China, in the listed author order.

Creating this repository does not constitute institutional approval, an
independent scientific audit, or publication of the manuscript.
