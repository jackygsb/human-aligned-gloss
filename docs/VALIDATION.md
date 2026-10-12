# Lightweight-package validation

The standalone lightweight directory was executed on 9 October 2026, with all
outputs written to a separate temporary directory rather than copied into this
repository. Ten CPU jobs completed successfully and regenerated 66 artifacts.
All 22 final PDF assets matched canonical page counts, dimensions and extracted
text. All nine supplementary LaTeX tables matched their canonical normalized
content hashes. Results and actual script logs are retained in `validation/`;
the per-PDF audit is `figure_reproduction_pdf_audit.json`.

The scientific environment was Python 3.12.14 with the pinned NumPy/Pandas/SciPy/
Matplotlib/Pillow versions in environment/requirements-cpu.txt. The PDF assembly
used a second Python 3.12.14 environment with pypdf 6.10.0 and reportlab 4.4.9,
selected using --pdf-python. A single environment with all listed dependencies
can run the same command. No model inference, external upload or network request
was part of this validation. Full statistical re-estimation was not rerun.
