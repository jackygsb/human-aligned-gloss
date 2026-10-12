# Environments

The CPU figure and table pipeline uses `requirements-cpu.txt` (Python 3.12 tested).
These versions record the scientific runtime plus a compatible PDF stack; exact
executed versions for the current verification are written in the reproduction report.
The original acquisition environments are retained in `code/configs/`.
GPU acquisition used torch 2.6.0+cu124 and transformers 4.57.1. Use the recorded
model revisions, not a moving model alias. No model weights are redistributed.
Arial was installed on the original Windows workstation; it is not bundled.
If unavailable, Matplotlib's fallback font changes typography but not statistics.
Model runtimes, third-party source releases and GPU drivers are not vendored.
