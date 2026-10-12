# Data availability and release scope

## Data included in GitHub

`figures/inputs/` contains the compact numerical inputs, stimulus examples,
selected activation arrays and summaries needed by the CPU figure-reproduction
pipeline. Exact experimental configurations and selected response/controller
examples are also preserved. Use `docs/figure_reproduction.md` to map each main
and supplementary figure to its inputs and upstream analysis scripts.

These are research data and supporting records, but they are not the complete
collection of model responses or all saved internal states.

## Full archive prepared for Zenodo

The complete local release is named `Human_Aligned_Gloss_Release_v1`. It includes
primary and exploratory response records, complete matching trajectories,
analysis data, saved internal-state arrays, intervention records, reconstructed
references, tiny-network checkpoints and additional provenance. It has been
packaged into six independent ZIP archives totalling approximately 14.23 GB.
Those six archives extract under the same release-root directory. A separate
`07_GitHub_source_c0f8da12.zip` preserves the GitHub source at commit
`c0f8da12a3bdb3fe7bc3cb4ff49eaf7a2abc3083`. Together with three archive
description/checksum files, the upload plan contains ten files (seven ZIPs),
totalling 14,297,892,487 bytes (approximately 14.30 GB). Consult the archive
README for the separate source snapshot's layout.

[Zenodo draft 23309081](https://zenodo.org/deposit/23309081) has been created;
uploads and checksum verification are in progress. Its reserved DOI
`10.5281/zenodo.23309081` is not registered or published and must not be used
as a released-data citation. The draft requires authorized account access;
there is no public full-data download. Full archived analyses require those
additional data, as documented in the reproduction guide.

## Original data and third-party rights

The human behavioral dataset and original test stimuli originate from
[Morimoto and colleagues](https://github.com/takuma929/gloss_tinynetworks).
Reconstructed reference images are identified as reconstructions, not recovered
original reference images. The original Doge2 HDR environment is not included.
Its provenance and the unresolved redistribution scope of derived reference
assets must be reviewed before public release.

The project's MIT grant covers its original code only. Other data and assets
retain their applicable rights; see `../THIRD_PARTY_NOTICES.md`. Publication of
the repository or creation of a DOI will not change those underlying rights.

## Verification scope

`FILE_MANIFEST.json` records file sizes and SHA-256 hashes. Run
`python code/verify_release_manifest.py` from the repository root to check the
listed files. This verifies file integrity, not scientific correctness.

The previously executed CPU validation regenerated figures and tables from
preserved inputs. It did not independently rerun every model acquisition,
statistical estimator or permutation test. See `VALIDATION.md`.
