# fMRI Reproducibility for Galaxy

Galaxy tool for batch analysis of already preprocessed fMRI data. It runs the Python GLM/ROC/Kappa/reproducibility pipeline; it does not run MATLAB/SPM or the Windows GUI. The `code/` directory is a copy of the analysis modules from the fMRI Reproducibility project. No participant data or results are included.

## Inputs and outputs

Upload one ZIP per subject. Its top-level directory must match the Subject ID entered in Galaxy and contain `subject.json`, `runs/<run_id>/bold.nii.gz`, `logs/<run_id>.tsv`, and `brain_mask.nii.gz` (or `.nii`). The configured mask checks require at least 1,000 voxels and reject a mask covering more than 80% of the image volume. The manifest defines run order, events, conditions, contrasts and `x_star`. See `test-data/synthetic_subject.zip` for a generated example; its creation code is in `tests/make_fixture.py`. It contains no human data.

Galaxy returns `summary.json` and a ZIP with all outputs, including NIfTI maps and plots. The tool runs in a temporary directory and requests one CPU worker unless `GALAXY_SLOTS` is set by the server. Ensure the job node has sufficient memory and scratch disk for uncompressed 4D NIfTI inputs and results.

## Install on a Galaxy server

The administrator can place this repository (or the contents of `fmri_reproducibility-galaxy-0.2.0.zip`) in the Galaxy tools directory and add the tool entry from `tool_conf.xml.example` to the active `tool_conf.xml`. Keep `fmri_reproducibility.xml`, `run_galaxy.py`, `code/` and `test-data/` together. Enable the Galaxy Conda dependency resolver or provide an equivalent environment with the package versions specified in the XML. Restart or reload Galaxy according to the server's tool configuration.

The tool targets Linux with Python 3.12. The Windows executable distributed in the program repository is for desktop use and is not needed on the Galaxy server. This tool has not yet been run on the target Galaxy instance, whose operating system and administrator settings are unknown.

## Local verification

With the dependencies from `requirements.txt` installed:

```bash
python tests/make_fixture.py
python -m unittest tests.test_runner -v
planemo lint fmri_reproducibility.xml
planemo test --install_galaxy --conda_dependency_resolution --conda_auto_install --conda_auto_init --conda_channels conda-forge,bioconda fmri_reproducibility.xml
```

`planemo test` provisions a temporary Galaxy and creates a Conda environment for the tool requirements. With an existing local Galaxy, replace `--install_galaxy` with `--galaxy_root /path/to/galaxy`; keep the Conda options. Its XML test uses the synthetic subject, four voxels, OLS as the primary method and two ROC restarts. This test passed in the local WSL Galaxy used for PsiScan. The GitHub Actions workflow runs the synthetic analysis, Galaxy XML linter and a Galaxy integration test on Linux. A full scientific validation on an anonymized real subject and acceptance on the target Galaxy remain separate steps.

For a direct command-line run:

```bash
python run_galaxy.py --input test-data/synthetic_subject.zip --subject-id synthetic_01 --output results.zip --summary summary.json --max-voxels 4 --glm-method ols --roc-restarts 2
```

Before uploading real data, remove identifying metadata and absolute source paths from `subject.json`, logs, and NIfTI headers. The synthetic fixture demonstrates the required layout without exposing any research data.
