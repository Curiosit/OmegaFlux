# OmegaFlux Core Module

This directory contains the core Python scripts for the OmegaFlux daylight prediction pipeline. The pipeline operates entirely without neural networks, utilizing gradient-boosted tree ensembles on dimensionless radiometric features.

## Pipeline Scripts

*   `generate_doe.py`: Generates Latin Hypercube Sampling matrices for simulated architectural typologies.
*   `unzip_data.py`: Scans the `data/raw/` folder and extracts any downloaded simulation zip files.
*   `extract_features.py`: Converts room-level exports into a per-sensor tabular dataset of radiometric features (~40 min).
*   `train.py`: Trains the XGBoost regressors on the extracted tabular data and outputs models (~3 h).
*   `explain.py`: Generates TreeSHAP attributions from the trained models.
*   `make_figures.py`: Processes the results and generates visualization figures.
*   `export_onnx.py`: Converts the trained JSON models into web-ready ONNX formats.
*   `fetch_models.py`: Downloads pre-trained models directly from the repository's latest release.

## Usage
These scripts are intended to be run as modules from the root directory:
`python -m omegaflux.extract_features`