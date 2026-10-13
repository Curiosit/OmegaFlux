# Changelog

## v1.1.0 — 2026-10-13

Release accompanying the article *Daylight prediction from per-sensor geometric descriptors:
descriptor selection for transfer to unseen room shapes* (submitted to the Journal of Building
Engineering).

### Model
- The default model uses the 12 descriptors chosen by forward selection (`features.py`,
  `FEATURES_CORE`). The 26-descriptor model remains available with `--features all`.
- `Global_Sill_Clearance` is signed (−1 to 1): negative when the sill is above the 0.8 m work
  plane. In v1.0.0 it was clipped at zero for about a quarter of grid points. The web app and the
  Hops service compute it the same way.
- Every model folder contains `features.json`; `predict`, `export_onnx`, the web app and the Hops
  service read their inputs from it.
- ONNX export trims each booster at its early-stopping best iteration, so browser predictions match
  `predict_room`.

### Experiments added for the article
- Transfer tests E2 (rectangular → L-shaped), E3 (→ rotated) and E4 (1–2 → 3–4 windows), with
  three model seeds for E2 (`--model-seed`).
- Transfer-aware and in-distribution-only forward selection (`--fs-e2-weight`), and the
  one-descriptor ablations (`run_selection_control.bat`, `run_e2_robustness.bat`,
  `run_final_checks.bat`).
- Baselines on identical splits: window coordinates with XGBoost, and MLPs on 12 and 26
  descriptors (`baselines.py`, `run_baselines.bat`).
- A single model trained directly on UDI_useful (`--target useful`, `run_direct_useful.bat`).
- Distance-based applicability flag (`applicability.py`).
- Radiance convergence check: 150 held-out rooms re-simulated at `-ab 8`, ten L-shapes at `-ab 12`
  (`select_convergence_rooms.py`, `prepare_convergence_data.py`, `evaluate_convergence.py`,
  `data/convergence_check/`).
- All paper figures from one script (`make_paper_figures.py`), plus the two drawn figures
  (`fig_room_types.py`, `fig_solid_angle.py`).

### Fixes
- `train.py`: the in-script direct UDI_useful baseline (B5, `--steps baselines`) passed the nested
  low/up hyperparameter dictionary to XGBoost, which then trained with default settings. It now uses
  the UDI_low hyperparameters. The direct-model results reported in the article come from
  `--target useful`, which was not affected.
- `unzip_data.py` checked for the extracted files in `data/` instead of `data/raw/`, so it always
  re-extracted.
- `main.py` called a `make_figures` module that no longer exists; it now runs `make_paper_figures`.
- `explain` reads `data/processed/` (previously `data_processed/`, inconsistent with `train`).
- Raw data documentation: ground reflectance 0.1 and glazing transmittance 0.64 (previously listed as
  0.20 and 0.65).

### Data
- The two rectangular strata are now deposited as zip files (`study-ml-rect.xlsx.zip`,
  `study-ml-rotated.xlsx.zip`) alongside `study-ml-shapes.csv.zip`, so all three are tracked by git
  and archived by Zenodo.
- The raw convergence re-simulations (`data/raw/convergence/*.xlsx`) are tracked.

## v1.0.0 — 2026-08-03

First release, submitted with the original manuscript: 26-descriptor models, data and the web app.
