# Reproducing the article

All commands run from the repository root after `pip install -e ".[dev]"`. The `.bat` files are
Windows batch files; each line is an ordinary command and can be run on any system.

## 1. Data

```bash
python -m omegaflux.unzip_data
python -m omegaflux.extract_features          # -> data/processed/omegaflux_sensors_pooled.csv
```

## 2. Tables and figures

| Article | Command(s) | Output |
|---|---|---|
| Table 2, Appendix D (Table D1) | `omegaflux/features.py` | descriptor lists |
| Table 3, E1 and cross-validation | `python -m omegaflux.train --mode full --features core` and `--features all` | `OmegaFlux_Models/{core,all}/results_table.csv` |
| Table 3, E2 seed mean ± SD | `run_e2_robustness.bat` (section 1) | `OmegaFlux_Models/seed{1,2}_*` |
| Table 3, E3 and E4 | `train ... --steps topology --topology E3,E4` (included in `--mode full`) | `results_table.csv` |
| Appendix B (Tables B1–B3) | `train --steps tune` (included in `--mode full`) | `hyperparameters.json`, `results_table.csv` |
| Section 3.5 selection, Table D2, Figure 7a | `train --steps fs` (transfer path, default `--fs-e2-weight 0.6`) and `--fs-e2-weight 0` (control); `run_selection_control.bat` | `feature_selection_path.csv`, `OmegaFlux_Models/fs*_k*` |
| Figure 7b (one-descriptor ablation) | `run_e2_robustness.bat` (section 2), `run_final_checks.bat` | `OmegaFlux_Models/*abl_*` |
| Figure 6, Table E1 (coordinates, MLP) | `run_baselines.bat`, `run_final_checks.bat` | `OmegaFlux_Models/baselines/results_table.csv` |
| Section 4.2, direct UDI_useful model | `run_direct_useful.bat` | `OmegaFlux_Models/*direct_useful*` |
| Section 4.4, Figure 8 | `python -m omegaflux.applicability --context E2` and `--context E4` | `OmegaFlux_Models/core/applicability_E*/` |
| Section 3.2, Appendix C | `python -m omegaflux.prepare_convergence_data`, then `python -m omegaflux.evaluate_convergence` | `data/convergence_check/`, Tables C1–C3 |
| Figures 5–8, C1 | `python -m omegaflux.make_paper_figures` | `figures/paper/` |
| Figures 2 and 4 | `python -m omegaflux.fig_room_types`, `python -m omegaflux.fig_solid_angle` | `figures/paper/` |
| Section 4.5, timing | logged by `extract_features` (`manifest_extraction.json`) and `train` (LATENCY block) | |

Room-level Spearman ρ (Section 6.3) is the `room_spearman_rho` column of every `results_table.csv`.

## 3. Notes

- Data splits use seed 42 throughout; `--model-seed` changes only model fitting.
- The 254 L-shaped training rooms reserved for tuning and selection are excluded from the E2 test set.
- The test rooms (E1, 675 rooms) are not used for tuning, early stopping or selection.
