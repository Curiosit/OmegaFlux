# The `omegaflux` package

Run every module from the repository root, e.g. `python -m omegaflux.extract_features`.

## Pipeline

| Module | What it does |
|---|---|
| `features.py` | Descriptor sets: all 26 (`FEATURES_ALL`) and the 12 selected (`FEATURES_CORE`). Single source of truth for every consumer. |
| `generate_doe.py` | Latin Hypercube sampling matrices for the six bins (Appendix A). |
| `unzip_data.py` | Extracts `data/raw/*.zip`. |
| `extract_features.py` | Raw room exports → one row per grid point with 26 descriptors and the two targets (about 40 min). |
| `train.py` | XGBoost models, hyperparameter search, cross-validation, transfer tests E2–E4, forward selection, model size sweep, TreeSHAP. |
| `predict.py` | `predict_room(room, windows, grid)` for a trained model folder. |
| `export_onnx.py` | ONNX export (trimmed at the best iteration) for the web app. |
| `fetch_models.py` | Downloads trained models from the GitHub release. |

## Experiments for the article

| Module | What it does |
|---|---|
| `baselines.py` | Window-coordinate XGBoost and MLP (12 and 26 descriptors) on the same splits as `train.py`. MLP: hidden layers 256-128-64, ReLU, Adam (learning rate 1e-3, batch 2,048), up to 60 epochs with early stopping (patience 6), standardised inputs, 1,000,000 training rows. |
| `applicability.py` | Distance-based applicability flag: mean distance to the 10 nearest training points in standardised descriptor space, threshold at the 95th percentile of held-out rooms of the training family. |
| `select_convergence_rooms.py` | Reproduces the held-out split and selects the 150 convergence-check rooms. |
| `prepare_convergence_data.py` | Cleans the `-ab 8` / `-ab 12` re-simulations into `data/convergence_check/`. |
| `evaluate_convergence.py` | Surrogate error against the original and the converged labels (Table C3). |
| `explain.py` | TreeSHAP attribution for a model folder. |
| `make_paper_figures.py` | Figures 5–8 and C1 from the runs on disk; skips figures whose runs are missing. |
| `fig_room_types.py` | Figure 2 (drawn from the sampling ranges, no data needed). |
| `fig_solid_angle.py` | Figure 4 (drawn, no data needed). |

See `docs/REPRODUCE.md` for which command produces which table and figure.
