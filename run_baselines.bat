@echo off
REM Same-data baselines (WO-B, B5). Each run appends to OmegaFlux_Models/baselines/results_table.csv,
REM so tests run one at a time to stay within memory; a crash keeps everything already finished.
python -m omegaflux.baselines --models coord_xgb --tests E1
python -m omegaflux.baselines --models coord_xgb --tests E2
python -m omegaflux.baselines --models coord_xgb --tests E3
python -m omegaflux.baselines --models coord_xgb --tests E4
python -m omegaflux.baselines --models mlp12,mlp26 --tests E1
python -m omegaflux.baselines --models mlp12,mlp26 --tests E2
