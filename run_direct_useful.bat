@echo off
REM Ablation requested by Reviewer 3: one model predicting UDI_useful directly,
REM against the two-component models (UDI_low + UDI_up). Same 12 descriptors, same splits,
REM UDI_low hyperparameters reused (not re-tuned for UDI_useful).
set P=--target useful --features core --params-from OmegaFlux_Models/core --force

REM 1. Main model (E1) plus transfer to L-shapes (E2), seed 42
python -m omegaflux.train %P% --steps topology --topology E2 --out OmegaFlux_Models/direct_useful

REM 2. E3 and E4 (separate run, to stay within memory)
python -m omegaflux.train %P% --steps topology --topology E3,E4 --skip-main --out OmegaFlux_Models/direct_useful_E3E4

REM 3. E2 with two more model seeds, to compare with 0.856 +/- 0.001 of the component model
for %%S in (1 2) do (
  python -m omegaflux.train %P% --steps topology --topology E2 --skip-main --model-seed %%S --out OmegaFlux_Models/seed%%S_direct_useful
)
