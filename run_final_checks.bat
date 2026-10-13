@echo off
REM Final runs for the paper figures:
REM  1. Seed repeats of the one-descriptor ablation (and of its base model), so the ablation panel
REM     shows a spread instead of single runs. Seed 42 results already exist.
REM  2. MLP baselines on E3 and E4, to fill the "not run" gaps in the baseline figure.
set P=--steps topology --topology E2 --skip-main --params-from OmegaFlux_Models/core --force
set E1_9=Window_Sky_Weighted_Omega,Global_Cos_Theta,Dominant_Win_Ray_Forward,South_Weighted_Omega,Outer_Corner_Dmin,Smooth_Depth_StdDev,Annual_Direct_Sun_Fraction,Total_Omega,Window_Angular_Spread

for %%S in (1 2) do (
  python -m omegaflux.train %P% --model-seed %%S --feature-list "%E1_9%"                      --out OmegaFlux_Models/seed%%S_fsE1_k9
  python -m omegaflux.train %P% --model-seed %%S --feature-list "%E1_9%,Global_inv_dmin2"     --out OmegaFlux_Models/seed%%S_abl_inv_dmin2
  python -m omegaflux.train %P% --model-seed %%S --feature-list "%E1_9%,Opaque_Wall_dmin"     --out OmegaFlux_Models/seed%%S_abl_opaque_wall
  python -m omegaflux.train %P% --model-seed %%S --feature-list "%E1_9%,Occlusion_Multiplier" --out OmegaFlux_Models/seed%%S_abl_occlusion
)

python -m omegaflux.baselines --models mlp12,mlp26 --tests E3
python -m omegaflux.baselines --models mlp12,mlp26 --tests E4

python -m omegaflux.make_paper_figures
