@echo off
REM E2 robustness: is the E2 ordering (core-12 > all-26 > in-distribution-12) larger than run-to-run noise,
REM and which of the three plan-geometry descriptors causes the drop?
REM Data splits are fixed; --model-seed changes only model fitting. Seed 42 results already exist.
set P=--steps topology --topology E2 --skip-main --params-from OmegaFlux_Models/core --force
set E1_12=Window_Sky_Weighted_Omega,Global_Cos_Theta,Dominant_Win_Ray_Forward,South_Weighted_Omega,Outer_Corner_Dmin,Smooth_Depth_StdDev,Annual_Direct_Sun_Fraction,Total_Omega,Window_Angular_Spread,Global_inv_dmin2,Occlusion_Multiplier,Opaque_Wall_dmin
set E1_9=Window_Sky_Weighted_Omega,Global_Cos_Theta,Dominant_Win_Ray_Forward,South_Weighted_Omega,Outer_Corner_Dmin,Smooth_Depth_StdDev,Annual_Direct_Sun_Fraction,Total_Omega,Window_Angular_Spread

REM --- 1. Seed repeats for the three models in the claim (6 runs) ---
for %%S in (1 2) do (
  python -m omegaflux.train %P% --model-seed %%S --features core --out OmegaFlux_Models/seed%%S_core12
  python -m omegaflux.train %P% --model-seed %%S --features all  --out OmegaFlux_Models/seed%%S_all26
  python -m omegaflux.train %P% --model-seed %%S --feature-list "%E1_12%" --out OmegaFlux_Models/seed%%S_fsE1_k12
)

REM --- 2. Which plan-geometry descriptor hurts: in-distribution 9 + one at a time (3 runs) ---
python -m omegaflux.train %P% --feature-list "%E1_9%,Global_inv_dmin2"     --out OmegaFlux_Models/abl_inv_dmin2
python -m omegaflux.train %P% --feature-list "%E1_9%,Occlusion_Multiplier" --out OmegaFlux_Models/abl_occlusion
python -m omegaflux.train %P% --feature-list "%E1_9%,Opaque_Wall_dmin"     --out OmegaFlux_Models/abl_opaque_wall
