@echo off
REM E2 (rectangles -> L-shapes) for the descriptor subsets on both forward-selection paths.
REM Same hyperparameters as the core model; E1 main model skipped; only the E2 test runs.
set P=--steps topology --topology E2 --skip-main --params-from OmegaFlux_Models/core --force

REM --- Decisive run: the 12 chosen for in-distribution accuracy only ---
python -m omegaflux.train %P% --out OmegaFlux_Models/fsE1_k12 --feature-list "Window_Sky_Weighted_Omega,Global_Cos_Theta,Dominant_Win_Ray_Forward,South_Weighted_Omega,Outer_Corner_Dmin,Smooth_Depth_StdDev,Annual_Direct_Sun_Fraction,Total_Omega,Window_Angular_Spread,Global_inv_dmin2,Occlusion_Multiplier,Opaque_Wall_dmin"

REM --- In-distribution-only path, shorter subsets ---
python -m omegaflux.train %P% --out OmegaFlux_Models/fsE1_k9 --feature-list "Window_Sky_Weighted_Omega,Global_Cos_Theta,Dominant_Win_Ray_Forward,South_Weighted_Omega,Outer_Corner_Dmin,Smooth_Depth_StdDev,Annual_Direct_Sun_Fraction,Total_Omega,Window_Angular_Spread"
python -m omegaflux.train %P% --out OmegaFlux_Models/fsE1_k6 --feature-list "Window_Sky_Weighted_Omega,Global_Cos_Theta,Dominant_Win_Ray_Forward,South_Weighted_Omega,Outer_Corner_Dmin,Smooth_Depth_StdDev"

REM --- Transfer-aware path (the core 12 in selection order), shorter subsets ---
python -m omegaflux.train %P% --out OmegaFlux_Models/fsTr_k9 --feature-list "Window_Sky_Weighted_Omega,Smooth_Depth_StdDev,Global_Cos_Theta,South_Weighted_Omega,Window_Angular_Spread,Outer_Corner_Dmin,Annual_Direct_Sun_Fraction,East_Weighted_Omega,Total_Omega"
python -m omegaflux.train %P% --out OmegaFlux_Models/fsTr_k6 --feature-list "Window_Sky_Weighted_Omega,Smooth_Depth_StdDev,Global_Cos_Theta,South_Weighted_Omega,Window_Angular_Spread,Outer_Corner_Dmin"
