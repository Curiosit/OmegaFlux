# Surrogate against original (ab5) and converged (ab8) labels

| class | rooms | surrogate_vs_ab5_RMSE | surrogate_vs_ab8_RMSE | surrogate_vs_ab5_MAE | surrogate_vs_ab8_MAE | surrogate_vs_ab5_bias | surrogate_vs_ab8_bias | label_shift_MAE | label_shift_bias | rho_rooms_vs_ab5 | rho_rooms_vs_ab8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L-shaped | 50 | 4.373 | 4.299 | 2.815 | 2.747 | 0.321 | -0.161 | 0.944 | 0.483 | 0.991 | 0.988 |
| Rectangular | 50 | 1.401 | 1.469 | 0.994 | 1.061 | 0.011 | -0.142 | 0.475 | 0.152 | 0.988 | 0.982 |
| Rotated | 50 | 1.791 | 1.814 | 1.199 | 1.254 | -0.066 | -0.309 | 0.47 | 0.243 | 0.991 | 0.988 |
| all | 150 | 3.428 | 3.385 | 2.049 | 2.038 | 0.166 | -0.189 | 0.734 | 0.355 | 0.994 | 0.992 |


Ladder rooms (10 L-shapes): vs ab5: RMSE 5.78, bias +1.29, vs ab8: RMSE 5.41, bias +0.46, vs ab12: RMSE 5.44, bias +0.38
