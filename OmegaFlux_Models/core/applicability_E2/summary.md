# Applicability flag - E2

Model on 12 descriptors; threshold = 95% quantile of in-family held-out distances (0.856); k = 10; reference 200,000 points.

| set | points | share flagged | MAE | RMSE | P95 abs |
|---|---|---|---|---|---|
| held-out, training family | 216,453 | 5.0% | 0.89 | 1.27 | 2.64 |
| transfer test, unflagged | 240,943 | 0% | 6.87 | 10.69 | 25.59 |
| transfer test, flagged | 622,305 | 100% | 5.39 | 8.79 | 20.77 |

Share of transfer-test points flagged: 72.1%. Spearman(distance, |error|) = -0.034.

Rooms: 635 of 871 flagged (>= 50% of points); mean room MAE flagged 5.28 vs unflagged 5.98; worst 10% of rooms caught: 66%; Spearman(share flagged, room MAE) = -0.014.

## Error by what the point sees (transfer test)

| view | share of points | share of abs. error | simulated UDI useful | bias useful | MAE useful | bias low | bias up | share flagged |
|---|---|---|---|---|---|---|---|---|
| clear view of a window | 74.7% | 69.6% | 61.8 | +3.77 | 5.41 | -3.68 | -0.09 | 71.3% |
| view partly blocked by own walls | 21.2% | 25.5% | 58.0 | +3.26 | 6.96 | -3.37 | +0.11 | 69.4% |
| no window in view | 4.1% | 4.9% | 5.1 | +0.22 | 6.91 | -2.59 | +2.38 | 100.0% |

Bias by decile of total window solid angle (useful / low / up): [0.000-0.008] +9.9/-10.9/+1.0; [0.008-0.017] +10.6/-10.6/+0.0; [0.017-0.027] +5.7/-5.7/-0.0; [0.027-0.040] +4.0/-4.0/-0.0; [0.040-0.058] +2.4/-2.4/-0.0; [0.058-0.084] +1.6/-1.5/-0.1; [0.084-0.126] +1.3/-1.0/-0.3; [0.126-0.197] +0.4/-0.3/-0.1; [0.197-0.346] -0.2/+0.2/+0.0; [0.346-1.000] -0.5/+0.4/+0.0
