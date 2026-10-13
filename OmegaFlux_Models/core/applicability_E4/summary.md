# Applicability flag - E4

Model on 12 descriptors; threshold = 95% quantile of in-family held-out distances (0.965); k = 10; reference 200,000 points.

| set | points | share flagged | MAE | RMSE | P95 abs |
|---|---|---|---|---|---|
| held-out, training family | 177,493 | 5.0% | 1.53 | 2.78 | 5.16 |
| transfer test, unflagged | 1,292,061 | 0% | 1.63 | 2.66 | 5.17 |
| transfer test, flagged | 155,967 | 100% | 2.41 | 3.45 | 7.15 |

Share of transfer-test points flagged: 10.8%. Spearman(distance, |error|) = 0.185.

Rooms: 94 of 3187 flagged (>= 50% of points); mean room MAE flagged 2.65 vs unflagged 1.52; worst 10% of rooms caught: 15%; Spearman(share flagged, room MAE) = 0.492.
