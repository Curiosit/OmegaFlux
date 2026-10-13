# Radiance convergence check

Re-simulation of 150 held-out rooms with stricter Radiance settings, to test whether
the training labels (`-ab 5 -ad 5000 -lw 2e-05`) under-resolve interreflection.
Rooms were selected by `select_convergence_rooms.py` (stratified, held-out partition only);
the ten `-ab 12` rooms are the eight most-shifted L-shapes at `-ab 8` plus the two darkest.
Only the Radiance parameters differ between runs; geometry, sensors, weather, materials,
thresholds and schedule are identical (see `convergence_settings.json`).

## convergence_sensors.csv.gz (85,782 rows, one per sensor)
| column | meaning |
|---|---|
| room_id | room name as in the training data (e.g. `B_373`, `C_Rotated_16`, `Expert_LShape_408`) |
| class | `rect`, `rect_rotated` or `randomized` (L-shaped) |
| sensor_idx | 0-based sensor order, identical across runs and to the training data |
| x_m, y_m | sensor position (m); sensors at 0.8 m above floor |
| udi_low_<s>, udi_useful_<s>, udi_up_<s> | % of occupied hours below 100 lux / 100-3000 lux / above 3000 lux, for setting `<s>` |

Settings: `ab5` original labels, `ab8` convergence reference, `ab12` ladder (L-shapes only). Empty = not run.

## convergence_rooms.csv (one per room)
Room metadata (`stratum`, `selection_reason`, `preselected_v2`), settings present, room-mean UDI per
setting, and per-room shifts in useful UDI: `useful_bias_a_to_b` (mean signed change, pp) and
`useful_mae_a_to_b` (mean absolute change, pp).

## Notes
- `convergence_cleaning_log.txt` lists every row excluded and why (missing results, rooms run by
  mistake outside the selection, duplicates).
- Values rounded to 2 decimals; low + useful + up = 100 within 0.1 pp at every sensor.
