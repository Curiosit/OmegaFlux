# Data

Everything under `data/raw/` is simulation output: the direct product of the Radiance runs, before
any descriptor extraction. It is the input to stage 2 of the pipeline and the thing another
researcher needs in order to reproduce the study without re-running 6,749 annual simulations.

## Files

| File | Stratum | Rooms | Rows | Size |
|---|---|---|---|---|
| `bins_A-D_rectangular.xlsx.zip` | axis-aligned rectangular, four fenestration patterns | 4,500 | 4,500 | 15 MB |
| `bin_E_rotated.xlsx.zip` | as A–D, at 90°, 180° and 270° | 1,124 | 1,125 | 6 MB |
| `bin_F_lshaped.csv.zip` | non-convex L-shaped, 1–3 apertures on any eligible wall | 1,125 | 1,125 | 14 MB |

One row is one simulated room. Bin E contains 1,125 rows but 1,124 distinct configurations; the
duplicate is retained as exported and is de-duplicated by group key during extraction.

## Column schema

Identical across the three files apart from how room geometry is expressed.

### Identity

| Column | Type | Meaning |
|---|---|---|
| `Iteration_Name` | string | Configuration identifier, unique within a stratum |
| `Iteration_Nr` | string / int | Generator index |
| `DateTime` | string | Simulation timestamp (XLSX files only) |

### Geometry — rectangular strata (bins A–E)

| Column | Unit | Meaning |
|---|---|---|
| `Room-width`, `Room-depth` | m | Plan dimensions |
| `Orientation` | ° | 0 for bins A–D; 90, 180 or 270 for bin E |
| `Canyon_HW` | — | External obstruction height-to-width ratio. **Zero throughout**; see below |
| `Front_Win{1,2}_Center_H_m` | m | Aperture centre along the south wall, from its origin |
| `Front_Win{1,2}_Center_V_m` | m | Aperture centre height above floor |
| `Front_Win{1,2}_Width_m` | m | Aperture width; 0 means the aperture is absent |
| `Front_Win{1,2}_Height_m` | m | Aperture height |
| `Side_Win{1,2}_*` | | The same four fields for the east wall |

### Geometry — non-convex stratum (bin F)

| Column | Format | Meaning |
|---|---|---|
| `Room_corners` | `{x,y}{x,y}…` | Plan boundary, counter-clockwise, six vertices |
| `Window_centerpoints` | `{x,y,z}{x,y,z}…` | Aperture centres in 3-D |
| `Window_widths` | `w;w;…` | Aperture widths, semicolon-separated |
| `Window_heights` | `h;h;…` | Aperture heights |

### Sensor grid and results — all strata

| Column | Unit | Meaning |
|---|---|---|
| `Sensor_location_matrix` | `{x,y,z}…` | Grid point coordinates, z = 0.8 m throughout |
| `raw_UDI_low` | % | **Target.** Occupied hours below the lower threshold, per grid point |
| `raw_UDI_up` | % | **Target.** Occupied hours above the upper threshold, per grid point |
| `raw_UDI` | % | Useful fraction as exported; equals 100 − low − up |
| `raw_DA`, `raw_cDA` | % | Daylight Autonomy and continuous DA, per grid point |
| `sDA`, `culled_sDA` | % | Spatial Daylight Autonomy, room aggregate |
| `aDA`, `culled_aDA` | % | Average DA, room aggregate |
| `ASE_results`, `ASE_culled_results` | % / h | Annual Sunlight Exposure, room aggregate |

Per-grid-point arrays are semicolon-separated and align element-by-element with
`Sensor_location_matrix`. 

The `culled_` variants exclude a margin near the walls. The study uses the uncounted values.

## Simulation settings

| | |
|---|---|
| Engine | Radiance via Honeybee / Ladybug Tools |
| Recipe | Annual daylight, daylight-coefficient method |
| Sky | Perez all-weather |
| Weather | Warsaw Okęcie, 52.17° N, 20.97° E |
| Ambient parameters | `-ab 5 -ad 5000 -lw 2e-05` |
| Reflectances | ceiling 0.80, walls 0.50, floor 0.20, ground 0.20 |
| Glazing | visible transmittance 0.65, frameless |
| Grid | 0.25 m spacing, 0.80 m above floor |
| External obstruction | **None.** `Canyon_HW` = 0 in every row |
| UDI thresholds | 100 lux and 3000 lux |


## Processed data

`python -m omegaflux.extract_features` reads all three raw files and writes
`data/processed/omegaflux_sensors_pooled.csv`: 3,266,627 rows, one per grid point, carrying the 26
descriptors, the two targets, a globally unique `Group_ID` and a set of `Meta_*` columns used only
for the baseline comparison.

The file is roughly 1.5 GB and is **not** committed. It is a deterministic function of the raw data
and the extractor, and `manifest_extraction.json`, written alongside it, records the seed, the site
latitude, every configuration constant and the record counts so a regenerated table can be checked
against the one used for the paper.

## Reuse

The raw data is usable independently of this model, citing the author and the journal article. 
It is, as far as I am aware, the largest openly
available set of per-grid-point annual daylight results for parametrically varied room geometry,
and it may suit work on other daylight metrics, on alternative surrogate architectures, or as a
benchmark for geometric generalisation. Licence is CC BY 4.0; the citation is in the main README.
