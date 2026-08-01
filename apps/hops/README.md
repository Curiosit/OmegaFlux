<div align="center">

<img src="../../docs/assets/logo.png" alt="OmegaFlux" width="96">

# OmegaFlux Hops Server (`apps/hops`)

**Real-time Grasshopper daylight predictions via Hops and HTTP REST API**
*Physics-informed 26-feature surrogate model server for McNeel Rhino / Grasshopper.*

[![License: MIT](https://img.shields.io/badge/code-MIT-green)](../../LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](../../pyproject.toml)

</div>

---

The **OmegaFlux Hops Server** exposes the trained 26-descriptor XGBoost daylight surrogate model to **McNeel Rhino / Grasshopper** via [Grasshopper Hops](https://developer.rhino3d.com/guides/grasshopper/hops-component/).

It receives 3D room perimeter points, window corner geometry, and sensor grid points directly from Grasshopper, computes the 26 dimensionless radiometric features on the fly, and streams back Useful Daylight Illuminance ($\text{UDI}_{\text{useful}}$), under-lit ($\text{UDI}_{\text{low}}$), and over-lit/glare ($\text{UDI}_{\text{up}}$) predictions in milliseconds.

---

## Key Features

* **Grasshopper Integration:** Solves entire sensor fields directly inside Rhino/Grasshopper without leaving the CAD environment.
* **Closed-Form Feature Extraction:** Computes solid angles (Van Oosterom–Strackee formula), annual direct sun path fractions, and dominant aperture frames on the fly for each sensor point.
* **Ground-Truth Validation:** Optionally accepts simulated Radiance/Honeybee UDI values to compute real-time validation metrics ($R^2$, $\text{RMSE}$, $\text{MSE}$) inside Grasshopper.
* **Portable Path Resolution:** Automatically locates pre-trained XGBoost model binaries from the repository root `models/` folder or environment overrides.

---

## Folder Structure

```
apps/hops/
├── gh.py               # Flask + Hops server implementation
└── README.md           # Hops server documentation
```

---

## Getting Started

### 1. Prerequisites

Ensure you have installed the core project dependencies with the `hops` optional extras:

```bash
# From repository root
pip install -e ".[hops]"
```

Or install the required packages manually:

```bash
pip install ghhops-server flask joblib numpy scikit-learn xgboost
```

### 2. Model Files

The server requires the pre-trained XGBoost models saved as `.joblib` binaries in the root `models/` directory:

```
omegaflux/
├── models/
│   ├── omegaflux_xgb_low.joblib
│   └── omegaflux_xgb_up.joblib
└── apps/
    └── hops/
        └── gh.py
```

*Note: If the `.joblib` files are not present, download them via `python -m omegaflux.fetch_models` or specify a custom directory using the `OMEGAFLUX_MODEL_DIR` environment variable.*

---

## 🛠 Running the Server

Start the local HTTP Hops server:

```bash
# From repository root
python apps/hops/gh.py
```

The Flask server will start locally on port **5000**:
```
 Running on [http://127.0.0.1:5000](http://127.0.0.1:5000)
```

---

## Connecting to Grasshopper

1. Open **Rhino 7 / 8** and launch **Grasshopper**.
2. Install the **Hops** component via the Package Manager (`PackageManager` -> Search for `Hops`).
3. Add a **Hops** component to your Grasshopper canvas.
4. Right-click the Hops component, select **Path**, and enter:
   ```
   http://localhost:5000/omegaflux_predict
   ```

---

## Component Specification

### Inputs

| Name | Key | Type | Description |
|---|---|---|---|
| **Points** | `Pts` | Point (List) | Grid of sensor points on the workplane ($z = 0.8\text{ m}$). |
| **Room Corners** | `RoomPts` | Point (List) | Ordered perimeter points defining the room boundary in 2D/3D. |
| **Window Corners** | `Corners` | Point (List) | Flat list of 4 corner points per aperture (4 points per window). |
| **Simulated Useful UDI** | `Simulated` | Number (List) | *(Optional)* Ground-truth UDI values to compute error metrics. |

### Outputs

| Name | Key | Type | Description |
|---|---|---|---|
| **Useful UDI** | `Useful` | Number (List) | Predicted Useful UDI percentage ($100 - \text{Low} - \text{Up}$). |
| **UDI Low** | `Low` | Number (List) | Predicted under-illuminated percentage. |
| **UDI Up** | `Up` | Number (List) | Predicted over-illuminated/glare percentage. |
| **R2 Score** | `R2` | Number (Item) | Coefficient of Determination against optional ground truth. |
| **RMSE** | `RMSE` | Number (Item) | Root Mean Squared Error (percentage points). |
| **MSE** | `MSE` | Number (Item) | Mean Squared Error. |

---
