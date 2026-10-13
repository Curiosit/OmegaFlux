<div align="center">

<img src="public/favicon-96x96.png" alt="OmegaFlux" width="96">

# OmegaFlux Web Application (`apps/web`)

An interactive, browser-based 2D daylighting visualization tool powered by the **OmegaFlux** surrogate model.

The application allows users to dynamically configure room dimensions, adjust fenestration parameters (window widths, positions, sill/head heights) across all cardinal orientations, and visualize real-time annual **Useful Daylight Illuminance (UDI)** distributions at 0.25 m spatial resolution.


[![Code licence: MIT](https://img.shields.io/badge/code-MIT-green)](LICENSE)
[![Data licence: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-green)](LICENSE-DATA)


</div>

---

![web.png](web.png)

---
## Tech Stack

* **Framework:** React 18 + TypeScript + Vite
* **ML Inference:** `onnxruntime-web` (ONNX Runtime WebAssembly execution)
* **Styling:** CSS Modules / CSS Variables
* **Build System:** Vite

---

## Project Structure



```
apps/web/
├── public/
│   ├── models/
│   │   ├── OmegaFluxLow.onnx      # Pre-trained ONNX model (UDI Low)
│   │   └── OmegaFluxUp.onnx       # Pre-trained ONNX model (UDI Up)
│   ├── favicon.svg
│   └── site.webmanifest
├── src/
│   ├── components/
│   │   ├── Room.tsx               # Main application container & simulation loop
│   │   ├── RoomVisualizer.tsx     # 2D Canvas/SVG grid rendering
│   │   ├── Slider.tsx             # Reusable UI range slider
│   │   └── WindowDashboard.tsx    # Active window parameters view
│   ├── engine.ts                  # Client-side geometric extractor (26 candidate descriptors)
│   ├── types.ts                   # TypeScript interfaces (RoomDimensions, WindowSpecs)
│   ├── App.tsx                    # Root React component
│   └── main.tsx                   # Application entry point
├── package.json
├── tsconfig.json
└── vite.config.ts
```

---

## Getting Started

### Prerequisites

Ensure you have **Node.js** (v18+) and **npm** installed.

### Installation

From the repository root or the `apps/web/` folder:

```bash
cd apps/web
npm install
```

Development Server
Start the Vite development server:


```Bash
npm run dev
Open your browser at http://localhost:5173
```

---

## Architecture & Inference Engine

1. **Feature Extraction (`src/engine.ts`):** 
   Whenever the user updates room topology or window sliders, the application samples the floor plate at `0.25 m` intervals. For each coordinate $(x, y)$, `extractFeatures()` constructs 3D aperture geometries and evaluates the 26 candidate descriptors exactly as the `omegaflux` Python extractor does, and passes the ones listed in `public/models/features.json` (12 for v1.1.0 models) to the model.

2. **Batch Inference (`src/components/Room.tsx`):**
   The model's descriptor columns for all grid points $[N \times F]$, with $F$ = 12 for v1.1.0 models, are flattened into a single `Float32Array` and wrapped inside an `onnxruntime-web` tensor.

3. **Concurrency & Thread Safety:**
   ONNX sessions (`OmegaFluxLow.onnx` and `OmegaFluxUp.onnx`) run sequentially using a global promise mutex queue. This prevents WebAssembly thread lock collisions during rapid slider movements and React Strict Mode re-mounts.

4. **Daylight Calculation:**
   $$\text{UDI}_{\text{useful}} = \max(0, 100 - \text{UDI}_{\text{low}} - \text{UDI}_{\text{up}})$$

---
