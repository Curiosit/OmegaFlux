import React, { useState, useMemo, useEffect } from 'react';
import * as ort from 'onnxruntime-web';
import { type RoomDimensions, type WindowSpecs, type WallType, type GridPoint, type ColorPalette, PALETTE_CONFIG } from '../types';
import { extractFeatures } from '../engine';
import Slider from './Slider';
import RoomVisualizer from './RoomVisualizer';

// Import lightweight JavaScript models (m2cgen export)
import { score as jsScoreLow } from './OmegaFluxLow.js';
import { score as jsScoreUp } from './OmegaFluxUp.js';

interface UIWindow extends WindowSpecs {
  id: string;
}

const WALLS: WallType[] = ['South', 'West', 'North', 'East'];

// Configure ONNX WASM path for browser execution
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@latest/dist/';

let ortSessionLow: ort.InferenceSession | null = null;
let ortSessionUp: ort.InferenceSession | null = null;

async function loadOnnxSessions() {
  if (!ortSessionLow) {
    ortSessionLow = await ort.InferenceSession.create('./OmegaFlux_Models/omegaflux_low.onnx');
  }
  if (!ortSessionUp) {
    ortSessionUp = await ort.InferenceSession.create('./OmegaFlux_Models/omegaflux_up.onnx');
  }
}

async function runPrediction(features: number[], modelType: 'js' | 'onnx'): Promise<[number, number]> {
  if (modelType === 'js') {
    const low = jsScoreLow(features);
    const up = jsScoreUp(features);
    return [low, up];
  } else {
    await loadOnnxSessions();
    const tensor = new ort.Tensor('float32', Float32Array.from(features), [1, features.length]);
    if (!ortSessionLow || !ortSessionUp) throw new Error("ONNX sessions failed to load.");

    const resultsLow = await ortSessionLow.run({ [ortSessionLow.inputNames[0]]: tensor });
    const resultsUp = await ortSessionUp.run({ [ortSessionUp.inputNames[0]]: tensor });

    const lowVal = Number(resultsLow[ortSessionLow.outputNames[0]].data[0]);
    const upVal = Number(resultsUp[ortSessionUp.outputNames[0]].data[0]);

    return [lowVal, upVal];
  }
}

export default function Room() {
  const [room, setRoom] = useState<RoomDimensions>({ width: 5.2, depth: 4.5, height: 3.0 });
  const [activeTab, setActiveTab] = useState<WallType>('South');
  
  // Master window array tracking all windows across all 4 walls
  const [windows, setWindows] = useState<UIWindow[]>([
    { id: 'default-1', wall: 'South', width: 2.0, center: 2.6, sill: 0.8, head: 2.3 }
  ]);

  const [palette, setPalette] = useState<ColorPalette>('omega');
  const [viewMode, setViewMode] = useState<'useful' | 'low' | 'up'>('useful');
  const [modelType, setModelType] = useState<'js' | 'onnx'>('js'); // Toggle between JS and ONNX models
  const [zoom, setZoom] = useState<number>(1.2); // Default zoom scale multiplier
  const [gridData, setGridData] = useState<GridPoint[]>([]);
  const [calcTime, setCalcTime] = useState<number>(0);
  const [avgUdi, setAvgUdi] = useState<number>(0);

  const resolution = 0.25;

  // Filter windows for the currently selected tab
  const activeWindows = useMemo(
    () => windows.filter((w) => w.wall === activeTab),
    [windows, activeTab]
  );

  const handleAddWindow = () => {
    if (activeWindows.length >= 3) return;
    const maxAxis = activeTab === 'South' || activeTab === 'North' ? room.width : room.depth;
    
    const newWindow: UIWindow = {
      id: crypto.randomUUID(),
      wall: activeTab,
      center: maxAxis / 2,
      width: 2.0,
      sill: 0.8,
      head: 2.3,
    };
    setWindows([...windows, newWindow]);
  };

  const handleRemoveWindow = (id: string) => {
    setWindows(windows.filter((w) => w.id !== id));
  };

  const handleWindowChange = (id: string, field: keyof WindowSpecs, value: number) => {
    setWindows(windows.map((w) => (w.id === id ? { ...w, [field]: value } : w)));
  };

  // Run spatial inference loop whenever room dimensions, windows configuration, viewMode, or modelType changes
  useEffect(() => {
    let isMounted = true;
    const runSimulation = async () => {
      const start = performance.now();
      const data: GridPoint[] = [];
      let totalUdi = 0;

      for (let x = 0.13; x < room.width; x += resolution) {
        for (let y = 0.13; y < room.depth; y += resolution) {
          if (!isMounted) return;
          const features = extractFeatures(x, y, room, windows);
          const [low, up] = await runPrediction(features, modelType);
          const useful = Math.max(0, 100 - low - up);

          const value = viewMode === 'useful' ? useful : (viewMode === 'low' ? low : up);
          data.push({ x, y, udi: value });
          totalUdi += value;
        }
      }
      
      if (!isMounted) return;
      const end = performance.now();
      setGridData(data);
      setCalcTime(end - start);
      setAvgUdi(data.length > 0 ? totalUdi / data.length : 0);
    };

    runSimulation();

    return () => {
      isMounted = false;
    };
  }, [room, windows, resolution, viewMode, modelType]);

  return (
    <div style={{ padding: '2rem', maxWidth: '1200px', margin: '0 auto' }}>
      
      {/* HEADER */}
      <header style={{ 
        display: 'flex', alignItems: 'center', gap: '1.5rem', marginBottom: '2.5rem', 
        borderBottom: '4px solid var(--color-secondary)', paddingBottom: '1.5rem' 
      }}>
        <img src="/web-app-manifest-192x192.png" alt="OmegaFlux Logo" style={{ width: '80px', height: '80px', objectFit: 'contain' }} />
        <div>
          <h1 style={{ fontWeight: '800', fontSize: '2.25rem', margin: 0, color: 'var(--color-primary)' }}>
            ΩΦ OmegaFlux Web Engine
          </h1>
          <p style={{ color: 'var(--text-muted)', marginTop: '0.5rem', marginBottom: 0, fontSize: '1.1rem' }}>
            Multi-Sided Daylighting Optimization Surrogate Model via Physics-Informed Spatial Extraction.
          </p>
        </div>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: '340px 1fr', gap: '2.5rem', alignItems: 'start' }}>
        
        {/* SIDEBAR CONTROLS */}
        <aside style={{ backgroundColor: 'var(--bg-panel)', padding: '1.5rem', borderRadius: '12px', border: '1px solid var(--border-light)', borderTop: '4px solid var(--color-secondary)' }}>
          
          <h2 style={{ fontSize: '1.25rem', fontWeight: '700', marginBottom: '1.25rem', borderBottom: '2px solid var(--border-medium)', paddingBottom: '0.5rem', color: 'var(--color-primary)' }}>
            Room Topology
          </h2>
          <Slider label="Room Width (m)" min={3.0} max={12.0} step={0.1} value={room.width} onChange={(w) => setRoom({ ...room, width: w })} />
          <Slider label="Room Depth (m)" min={3.0} max={12.0} step={0.1} value={room.depth} onChange={(d) => setRoom({ ...room, depth: d })} />
          
          {/* WALL TABS */}
          <h2 style={{ fontSize: '1.25rem', fontWeight: '700', marginTop: '2rem', marginBottom: '1rem', color: 'var(--color-primary)' }}>
            Fenestration Walls
          </h2>
          <div style={{ display: 'flex', gap: '0.25rem', marginBottom: '1.25rem', borderBottom: '2px solid var(--border-medium)', paddingBottom: '0.5rem' }}>
            {WALLS.map((wall) => {
              const count = windows.filter(w => w.wall === wall).length;
              return (
                <button
                  key={wall}
                  onClick={() => setActiveTab(wall)}
                  style={{
                    flex: 1, padding: '0.4rem 0.2rem', fontSize: '0.8rem',
                    backgroundColor: activeTab === wall ? 'var(--color-secondary)' : '#f3f4f6',
                    color: activeTab === wall ? 'var(--color-primary)' : '#4b5563',
                    border: 'none', borderRadius: '4px', fontWeight: 'bold', cursor: 'pointer'
                  }}
                >
                  {wall} ({count})
                </button>
              );
            })}
          </div>

          {/* ACTIVE WALL WINDOW SLIDERS */}
          <div style={{ minHeight: '180px' }}>
            {activeWindows.length === 0 ? (
              <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)', fontStyle: 'italic', textAlign: 'center', padding: '1rem 0' }}>
                No windows on {activeTab} wall.
              </p>
            ) : (
              activeWindows.map((win, idx) => {
                const maxAxis = activeTab === 'South' || activeTab === 'North' ? room.width : room.depth;
                return (
                  <div key={win.id} style={{ marginBottom: '1.5rem', paddingBottom: '1rem', borderBottom: '1px dashed var(--border-medium)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                      <span style={{ fontSize: '0.9rem', fontWeight: 'bold', color: 'var(--color-primary)' }}>{activeTab} Window {idx + 1}</span>
                      <button onClick={() => handleRemoveWindow(win.id)} style={{ color: '#ef4444', background: 'none', border: 'none', cursor: 'pointer', fontSize: '0.8rem' }}>Remove</button>
                    </div>
                    <Slider label="Width (m)" min={0.5} max={maxAxis} step={0.1} value={win.width} onChange={(w) => handleWindowChange(win.id, 'width', w)} />
                    <Slider label="Center (m)" min={0} max={maxAxis} step={0.1} value={win.center} onChange={(c) => handleWindowChange(win.id, 'center', c)} />
                    <Slider label="Head (m)" min={1.0} max={2.7} step={0.1} value={win.head} onChange={(h) => handleWindowChange(win.id, 'head', h)} />
                    <Slider label="Sill (m)" min={0.0} max={2.0} step={0.1} value={win.sill} onChange={(s) => handleWindowChange(win.id, 'sill', s)} />
                  </div>
                );
              })
            )}

            {activeWindows.length < 3 && (
              <button 
                onClick={handleAddWindow}
                style={{ width: '100%', padding: '0.6rem', border: '2px dashed var(--border-medium)', background: 'none', borderRadius: '6px', color: 'var(--text-muted)', fontWeight: '600', cursor: 'pointer', fontSize: '0.9rem' }}
              >
                + Add Window to {activeTab}
              </button>
            )}
          </div>

          {/* DATA METRIC */}
          <h2 style={{ fontSize: '1.25rem', fontWeight: '700', marginTop: '2rem', marginBottom: '1rem', borderBottom: '2px solid var(--border-medium)', paddingBottom: '0.5rem', color: 'var(--color-primary)' }}>
            Data Metric
          </h2>
          <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1.5rem' }}>
            {(['useful', 'low', 'up'] as const).map((mode) => (
              <button
                key={mode}
                onClick={() => setViewMode(mode)}
                style={{
                  flex: 1, padding: '0.5rem',
                  backgroundColor: viewMode === mode ? 'var(--color-secondary)' : '#f3f4f6',
                  border: 'none', borderRadius: '6px', fontWeight: 'bold', cursor: 'pointer', textTransform: 'capitalize', fontSize: '0.85rem'
                }}
              >
                {mode}
              </button>
            ))}
          </div>

          {/* MODEL RUNNER SELECTION */}
          <h2 style={{ fontSize: '1.25rem', fontWeight: '700', marginTop: '1.5rem', marginBottom: '1rem', borderBottom: '2px solid var(--border-medium)', paddingBottom: '0.5rem', color: 'var(--color-primary)' }}>
            Inference Engine
          </h2>
          <select 
            value={modelType} 
            onChange={(e) => setModelType(e.target.value as 'js' | 'onnx')}
            style={{ width: '100%', padding: '0.6rem', borderRadius: '6px', border: '1px solid var(--border-medium)', backgroundColor: '#ffffff', color: 'var(--color-primary)', fontWeight: '600', cursor: 'pointer', outline: 'none', marginBottom: '1.5rem' }}
          >
            <option value="onnx">ONNX Runtime (Large Model)</option>
            <option value="js">JavaScript m2cgen (Shallow Model)</option>
          </select>

          {/* COLOR GRADIENT */}
          <h2 style={{ fontSize: '1.25rem', fontWeight: '700', marginTop: '0.5rem', marginBottom: '1rem', borderBottom: '2px solid var(--border-medium)', paddingBottom: '0.5rem', color: 'var(--color-primary)' }}>
            Visualization Palette
          </h2>
          <select 
            value={palette} 
            onChange={(e) => setPalette(e.target.value as ColorPalette)}
            style={{ width: '100%', padding: '0.6rem', borderRadius: '6px', border: '1px solid var(--border-medium)', backgroundColor: '#ffffff', color: 'var(--color-primary)', fontWeight: '600', cursor: 'pointer', outline: 'none' }}
          >
            <option value="omega">Omega Flux</option>
            <option value="udi">Classic UDI</option>
            <option value="blueprint">Blueprint Monochromatic</option>
          </select>

        </aside>

        {/* MAIN VISUALIZER CANVAS */}
        <main style={{ 
          display: 'flex', flexDirection: 'column', alignItems: 'center', 
          backgroundColor: '#ffffff', borderRadius: '12px', padding: '1.5rem', 
          border: '1px solid var(--border-light)', boxShadow: '0 1px 3px rgba(0,0,0,0.05)'
        }}>
          <div style={{ marginBottom: '1.5rem', width: '100%', display: 'flex', justifyContent: 'space-between', alignItems: 'center', color: 'var(--text-muted)', fontSize: '0.875rem', fontWeight: '600', textTransform: 'uppercase' }}>
            <span>
              Plan View: {viewMode === 'useful' ? 'Useful Daylight (UDI)' : viewMode === 'low' ? 'UDI Under-lit' : 'UDI Glare (Up)'}
            </span>
            
            <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
              {/* --- ZOOM CONTROLS --- */}
              <div style={{ display: 'flex', alignItems: 'center', backgroundColor: '#f3f4f6', borderRadius: '8px', padding: '0.2rem' }}>
                <button 
                  onClick={() => setZoom(Math.max(0.5, zoom - 0.2))} 
                  title="Zoom Out"
                  style={{ background: 'none', border: 'none', padding: '0.2rem 0.6rem', cursor: 'pointer', fontWeight: 'bold', fontSize: '1rem', color: '#374151' }}
                >
                  -
                </button>
                <span style={{ fontSize: '0.75rem', fontWeight: '700', padding: '0 0.4rem', color: '#374151' }}>
                  {Math.round(zoom * 100)}%
                </span>
                <button 
                  onClick={() => setZoom(Math.min(3.0, zoom + 0.2))} 
                  title="Zoom In"
                  style={{ background: 'none', border: 'none', padding: '0.2rem 0.6rem', cursor: 'pointer', fontWeight: 'bold', fontSize: '1rem', color: '#374151' }}
                >
                  +
                </button>
              </div>

              <span style={{ backgroundColor: '#eef2ff', color: '#4f46e5', padding: '0.35rem 0.85rem', borderRadius: '9999px', fontWeight: '700' }}>
                Avg UDI: {avgUdi.toFixed(1)}%
              </span>
              <span style={{ backgroundColor: '#f3f4f6', color: '#374151', padding: '0.35rem 0.85rem', borderRadius: '9999px', fontWeight: '700' }}>
                {calcTime.toFixed(0)} ms
              </span>
              <span style={{ backgroundColor: 'var(--color-secondary)', color: 'var(--color-primary)', padding: '0.35rem 0.85rem', borderRadius: '9999px', fontWeight: '700' }}>
                Sensors: {gridData.length}
              </span>

            </div>
          </div>
         {/* --- COLOR GRADIENT LEGEND --- */}
          <div style={{ width: '100%', marginTop: '1.5rem', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <div style={{ fontSize: '0.8rem', fontWeight: '600', color: 'var(--text-muted)', marginBottom: '0.4rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              {PALETTE_CONFIG[palette].name} Scale (%)
            </div>
            
            <div style={{
              width: '100%',
              maxWidth: '400px',
              height: '14px',
              borderRadius: '7px',
              background: PALETTE_CONFIG[palette].cssGradient,
              boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.1)',
              border: '1px solid var(--border-medium)'
            }} />

            <div style={{ width: '100%', maxWidth: '400px', display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: '600', marginTop: '0.25rem' }}>
              <span>0%</span>
              <span>50%</span>
              <span>100%</span>
            </div>
          </div> 
          <RoomVisualizer room={room} windows={windows} gridData={gridData} resolution={resolution} palette={palette} zoom={zoom} />
        
        </main>

      </div>
    </div>
  );
}