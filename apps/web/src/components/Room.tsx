import { useState, useMemo, useEffect, useRef } from 'react';
import * as ort from 'onnxruntime-web';
import {
  type RoomDimensions,
  type WindowSpecs,
  type WallType,
  type GridPoint,
  type ColorPalette,
  PALETTE_CONFIG
} from '../types';

import { extractFeatures, FEATURE_NAMES } from '../engine';
import Slider from './Slider';
import RoomVisualizer from './RoomVisualizer';

interface UIWindow extends WindowSpecs {
  id: string;
}

const WALLS: WallType[] = ['South', 'West', 'North', 'East'];

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/';
ort.env.wasm.numThreads = 1;

let ortSessionLow: ort.InferenceSession | null = null;
let ortSessionUp: ort.InferenceSession | null = null;
// Positions (within the 26 engine features) of the columns the loaded model expects.
let modelFeatureIdx: number[] | null = null;

async function loadFeatureIndices(path: string): Promise<number[]> {
  const res = await fetch(path);
  if (!res.ok) {
    // v1.0.0 models shipped without features.json and used all 26 descriptors.
    console.warn(`${path} not found; assuming all ${FEATURE_NAMES.length} features.`);
    return FEATURE_NAMES.map((_, i) => i);
  }
  const spec = await res.json() as { features: string[] };
  return spec.features.map((name) => {
    const i = (FEATURE_NAMES as readonly string[]).indexOf(name);
    if (i < 0) throw new Error(`Model feature "${name}" is not computed by engine.ts.`);
    return i;
  });
}

const BASE = import.meta.env.BASE_URL;   // '/OmegaFlux/'

let inferenceQueue: Promise<unknown> = Promise.resolve();

async function loadSingleModel(path: string): Promise<ort.InferenceSession> {
  const res = await fetch(path);

  if (!res.ok) {
    throw new Error(
      `Failed to fetch ${path}: ${res.status} ${res.statusText}`
    );
  }

  const buffer = await res.arrayBuffer();

  if (buffer.byteLength === 0) {
    throw new Error(`${path} is empty.`);
  }

  const text = new TextDecoder().decode(buffer.slice(0, 64));
  if (
    text.startsWith('<!DOCTYPE') ||
    text.startsWith('<html') ||
    text.includes('<head')
  ) {
    throw new Error(
      `${path} is not an ONNX model. Server returned HTML instead.`
    );
  }

  return await ort.InferenceSession.create(buffer);
}


async function predictGrid(
  featureBatch: number[][]
): Promise<Array<[number, number]>> {
  const currentTask = new Promise<Array<[number, number]>>((resolve, reject) => {
    inferenceQueue = inferenceQueue.then(async () => {
      try {
        if (!ortSessionLow || !ortSessionUp) {
          throw new Error('ONNX sessions are not initialized.');
        }

        const idx = modelFeatureIdx ?? FEATURE_NAMES.map((_, i) => i);
        const batch = featureBatch.map((row) => idx.map((i) => row[i]));
        const numSensors = batch.length;
        const numFeatures = idx.length;
        const flatFeatures = new Float32Array(numSensors * numFeatures);

        for (let i = 0; i < numSensors; i++) {
          flatFeatures.set(batch[i], i * numFeatures);
        }

        const tensor = new ort.Tensor('float32', flatFeatures, [numSensors, numFeatures]);

        // Sequential execution avoids WASM thread collisions
        const lowOutput = await ortSessionLow.run({ [ortSessionLow.inputNames[0]]: tensor });
        const upOutput = await ortSessionUp.run({ [ortSessionUp.inputNames[0]]: tensor });

        const lowData = lowOutput[ortSessionLow.outputNames[0]].data as Float32Array;
        const upData = upOutput[ortSessionUp.outputNames[0]].data as Float32Array;

        const results: Array<[number, number]> = [];
        for (let i = 0; i < numSensors; i++) {
          const lowVal = Math.max(0, Math.min(100, lowData[i]));
          const upVal = Math.max(0, Math.min(100, upData[i]));
          results.push([lowVal, upVal]);
        }

        resolve(results);
      } catch (err) {
        reject(err);
      }
    });
  });

  return currentTask;
}

export default function Room() {
  const [room, setRoom] = useState<RoomDimensions>({ width: 5.2, depth: 4.5 });
  const [activeTab, setActiveTab] = useState<WallType>('South');

  const [windows, setWindows] = useState<UIWindow[]>([
    { id: 'default-1', wall: 'South', width: 2.0, center: 2.6, sill: 0.8, head: 2.3 }
  ]);

  const [palette, setPalette] = useState<ColorPalette>('omega');
  const [viewMode, setViewMode] = useState<'useful' | 'low' | 'up'>('useful');
  const [zoom, setZoom] = useState<number>(1.2);
  const [gridData, setGridData] = useState<GridPoint[]>([]);
  const [calcTime, setCalcTime] = useState<number>(0);
  const [avgUdi, setAvgUdi] = useState<number>(0);


  const [isModelReady, setIsModelReady] = useState<boolean>(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  const resolution = 0.25;


  const isInitializing = useRef<boolean>(false);
  const currentRunId = useRef<number>(0);


  useEffect(() => {
    if (ortSessionLow && ortSessionUp) {
      setIsModelReady(true);
      return;
    }

    if (isInitializing.current) return;
    isInitializing.current = true;

    async function initModels() {
      try {
        console.log('Initializing ONNX models on mount...');
        const [low, up, idx] = await Promise.all([
          loadSingleModel(`${BASE}models/OmegaFluxLow.onnx`),
          loadSingleModel(`${BASE}models/OmegaFluxUp.onnx`),
          loadFeatureIndices(`${BASE}models/features.json`)
        ]);
        modelFeatureIdx = idx;

        ortSessionLow = low;
        ortSessionUp = up;

        console.log('✓ ONNX models loaded and ready.');
        setIsModelReady(true);
      } catch (err: any) {
        console.error('Failed to initialize ONNX models:', err);
        setLoadError(err.message || 'Failed to load ONNX models.');
      }
    }

    initModels();
  }, []);

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


  useEffect(() => {

    if (!isModelReady) return;

    let isMounted = true;
    const runId = ++currentRunId.current;

    const runSimulation = async () => {
      const start = performance.now();
      const coords: Array<{ x: number; y: number }> = [];
      const featureBatch: number[][] = [];


      for (let x = 0.13; x < room.width; x += resolution) {
        for (let y = 0.13; y < room.depth; y += resolution) {
          coords.push({ x, y });
          featureBatch.push(extractFeatures(x, y, room, windows));
        }
      }

      if (coords.length === 0) return;

      try {
        const predictions = await predictGrid(featureBatch);


        if (!isMounted || runId !== currentRunId.current) return;

        const data: GridPoint[] = [];
        let totalUdi = 0;

        for (let i = 0; i < coords.length; i++) {
          const [low, up] = predictions[i];
          const useful = Math.max(0, 100 - low - up);
          const value = viewMode === 'useful' ? useful : (viewMode === 'low' ? low : up);

          data.push({ x: coords[i].x, y: coords[i].y, udi: value });
          totalUdi += value;
        }

        const end = performance.now();
        setGridData(data);
        setCalcTime(end - start);
        setAvgUdi(data.length > 0 ? totalUdi / data.length : 0);
      } catch (err) {
        if (isMounted && runId === currentRunId.current) {
          console.error('Simulation run failed:', err);
        }
      }
    };

    runSimulation();

    return () => {
      isMounted = false;
    };
  }, [isModelReady, room, windows, resolution, viewMode]);

  return (
    <div style={{ padding: '2rem', maxWidth: '1200px', margin: '0 auto' }}>
      
      <header style={{ 
        display: 'flex', alignItems: 'center', gap: '1.5rem', marginBottom: '2.5rem', 
        borderBottom: '4px solid var(--color-secondary)', paddingBottom: '1.5rem' 
      }}>
        <img src={`${import.meta.env.BASE_URL}web-app-manifest-192x192.png`} alt="OmegaFlux Logo" style={{ width: '80px', height: '80px', objectFit: 'contain' }} />
        <div>
          <h1 style={{ fontWeight: '800', fontSize: '2.25rem', margin: 0, color: 'var(--color-primary)' }}>
            ΩΦ OmegaFlux Web Engine
          </h1>
          <p style={{ color: 'var(--text-muted)', marginTop: '0.5rem', marginBottom: 0, fontSize: '1.5rem' }}>
            Daylight prediction without neural networks. Online daylight prediction check.
          </p>
          <p style={{ color: 'var(--text-muted)', fontWeight: '600', marginTop: '0.5rem', marginBottom: 0, fontSize: '1.1rem' }}>
          Analytic radiometric descriptors enable geometry-flexible per-sensor daylight prediction without learned spatial encodings
          </p>
          <p style={{ color: 'var(--text-muted)', marginTop: '0.5rem', marginBottom: 0, fontSize: '1.1rem' }}>
            <i>Tabular data is all you need!</i>
          </p>
        </div>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: '340px 1fr', gap: '2.5rem', alignItems: 'start' }}>
        

        <aside style={{ backgroundColor: 'var(--bg-panel)', padding: '1.5rem', borderRadius: '12px', border: '1px solid var(--border-light)', borderTop: '4px solid var(--color-secondary)' }}>
          
          <h2 style={{ fontSize: '1.25rem', fontWeight: '700', marginBottom: '1.25rem', borderBottom: '2px solid var(--border-medium)', paddingBottom: '0.5rem', color: 'var(--color-primary)' }}>
            Room Topology
          </h2>
          <Slider label="Room Width (m)" min={3.0} max={12.0} step={0.1} value={room.width} onChange={(w) => setRoom({ ...room, width: w })} />
          <Slider label="Room Depth (m)" min={3.0} max={12.0} step={0.1} value={room.depth} onChange={(d) => setRoom({ ...room, depth: d })} />
          

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

              {!isModelReady && (
                <span style={{ backgroundColor: '#fef3c7', color: '#d97706', padding: '0.35rem 0.85rem', borderRadius: '9999px', fontWeight: '700' }}>
                  {loadError ? 'Model Error' : 'Loading ONNX...'}
                </span>
              )}

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