import React, { useState, useMemo, useEffect } from 'react';
import type { RoomDimensions, WindowSpecs, GridPoint, ColorPalette } from '../types';
import { extractFeatures, predictOmegaFlux } from '../engine';
import Slider from './Slider';
import RoomVisualizer from './RoomVisualizer';

export default function Room() {
  const [room, setRoom] = useState<RoomDimensions>({ width: 5.2, depth: 4.5, height: 3.0 });
  const [win, setWin] = useState<WindowSpecs>({ width: 2.0, center: 2.6, sill: 0.8, head: 2.3 });
  
  // NEW: Palette State
  const [palette, setPalette] = useState<ColorPalette>('omega');
  
  const resolution = 0.25;

const [gridData, setGridData] = useState<GridPoint[]>([]);
const [calcTime, setCalcTime] = useState<number>(0);
const [avgUdi, setAvgUdi] = useState<number>(0); // NEW: State for average

useEffect(() => {
  const start = performance.now();
  
  const data: GridPoint[] = [];
  let totalUdi = 0; // Running total

  for (let x = 0.13; x < room.width; x += resolution) {
    for (let y = 0.13; y < room.depth; y += resolution) {
      const features = extractFeatures(x, y, room, win);
      const predictedUDI = predictOmegaFlux(features);
      data.push({ x, y, udi: predictedUDI });
      totalUdi += predictedUDI; // Accumulate
    }
  }
  
  const end = performance.now();
  setGridData(data);
  setCalcTime(end - start);
  // Calculate average, guard against division by zero
  setAvgUdi(data.length > 0 ? totalUdi / data.length : 0);
}, [room, win, resolution]);

  return (
    <div style={{ padding: '2rem', maxWidth: '1100px', margin: '0 auto' }}>
      
      {/* HEADER */}
      <header style={{ 
        display: 'flex', 
        alignItems: 'center', 
        gap: '1.5rem', 
        marginBottom: '2.5rem', 
        borderBottom: '4px solid var(--color-secondary)', 
        paddingBottom: '1.5rem' 
      }}>
        <img
          src="/web-app-manifest-192x192.png" 
          alt="OmegaFlux Logo" 
          style={{ width: '80px', height: '80px', objectFit: 'contain' }} 
        />
        
        <div>
          <h1 style={{ fontWeight: '800', fontSize: '2.25rem', margin: 0, color: 'var(--color-primary)' }}>
            ΩΦ OmegaFlux Web Engine
          </h1>
          <p style={{ color: 'var(--text-muted)', marginTop: '0.5rem', marginBottom: 0, fontSize: '1.1rem' }}>
            A Grid-Point Shape Agnostic Machine Learning Surrogate Model OmegaFlux for Multi-Sided Daylighting Optimization: Bypassing the Curse of Dimensionality via Physics-Informed Spatial Extraction and Component-Based Target Mapping.
          </p>
        </div>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: '300px 1fr', gap: '3rem', alignItems: 'start' }}>
        
        <aside style={{ backgroundColor: 'var(--bg-panel)', padding: '1.5rem', borderRadius: '12px', border: '1px solid var(--border-light)', borderTop: '4px solid var(--color-secondary)' }}>
          <h2 style={{ fontSize: '1.25rem', fontWeight: '700', marginBottom: '1.5rem', borderBottom: '2px solid var(--border-medium)', paddingBottom: '0.5rem', color: 'var(--color-primary)' }}>Room Topology</h2>
          <Slider label="Room Width (m)" min={3.0} max={8.0} step={0.1} value={room.width} onChange={(w) => setRoom({ ...room, width: w })} />
          <Slider label="Room Depth (m)" min={3.0} max={8.0} step={0.1} value={room.depth} onChange={(d) => setRoom({ ...room, depth: d })} />

          <h2 style={{ fontSize: '1.25rem', fontWeight: '700', marginTop: '2rem', marginBottom: '1.5rem', borderBottom: '2px solid var(--border-medium)', paddingBottom: '0.5rem', color: 'var(--color-primary)' }}>Front Fenestration</h2>
          <Slider label="Window Width (m)" min={0} max={room.width} step={0.1} value={win.width} onChange={(w) => setWin({ ...win, width: w })} />
          <Slider label="Center X (m)" min={0} max={room.width} step={0.1} value={win.center} onChange={(c) => setWin({ ...win, center: c })} />
          <Slider label="Head Height (m)" min={1.0} max={room.height} step={0.1} value={win.head} onChange={(h) => setWin({ ...win, head: h })} />
          <Slider label="Sill Height (m)" min={0.0} max={2.0} step={0.1} value={win.sill} onChange={(s) => setWin({ ...win, sill: s })} />

          {/* NEW: VISUALIZATION SETTINGS */}
          <h2 style={{ fontSize: '1.25rem', fontWeight: '700', marginTop: '2rem', marginBottom: '1rem', borderBottom: '2px solid var(--border-medium)', paddingBottom: '0.5rem', color: 'var(--color-primary)' }}>
            Visualization
          </h2>
          <label style={{ display: 'block', fontSize: '0.9rem', fontWeight: '500', color: 'var(--text-main)' }}>
            <div style={{ marginBottom: '0.5rem' }}>Color Gradient</div>
            <select 
              value={palette} 
              onChange={(e) => setPalette(e.target.value as ColorPalette)}
              style={{
                width: '100%',
                padding: '0.6rem',
                borderRadius: '6px',
                border: '1px solid var(--border-medium)',
                backgroundColor: '#ffffff',
                color: 'var(--color-primary)',
                fontWeight: '600',
                cursor: 'pointer',
                outline: 'none'
              }}
            >
              <option value="omega">Omega Brand Theme</option>
              <option value="udi">Classic UDI (Traffic Light)</option>
              <option value="blueprint">Blueprint Monochromatic</option>
            </select>
          </label>
        </aside>

        <main style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', backgroundColor: 'var(--bg-canvas-container)', borderRadius: '12px', padding: '2rem', border: '1px solid var(--border-light)' }}>
          <div style={{ marginBottom: '2rem', width: '100%', display: 'flex', justifyContent: 'space-between', alignItems: 'center', color: 'var(--text-muted)', fontSize: '0.875rem', fontWeight: '600', textTransform: 'uppercase' }}>
            <span>Plan View (Real-time Prediction)</span>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              {/* AVG UDI PILL */}
              <span style={{ backgroundColor: '#eef2ff', color: '#4f46e5', padding: '0.35rem 0.85rem', borderRadius: '9999px', fontWeight: '700' }}>
                Avg UDI: {avgUdi.toFixed(1)}%
              </span>

              {/* TIME PILL */}
              <span style={{ backgroundColor: '#f3f4f6', color: '#374151', padding: '0.35rem 0.85rem', borderRadius: '9999px', fontWeight: '700' }}>
                {calcTime.toFixed(0)} ms
              </span>

              {/* COUNT PILL */}
              <span style={{ backgroundColor: 'var(--color-secondary)', color: 'var(--color-primary)', padding: '0.35rem 0.85rem', borderRadius: '9999px', fontWeight: '700', boxShadow: '0 1px 2px rgba(0,0,0,0.05)' }}>
                Sensors: {gridData.length}
              </span>
            </div>
          </div>
          
          <RoomVisualizer room={room} window={win} gridData={gridData} resolution={resolution} palette={palette} />
        </main>

      </div>
    </div>
  );
}