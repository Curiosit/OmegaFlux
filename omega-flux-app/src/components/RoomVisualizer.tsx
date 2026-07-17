import React, { useRef, useEffect } from 'react';
import type { RoomDimensions, WindowSpecs, GridPoint, ColorPalette } from '../types';

interface RoomVisualizerProps {
  room: RoomDimensions;
  window: WindowSpecs;
  gridData: GridPoint[];
  resolution: number;
  palette: ColorPalette;
}

// --- COLOR MATH ENGINE ---
const lerpColor = (c1: number[], c2: number[], t: number) => {
  return c1.map((v, i) => Math.round(v + (c2[i] - v) * t));
};

const getGradientColor = (value: number, palette: ColorPalette) => {
  const t = Math.max(0, Math.min(100, value)) / 100;
  let stops: { t: number; c: number[] }[] = [];

  if (palette === 'udi') {
    // Standard UDI: Red (0) -> Yellow (50) -> Green (100)
    stops = [
      { t: 0, c: [239, 68, 68] },
      { t: 0.5, c: [234, 179, 8] },
      { t: 1, c: [34, 197, 94] }
    ];
  } else if (palette === 'omega') {
    // Brand Theme: Navy (0) -> Accent Blue (50) -> Sunshine Yellow (100)
    stops = [
      { t: 0, c: [15, 23, 42] },
      { t: 0.5, c: [14, 165, 233] },
      { t: 1, c: [243, 239, 168] }
    ];
  } else {
    // Blueprint: Dark Blue (0) -> White (100)
    stops = [
      { t: 0, c: [15, 23, 42] },
      { t: 1, c: [255, 255, 255] }
    ];
  }

  for (let i = 0; i < stops.length - 1; i++) {
    if (t >= stops[i].t && t <= stops[i + 1].t) {
      const range = stops[i + 1].t - stops[i].t;
      const localT = (t - stops[i].t) / range;
      const blended = lerpColor(stops[i].c, stops[i + 1].c, localT);
      return `rgb(${blended[0]}, ${blended[1]}, ${blended[2]})`;
    }
  }
  const last = stops[stops.length - 1].c;
  return `rgb(${last[0]}, ${last[1]}, ${last[2]})`;
};

// --- LEGEND CSS GENERATOR ---
const getLegendGradient = (palette: ColorPalette) => {
  if (palette === 'udi') return 'linear-gradient(to right, rgb(239,68,68), rgb(234,179,8), rgb(34,197,94))';
  if (palette === 'omega') return 'linear-gradient(to right, rgb(15,23,42), rgb(14,165,233), rgb(243,239,168))';
  return 'linear-gradient(to right, rgb(15,23,42), rgb(255,255,255))';
};

export default function RoomVisualizer({ room, window, gridData, resolution, palette }: RoomVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const scale = 80;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = room.width * scale;
    canvas.height = room.depth * scale;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 1. Render Heatmap
    gridData.forEach((pt) => {
      ctx.fillStyle = getGradientColor(pt.udi, palette);
      ctx.fillRect(pt.x * scale, (room.depth - pt.y) * scale, resolution * scale, resolution * scale);
    });

// 2. Render Black Walls
    const wallThick = 8;
    ctx.fillStyle = '#000000'; 
    ctx.fillRect(0, 0, wallThick, canvas.height); // Left
    ctx.fillRect(canvas.width - wallThick, 0, wallThick, canvas.height); // Right
    ctx.fillRect(0, 0, canvas.width, wallThick); // Top
    ctx.fillRect(0, canvas.height - wallThick, canvas.width, wallThick); // Bottom
    
    // 3. Render Window (Make it slightly thicker than the wall so it 'pops')
    const windowThick = 12; 
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(
      (window.center - window.width / 2) * scale, 
      canvas.height - windowThick, // Position to overlay the wall
      window.width * scale, 
      windowThick
    );
    
    // Optional: Add a subtle border to the window to make it "snap" out
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.lineWidth = 1;
    ctx.strokeRect(
      (window.center - window.width / 2) * scale, 
      canvas.height - windowThick, 
      window.width * scale, 
      windowThick
    );
  }, [room, window, gridData, resolution, palette]);

  return (
    <div style={{ width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      {/* GRADIENT LEGEND */}
      <div style={{ width: '100%', maxWidth: `${room.width * scale}px`, marginBottom: '1.5rem' }}>
        <div style={{ 
          height: '12px', 
          width: '100%', 
          borderRadius: '6px', 
          background: getLegendGradient(palette),
          boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.1)'
        }} />
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '0.5rem', fontSize: '0.75rem', fontWeight: '600', color: 'var(--text-muted)' }}>
          <span>0% UDI</span>
          <span>50%</span>
          <span>100% UDI</span>
        </div>
      </div>

      {/* CANVAS */}
      <canvas
        ref={canvasRef}
        style={{
          backgroundColor: '#ffffff',
          boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)',
          borderRadius: '8px',
          maxWidth: '100%',
          height: 'auto'
        }}
      />
    </div>
  );
}