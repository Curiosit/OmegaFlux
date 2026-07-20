import React, { useRef, useEffect, useState } from 'react';
import type { RoomDimensions, WindowSpecs, GridPoint, ColorPalette } from '../types';
import { PALETTE_CONFIG } from '../types';

interface UIWindow extends WindowSpecs {
  id: string;
}

interface RoomVisualizerProps {
  room: RoomDimensions;
  windows: UIWindow[];
  gridData: GridPoint[];
  resolution: number;
  palette: ColorPalette;
  zoom: number;
}

export default function RoomVisualizer({ room, windows, gridData, resolution, palette, zoom }: RoomVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [showFacades, setShowFacades] = useState<boolean>(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const baseScale = 55; 
    const currentZoom = zoom || 1;
    const scale = baseScale * currentZoom; 
    
    const wallLineWidth = Math.max(6, 8 * currentZoom);
    const halfLine = wallLineWidth / 2;
    const roomHeightPx = (room.height || 2.7) * scale;
    const facadeOffset = 30 + halfLine;

    // Dynamically calculate padding based on room height + facade offset so nothing ever clips
    const padding = showFacades 
      ? Math.ceil(facadeOffset + roomHeightPx + 50) 
      : Math.round(50 * Math.max(1, currentZoom * 0.5));
    
    const roomPixelWidth = room.width * scale;
    const roomPixelDepth = room.depth * scale;

    canvas.width = roomPixelWidth + padding * 2;
    canvas.height = roomPixelDepth + padding * 2;

    // Clear canvas background
    ctx.fillStyle = '#f8fafc';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const originX = padding;
    const originY = padding;
    const activePalette = PALETTE_CONFIG[palette] || PALETTE_CONFIG.omega;

    // 1. Room Background Fill (Analytical Grid Area)
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(originX, originY, roomPixelWidth, roomPixelDepth);

    // 2. Grid Heatmap Points
    if (gridData && gridData.length > 0) {
      gridData.forEach((point) => {
        const cx = originX + point.x * scale;
        const cy = originY + (room.depth - point.y) * scale;
        const size = Math.max(2, resolution * scale);

        const val = Math.max(0, Math.min(100, point.udi));
        ctx.fillStyle = activePalette.getColor(val);
        ctx.fillRect(cx - size / 2, cy - size / 2, size, size);
      });
    }

    // 3. Thick Room Wall Outline Box - Extended Fully Outward
    ctx.strokeStyle = '#0f172a';
    ctx.lineWidth = wallLineWidth;
    ctx.strokeRect(
      originX - halfLine, 
      originY - halfLine, 
      roomPixelWidth + wallLineWidth, 
      roomPixelDepth + wallLineWidth
    );

    // 4. Draw Windows as White Openings fully inside the black wall stroke thickness
    if (windows && windows.length > 0) {
      windows.forEach((win) => {
        const halfW = (win.width || 1) / 2;
        const center = win.center || 0;

        ctx.strokeStyle = '#ffffff'; 
        ctx.lineWidth = wallLineWidth;
        ctx.beginPath();

        if (win.wall === 'South') {
          const wx1 = originX + (center - halfW) * scale;
          const wx2 = originX + (center + halfW) * scale;
          const wy = originY + roomPixelDepth + halfLine;
          ctx.moveTo(wx1, wy);
          ctx.lineTo(wx2, wy);
        } else if (win.wall === 'North') {
          const wx1 = originX + (center - halfW) * scale;
          const wx2 = originX + (center + halfW) * scale;
          const wy = originY - halfLine;
          ctx.moveTo(wx1, wy);
          ctx.lineTo(wx2, wy);
        } else if (win.wall === 'West') {
          const wx = originX - halfLine;
          const wy1 = originY + roomPixelDepth - (center - halfW) * scale;
          const wy2 = originY + roomPixelDepth - (center + halfW) * scale;
          ctx.moveTo(wx, wy1);
          ctx.lineTo(wx, wy2);
        } else if (win.wall === 'East') {
          const wx = originX + roomPixelWidth + halfLine;
          const wy1 = originY + roomPixelDepth - (center - halfW) * scale;
          const wy2 = originY + roomPixelDepth - (center + halfW) * scale;
          ctx.moveTo(wx, wy1);
          ctx.lineTo(wx, wy2);
        }
        ctx.stroke();
      });
    }

    // 5. Folded Flat Facades
    if (showFacades) {
      // South Facade (Below room)
      const southX = originX;
      const southY = originY + roomPixelDepth + facadeOffset;
      ctx.strokeStyle = '#cbd5e1';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(southX, southY, roomPixelWidth, roomHeightPx);
      ctx.fillStyle = '#f1f5f9';
      ctx.fillRect(southX, southY, roomPixelWidth, roomHeightPx);

      // North Facade (Above room)
      const northX = originX;
      const northY = originY - facadeOffset - roomHeightPx;
      ctx.strokeStyle = '#cbd5e1';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(northX, northY, roomPixelWidth, roomHeightPx);
      ctx.fillStyle = '#f1f5f9';
      ctx.fillRect(northX, northY, roomPixelWidth, roomHeightPx);

      // West Facade (Left)
      const westX = originX - facadeOffset - roomHeightPx;
      const westY = originY;
      ctx.strokeStyle = '#cbd5e1';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(westX, westY, roomHeightPx, roomPixelDepth);
      ctx.fillStyle = '#f1f5f9';
      ctx.fillRect(westX, westY, roomHeightPx, roomPixelDepth);

      // East Facade (Right)
      const eastX = originX + roomPixelWidth + facadeOffset;
      const eastY = originY;
      ctx.strokeStyle = '#cbd5e1';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(eastX, eastY, roomHeightPx, roomPixelDepth);
      ctx.fillStyle = '#f1f5f9';
      ctx.fillRect(eastX, eastY, roomHeightPx, roomPixelDepth);

      // Render Windows on Facades
      windows.forEach((win) => {
        const halfW = (win.width || 1) / 2;
        const center = win.center || 0;
        
        const sillPx = (win.sill || 0) * scale;
        const headPx = (win.head || 2.1) * scale;
        const winH = Math.max(5, headPx - sillPx); 
        const winW = (win.width || 1.0) * scale;

        ctx.fillStyle = '#93c5fd';
        ctx.strokeStyle = '#1e3a8a';
        ctx.lineWidth = 1;

        if (win.wall === 'South') {
          const wx = southX + (center - halfW) * scale;
          const wy = (southY + roomHeightPx) - sillPx - winH;
          ctx.fillRect(wx, wy, winW, winH);
          ctx.strokeRect(wx, wy, winW, winH);
        } else if (win.wall === 'North') {
          const wx = northX + (center - halfW) * scale;
          const wy = (northY + roomHeightPx) - sillPx - winH;
          ctx.fillRect(wx, wy, winW, winH);
          ctx.strokeRect(wx, wy, winW, winH);
        } else if (win.wall === 'West') {
          const wy = westY + roomPixelDepth - (center + halfW) * scale;
          const wx = westX + sillPx; 
          ctx.fillRect(wx, wy, winH, winW);
          ctx.strokeRect(wx, wy, winH, winW);
        } else if (win.wall === 'East') {
          const wy = eastY + roomPixelDepth - (center + halfW) * scale;
          const wx = (eastX + roomHeightPx) - sillPx - winH;
          ctx.fillRect(wx, wy, winH, winW);
          ctx.strokeRect(wx, wy, winH, winW);
        }
      });
    }

  }, [room, windows, gridData, resolution, palette, zoom, showFacades]);

  return (
    <div style={{ width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <div style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <button
          onClick={() => setShowFacades(!showFacades)}
          style={{
            backgroundColor: showFacades ? 'var(--color-secondary)' : '#f3f4f6',
            color: showFacades ? 'var(--color-primary)' : '#4b5563',
            border: '1px solid var(--border-medium)',
            padding: '0.35rem 0.85rem',
            borderRadius: '6px',
            fontSize: '0.8rem',
            fontWeight: 'bold',
            cursor: 'pointer',
            transition: 'all 0.2s'
          }}
        >
          {showFacades ? 'Hide Facades' : 'Show Facades'}
        </button>
      </div>

      <div style={{ width: '100%', overflowX: 'auto', display: 'flex', justifyContent: 'center' }}>
        <canvas 
          ref={canvasRef} 
          style={{ 
            display: 'block',
            height: 'auto',
            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)', 
            borderRadius: '8px', 
            backgroundColor: '#ffffff',
            border: '1px solid #e2e8f0'
          }} 
        />
      </div>
    </div>
  );
}