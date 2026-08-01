export type WallType = 'North' | 'South' | 'East' | 'West';

export interface WindowSpecs {
  wall: WallType;
  center: number; // Distance from the left edge of that specific wall
  width: number;
  sill: number;
  head: number;
}

export interface RoomDimensions {
  width: number; // X-axis length
  depth: number; // Y-axis length
}

export interface GridPoint {
  x: number;
  y: number;
  udi: number;
}

export type ColorPalette = 'omega' | 'udi' | 'blueprint';

interface ColorStop {
  t: number; // Position from 0 to 1
  c: [number, number, number]; // RGB array
}


const interpolateColor = (val: number, stops: ColorStop[]): string => {
  const t = Math.max(0, Math.min(1, val / 100));
  

  let lower = stops[0];
  let upper = stops[stops.length - 1];

  for (let i = 0; i < stops.length - 1; i++) {
    if (t >= stops[i].t && t <= stops[i + 1].t) {
      lower = stops[i];
      upper = stops[i + 1];
      break;
    }
  }


  const range = upper.t - lower.t;
  const localT = range === 0 ? 0 : (t - lower.t) / range;

  const r = Math.round(lower.c[0] + (upper.c[0] - lower.c[0]) * localT);
  const g = Math.round(lower.c[1] + (upper.c[1] - lower.c[1]) * localT);
  const b = Math.round(lower.c[2] + (upper.c[2] - lower.c[2]) * localT);

  return `rgb(${r}, ${g}, ${b})`;
};

export const PALETTE_CONFIG = {
  omega: {
    name: 'Omega Flux Theme',
    cssGradient: 'linear-gradient(to right, rgb(15, 23, 42) 0%, rgb(14, 165, 233) 50%, rgb(243, 239, 168) 100%)',
    getColor: (val: number) => {
      const stops: ColorStop[] = [
        { t: 0, c: [15, 23, 42] },
        { t: 0.5, c: [14, 165, 233] },
        { t: 1, c: [243, 239, 168] }
      ];
      return interpolateColor(val, stops);
    }
  },
  udi: {
    name: 'Classic UDI',
    cssGradient: 'linear-gradient(to right, rgb(239, 68, 68) 0%, rgb(234, 179, 8) 50%, rgb(34, 197, 94) 100%)',
    getColor: (val: number) => {
      const stops: ColorStop[] = [
        { t: 0, c: [239, 68, 68] },
        { t: 0.5, c: [234, 179, 8] },
        { t: 1, c: [34, 197, 94] }
      ];
      return interpolateColor(val, stops);
    }
  },
  blueprint: {
    name: 'Blueprint Monochromatic',
    cssGradient: 'linear-gradient(to right, rgb(15, 23, 42) 0%, rgb(235, 243, 255) 100%)',
    getColor: (val: number) => {
      const stops: ColorStop[] = [
        { t: 0, c: [15, 23, 42] },
        { t: 1, c: [235, 243, 255] }
      ];
      return interpolateColor(val, stops);
    }
  }
};