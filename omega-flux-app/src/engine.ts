import type { RoomDimensions, WindowSpecs } from './types';
import { score as scoreLow } from './components/OmegaFluxLow';
import { score as scoreUp } from './components/OmegaFluxUp';

export const analyticalSolidAngle = (x1: number, x2: number, z1: number, z2: number, d: number): number => {
  const dist = Math.max(d, 0.1); // Minimum 10cm safety buffer to prevent blowups
  const f = (x: number, z: number) => Math.atan2(x * z, dist * Math.sqrt(x ** 2 + dist ** 2 + z ** 2));
  const val = Math.abs(f(x2, z2) - f(x1, z2) - f(x2, z1) + f(x1, z1));
  return isNaN(val) || !isFinite(val) ? 0 : val;
};

export const extractFeatures = (x: number, y: number, room: RoomDimensions, windows: WindowSpecs[]): number[] => {
  const sz = 0.8; // Standard sensor height (desk level)
  
  let totalOmega = 0;
  let sumInvDmin2 = 0;
  let totalArea = 0;
  let areaWeightedCosTheta = 0;
  let areaWeightedSillClearance = 0;

  if (!windows || windows.length === 0) {
    return [0, 0, 0, 0, Math.min(x, room.width - x, y, room.depth - y), 6.28];
  }

  windows.forEach((win) => {
    let d = 0.1; 
    let lateralPos = 0; 

    // Transform coordinates based on target wall
    if (win.wall === 'South') {
      d = y;
      lateralPos = x;
    } else if (win.wall === 'North') {
      d = room.depth - y;
      lateralPos = x;
    } else if (win.wall === 'East') {
      d = room.width - x;
      lateralPos = y;
    } else if (win.wall === 'West') {
      d = x;
      lateralPos = y;
    }

    // Clamp perpendicular distance to safe room bounds
    d = Math.max(d, 0.05); 

    const cz = (win.sill + win.head) / 2.0;
    const h = Math.max(win.head - win.sill, 0.1);
    const winWidth = Math.max(win.width, 0.1);
    const winArea = winWidth * h;

    // 1. Solid Angle Calculation
    const winOmega = analyticalSolidAngle(
      win.center - winWidth / 2 - lateralPos,
      win.center + winWidth / 2 - lateralPos,
      win.sill - sz,
      win.head - sz,
      d
    );
    totalOmega += winOmega;

    // 2. Bounded Inverse-Square Distance (prevents the 10,000 spikes)
    const dx = Math.max(0, Math.abs(win.center - lateralPos) - winWidth / 2);
    const dz = Math.max(0, Math.abs(cz - sz) - h / 2);
    const dmin2 = dx ** 2 + d ** 2 + dz ** 2;
    // Cap the denominator so inverse distance doesn't explode near borders
    sumInvDmin2 += 1.0 / Math.max(dmin2, 0.25); 

    // 3. Area-Weighted Aggregates
    const d3dCent = Math.max(Math.sqrt((win.center - lateralPos) ** 2 + d ** 2 + (cz - sz) ** 2), 0.1);
    const cosTheta = Math.min(Math.max(d / d3dCent, 0), 1); // Clamp cosine between 0 and 1
    
    areaWeightedCosTheta += cosTheta * winArea;
    areaWeightedSillClearance += Math.max(0, sz - win.sill) * winArea;
    totalArea += winArea;
  });

  // Finalize Global Aggregates
  const globalCosTheta = totalArea > 0 ? areaWeightedCosTheta / totalArea : 0;
  const globalSillClearance = totalArea > 0 ? areaWeightedSillClearance / totalArea : 0;
  
  // Safe minimum distance to opaque walls
  const opaqueWallDmin = Math.max(0.05, Math.min(y, room.depth - y, x, room.width - x));
  
  // Remaining solid angle of opaque enclosure (hemi-sphere is ~6.28 sr)
  const opaqueOmega = Math.max(0, 6.28 - totalOmega); 

  const features = [
    totalOmega, 
    globalCosTheta, 
    globalSillClearance, 
    sumInvDmin2, 
    opaqueWallDmin, 
    opaqueOmega
  ];

  // Final validation pass to guarantee zero NaNs/Infinities
  return features.map(val => (isNaN(val) || !isFinite(val) ? 0 : val));
};

export const predictOmegaFlux = (features: number[]) => {
  const low = scoreLow(features);
  const up = scoreUp(features);
  
  return {
    low: Math.max(0, Math.min(100, isNaN(low) ? 0 : low)),
    up: Math.max(0, Math.min(100, isNaN(up) ? 0 : up)),
  };
};