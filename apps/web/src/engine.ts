import type { RoomDimensions, WindowSpecs } from './types';

const TWO_PI = 2.0 * Math.PI;


function smoothstep(edge0: number, edge1: number, x: number): number {
  if (edge1 - edge0 === 0) return 0.0;
  const t = Math.max(0.0, Math.min(1.0, (x - edge0) / (edge1 - edge0)));
  return t * t * (3.0 - 2.0 * t);
}

function distPtSeg2D(px: number, py: number, ax: number, ay: number, bx: number, by: number) {
  const abX = bx - ax;
  const abY = by - ay;
  const apX = px - ax;
  const apY = py - ay;
  const abLenSq = abX * abX + abY * abY;
  if (abLenSq === 0) return Math.hypot(px - ax, py - ay);
  const t = Math.max(0.0, Math.min(1.0, (apX * abX + apY * abY) / abLenSq));
  return Math.hypot(px - (ax + t * abX), py - (ay + t * abY));
}

function solidAngleTri(px: number, py: number, pz: number, a: number[], b: number[], c: number[]): number {
  const r1 = [a[0] - px, a[1] - py, a[2] - pz];
  const r2 = [b[0] - px, b[1] - py, b[2] - pz];
  const r3 = [c[0] - px, c[1] - py, c[2] - pz];

  const l1 = Math.hypot(r1[0], r1[1], r1[2]);
  const l2 = Math.hypot(r2[0], r2[1], r2[2]);
  const l3 = Math.hypot(r3[0], r3[1], r3[2]);

  if (l1 === 0 || l2 === 0 || l3 === 0) return 0.0;

  const cx = r2[1] * r3[2] - r2[2] * r3[1];
  const cy = r2[2] * r3[0] - r2[0] * r3[2];
  const cz = r2[0] * r3[1] - r2[1] * r3[0];
  const scalarTriple = r1[0] * cx + r1[1] * cy + r1[2] * cz;

  const den =
    l1 * l2 * l3 +
    (r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2]) * l3 +
    (r2[0] * r3[0] + r2[1] * r3[1] + r2[2] * r3[2]) * l1 +
    (r3[0] * r1[0] + r3[1] * r1[1] + r3[2] * r1[2]) * l2;

  if (den === 0) return 0.0;
  return Math.abs(2.0 * Math.atan2(scalarTriple, den));
}

function solidAngleQuad(px: number, py: number, pz: number, corners: number[][]): number {
  return (
    solidAngleTri(px, py, pz, corners[0], corners[1], corners[2]) +
    solidAngleTri(px, py, pz, corners[0], corners[2], corners[3])
  );
}

function rayCast2D(ox: number, oy: number, dx: number, dy: number, segments: number[][][], lChar: number): number {
  let minDist = Infinity;
  for (const w of segments) {
    const vx = w[1][0] - w[0][0];
    const vy = w[1][1] - w[0][1];
    const qx = ox - w[0][0];
    const qy = oy - w[0][1];
    const cross = dx * vy - dy * vx;
    if (Math.abs(cross) < lChar * 1e-6) continue;
    const u = (vx * qy - vy * qx) / cross;
    const t = (dx * qy - dy * qx) / cross;
    if (u > lChar * 1e-5 && t >= -1e-4 && t <= 1.0001) {
      if (u < minDist) minDist = u;
    }
  }
  return minDist !== Infinity ? minDist : lChar * 2.0;
}


function generateAnnualSunVectors(latitudeDeg = 52.17): number[][] {
  const declinations = [-23.45, -11.5, 0.0, 11.5, 23.45];
  const minElev = 2.0 * (Math.PI / 180);
  const lat = latitudeDeg * (Math.PI / 180);
  const rays: number[][] = [];

  for (const decDeg of declinations) {
    const dec = decDeg * (Math.PI / 180);
    for (let haDeg = -120; haDeg <= 120; haDeg += 10) {
      const ha = haDeg * (Math.PI / 180);
      const sinElev = Math.sin(lat) * Math.sin(dec) + Math.cos(lat) * Math.cos(dec) * Math.cos(ha);
      const elev = Math.asin(Math.max(-1.0, Math.min(1.0, sinElev)));
      if (elev > minElev) {
        const dx = -Math.cos(dec) * Math.sin(ha);
        const dy = Math.sin(dec) * Math.cos(lat) - Math.cos(dec) * Math.sin(lat) * Math.cos(ha);
        const dz = Math.sin(elev);
        const norm = Math.sqrt(dx * dx + dy * dy + dz * dz);
        rays.push([dx / norm, dy / norm, dz / norm]);
      }
    }
  }
  return rays;
}

const SUN_VECTORS = generateAnnualSunVectors(52.17);


export const extractFeatures = (
  sx: number,
  sy: number,
  room: RoomDimensions,
  windows: WindowSpecs[]
): number[] => {
  const sz = 0.8;
  const roomHeight = 2.7;


  const roomCorners: number[][] = [
    [0, 0],
    [room.width, 0],
    [room.width, room.depth],
    [0, room.depth]
  ];
  const wallSegments: number[][][] = [
    [[0, 0], [room.width, 0]],
    [[room.width, 0], [room.width, room.depth]],
    [[room.width, room.depth], [0, room.depth]],
    [[0, room.depth], [0, 0]]
  ];

  const lChar = Math.max(1.0, Math.hypot(room.width, room.depth));


  let totalLen = 0;
  let weightedDistSum = 0;
  const variances: { d: number; l: number }[] = [];

  for (const seg of wallSegments) {
    const d = distPtSeg2D(sx, sy, seg[0][0], seg[0][1], seg[1][0], seg[1][1]);
    const l = Math.hypot(seg[1][0] - seg[0][0], seg[1][1] - seg[0][1]);
    weightedDistSum += d * l;
    totalLen += l;
    variances.push({ d, l });
  }

  const smoothMeanDepth = totalLen > 0 ? weightedDistSum / totalLen : 0;
  const smoothDepthStdDev =
    totalLen > 0
      ? Math.sqrt(variances.reduce((acc, v) => acc + v.l * Math.pow(v.d - smoothMeanDepth, 2), 0) / totalLen)
      : 0;


  const apertures: Array<{
    corners: number[][];
    normal: [number, number];
    center: [number, number, number];
    width: number;
    height: number;
    skyVisRatio: number;
  }> = [];

  if (windows && windows.length > 0) {
    windows.forEach((win) => {
      let center2D: [number, number] = [0, 0];
      let normal: [number, number] = [0, -1];
      let rx = 1, ry = 0;

      if (win.wall === 'South') {
        center2D = [win.center, 0];
        normal = [0, -1];
        rx = 1; ry = 0;
      } else if (win.wall === 'North') {
        center2D = [win.center, room.depth];
        normal = [0, 1];
        rx = -1; ry = 0;
      } else if (win.wall === 'East') {
        center2D = [room.width, win.center];
        normal = [1, 0];
        rx = 0; ry = 1;
      } else if (win.wall === 'West') {
        center2D = [0, win.center];
        normal = [-1, 0];
        rx = 0; ry = -1;
      }

      const w = Math.max(win.width, 0.1);
      const z1 = win.sill;
      const z2 = win.head;
      const h = Math.max(z2 - z1, 0.1);
      const cz = (z1 + z2) / 2.0;

      const corners3D = [
        [center2D[0] - rx * (w / 2), center2D[1] - ry * (w / 2), z1],
        [center2D[0] + rx * (w / 2), center2D[1] + ry * (w / 2), z1],
        [center2D[0] + rx * (w / 2), center2D[1] + ry * (w / 2), z2],
        [center2D[0] - rx * (w / 2), center2D[1] - ry * (w / 2), z2]
      ];

      apertures.push({
        corners: corners3D,
        normal,
        center: [center2D[0], center2D[1], cz],
        width: w,
        height: h,
        skyVisRatio: 1.0 
      });
    });
  }


  let solarHits = 0;
  for (const [rdx, rdy, rdz] of SUN_VECTORS) {
    let hit = false;
    for (const ap of apertures) {
      const [nx, ny] = ap.normal;

      const dot = rdx * nx + rdy * ny;
      if (dot > 0.01) {
        const [cx, cy, cz] = ap.center;
        const dPlane = (cx - sx) * nx + (cy - sy) * ny;
        if (dPlane > 0) {
          const t = dPlane / dot;
          const hx = sx + t * rdx;
          const hy = sy + t * rdy;
          const hz = sz + t * rdz;

          if (hz >= ap.center[2] - ap.height / 2 && hz <= ap.center[2] + ap.height / 2) {
            const hDist = Math.hypot(hx - cx, hy - cy);
            if (hDist <= ap.width / 2) {
              hit = true;
              break;
            }
          }
        }
      }
    }
    if (hit) solarHits++;
  }


  let totalOmega = 0;
  let rawTotalOmega = 0;
  let southW = 0, eastW = 0, westW = 0, skyW = 0;
  let totalArea = 0;
  let areaCosTheta = 0;

  const dminSqList: number[] = [];
  const clearanceData: { sill: number; header: number; weight: number }[] = [];

  let visibleCount = 0;
  let sumVisibleDist = 0;
  let maxWinOmega = 0;
  let dominantNormal: [number, number] | null = null;
  let dominantCenter: [number, number] | null = null;
  let totalAngularSpread = 0;

  for (const ap of apertures) {
    const rawOmega = solidAngleQuad(sx, sy, sz, ap.corners);
    rawTotalOmega += rawOmega;
    const winOmega = rawOmega;  
    totalOmega += winOmega;

    const [cx, cy, cz] = ap.center;
    const d = Math.max(0.1, Math.hypot(cx - sx, cy - sy));
    const [wnx, wny] = ap.normal;

    visibleCount++;
    sumVisibleDist += d;

    if (winOmega > maxWinOmega) {
      maxWinOmega = winOmega;
      dominantNormal = ap.normal;
      dominantCenter = [cx, cy];
    }


    const angles = ap.corners.map(c => Math.atan2(c[1] - sy, c[0] - sx));
    let spread = 0;
    for (let i = 0; i < 4; i++) {
      for (let j = i + 1; j < 4; j++) {
        let diff = Math.abs(angles[i] - angles[j]);
        if (diff > Math.PI) diff = 2 * Math.PI - diff;
        if (diff > spread) spread = diff;
      }
    }
    totalAngularSpread += spread;


    southW += winOmega * Math.max(0.0, -wny);
    eastW += winOmega * Math.max(0.0, wnx);
    westW += winOmega * Math.max(0.0, -wnx);
    skyW += rawOmega * ap.skyVisRatio;

    const winArea = ap.width * ap.height;
    const d3d = Math.max(Math.sqrt((cx - sx) ** 2 + (cy - sy) ** 2 + (cz - sz) ** 2), 1e-3);
    areaCosTheta += (d / d3d) * winArea;
    totalArea += winArea;

    const zMin = ap.center[2] - ap.height / 2;
    const zMax = ap.center[2] + ap.height / 2;

    dminSqList.push(
      Math.pow(ap.width / 2.0, 2) + Math.pow(d, 2) + Math.pow(Math.max(0.0, Math.abs(cz - sz) - ap.height / 2.0), 2)
    );

    clearanceData.push({
      sill: sz - zMin,
      header: zMax - sz,
      weight: winArea / (Math.pow(d, 3) + 1e-3)
    });
  }


  let planeDist = lChar * 2.0;
  let hOffset = lChar * 2.0;
  let rayFwd = lChar * 2.0;
  let rayBwd = lChar * 2.0;
  let rayLft = lChar * 2.0;
  let rayRgt = lChar * 2.0;

  if (dominantNormal !== null && dominantCenter !== null) {
    const [dnx, dny] = dominantNormal;
    const [dcx, dcy] = dominantCenter;
    planeDist = Math.abs((sx - dcx) * dnx + (sy - dcy) * dny);
    hOffset = Math.abs(-(sx - dcx) * dny + (sy - dcy) * dnx);
    rayFwd = rayCast2D(sx, sy, -dnx, -dny, wallSegments, lChar);
    rayBwd = rayCast2D(sx, sy, dnx, dny, wallSegments, lChar);
    rayLft = rayCast2D(sx, sy, -dny, dnx, wallSegments, lChar);
    rayRgt = rayCast2D(sx, sy, dny, -dnx, wallSegments, lChar);
  }


  let minOpaque = Infinity;
  for (const seg of wallSegments) {
    const d = distPtSeg2D(sx, sy, seg[0][0], seg[0][1], seg[1][0], seg[1][1]);
    if (d < minOpaque) minOpaque = d;
  }

  const outerCornerDmin = Math.max(0.01, Math.min(...roomCorners.map(c => Math.hypot(sx - c[0], sy - c[1]))));


  const cosTheta = totalArea > 0 ? areaCosTheta / totalArea : 0.0;
  const wSum = clearanceData.reduce((acc, c) => acc + c.weight, 0);

  const clip = (v: number) => Math.max(0.0, Math.min(1.0, isNaN(v) || !isFinite(v) ? 0.0 : v));

  const features = [
    clip(totalOmega / TWO_PI),                                                  // 0: Total_Omega
    clip(cosTheta),                                                             // 1: Global_Cos_Theta
    clip(wSum > 0 ? clearanceData.reduce((acc, c) => acc + c.sill * c.weight, 0) / wSum / roomHeight : 0.0),   // 2: Global_Sill_Clearance
    clip(wSum > 0 ? clearanceData.reduce((acc, c) => acc + c.header * c.weight, 0) / wSum / roomHeight : 0.0), // 3: Global_Header_Clearance
    (dminSqList.length > 0 ? 1.0 / Math.max(Math.min(...dminSqList), 1e-4) : 0.0) * Math.pow(lChar, 2),        // 4: Global_inv_dmin2
    clip(Math.max(0.05, minOpaque) / lChar),                                    // 5: Opaque_Wall_dmin
    clip(1.0),                                                                  // 6: Inner_Corner_Dmin (1.0 for convex rooms)
    clip(outerCornerDmin / lChar),                                              // 7: Outer_Corner_Dmin
    clip(southW / TWO_PI),                                                      // 8: South_Weighted_Omega
    clip(eastW / TWO_PI),                                                       // 9: East_Weighted_Omega
    clip(westW / TWO_PI),                                                       // 10: West_Weighted_Omega
    clip(SUN_VECTORS.length > 0 ? solarHits / SUN_VECTORS.length : 0.0),        // 11: Annual_Direct_Sun_Fraction
    clip(smoothMeanDepth / lChar),                                              // 12: Smooth_Mean_Depth
    clip(smoothDepthStdDev / lChar),                                            // 13: Smooth_Depth_StdDev
    clip(2.0),                                                                  // 14: Min_Occluder_Distance (2.0 default unoccluded)
    clip(totalAngularSpread / TWO_PI),                                          // 15: Window_Angular_Spread
    clip(skyW / TWO_PI),                                                        // 16: Window_Sky_Weighted_Omega
    clip(rawTotalOmega > 0 ? totalOmega / rawTotalOmega : 0.0),                 // 17: Occlusion_Multiplier
    clip(visibleCount / 10.0),                                                  // 18: Visible_Window_Count
    clip((visibleCount > 0 ? sumVisibleDist / visibleCount : lChar * 2.0) / lChar), // 19: Mean_Window_Distance
    clip(planeDist / lChar),                                                    // 20: Dominant_Win_Plane_Dist
    clip(hOffset / lChar),                                                      // 21: Dominant_Win_H_Offset
    clip(rayFwd / lChar),                                                       // 22: Dominant_Win_Ray_Forward
    clip(rayBwd / lChar),                                                       // 23: Dominant_Win_Ray_Backward
    clip(rayLft / lChar),                                                       // 24: Dominant_Win_Ray_Left
    clip(rayRgt / lChar)                                                        // 25: Dominant_Win_Ray_Right
  ];

  return features.map(val => (isNaN(val) || !isFinite(val) ? 0 : val));
};
