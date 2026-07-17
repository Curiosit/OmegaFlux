export interface RoomDimensions {
  width: number;
  depth: number;
  height: number;
}

export interface WindowSpecs {
  width: number;
  center: number;
  sill: number;
  head: number;
}

export interface GridPoint {
  x: number;
  y: number;
  udi: number;
}

export type ColorPalette = 'udi' | 'omega' | 'blueprint';