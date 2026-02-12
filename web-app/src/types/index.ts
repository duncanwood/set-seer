// Shared type definitions for Set Seer

/** Bounding box in normalized coordinates [0, 1] */
export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** A single detection result from the YOLO model */
export interface Detection {
  id: string;
  className: string;
  confidence: number;
  boundingBox: BoundingBox;
}

/** Card feature enums */
export type SetNumber = 1 | 2 | 3;
export type SetColor = 'red' | 'green' | 'purple';
export type SetShape = 'diamond' | 'oval' | 'squiggle';
export type SetShading = 'solid' | 'striped' | 'empty';

/** A parsed Set card with all four attributes */
export interface SetCard {
  id: string;
  number: SetNumber;
  color: SetColor;
  shape: SetShape;
  shading: SetShading;
  boundingBox: BoundingBox;
}

/** A valid Set (group of 3 cards) */
export interface SetResult {
  id: string;
  cards: [SetCard, SetCard, SetCard];
  colorIndex: number;
}

/** Inference configuration */
export interface InferenceConfig {
  confidenceThreshold: number;
  iouThreshold: number;
  modelInputSize: number;
}

/** Default config matching the Swift app's tuned values */
export const DEFAULT_INFERENCE_CONFIG: InferenceConfig = {
  confidenceThreshold: 0.3,
  iouThreshold: 0.45,
  modelInputSize: 640,
};
