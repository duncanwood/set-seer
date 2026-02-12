/**
 * YOLO inference engine for Set Seer.
 *
 * Handles ONNX model loading, image preprocessing, YOLOv8 output parsing,
 * and Non-Maximum Suppression. Runs entirely client-side via ONNX Runtime Web.
 */
import * as ort from 'onnxruntime-web';
import { getAssetPath } from '@/lib/utils';
import { CLASS_NAMES, NUM_CLASSES } from './classNames';
import type { Detection, BoundingBox, InferenceConfig } from '@/types';
import { DEFAULT_INFERENCE_CONFIG } from '@/types';

// ─── Model Session Management ──────────────────────────────────────────────

let session: ort.InferenceSession | null = null;
let isLoading = false;

/**
 * Load the ONNX model. Returns the cached session if already loaded.
 * Uses the WASM execution provider for broad browser compatibility.
 */
export async function loadModel(
  modelPath: string = '/models/setDetectionYOLO.onnx'
): Promise<ort.InferenceSession> {
  // Respect base path for deployment
  const fullPath = getAssetPath(modelPath);
  if (session) return session;
  if (isLoading) {
    // Wait for in-flight load to complete
    while (isLoading) {
      await new Promise((r) => setTimeout(r, 100));
    }
    if (session) return session;
  }

  isLoading = true;
  try {
    // Configure WASM paths for onnxruntime-web
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = getAssetPath('/');

    session = await ort.InferenceSession.create(fullPath, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });

    console.log('[SetSeer] Model loaded successfully');
    return session;
  } catch (error) {
    console.error('[SetSeer] Failed to load model:', error);
    throw error;
  } finally {
    isLoading = false;
  }
}

// ─── Preprocessing ──────────────────────────────────────────────────────────

/**
 * Preprocess a video frame for YOLOv8 inference.
 *
 * 1. Draws the video/canvas onto a square canvas (letterboxed)
 * 2. Extracts pixel data and converts to NCHW float32 tensor
 * 3. Normalizes pixel values to [0, 1]
 *
 * Returns the tensor and the letterbox metadata needed for coordinate mapping.
 */
export function preprocess(
  source: HTMLVideoElement | HTMLCanvasElement,
  modelSize: number = DEFAULT_INFERENCE_CONFIG.modelInputSize
): { tensor: ort.Tensor; letterbox: LetterboxInfo } {
  const canvas = document.createElement('canvas');
  canvas.width = modelSize;
  canvas.height = modelSize;
  const ctx = canvas.getContext('2d')!;

  // Calculate letterbox dimensions (preserve aspect ratio)
  const srcWidth = source instanceof HTMLVideoElement ? source.videoWidth : source.width;
  const srcHeight = source instanceof HTMLVideoElement ? source.videoHeight : source.height;

  const scale = Math.min(modelSize / srcWidth, modelSize / srcHeight);
  const scaledWidth = Math.round(srcWidth * scale);
  const scaledHeight = Math.round(srcHeight * scale);
  const xOffset = Math.round((modelSize - scaledWidth) / 2);
  const yOffset = Math.round((modelSize - scaledHeight) / 2);

  // Fill with gray (common YOLO letterbox color)
  ctx.fillStyle = '#808080';
  ctx.fillRect(0, 0, modelSize, modelSize);

  // Draw the source image centered
  ctx.drawImage(source, xOffset, yOffset, scaledWidth, scaledHeight);

  // Extract pixel data
  const imageData = ctx.getImageData(0, 0, modelSize, modelSize);
  const { data } = imageData;

  // Convert RGBA HWC → RGB NCHW float32 normalized to [0, 1]
  const numPixels = modelSize * modelSize;
  const float32Data = new Float32Array(3 * numPixels);

  for (let i = 0; i < numPixels; i++) {
    const rgbaIdx = i * 4;
    float32Data[i] = data[rgbaIdx] / 255.0;                    // R channel
    float32Data[numPixels + i] = data[rgbaIdx + 1] / 255.0;    // G channel
    float32Data[2 * numPixels + i] = data[rgbaIdx + 2] / 255.0; // B channel
  }

  const tensor = new ort.Tensor('float32', float32Data, [1, 3, modelSize, modelSize]);

  return {
    tensor,
    letterbox: { xOffset, yOffset, scale, modelSize, srcWidth, srcHeight },
  };
}

/** Metadata for reversing letterbox coordinate transform */
export interface LetterboxInfo {
  xOffset: number;
  yOffset: number;
  scale: number;
  modelSize: number;
  srcWidth: number;
  srcHeight: number;
}

// ─── Inference ──────────────────────────────────────────────────────────────

/**
 * Run a full inference cycle: preprocess → model → postprocess.
 */
export async function runInference(
  source: HTMLVideoElement | HTMLCanvasElement,
  config: InferenceConfig = DEFAULT_INFERENCE_CONFIG
): Promise<Detection[]> {
  const modelSession = await loadModel();

  const { tensor, letterbox } = preprocess(source, config.modelInputSize);

  // Run the model
  const inputName = modelSession.inputNames[0];
  const results = await modelSession.run({ [inputName]: tensor });

  // Get the output tensor
  const outputName = modelSession.outputNames[0];
  const outputTensor = results[outputName];

  return postprocess(outputTensor, letterbox, config);
}

// ─── Postprocessing ─────────────────────────────────────────────────────────

interface RawDetection {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  confidence: number;
  classIdx: number;
}

/**
 * Parse YOLOv8 output tensor and apply NMS.
 *
 * YOLOv8 output shape: [1, (4 + numClasses), numAnchors]
 *   - First 4 rows: cx, cy, w, h (in model pixel coordinates)
 *   - Next numClasses rows: per-class confidence scores
 *
 * This mirrors the parsing logic in CameraManager.swift's parseYOLOOutput().
 */
function postprocess(
  tensor: ort.Tensor,
  letterbox: LetterboxInfo,
  config: InferenceConfig
): Detection[] {
  const data = tensor.data as Float32Array;
  const [, features, numAnchors] = tensor.dims;

  // Sanity check: features should be 4 + NUM_CLASSES
  const detectedClasses = features - 4;
  if (detectedClasses !== NUM_CLASSES) {
    console.warn(
      `[SetSeer] Expected ${NUM_CLASSES} classes but model has ${detectedClasses}`
    );
  }

  const candidates: RawDetection[] = [];

  for (let i = 0; i < numAnchors; i++) {
    // Extract box coordinates (center format)
    const cx = data[0 * numAnchors + i];
    const cy = data[1 * numAnchors + i];
    const w = data[2 * numAnchors + i];
    const h = data[3 * numAnchors + i];

    // Find best class score
    let maxScore = 0;
    let maxClassIdx = 0;
    for (let c = 0; c < detectedClasses; c++) {
      const score = data[(4 + c) * numAnchors + i];
      if (score > maxScore) {
        maxScore = score;
        maxClassIdx = c;
      }
    }

    if (maxScore >= config.confidenceThreshold) {
      // Convert center → corner format
      const x0 = cx - w / 2;
      const y0 = cy - h / 2;
      const x1 = cx + w / 2;
      const y1 = cy + h / 2;

      candidates.push({ x0, y0, x1, y1, confidence: maxScore, classIdx: maxClassIdx });
    }
  }

  // Apply Non-Maximum Suppression
  const nmsResults = applyNMS(candidates, config.iouThreshold);

  // Convert to normalized coordinates (remove letterbox, normalize to [0,1])
  return nmsResults.map((det, idx) => {
    const bbox = modelToNormalized(det, letterbox);
    return {
      id: `det-${idx}-${Date.now()}`,
      className: CLASS_NAMES[det.classIdx] ?? `unknown-${det.classIdx}`,
      confidence: det.confidence,
      boundingBox: bbox,
    };
  });
}

/**
 * Convert model-space coordinates to normalized [0,1] coordinates
 * by removing the letterbox offset and scaling.
 *
 * Mirrors CameraManager.swift's letterbox adjustment logic.
 */
function modelToNormalized(det: RawDetection, lb: LetterboxInfo): BoundingBox {
  const activeWidth = lb.modelSize - 2 * lb.xOffset;
  const activeHeight = lb.modelSize - 2 * lb.yOffset;

  const normX = (det.x0 - lb.xOffset) / activeWidth;
  const normY = (det.y0 - lb.yOffset) / activeHeight;
  const normW = (det.x1 - det.x0) / activeWidth;
  const normH = (det.y1 - det.y0) / activeHeight;

  return {
    x: Math.max(0, Math.min(1, normX)),
    y: Math.max(0, Math.min(1, normY)),
    width: Math.max(0, Math.min(1 - Math.max(0, normX), normW)),
    height: Math.max(0, Math.min(1 - Math.max(0, normY), normH)),
  };
}

// ─── Non-Maximum Suppression ────────────────────────────────────────────────

/**
 * Per-class Non-Maximum Suppression.
 * Mirrors CameraManager.swift's applyNMS() method.
 */
function applyNMS(detections: RawDetection[], iouThreshold: number): RawDetection[] {
  // Sort by confidence descending
  const sorted = [...detections].sort((a, b) => b.confidence - a.confidence);
  const keep = new Array(sorted.length).fill(true);

  for (let i = 0; i < sorted.length; i++) {
    if (!keep[i]) continue;

    for (let j = i + 1; j < sorted.length; j++) {
      if (!keep[j]) continue;

      const iou = calculateIoU(sorted[i], sorted[j]);
      if (iou > iouThreshold) {
        keep[j] = false;
      }
    }
  }

  return sorted.filter((_, i) => keep[i]);
}

function calculateIoU(a: RawDetection, b: RawDetection): number {
  const x0 = Math.max(a.x0, b.x0);
  const y0 = Math.max(a.y0, b.y0);
  const x1 = Math.min(a.x1, b.x1);
  const y1 = Math.min(a.y1, b.y1);

  const intersection = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
  const areaA = (a.x1 - a.x0) * (a.y1 - a.y0);
  const areaB = (b.x1 - b.x0) * (b.y1 - b.y0);
  const union = areaA + areaB - intersection;

  return union > 0 ? intersection / union : 0;
}
