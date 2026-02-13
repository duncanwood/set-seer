'use client';

/**
 * Camera component — captures video frames, runs YOLO inference,
 * and renders detection overlays with Set visualization.
 *
 * Uses react-webcam for camera access and canvas for drawing.
 */
import React, { useRef, useCallback, useEffect, useState } from 'react';
import Webcam from 'react-webcam';
import { loadModel, runInference } from '@/lib/yolo';
import { findSetsFromDetections } from '@/lib/setEngine';
import type { Detection, SetCard, SetResult, InferenceConfig } from '@/types';
import { DEFAULT_INFERENCE_CONFIG } from '@/types';
import { APP_VERSION, COMMIT_HASH, BUILD_TIME } from '@/lib/constants';

// ─── Color palette for Set visualization ────────────────────────────────────

const SET_COLORS = [
  '#FF4444', '#4488FF', '#FF8800', '#AA44FF',
  '#FF44AA', '#00CCCC', '#FFCC00', '#44DDAA', '#6644FF',
];

function formatLabel(className: string): string {
  return className
    .split('-')
    .map((s) => s.charAt(0).toUpperCase() + s.slice(1))
    .join(' ');
}

// ─── Component ──────────────────────────────────────────────────────────────

interface CameraProps {
  targetFps?: number;
}

export default function Camera({ targetFps = 15 }: CameraProps) {
  const webcamRef = useRef<Webcam>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const requestIdRef = useRef<number>(0);
  const lastFrameTimeRef = useRef<number>(0);

  const [modelReady, setModelReady] = useState(false);
  const [modelError, setModelError] = useState<string | null>(null);
  const [facingMode, setFacingMode] = useState<'environment' | 'user'>('environment');
  const [showSets, setShowSets] = useState(false);
  const [debugMode, setDebugMode] = useState(false);
  const [fps] = useState(targetFps);

  // Detection state
  const [detections, setDetections] = useState<Detection[]>([]);
  const [sets, setSets] = useState<SetResult[]>([]);
  const [cards, setCards] = useState<SetCard[]>([]);
  const [inferenceTime, setInferenceTime] = useState<number>(0);

  // Refs for the animation loop (avoids stale closures)
  const fpsRef = useRef(fps);
  const showSetsRef = useRef(showSets);
  const configRef = useRef<InferenceConfig>(DEFAULT_INFERENCE_CONFIG);
  const isProcessingRef = useRef(false);

  useEffect(() => { fpsRef.current = fps; }, [fps]);
  useEffect(() => { showSetsRef.current = showSets; }, [showSets]);

  // ── Load model on mount ──
  useEffect(() => {
    loadModel()
      .then(() => setModelReady(true))
      .catch((err) => setModelError(err.message));
  }, []);

  // ── Inference loop ──
  const runDetection = useCallback(async () => {
    if (!webcamRef.current?.video || !overlayRef.current || !modelReady) {
      requestIdRef.current = requestAnimationFrame(runDetection);
      return;
    }

    const now = performance.now();
    const frameInterval = 1000 / fpsRef.current;

    if (now - lastFrameTimeRef.current < frameInterval) {
      requestIdRef.current = requestAnimationFrame(runDetection);
      return;
    }

    if (isProcessingRef.current) {
      requestIdRef.current = requestAnimationFrame(runDetection);
      return;
    }

    isProcessingRef.current = true;
    lastFrameTimeRef.current = now;

    try {
      const video = webcamRef.current.video;
      if (video.readyState < 2) {
        requestIdRef.current = requestAnimationFrame(runDetection);
        isProcessingRef.current = false;
        return;
      }

      const t0 = performance.now();
      const dets = await runInference(video, configRef.current);
      const t1 = performance.now();

      const { cards: parsedCards, sets: foundSets } = findSetsFromDetections(dets);

      setDetections(dets);
      setCards(parsedCards);
      setSets(foundSets);
      setInferenceTime(Math.round(t1 - t0));

      // Draw overlays
      drawOverlay(dets, parsedCards, foundSets, video, showSetsRef.current);
    } catch (err) {
      console.error('[SetSeer] Inference error:', err);
    }

    isProcessingRef.current = false;
    requestIdRef.current = requestAnimationFrame(runDetection);
  }, [modelReady]);

  useEffect(() => {
    if (modelReady) {
      requestIdRef.current = requestAnimationFrame(runDetection);
    }
    return () => cancelAnimationFrame(requestIdRef.current);
  }, [modelReady, runDetection]);

  // ── Drawing ──
  const drawOverlay = useCallback(
    (
      dets: Detection[],
      currentCards: SetCard[],
      currentSets: SetResult[],
      video: HTMLVideoElement,
      shouldShowSets: boolean
    ) => {
      const canvas = overlayRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Handle High DPI screens
      const dpr = window.devicePixelRatio || 1;
      const rect = video.getBoundingClientRect();

      // Set physical pixel size
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;

      // Scale context to match logical pixels
      ctx.scale(dpr, dpr);

      // Use logical dimensions for coordinate calculation
      const w = rect.width;
      const h = rect.height;

      for (const det of dets) {
        const x = det.boundingBox.x * w;
        const y = det.boundingBox.y * h;
        const bw = det.boundingBox.width * w;
        const bh = det.boundingBox.height * h;

        if (shouldShowSets) {
          // Find which sets this detection belongs to
          const participating = currentSets.filter((s) =>
            s.cards.some((c) => c.id === det.id)
          );

          if (participating.length > 0) {
            for (let si = 0; si < participating.length; si++) {
              const expansion = si * 6;
// ... (rest of loop content implicitly preserved if not matched?)
// Wait, I need to match the block correctly. I will replace the condition start.
              const color = SET_COLORS[participating[si].colorIndex % SET_COLORS.length];

              ctx.strokeStyle = color;
              ctx.lineWidth = 3;
              ctx.strokeRect(
                x - expansion,
                y - expansion,
                bw + expansion * 2,
                bh + expansion * 2
              );

              // Subtle fill
              ctx.fillStyle = color + '30';
              ctx.fillRect(
                x - expansion,
                y - expansion,
                bw + expansion * 2,
                bh + expansion * 2
              );
            }
          } else {
            // Not in any set — transparent grey border
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
            ctx.lineWidth = 1.5;
            ctx.strokeRect(x, y, bw, bh);
          }
        } else {
          // Default: green detection box
          ctx.strokeStyle = '#00FF88';
          ctx.lineWidth = 2.5;
          ctx.strokeRect(x, y, bw, bh);

          // Semi-transparent fill
          ctx.fillStyle = 'rgba(0, 255, 136, 0.08)';
          ctx.fillRect(x, y, bw, bh);

          // Label only in debug mode
          if (debugMode) {
            const label = `${formatLabel(det.className)} ${Math.round(det.confidence * 100)}%`;
            ctx.font = 'bold 11px Inter, system-ui, sans-serif';
            const metrics = ctx.measureText(label);
            const labelH = 18;
            const labelY = y > labelH + 4 ? y - labelH - 2 : y + bh + 2;

            ctx.fillStyle = 'rgba(0, 0, 0, 0.75)';
            ctx.fillRect(x, labelY, metrics.width + 8, labelH);

            ctx.fillStyle = '#00FF88';
            ctx.fillText(label, x + 4, labelY + 13);
          }
        }
      }
    },
    []
  );

  // Redraw when showSets or debugMode changes
  useEffect(() => {
    if (webcamRef.current?.video) {
      drawOverlay(detections, cards, sets, webcamRef.current.video, showSets);
    }
  }, [showSets, debugMode, detections, cards, sets, drawOverlay]);

  const toggleCamera = () => {
    setFacingMode((prev) => (prev === 'environment' ? 'user' : 'environment'));
  };

  return (
    <div className="camera-container">
      {/* Video + overlay */}
      <div className="camera-viewport">
        <Webcam
          ref={webcamRef}
          audio={false}
          videoConstraints={{
            facingMode,
            width: { ideal: 1280 },
            height: { ideal: 720 },
          }}
          className="camera-video"
          mirrored={facingMode === 'user'}
        />
        <canvas ref={overlayRef} className="camera-overlay" />
      </div>

      {/* Loading state */}
      {!modelReady && !modelError && (
        <div className="loading-overlay">
          <div className="loading-spinner" />
          <p>Loading model…</p>
        </div>
      )}

      {modelError && (
        <div className="error-overlay">
          <p>⚠ Failed to load model</p>
          <p className="error-detail">{modelError}</p>
        </div>
      )}

      {/* HUD */}
      {modelReady && (
        <div className="hud">
          {/* Top bar */}
          <div className="hud-top">
            <button
              className={`hud-btn hud-btn-sets ${showSets ? 'hud-btn-active' : ''} ${
                sets.length === 0 ? 'hud-btn-no-sets' : ''
              }`}
              onClick={() => setShowSets(!showSets)}
            >
              {sets.length > 0 ? (showSets ? 'Hide Sets' : 'Show Sets') : 'No Sets!'}
            </button>

            {debugMode && (
              <div className="hud-debug-group">
                 <div className="hud-version" title={`Built: ${BUILD_TIME}`}>
                  v{APP_VERSION} <span style={{ opacity: 0.6 }}>#{COMMIT_HASH}</span>
                </div>
                <div className="hud-stats">
                  <span>{detections.length} cards</span>
                  <span className="hud-dot">·</span>
                  <span>{inferenceTime}ms</span>
                </div>
              </div>
            )}
          </div>

          {/* Bottom controls */}
          <div className="hud-bottom">
            <button className="hud-btn hud-btn-icon" onClick={toggleCamera} title="Switch camera">
               <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                <path d="M3 3v5h5" />
                <path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16" />
                <path d="M21 21v-5h-5" />
               </svg>
            </button>

            <button
              className={`hud-btn hud-btn-icon hud-btn-debug ${debugMode ? 'hud-btn-active' : ''}`}
              onClick={() => setDebugMode(!debugMode)}
              title="Debug Mode"
            >
              <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z" />
              </svg>
            </button>
          </div>

          {/* Set legend when showing sets */}
          {showSets && sets.length > 0 && (
            <div className="set-legend">
              {sets.map((s) => (
                <div key={s.id} className="set-legend-item">
                  <span
                    className="set-legend-swatch"
                    style={{ backgroundColor: SET_COLORS[s.colorIndex % SET_COLORS.length] }}
                  />
                  <span className="set-legend-text">
                    {s.cards.map((c) => formatLabel(`${c.number}-${c.color}-${c.shading}-${c.shape}`)).join(', ')}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
