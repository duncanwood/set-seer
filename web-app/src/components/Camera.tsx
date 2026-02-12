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
  const [fps, setFps] = useState(targetFps);
  const [showFpsControl, setShowFpsControl] = useState(false);

  // Detection state
  const [detections, setDetections] = useState<Detection[]>([]);
  const [sets, setSets] = useState<SetResult[]>([]);
  const [cards, setCards] = useState<SetCard[]>([]);
  const [inferenceTime, setInferenceTime] = useState<number>(0);

  // Refs for the animation loop (avoids stale closures)
  const fpsRef = useRef(fps);
  const configRef = useRef<InferenceConfig>(DEFAULT_INFERENCE_CONFIG);
  const isProcessingRef = useRef(false);

  useEffect(() => { fpsRef.current = fps; }, [fps]);

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
      drawOverlay(dets, parsedCards, foundSets, video);
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
      video: HTMLVideoElement
    ) => {
      const canvas = overlayRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Match canvas to displayed video size
      const rect = video.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const w = canvas.width;
      const h = canvas.height;

      for (const det of dets) {
        const x = det.boundingBox.x * w;
        const y = det.boundingBox.y * h;
        const bw = det.boundingBox.width * w;
        const bh = det.boundingBox.height * h;

        if (showSets) {
          // Find which sets this detection belongs to
          const participating = currentSets.filter((s) =>
            s.cards.some((c) => c.id === det.id)
          );

          if (participating.length > 0) {
            for (let si = 0; si < participating.length; si++) {
              const expansion = si * 6;
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
            // Not in any set — dim border
            ctx.strokeStyle = 'rgba(200, 200, 200, 0.4)';
            ctx.lineWidth = 1;
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

          // Label
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
    },
    [showSets]
  );

  // Redraw when showSets changes
  useEffect(() => {
    if (webcamRef.current?.video) {
      drawOverlay(detections, cards, sets, webcamRef.current.video);
    }
  }, [showSets, detections, cards, sets, drawOverlay]);

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
            <div className="hud-badge">
              <span className="hud-badge-label">Sets</span>
              <span className="hud-badge-value">{sets.length}</span>
            </div>

            <div className="hud-stats">
              <span>{detections.length} cards</span>
              <span className="hud-dot">·</span>
              <span>{inferenceTime}ms</span>
            </div>

            <button
              className={`hud-btn ${showSets ? 'hud-btn-active' : ''}`}
              onClick={() => setShowSets(!showSets)}
            >
              {showSets ? 'Hide Sets' : 'Show Sets'}
            </button>
          </div>

          {/* Bottom controls */}
          <div className="hud-bottom">
            <button className="hud-btn hud-btn-icon" onClick={toggleCamera} title="Switch camera">
              <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M16 3H8l-2 3H3a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h18a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-3l-2-3z" />
                <circle cx="12" cy="13" r="4" />
              </svg>
            </button>

            <button
              className="hud-btn hud-btn-icon"
              onClick={() => setShowFpsControl(!showFpsControl)}
              title="FPS settings"
            >
              <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="3" />
                <path d="M12 1v2m0 18v2M4.22 4.22l1.42 1.42m12.72 12.72 1.42 1.42M1 12h2m18 0h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
              </svg>
            </button>

            {showFpsControl && (
              <div className="fps-control">
                <label>FPS: {fps}</label>
                <input
                  type="range"
                  min="1"
                  max="30"
                  value={fps}
                  onChange={(e) => setFps(Number(e.target.value))}
                />
              </div>
            )}
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
                    {s.cards.map((c) => formatLabel(c.number + '-' + c.color + '-' + c.shape)).join(', ')}
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
