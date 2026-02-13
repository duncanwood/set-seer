import { Detection, BoundingBox } from '@/types';

/**
 * A track represents a single card being tracked over time.
 */
interface Track {
  id: string;
  box: BoundingBox;
  smoothedBox: BoundingBox;
  hitCount: number;
  missCount: number;
  active: boolean;
  className: string;
  confidence: number;
}

export interface TemporalTrackerConfig {
  iouThreshold: number;
  maxMissCount: number;
  minHitCount: number;
  smoothingAlpha: number;
}

const DEFAULT_CONFIG: TemporalTrackerConfig = {
  iouThreshold: 0.5,
  maxMissCount: 2, // Further reduced to minimize ghosts during rapid motion
  minHitCount: 2,
  smoothingAlpha: 0.5,
};

export class TemporalTracker {
  private tracks: Track[] = [];
  private config: TemporalTrackerConfig;
  private nextId = 0;

  constructor(config: Partial<TemporalTrackerConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Update tracks with new detections from the current frame.
   */
  update(detections: Detection[]): Detection[] {
    const matchedDetections = new Set<number>();
    const matchedTracks = new Set<number>();

    // 1. IoU Matching (Greedy)
    const matches: { trackIdx: number; detIdx: number; iou: number }[] = [];

    for (let t = 0; t < this.tracks.length; t++) {
      for (let d = 0; d < detections.length; d++) {
        const iou = this.calculateIoU(this.tracks[t].box, detections[d].boundingBox);
        if (iou > this.config.iouThreshold) {
          matches.push({ trackIdx: t, detIdx: d, iou });
        }
      }
    }

    // Sort matches by IoU descending
    matches.sort((a, b) => b.iou - a.iou);

    const matchPairs: [number, number][] = [];
    for (const match of matches) {
      if (!matchedTracks.has(match.trackIdx) && !matchedDetections.has(match.detIdx)) {
        matchedTracks.add(match.trackIdx);
        matchedDetections.add(match.detIdx);
        matchPairs.push([match.trackIdx, match.detIdx]);
      }
    }

    // 2. Estimate Global Motion (Median dx, dy)
    let globalDx = 0;
    let globalDy = 0;

    if (matchPairs.length >= 2) {
      const dxs: number[] = [];
      const dys: number[] = [];

      for (const [tIdx, dIdx] of matchPairs) {
        const track = this.tracks[tIdx];
        const det = detections[dIdx];
        
        // Calculate motion based on centers
        const trackCx = track.box.x + track.box.width / 2;
        const trackCy = track.box.y + track.box.height / 2;
        const detCx = det.boundingBox.x + det.boundingBox.width / 2;
        const detCy = det.boundingBox.y + det.boundingBox.height / 2;

        dxs.push(detCx - trackCx);
        dys.push(detCy - trackCy);
      }

      globalDx = this.median(dxs);
      globalDy = this.median(dys);
    }

    // 3. Update Matched Tracks
    for (const [tIdx, dIdx] of matchPairs) {
      const track = this.tracks[tIdx];
      const det = detections[dIdx];

      const alpha = this.config.smoothingAlpha;
      
      track.box = { ...det.boundingBox };
      track.smoothedBox = {
        x: alpha * det.boundingBox.x + (1 - alpha) * track.smoothedBox.x,
        y: alpha * det.boundingBox.y + (1 - alpha) * track.smoothedBox.y,
        width: alpha * det.boundingBox.width + (1 - alpha) * track.smoothedBox.width,
        height: alpha * det.boundingBox.height + (1 - alpha) * track.smoothedBox.height,
      };

      track.hitCount++;
      track.missCount = 0;
      track.className = det.className;
      track.confidence = det.confidence;
      if (track.hitCount >= this.config.minHitCount) {
        track.active = true;
      }
    }

    // 4. Handle Unmatched Tracks (Propagation)
    for (let t = 0; t < this.tracks.length; t++) {
      if (!matchedTracks.has(t)) {
        const track = this.tracks[t];
        track.missCount++;
        
        // Propagate box by global motion
        track.box.x += globalDx;
        track.box.y += globalDy;
        track.smoothedBox.x += globalDx;
        track.smoothedBox.y += globalDy;
      }
    }

    // 5. Handle New Detections
    for (let d = 0; d < detections.length; d++) {
      if (!matchedDetections.has(d)) {
        const det = detections[d];
        this.tracks.push({
          id: `track-${this.nextId++}-${Date.now()}`,
          box: { ...det.boundingBox },
          smoothedBox: { ...det.boundingBox },
          hitCount: 1,
          missCount: 0,
          active: false,
          className: det.className,
          confidence: det.confidence,
        });
      }
    }

    // 6. Cleanup
    this.tracks = this.tracks.filter((t) => t.missCount < this.config.maxMissCount);

    // 7. Track-to-Track Suppression (NMS on active tracks)
    const activeTracks = this.tracks.filter((t) => t.active);
    const keptTracks: Track[] = [];

    // Sort active tracks by hitCount descending to keep more stable ones
    const sortedActive = [...activeTracks].sort((a, b) => b.hitCount - a.hitCount);

    for (const track of sortedActive) {
      let isDuplicate = false;
      for (const kept of keptTracks) {
        if (this.calculateIoU(track.smoothedBox, kept.smoothedBox) > 0.5) {
          isDuplicate = true;
          break;
        }
      }
      if (!isDuplicate) {
        keptTracks.push(track);
      }
    }

    // 8. Return Suppressed Active Tracks as Detections
    return keptTracks.map((t) => ({
      id: t.id,
      className: t.className,
      confidence: t.confidence,
      boundingBox: t.smoothedBox,
    }));
  }

  private calculateIoU(a: BoundingBox, b: BoundingBox): number {
    const x0 = Math.max(a.x, b.x);
    const y0 = Math.max(a.y, b.y);
    const x1 = Math.min(a.x + a.width, b.x + b.width);
    const y1 = Math.min(a.y + a.height, b.y + b.height);

    const intersection = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
    const areaA = a.width * a.height;
    const areaB = b.width * b.height;
    const union = areaA + areaB - intersection;

    return union > 0 ? intersection / union : 0;
  }

  private median(values: number[]): number {
    if (values.length === 0) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    const half = Math.floor(sorted.length / 2);
    if (sorted.length % 2) return sorted[half];
    return (sorted[half - 1] + sorted[half]) / 2.0;
  }
}
