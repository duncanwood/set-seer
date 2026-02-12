/**
 * Set Game Engine — finds valid Sets from detected cards.
 *
 * A "Set" consists of 3 cards where, for each of the four features
 * (number, color, shading, shape), the values are either all the same
 * or all different across the three cards.
 *
 * Ported from SetGameEngine in Prediction.swift.
 */
import type {
  Detection,
  SetCard,
  SetResult,
  SetNumber,
  SetColor,
  SetShape,
  SetShading,
} from '@/types';

// ─── Feature Values ────────────────────────────────────────────────────────

const NUMBERS: SetNumber[] = [1, 2, 3];
const COLORS: SetColor[] = ['red', 'green', 'purple'];
const SHAPES: SetShape[] = ['diamond', 'oval', 'squiggle'];
const SHADINGS: SetShading[] = ['solid', 'striped', 'empty'];

// ─── Card Parsing ───────────────────────────────────────────────────────────

/**
 * Parse a detection's class name into a SetCard.
 * Class name format: "{number}-{color}-{shading}-{shape}"
 * 
 * Returns null if the class name doesn't match the expected format.
 */
export function parseDetection(detection: Detection): SetCard | null {
  const parts = detection.className.toLowerCase().split('-');
  if (parts.length < 4) return null;

  const number = parseInt(parts[0], 10) as SetNumber;
  if (!NUMBERS.includes(number)) return null;

  const color = parts[1] as SetColor;
  if (!COLORS.includes(color)) return null;

  const shading = parts[2] as SetShading;
  if (!SHADINGS.includes(shading)) return null;

  const shape = parts[3] as SetShape;
  if (!SHAPES.includes(shape)) return null;

  return {
    id: detection.id,
    number,
    color,
    shape,
    shading,
    boundingBox: detection.boundingBox,
  };
}

// ─── Set Finding ────────────────────────────────────────────────────────────

/**
 * Compute a unique signature for a card based on its features.
 * Used as a hash key for O(n²) set-finding and for stable sorting.
 */
export function cardSignature(card: SetCard): string {
  return `${card.number}-${card.color}-${card.shading}-${card.shape}`;
}

/**
 * Given two feature values, determine the completing third value
 * that would make a valid Set (all same or all different).
 */
function getCompleting<T>(a: T, b: T, allValues: T[]): T {
  if (a === b) return a;
  // All different — return the one that's neither a nor b
  return allValues.find((v) => v !== a && v !== b)!;
}

/**
 * Find all valid Sets among the given cards.
 * 
 * Uses an O(n²) algorithm: for each pair of cards, compute the required
 * third card and check if it exists in a map. This is the same approach
 * as SetGameEngine.findSets() in the Swift app.
 */
export function findSets(cards: SetCard[]): SetResult[] {
  if (cards.length < 3) return [];

  // Build a lookup map: signature → cards with that signature
  const cardMap = new Map<string, SetCard[]>();
  for (const card of cards) {
    const sig = cardSignature(card);
    const existing = cardMap.get(sig) ?? [];
    existing.push(card);
    cardMap.set(sig, existing);
  }

  const results: SetResult[] = [];
  const foundSetIds = new Set<string>();

  for (let i = 0; i < cards.length; i++) {
    for (let j = i + 1; j < cards.length; j++) {
      const a = cards[i];
      const b = cards[j];

      // Compute the required third card
      const targetNumber = getCompleting(a.number, b.number, NUMBERS);
      const targetColor = getCompleting(a.color, b.color, COLORS);
      const targetShading = getCompleting(a.shading, b.shading, SHADINGS);
      const targetShape = getCompleting(a.shape, b.shape, SHAPES);

      const targetSig = `${targetNumber}-${targetColor}-${targetShading}-${targetShape}`;
      const matches = cardMap.get(targetSig);

      if (matches) {
        for (const c of matches) {
          if (c.id === a.id || c.id === b.id) continue;

          // Deduplicate: sort IDs and check
          const setSignature = [a.id, b.id, c.id].sort().join(',');
          if (foundSetIds.has(setSignature)) continue;

          foundSetIds.add(setSignature);
          // Stable color index based on feature hash (prevents cycling as IDs change)
          const setFeatures = [cardSignature(a), cardSignature(b), cardSignature(c)].sort().join('|');
          const setHash = setFeatures.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);

          results.push({
            id: `set-${results.length}`,
            cards: [a, b, c],
            colorIndex: setHash,
          });
        }
      }
    }
  }

  return results;
}

/**
 * Convenience: parse detections and find Sets in one call.
 */
export function findSetsFromDetections(detections: Detection[]): {
  cards: SetCard[];
  sets: SetResult[];
} {
  const cards = detections.map(parseDetection).filter((c): c is SetCard => c !== null);
  const sets = findSets(cards);

  // ─── Stable Sorting ───

  // 1. Sort cards within each set
  sets.forEach((set) => {
    set.cards.sort((a, b) => cardSignature(a).localeCompare(cardSignature(b)));
  });

  // 2. Sort the list of sets based on the signature of their first card
  sets.sort((a, b) => {
    const sigA = cardSignature(a.cards[0]);
    const sigB = cardSignature(b.cards[0]);
    return sigA.localeCompare(sigB);
  });

  return { cards, sets };
}
