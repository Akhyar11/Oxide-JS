import Matrix from "../matrix";

export type PaddingSide = "left" | "right";

export interface TrimPaddingBatchResult {
  x: Matrix;
  y: Matrix;
  positionOffset: number;
  effectiveSeqLen: number;
  trimmed: boolean;
}

/**
 * Dynamically trims PAD tokens from the beginning (left) or end (right) of a
 * batch to reduce the effective sequence length processed by the model.
 *
 * Layout contract: data[pos * batchSize + b]
 *   x._shape = [seqLen, batchSize]
 *   y._shape = [seqLen, batchSize]
 *
 * A position is considered "useful" in a given batch sample if either the
 * source token (x) or the target token (y) at that position is not padId.
 *
 * For right-padding:
 *   - Finds the maximum lastUsefulPos across all batch samples.
 *   - Trims to range [0, lastUsefulPos] (inclusive).
 *   - positionOffset = 0 (real tokens still start at absolute position 0).
 *
 * For left-padding:
 *   - Finds the minimum firstUsefulPos across all batch samples.
 *   - Trims to range [firstUsefulPos, seqLen-1] (inclusive).
 *   - positionOffset = firstUsefulPos, so the caller can shift positional
 *     encoding to preserve absolute positions of real tokens.
 */
export function trimPaddingBatch(
  x: Matrix,
  y: Matrix,
  padId: number,
  paddingSide: PaddingSide,
): TrimPaddingBatchResult {
  const [seqLen, batchSize] = x._shape;

  // Guard: only supported for matching shapes and seqLen > 1
  if (
    x._shape[0] !== y._shape[0] ||
    x._shape[1] !== y._shape[1] ||
    seqLen <= 1
  ) {
    return { x, y, positionOffset: 0, effectiveSeqLen: seqLen, trimmed: false };
  }

  const xData = x._data;
  const yData = y._data;

  if (paddingSide === "right") {
    // Find the last position (across all batch samples) that has a non-PAD token.
    let lastUsefulPos = -1;
    for (let pos = 0; pos < seqLen; pos++) {
      for (let b = 0; b < batchSize; b++) {
        const idx = pos * batchSize + b;
        if (xData[idx] !== padId || yData[idx] !== padId) {
          if (pos > lastUsefulPos) lastUsefulPos = pos;
        }
      }
    }

    // If no useful token found, return original to avoid crash.
    if (lastUsefulPos < 0) {
      return { x, y, positionOffset: 0, effectiveSeqLen: seqLen, trimmed: false };
    }

    const newSeqLen = lastUsefulPos + 1;
    if (newSeqLen >= seqLen) {
      return { x, y, positionOffset: 0, effectiveSeqLen: seqLen, trimmed: false };
    }

    // Slice x and y to [0, lastUsefulPos] inclusive.
    const trimmedX = sliceRows(x, 0, newSeqLen, batchSize);
    const trimmedY = sliceRows(y, 0, newSeqLen, batchSize);

    return {
      x: trimmedX,
      y: trimmedY,
      positionOffset: 0,
      effectiveSeqLen: newSeqLen,
      trimmed: true,
    };
  } else {
    // paddingSide === "left"
    // Find the first position (across all batch samples) that has a non-PAD token.
    let firstUsefulPos = seqLen; // sentinel: beyond end
    for (let pos = 0; pos < seqLen; pos++) {
      for (let b = 0; b < batchSize; b++) {
        const idx = pos * batchSize + b;
        if (xData[idx] !== padId || yData[idx] !== padId) {
          if (pos < firstUsefulPos) firstUsefulPos = pos;
        }
      }
    }

    // If no useful token found, return original.
    if (firstUsefulPos >= seqLen) {
      return { x, y, positionOffset: 0, effectiveSeqLen: seqLen, trimmed: false };
    }

    if (firstUsefulPos === 0) {
      return { x, y, positionOffset: 0, effectiveSeqLen: seqLen, trimmed: false };
    }

    const newSeqLen = seqLen - firstUsefulPos;
    const trimmedX = sliceRows(x, firstUsefulPos, newSeqLen, batchSize);
    const trimmedY = sliceRows(y, firstUsefulPos, newSeqLen, batchSize);

    return {
      x: trimmedX,
      y: trimmedY,
      positionOffset: firstUsefulPos,
      effectiveSeqLen: newSeqLen,
      trimmed: true,
    };
  }
}

/**
 * Returns a new Matrix that contains rows [startRow, startRow + count) of src.
 * Data layout: data[pos * batchSize + b] – rows are contiguous in memory,
 * so we can copy the entire block with a single subarray operation.
 */
function sliceRows(src: Matrix, startRow: number, count: number, batchSize: number): Matrix {
  const start = startRow * batchSize;
  const newData = src._data.slice(start, start + count * batchSize);
  return Matrix.fromFlat(newData, [count, batchSize]);
}
