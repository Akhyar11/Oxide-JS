/**
 * Tests for PR: residual buffer reuse and addInto optimization
 *
 * Covers:
 * 1. add/sub with optional `out` parameter (new API)
 * 2. addInto / subInto helper functions (new API)
 * 3. Multiple buffer reuse cycles
 * 4. Value consistency: legacy path vs `out` path
 * 5. Transformer residual path stability (multiple forward passes)
 * 6. Embedding forward zero-fill behavior
 *
 * Jalankan: npx ts-node test/test_addinto_buffer_reuse.ts
 */

import mj from "../src/math";
import { Embedding } from "../src/layers";
import { Transformers } from "../src/models";

let passed = 0;
let failed = 0;

function assert(condition: boolean, name: string) {
  if (condition) {
    console.log(`  ✅ PASS: ${name}`);
    passed++;
  } else {
    console.error(`  ❌ FAIL: ${name}`);
    failed++;
  }
}

function assertClose(a: number, b: number, tol = 1e-5, name: string = "") {
  assert(Math.abs(a - b) < tol, `${name} (got ${a}, expected ${b})`);
}

function allFinite(m: any): boolean {
  for (let i = 0; i < m._data.length; i++) {
    if (!Number.isFinite(m._data[i])) return false;
  }
  return true;
}

// ============================================================
// SECTION 1: add() with optional `out` parameter
// ============================================================
console.log("\n=== 1. add() with out parameter ===");

{
  const a = mj.matrix([[1, 2], [3, 4]]);
  const b = mj.matrix([[10, 20], [30, 40]]);
  const out = mj.zeros([2, 2]);

  const result = mj.add(a, b, out);

  // Returns the out reference, not a new matrix
  assert(result === out, "add(a,b,out) returns out reference");

  // Values are correct
  assert(out._value[0][0] === 11 && out._value[0][1] === 22, "add(a,b,out) row 0 correct");
  assert(out._value[1][0] === 33 && out._value[1][1] === 44, "add(a,b,out) row 1 correct");

  // The underlying typed array buffer is reused (same reference)
  const dataBefore = out._data;
  mj.add(a, b, out);
  assert(out._data === dataBefore, "add(a,b,out) reuses _data buffer across calls");
}

{
  // Scalar + matrix with out
  const b = mj.matrix([[5, 6], [7, 8]]);
  const out = mj.zeros([2, 2]);
  const result = mj.add(10, b, out);
  assert(result === out, "add(scalar, matrix, out) returns out reference");
  assert(out._value[0][0] === 15 && out._value[1][1] === 18, "add(scalar, matrix, out) values correct");
}

{
  // Matrix + scalar with out
  const a = mj.matrix([[1, 2], [3, 4]]);
  const out = mj.zeros([2, 2]);
  const result = mj.add(a, 100, out);
  assert(result === out, "add(matrix, scalar, out) returns out reference");
  assert(out._value[0][0] === 101 && out._value[1][1] === 104, "add(matrix, scalar, out) values correct");
}

{
  // Shape mismatch in out parameter should throw
  const a = mj.matrix([[1, 2], [3, 4]]);
  const b = mj.matrix([[5, 6], [7, 8]]);
  const wrongOut = mj.zeros([3, 3]);
  let threw = false;
  try {
    mj.add(a, b, wrongOut);
  } catch (e) {
    threw = true;
  }
  assert(threw, "add(a,b,out) throws on out shape mismatch");
}

// ============================================================
// SECTION 2: sub() with optional `out` parameter
// ============================================================
console.log("\n=== 2. sub() with out parameter ===");

{
  const a = mj.matrix([[10, 20], [30, 40]]);
  const b = mj.matrix([[1, 2], [3, 4]]);
  const out = mj.zeros([2, 2]);

  const result = mj.sub(a, b, out);

  assert(result === out, "sub(a,b,out) returns out reference");
  assert(out._value[0][0] === 9 && out._value[0][1] === 18, "sub(a,b,out) row 0 correct");
  assert(out._value[1][0] === 27 && out._value[1][1] === 36, "sub(a,b,out) row 1 correct");

  // Buffer reuse
  const dataBefore = out._data;
  mj.sub(a, b, out);
  assert(out._data === dataBefore, "sub(a,b,out) reuses _data buffer across calls");
}

{
  // Scalar - matrix with out
  const b = mj.matrix([[1, 2], [3, 4]]);
  const out = mj.zeros([2, 2]);
  const result = mj.sub(100, b, out);
  assert(result === out, "sub(scalar, matrix, out) returns out reference");
  assert(out._value[0][0] === 99 && out._value[1][1] === 96, "sub(scalar, matrix, out) values correct");
}

{
  // Matrix - scalar with out
  const a = mj.matrix([[10, 20], [30, 40]]);
  const out = mj.zeros([2, 2]);
  const result = mj.sub(a, 5, out);
  assert(result === out, "sub(matrix, scalar, out) returns out reference");
  assert(out._value[0][0] === 5 && out._value[1][1] === 35, "sub(matrix, scalar, out) values correct");
}

{
  // Shape mismatch in out parameter should throw
  const a = mj.matrix([[1, 2], [3, 4]]);
  const b = mj.matrix([[5, 6], [7, 8]]);
  const wrongOut = mj.zeros([1, 4]);
  let threw = false;
  try {
    mj.sub(a, b, wrongOut);
  } catch (e) {
    threw = true;
  }
  assert(threw, "sub(a,b,out) throws on out shape mismatch");
}

// ============================================================
// SECTION 3: addInto() helper
// ============================================================
console.log("\n=== 3. addInto() helper ===");

{
  const a = mj.matrix([[2, 4], [6, 8]]);
  const b = mj.matrix([[1, 1], [1, 1]]);
  const out = mj.zeros([2, 2]);

  const result = mj.addInto(a, b, out);

  assert(result === out, "addInto returns out reference");

  // All values checked
  const v = out._value;
  assert(v[0][0] === 3 && v[0][1] === 5 && v[1][0] === 7 && v[1][1] === 9, "addInto all values correct");

  // Buffer identity preserved
  const dataRef = out._data;
  mj.addInto(a, b, out);
  assert(out._data === dataRef, "addInto preserves _data buffer identity across calls");
}

{
  // Non-square matrix
  const a = mj.matrix([[1, 2, 3], [4, 5, 6]]);
  const b = mj.matrix([[10, 20, 30], [40, 50, 60]]);
  const out = mj.zeros([2, 3]);

  mj.addInto(a, b, out);
  const v = out._value;
  assert(v[0][0] === 11 && v[0][2] === 33 && v[1][0] === 44 && v[1][2] === 66, "addInto non-square matrix correct");
}

{
  // 1x1 matrix edge case
  const a = mj.matrix([[7]]);
  const b = mj.matrix([[3]]);
  const out = mj.zeros([1, 1]);
  mj.addInto(a, b, out);
  assert(out._value[0][0] === 10, "addInto 1x1 matrix correct");
}

// ============================================================
// SECTION 4: subInto() helper
// ============================================================
console.log("\n=== 4. subInto() helper ===");

{
  const a = mj.matrix([[10, 20], [30, 40]]);
  const b = mj.matrix([[3, 4], [5, 6]]);
  const out = mj.zeros([2, 2]);

  const result = mj.subInto(a, b, out);

  assert(result === out, "subInto returns out reference");

  const v = out._value;
  assert(v[0][0] === 7 && v[0][1] === 16 && v[1][0] === 25 && v[1][1] === 34, "subInto all values correct");

  // Buffer identity preserved
  const dataRef = out._data;
  mj.subInto(a, b, out);
  assert(out._data === dataRef, "subInto preserves _data buffer identity across calls");
}

{
  // Non-square matrix
  const a = mj.matrix([[100, 200, 300]]);
  const b = mj.matrix([[1, 2, 3]]);
  const out = mj.zeros([1, 3]);

  mj.subInto(a, b, out);
  const v = out._value;
  assert(v[0][0] === 99 && v[0][1] === 198 && v[0][2] === 297, "subInto non-square matrix correct");
}

{
  // 1x1 matrix edge case
  const a = mj.matrix([[100]]);
  const b = mj.matrix([[37]]);
  const out = mj.zeros([1, 1]);
  mj.subInto(a, b, out);
  assert(out._value[0][0] === 63, "subInto 1x1 matrix correct");
}

// ============================================================
// SECTION 5: Multiple reuse cycles with same buffer
// ============================================================
console.log("\n=== 5. Multiple buffer reuse cycles ===");

{
  const out = mj.zeros([2, 2]);

  // Cycle 1
  const a1 = mj.matrix([[1, 2], [3, 4]]);
  const b1 = mj.matrix([[10, 20], [30, 40]]);
  mj.addInto(a1, b1, out);
  assert(out._value[0][0] === 11 && out._value[1][1] === 44, "addInto reuse cycle 1 correct");

  // Cycle 2 (different values)
  const a2 = mj.matrix([[5, 5], [5, 5]]);
  const b2 = mj.matrix([[1, 2], [3, 4]]);
  mj.addInto(a2, b2, out);
  assert(out._value[0][0] === 6 && out._value[0][1] === 7 && out._value[1][1] === 9, "addInto reuse cycle 2 correct");

  // Cycle 3
  const a3 = mj.matrix([[0, 0], [0, 0]]);
  const b3 = mj.matrix([[7, 8], [9, 10]]);
  mj.addInto(a3, b3, out);
  assert(out._value[0][0] === 7 && out._value[1][1] === 10, "addInto reuse cycle 3 correct");
}

{
  const out = mj.zeros([2, 2]);

  // Cycle 1
  const a1 = mj.matrix([[100, 200], [300, 400]]);
  const b1 = mj.matrix([[1, 2], [3, 4]]);
  mj.subInto(a1, b1, out);
  assert(out._value[0][0] === 99 && out._value[1][1] === 396, "subInto reuse cycle 1 correct");

  // Cycle 2
  const a2 = mj.matrix([[50, 50], [50, 50]]);
  const b2 = mj.matrix([[25, 25], [25, 25]]);
  mj.subInto(a2, b2, out);
  assert(out._value[0][0] === 25 && out._value[1][1] === 25, "subInto reuse cycle 2 correct");

  // Cycle 3
  const a3 = mj.matrix([[10, 20], [30, 40]]);
  const b3 = mj.matrix([[10, 20], [30, 40]]);
  mj.subInto(a3, b3, out);
  assert(out._value[0][0] === 0 && out._value[1][1] === 0, "subInto reuse cycle 3 (zero result) correct");
}

// ============================================================
// SECTION 6: Value consistency - legacy vs out path
// ============================================================
console.log("\n=== 6. Legacy vs out path value consistency ===");

{
  // Various shapes to ensure consistency
  const shapes: [number, number][] = [[1, 1], [3, 3], [2, 5], [8, 8], [4, 1], [1, 6]];

  for (const [r, c] of shapes) {
    const a = mj.random([r, c]);
    const b = mj.random([r, c]);
    const legacy = mj.add(a, b);
    const out = mj.zeros([r, c]);
    mj.addInto(a, b, out);

    let allMatch = true;
    for (let i = 0; i < r * c; i++) {
      if (Math.abs(legacy._data[i] - out._data[i]) > 1e-6) {
        allMatch = false;
        break;
      }
    }
    assert(allMatch, `addInto vs add legacy consistent for shape [${r}x${c}]`);
  }
}

{
  const shapes: [number, number][] = [[1, 1], [3, 3], [2, 5], [8, 8], [4, 1], [1, 6]];

  for (const [r, c] of shapes) {
    const a = mj.random([r, c]);
    const b = mj.random([r, c]);
    const legacy = mj.sub(a, b);
    const out = mj.zeros([r, c]);
    mj.subInto(a, b, out);

    let allMatch = true;
    for (let i = 0; i < r * c; i++) {
      if (Math.abs(legacy._data[i] - out._data[i]) > 1e-6) {
        allMatch = false;
        break;
      }
    }
    assert(allMatch, `subInto vs sub legacy consistent for shape [${r}x${c}]`);
  }
}

// ============================================================
// SECTION 7: Transformer residual buffer stability
// ============================================================
console.log("\n=== 7. Transformer residual path stability ===");

{
  const transformer = new Transformers({
    units: 8,
    seqLen: 4,
    vocabSize: 20,
    heads: 2,
    dropoutRate: 0,
    alpha: 0.001,
    padTokenId: 0,
  });

  // Input: seqLen=4, batch=2
  const input1 = mj.matrix([[1, 2], [3, 0], [4, 1], [2, 3]]);
  const target1 = mj.matrix([[5, 8]]);

  // First forward pass
  const out1 = transformer.forward(input1);
  assert(out1._shape[0] === 20 && out1._shape[1] === 2, "transformer forward pass 1 output shape [vocabSize x batch]");
  assert(allFinite(out1), "transformer forward pass 1 output is all finite");

  // Backward
  transformer.backward(target1);
  assert(Number.isFinite(transformer.loss), "transformer backward pass 1 loss is finite");

  // Second forward pass (same input) — should not produce NaN from corrupted residual buffers
  const out2 = transformer.forward(input1);
  assert(allFinite(out2), "transformer forward pass 2 output is all finite (no buffer corruption)");
  assert(out2._shape[0] === 20 && out2._shape[1] === 2, "transformer forward pass 2 shape consistent");

  // Third forward with different input
  const input3 = mj.matrix([[5, 6], [7, 8], [9, 10], [11, 12]]);
  const out3 = transformer.forward(input3);
  assert(allFinite(out3), "transformer forward pass 3 with different input is all finite");
}

{
  // Transformer with single-sample batch (batch=1)
  const transformer = new Transformers({
    units: 8,
    seqLen: 6,
    vocabSize: 32,
    heads: 2,
    dropoutRate: 0,
    alpha: 0.001,
    padTokenId: 0,
  });

  const input = mj.matrix([[1], [2], [3], [0], [4], [5]]);
  const out = transformer.forward(input);
  assert(out._shape[0] === 32 && out._shape[1] === 1, "transformer single-batch forward output shape [vocabSize x 1]");
  assert(allFinite(out), "transformer single-batch forward output is all finite");

  transformer.backward(mj.matrix([[7]]));
  assert(Number.isFinite(transformer.loss), "transformer single-batch backward loss is finite");
}

// ============================================================
// SECTION 8: Embedding forward zero-fill behavior
// ============================================================
console.log("\n=== 8. Embedding forward zero-fill behavior ===");

{
  const vocabSize = 10;
  const embeddingDim = 4;
  const padTokenId = 0;

  const emb = new Embedding({ vocabSize, embeddingDim, padTokenId });

  // Input: seqLen=3, batch=1 (3 tokens)
  const input = mj.matrix([[1], [2], [3]]);
  const out = emb.forward(input);

  // Output shape should be [embeddingDim, seqLen*batch] = [4, 3]
  assert(out._shape[0] === embeddingDim && out._shape[1] === 3, "embedding forward output shape [embeddingDim x seqLen]");
  assert(allFinite(out), "embedding forward output is all finite");
}

{
  // Pad token produces zero vector
  const vocabSize = 10;
  const embeddingDim = 4;
  const padTokenId = 0;

  const emb = new Embedding({ vocabSize, embeddingDim, padTokenId });

  // Single pad token
  const input = mj.matrix([[0]]);
  const out = emb.forward(input);

  assert(out._shape[0] === embeddingDim && out._shape[1] === 1, "embedding pad-only input shape correct");

  // All values for pad token should be zero
  let allZero = true;
  for (let i = 0; i < out._data.length; i++) {
    if (out._data[i] !== 0) { allZero = false; break; }
  }
  assert(allZero, "embedding pad token produces zero vector");
}

{
  // Mixed pad and real tokens
  const vocabSize = 16;
  const embeddingDim = 8;
  const padTokenId = 0;

  const emb = new Embedding({ vocabSize, embeddingDim, padTokenId });

  // Input with pad token at beginning: [[0],[0],[5],[7]] as seqLen=4, batch=1
  const input = mj.matrix([[0], [0], [5], [7]]);
  const out = emb.forward(input);

  // Shape: [8, 4]
  assert(out._shape[0] === embeddingDim && out._shape[1] === 4, "embedding mixed input shape correct");
  assert(allFinite(out), "embedding mixed input output all finite");

  // Columns 0 and 1 (pad tokens) should be zero
  // layout: col j = all embeddingDim rows for token j
  let padCols0Zero = true;
  let padCols1Zero = true;
  for (let i = 0; i < embeddingDim; i++) {
    if (out._data[i * 4 + 0] !== 0) padCols0Zero = false;
    if (out._data[i * 4 + 1] !== 0) padCols1Zero = false;
  }
  assert(padCols0Zero, "embedding pad token at position 0 is all zeros");
  assert(padCols1Zero, "embedding pad token at position 1 is all zeros");
}

{
  // Multiple forward passes produce consistent results (buffer reuse correctness)
  const vocabSize = 12;
  const embeddingDim = 6;
  const padTokenId = 0;

  const emb = new Embedding({ vocabSize, embeddingDim, padTokenId });

  const input = mj.matrix([[1], [2], [3]]);

  const out1 = emb.forward(input);
  // Snapshot data from first pass
  const snap1 = Array.from(out1._data);

  const out2 = emb.forward(input);
  const snap2 = Array.from(out2._data);

  // Results should be identical across passes (deterministic lookup)
  let consistent = true;
  for (let i = 0; i < snap1.length; i++) {
    if (snap1[i] !== snap2[i]) { consistent = false; break; }
  }
  assert(consistent, "embedding forward produces identical results on repeated calls with same input");
}

// ============================================================
// SECTION 9: Regression / boundary cases
// ============================================================
console.log("\n=== 9. Regression and boundary cases ===");

{
  // addInto with all-negative values
  const a = mj.matrix([[-1, -2], [-3, -4]]);
  const b = mj.matrix([[-5, -6], [-7, -8]]);
  const out = mj.zeros([2, 2]);
  mj.addInto(a, b, out);
  assert(out._value[0][0] === -6 && out._value[1][1] === -12, "addInto with all-negative values correct");
}

{
  // subInto with all-negative values
  const a = mj.matrix([[-1, -2], [-3, -4]]);
  const b = mj.matrix([[-5, -6], [-7, -8]]);
  const out = mj.zeros([2, 2]);
  mj.subInto(a, b, out);
  assert(out._value[0][0] === 4 && out._value[1][1] === 4, "subInto with all-negative values correct");
}

{
  // addInto with zeros
  const a = mj.zeros([3, 3]);
  const b = mj.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
  const out = mj.zeros([3, 3]);
  mj.addInto(a, b, out);
  assert(out._value[0][0] === 1 && out._value[2][2] === 9, "addInto with zero matrix (identity property)");
}

{
  // subInto self-subtraction should give zero
  const a = mj.matrix([[5, 10], [15, 20]]);
  const out = mj.zeros([2, 2]);
  mj.subInto(a, a, out);
  assert(out._value[0][0] === 0 && out._value[0][1] === 0 && out._value[1][0] === 0 && out._value[1][1] === 0,
    "subInto self-subtraction gives all zeros");
}

{
  // Large matrix addInto correctness (column-major check)
  const rows = 16;
  const cols = 32;
  const aData: number[][] = [];
  const bData: number[][] = [];
  for (let i = 0; i < rows; i++) {
    aData.push([]);
    bData.push([]);
    for (let j = 0; j < cols; j++) {
      aData[i].push(i * cols + j);
      bData[i].push(1);
    }
  }
  const a = mj.matrix(aData);
  const b = mj.matrix(bData);
  const out = mj.zeros([rows, cols]);
  mj.addInto(a, b, out);

  let allCorrect = true;
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      const expected = i * cols + j + 1;
      const got = out._data[i * cols + j];
      if (Math.abs(got - expected) > 1e-5) { allCorrect = false; break; }
    }
    if (!allCorrect) break;
  }
  assert(allCorrect, "addInto large matrix [16x32] all elements correct");
}

{
  // addInto commutativity: add(a,b,out) === add(b,a,out2)
  const a = mj.matrix([[3, 5], [7, 9]]);
  const b = mj.matrix([[2, 4], [6, 8]]);
  const out1 = mj.zeros([2, 2]);
  const out2 = mj.zeros([2, 2]);
  mj.addInto(a, b, out1);
  mj.addInto(b, a, out2);

  let symmetric = true;
  for (let i = 0; i < out1._data.length; i++) {
    if (Math.abs(out1._data[i] - out2._data[i]) > 1e-6) { symmetric = false; break; }
  }
  assert(symmetric, "addInto is commutative (a+b === b+a)");
}

// ============================================================
// SUMMARY
// ============================================================
console.log(`\n${"=".repeat(50)}`);
console.log(`✅ PASSED: ${passed}  ❌ FAILED: ${failed}`);
console.log(`${"=".repeat(50)}`);
if (failed > 0) process.exit(1);