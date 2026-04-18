import mj from "../src/math";
import MultiHeadAttention from "../src/layers/multiHeadAttention";
import { setForceDisableNative } from "../src/math/rust_backend";

function assert(condition: boolean, message: string) {
  if (!condition) throw new Error(message);
}

function assertCloseArray(actual: ArrayLike<number>, expected: ArrayLike<number>, tol: number, label: string) {
  assert(actual.length === expected.length, `${label}: length mismatch ${actual.length} != ${expected.length}`);
  for (let i = 0; i < actual.length; i++) {
    const diff = Math.abs(actual[i] - expected[i]);
    if (diff > tol) {
      throw new Error(`${label}: mismatch at ${i} got=${actual[i]} expected=${expected[i]} diff=${diff}`);
    }
  }
}

function assertFiniteArray(values: ArrayLike<number>, label: string) {
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) {
      throw new Error(`${label}: non-finite value at ${i}: ${v}`);
    }
  }
}

function cloneMha(src: MultiHeadAttention): MultiHeadAttention {
  const cloned = new MultiHeadAttention({
    units: src.units,
    heads: src.heads,
    seqLen: src.seqLen,
    alpha: src.alpha,
    status: src.status,
  });
  cloned.load(src.save());
  cloned.compile({ alpha: src.alpha, optimizer: "sgd" });
  return cloned;
}

function assertMaskingBehavior(layer: MultiHeadAttention, padMask: boolean[], seqLen: number, batchSize: number, heads: number) {
  const attentionData = (layer as any).attentionData as Float32Array;
  for (let head = 0; head < heads; head++) {
    for (let batch = 0; batch < batchSize; batch++) {
      const sampleOffset = batch * seqLen;
      const attnOffset = (head * batchSize + batch) * seqLen * seqLen;
      for (let q = 0; q < seqLen; q++) {
        const qCol = sampleOffset + q;
        for (let k = 0; k < seqLen; k++) {
          const kCol = sampleOffset + k;
          const v = attentionData[attnOffset + k * seqLen + q];
          if (padMask[qCol]) {
            assert(v === 0, `padded query must have zero attention: head=${head} batch=${batch} q=${q} k=${k}`);
            continue;
          }
          if (padMask[kCol] || k > q) {
            assert(Math.abs(v) < 1e-7, `masked attention must be zero: head=${head} batch=${batch} q=${q} k=${k} value=${v}`);
          }
        }
      }
    }
  }
}

const units = 4;
const heads = 2;
const seqLen = 3;
const batchSize = 2;

const x = mj.matrix([
  [0.5, -0.1, 0.0, 0.3, 0.8, 0.0],
  [0.2, 0.4, 0.0, -0.2, 0.1, 0.0],
  [0.7, -0.5, 0.0, 0.6, -0.4, 0.0],
  [0.1, 0.9, 0.0, -0.3, 0.2, 0.0],
]);

const dErr = mj.matrix([
  [0.2, -0.1, 0.0, 0.1, 0.05, 0.0],
  [0.0, 0.3, 0.0, -0.2, 0.1, 0.0],
  [0.1, -0.2, 0.0, 0.3, -0.1, 0.0],
  [-0.1, 0.2, 0.0, 0.1, 0.2, 0.0],
]);

const padMask = [false, false, true, false, false, true];

const base = new MultiHeadAttention({ units, heads, seqLen, alpha: 0.01, status: "train" });
base.compile({ alpha: 0.01, optimizer: "sgd" });

setForceDisableNative(false);
const nativeLayer = cloneMha(base);
nativeLayer.setPadMask([...padMask]);
const nativeForward = nativeLayer.forward(x);
const nativeBackward = nativeLayer.backward(mj.matrix([[]]), dErr);

assert(nativeForward._shape[0] === units && nativeForward._shape[1] === seqLen * batchSize, "forward output shape mismatch");
assert(nativeBackward._shape[0] === units && nativeBackward._shape[1] === seqLen * batchSize, "backward output shape mismatch");

assertFiniteArray(nativeForward._data, "native forward output");
assertFiniteArray(nativeBackward._data, "native backward output");
assertFiniteArray((nativeLayer as any).dQAll._data as Float32Array, "native dQ");
assertFiniteArray((nativeLayer as any).dKAll._data as Float32Array, "native dK");
assertFiniteArray((nativeLayer as any).dVAll._data as Float32Array, "native dV");

assertMaskingBehavior(nativeLayer, padMask, seqLen, batchSize, heads);

setForceDisableNative(true);
const baselineLayer = cloneMha(base);
baselineLayer.setPadMask([...padMask]);
const baselineForward = baselineLayer.forward(x);
const baselineBackward = baselineLayer.backward(mj.matrix([[]]), dErr);

assertCloseArray(nativeForward._data, baselineForward._data, 1e-4, "native vs fallback forward");
assertCloseArray(nativeBackward._data, baselineBackward._data, 1e-3, "native vs fallback backward");
assertCloseArray(nativeLayer.q._data, baselineLayer.q._data, 1e-3, "native vs fallback q update");
assertCloseArray(nativeLayer.k._data, baselineLayer.k._data, 1e-3, "native vs fallback k update");
assertCloseArray(nativeLayer.v._data, baselineLayer.v._data, 1e-3, "native vs fallback v update");

setForceDisableNative(false);
console.log("mha_rust_regression passed");
