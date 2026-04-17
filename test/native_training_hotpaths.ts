import mj from "../src/math";
import LayerNormalization from "../src/layers/layerNormalization";
import MultiHeadAttention from "../src/layers/multiHeadAttention";
import { setForceDisableNative } from "../src/math/rust_backend";

function assert(condition: boolean, message: string) {
  if (!condition) {
    throw new Error(message);
  }
}

function assertCloseArray(actual: ArrayLike<number>, expected: ArrayLike<number>, tol: number, label: string) {
  assert(actual.length === expected.length, `${label}: length mismatch ${actual.length} !== ${expected.length}`);
  for (let i = 0; i < actual.length; i++) {
    const diff = Math.abs(actual[i] - expected[i]);
    if (diff > tol) {
      throw new Error(`${label}: mismatch at ${i}, got ${actual[i]}, expected ${expected[i]}, diff ${diff}`);
    }
  }
}

function cloneLayerNorm(src: LayerNormalization): LayerNormalization {
  const cloned = new LayerNormalization({ units: src.units, alpha: 0.01, optimizer: "sgd" });
  cloned.load(src.gamma._value, src.beta._value);
  cloned.compile({ alpha: 0.01, optimizer: "sgd" });
  return cloned;
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

const lnInput = mj.matrix([
  [1.0, 2.0, 0.0, 4.0],
  [2.0, 3.0, 0.0, 5.0],
  [3.0, 4.0, 0.0, 6.0],
  [4.0, 5.0, 0.0, 7.0],
]);
const lnErr = mj.matrix([
  [0.1, 0.2, 0.0, 0.3],
  [0.2, 0.1, 0.0, -0.2],
  [-0.1, 0.3, 0.0, 0.1],
  [0.4, -0.2, 0.0, 0.2],
]);

const baseLn = new LayerNormalization({ units: 4, alpha: 0.01, optimizer: "sgd" });

setForceDisableNative(true);
const jsLn = cloneLayerNorm(baseLn);
jsLn.forward(lnInput);
const jsLnBackward = jsLn.backward(mj.matrix([[]]), lnErr);

setForceDisableNative(false);
const nativeLn = cloneLayerNorm(baseLn);
nativeLn.forward(lnInput);
const nativeLnBackward = nativeLn.backward(mj.matrix([[]]), lnErr);

assertCloseArray(nativeLnBackward._data, jsLnBackward._data, 1e-4, "layernorm backward dx");
assertCloseArray(nativeLn.gamma._data, jsLn.gamma._data, 1e-4, "layernorm gamma update");
assertCloseArray(nativeLn.beta._data, jsLn.beta._data, 1e-4, "layernorm beta update");

const mhaInput = mj.matrix([
  [0.5, 0.4, 0.3, 0.0, 0.7, 0.6],
  [0.1, 0.2, 0.3, 0.0, 0.2, 0.1],
  [0.9, 0.8, 0.7, 0.0, 0.4, 0.5],
  [0.6, 0.5, 0.4, 0.0, 0.3, 0.2],
]);
const mhaErr = mj.matrix([
  [0.2, -0.1, 0.3, 0.0, 0.1, -0.2],
  [0.0, 0.2, -0.1, 0.0, -0.3, 0.4],
  [0.1, 0.0, 0.2, 0.0, 0.3, -0.1],
  [-0.2, 0.1, 0.0, 0.0, 0.2, 0.1],
]);

const mhaBase = new MultiHeadAttention({ units: 4, heads: 2, seqLen: 3, alpha: 0.01, status: "train" });
mhaBase.compile({ alpha: 0.01, optimizer: "sgd" });

setForceDisableNative(true);
const jsMha = cloneMha(mhaBase);
const jsMhaForward = jsMha.forward(mhaInput);
const jsMhaBackward = jsMha.backward(mj.matrix([[]]), mhaErr);

setForceDisableNative(false);
const nativeMha = cloneMha(mhaBase);
const nativeMhaForward = nativeMha.forward(mhaInput);
const nativeMhaBackward = nativeMha.backward(mj.matrix([[]]), mhaErr);

assertCloseArray(nativeMhaForward._data, jsMhaForward._data, 1e-4, "mha forward");
assertCloseArray(nativeMhaBackward._data, jsMhaBackward._data, 1e-3, "mha backward");

setForceDisableNative(false);
console.log("native_training_hotpaths passed");
