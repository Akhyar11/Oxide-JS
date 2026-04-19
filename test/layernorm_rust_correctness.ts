import mj from "../src/math";
import LayerNormalization from "../src/layers/layerNormalization";
import { setForceDisableNative } from "../src/math/rust_backend";

function assert(condition: boolean, message: string) {
  if (!condition) throw new Error(message);
}

function assertFiniteArray(values: ArrayLike<number>, label: string) {
  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i])) {
      throw new Error(`${label}: non-finite value at ${i}: ${values[i]}`);
    }
  }
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

function makeLayer(units: number): LayerNormalization {
  const layer = new LayerNormalization({
    units,
    status: "train",
    alpha: 0.01,
    optimizer: "sgd",
  });
  layer.compile({ alpha: 0.01, optimizer: "sgd" });
  return layer;
}

function cloneLayer(src: LayerNormalization): LayerNormalization {
  const cloned = makeLayer(src.units);
  const saved = src.save();
  cloned.load(saved.gamma, saved.beta);
  return cloned;
}

const rows = 4;
const cols = 6;

const x = mj.matrix([
  [0.5, -0.1, 0.0, 0.3, 0.8, -0.3],
  [0.2, 0.4, -0.7, -0.2, 0.1, 0.5],
  [0.7, -0.5, 0.6, 0.6, -0.4, 0.2],
  [0.1, 0.9, -0.2, -0.3, 0.2, -0.1],
]);

const err = mj.matrix([
  [0.2, -0.1, 0.0, 0.1, 0.05, -0.2],
  [0.0, 0.3, -0.4, -0.2, 0.1, 0.2],
  [0.1, -0.2, 0.2, 0.3, -0.1, 0.1],
  [-0.1, 0.2, -0.3, 0.1, 0.2, -0.1],
]);

const base = makeLayer(rows);
base.load(
  [[1.1], [0.9], [1.05], [0.95]],
  [[0.01], [-0.02], [0.03], [0.0]]
);

setForceDisableNative(false);
const nativeLayer = cloneLayer(base);
const nativeForward = nativeLayer.forward(x);
const nativeBackward = nativeLayer.backward(mj.matrix([[]]), err);

assert(nativeForward._shape[0] === rows && nativeForward._shape[1] === cols, "native forward shape mismatch");
assert(nativeBackward._shape[0] === rows && nativeBackward._shape[1] === cols, "native backward shape mismatch");
assertFiniteArray(nativeForward._data, "native forward");
assertFiniteArray(nativeBackward._data, "native backward");
assertFiniteArray((nativeLayer as any).dGammaBuffer._data as Float32Array, "native dGamma");
assertFiniteArray((nativeLayer as any).dBetaBuffer._data as Float32Array, "native dBeta");

setForceDisableNative(true);
const fallbackLayer = cloneLayer(base);
const fallbackForward = fallbackLayer.forward(x);
const fallbackBackward = fallbackLayer.backward(mj.matrix([[]]), err);

assertCloseArray(nativeForward._data, fallbackForward._data, 1e-5, "forward native vs fallback");
assertCloseArray(nativeBackward._data, fallbackBackward._data, 1e-5, "backward dx native vs fallback");
assertCloseArray((nativeLayer as any).dGammaBuffer._data, (fallbackLayer as any).dGammaBuffer._data, 1e-5, "dGamma native vs fallback");
assertCloseArray((nativeLayer as any).dBetaBuffer._data, (fallbackLayer as any).dBetaBuffer._data, 1e-5, "dBeta native vs fallback");
assertCloseArray(nativeLayer.gamma._data, fallbackLayer.gamma._data, 1e-5, "gamma update native vs fallback");
assertCloseArray(nativeLayer.beta._data, fallbackLayer.beta._data, 1e-5, "beta update native vs fallback");

console.log("layernorm_rust_correctness passed");
