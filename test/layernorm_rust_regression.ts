import mj from "../src/math";
import LayerNormalization from "../src/layers/layerNormalization";
import { setForceDisableNative } from "../src/math/rust_backend";

function assert(condition: boolean, message: string) {
  if (!condition) throw new Error(message);
}

function assertCloseArray(actual: ArrayLike<number>, expected: number[], tol: number, label: string) {
  assert(actual.length === expected.length, `${label}: length mismatch ${actual.length} != ${expected.length}`);
  for (let i = 0; i < actual.length; i++) {
    const diff = Math.abs(actual[i] - expected[i]);
    if (diff > tol) {
      throw new Error(`${label}: mismatch at ${i} got=${actual[i]} expected=${expected[i]} diff=${diff}`);
    }
  }
}

const layer = new LayerNormalization({
  units: 3,
  status: "train",
  alpha: 0.01,
  optimizer: "sgd",
});
layer.compile({ alpha: 0.01, optimizer: "sgd" });
layer.load(
  [[1.2], [0.8], [1.1]],
  [[0.1], [-0.05], [0.03]]
);

const x = mj.matrix([
  [0.3, -0.4, 0.7, 0.2],
  [0.1, 0.5, -0.2, -0.3],
  [0.8, -0.1, 0.0, 0.4],
]);
const err = mj.matrix([
  [0.2, -0.1, 0.05, 0.1],
  [-0.3, 0.2, -0.2, 0.4],
  [0.1, 0.3, -0.1, -0.2],
]);

const expectedForward = [
  -0.30759620666503906,
  -1.1828081607818604,
  1.758571743965149,
  0.5075962543487549,
  -0.8651924133300781,
  1.0190068483352661,
  -0.8101787567138672,
  -1.1369231939315796,
  1.524519443511963,
  -0.2639768719673157,
  -0.4451116919517517,
  1.15088951587677,
];

const expectedDx = [
  0.810823917388916,
  -0.4085305631160736,
  -0.00036411621840670705,
  0.39624667167663574,
  -0.5792113542556763,
  -0.20423509180545807,
  -0.001369345816783607,
  -0.1131114736199379,
  -0.2316124588251114,
  0.6127656698226929,
  0.0017335102893412113,
  -0.28313514590263367,
];

const expectedDGamma = [
  0.14204148948192596,
  0.21953195333480835,
  -0.10491622984409332,
];

const expectedDBeta = [
  0.25,
  0.09999999403953552,
  0.10000000894069672,
];

setForceDisableNative(false);
const forward = layer.forward(x);
const backward = layer.backward(mj.matrix([[]]), err);

assertCloseArray(forward._data, expectedForward, 1e-5, "forward regression");
assertCloseArray(backward._data, expectedDx, 1e-5, "dx regression");
assertCloseArray((layer as any).dGammaBuffer._data, expectedDGamma, 1e-5, "dGamma regression");
assertCloseArray((layer as any).dBetaBuffer._data, expectedDBeta, 1e-5, "dBeta regression");

console.log("layernorm_rust_regression passed");
