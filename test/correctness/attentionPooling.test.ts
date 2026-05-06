import { mj } from "@oxide-js/core";
import { Matrix } from "@oxide-js/core";
import { AttentionPooling } from "@oxide-js/layers";
import { setLayers } from "@oxide-js/layers";

function assert(condition: boolean, message: string): void {
  if (!condition) throw new Error(message);
}

function assertClose(actual: number, expected: number, eps: number, message: string): void {
  if (Math.abs(actual - expected) > eps) {
    throw new Error(`${message}: expected ${expected}, got ${actual}`);
  }
}

function createLayer(): AttentionPooling {
  const layer = new AttentionPooling({
    units: 2,
    maxTokens: 3,
    alpha: 0.1,
    optimizer: "sgd",
    clipGradient: false,
  });
  layer.load({
    units: 2,
    maxTokens: 3,
    alpha: 0.1,
    optimizer: "sgd",
    clipGradient: false,
    weight: [[1, 0]],
    bias: [[0]],
  });
  return layer;
}

function runMaskedForwardTest(): void {
  const layer = createLayer();
  layer.setValidLength(2);
  const x = mj.matrix([
    [1, 3, 99],
    [10, 20, 999],
  ]);

  const out = layer.forward(x);
  const s0 = Math.exp(1);
  const s1 = Math.exp(3);
  const z = s0 + s1;
  const w0 = s0 / z;
  const w1 = s1 / z;

  assertClose(out.get(0, 0), 1 * w0 + 3 * w1, 1e-6, "AttentionPooling harus menghitung weighted sum row 0");
  assertClose(out.get(1, 0), 10 * w0 + 20 * w1, 1e-6, "AttentionPooling harus menghitung weighted sum row 1");
}

function runBackwardShapeTest(): void {
  const layer = createLayer();
  layer.setValidLength(2);
  const x = mj.matrix([
    [1, 3, 99],
    [10, 20, 999],
  ]);
  layer.forward(x);
  const dx = layer.backward(mj.matrix([]), mj.matrix([[1], [1]]));

  assert(dx._shape[0] === 2 && dx._shape[1] === 3, "AttentionPooling backward harus mengembalikan shape input");
  assert(Number.isFinite(dx.get(0, 0)) && Number.isFinite(dx.get(1, 1)), "Gradient AttentionPooling harus finite");
  assertClose(dx.get(0, 2), 0, 1e-8, "Padding token harus tetap nol pada backward row 0");
  assertClose(dx.get(1, 2), 0, 1e-8, "Padding token harus tetap nol pada backward row 1");
}

function runSaveLoadRegistryTest(): void {
  const layer = createLayer();
  layer.setValidLength(2);
  const saved = layer.save();
  const restored = setLayers([saved])[0] as AttentionPooling;
  restored.setValidLength(2);

  const x = mj.matrix([
    [1, 3, 99],
    [10, 20, 999],
  ]);

  const before = layer.forward(x);
  const after = restored.forward(x);
  assertClose(before.get(0, 0), after.get(0, 0), 1e-8, "setLayers harus dapat restore AttentionPooling row 0");
  assertClose(before.get(1, 0), after.get(1, 0), 1e-8, "setLayers harus dapat restore AttentionPooling row 1");
}

export function runAttentionPoolingCorrectnessSuite(): void {
  runMaskedForwardTest();
  runBackwardShapeTest();
  runSaveLoadRegistryTest();
  console.log("AttentionPooling correctness tests passed");
}
