import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import mj from "../../src/math";
import { isNativeAvailable, setForceDisableNative } from "../../src/math/rust_backend";
import Transformers from "../../src/models/transformers";

function test(name: string, fn: () => void) {
  try {
    fn();
    console.log(`PASS ${name}`);
  } catch (error) {
    console.error(`FAIL ${name}`);
    throw error;
  }
}

function cloneModel(model: Transformers, config: ConstructorParameters<typeof Transformers>[0]): Transformers {
  const dir = mkdtempSync(join(tmpdir(), "ml-v2-transformer-"));
  const file = join(dir, "model.json");
  try {
    model.save(file);
    const clone = new Transformers(config);
    clone.load(file);
    return clone;
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

function shiftedTargets(x: number[][], padTokenId: number) {
  return mj.matrix(
    x.map((row, rowIndex) => {
      if (rowIndex === x.length - 1) {
        return row.map(() => padTokenId);
      }
      return x[rowIndex + 1];
    })
  );
}

function assertFiniteMatrix(name: string, matrix: { _data: Float32Array | number[] }) {
  for (const value of matrix._data) {
    assert.ok(Number.isFinite(value), `${name} contains non-finite value ${value}`);
  }
}

const baseConfig = {
  units: 8,
  seqLen: 4,
  vocabSize: 16,
  heads: 2,
  alpha: 1e-3,
  dropoutRate: 0.4,
  padTokenId: 0,
  clipGradient: false,
};

const multiBlockConfig = {
  ...baseConfig,
  numBlocks: 2,
};

test("Transformers training forward returns full-sequence logits", () => {
  const model = new Transformers(baseConfig);
  model.train();
  const x = mj.matrix([
    [0, 0],
    [5, 8],
    [6, 9],
    [7, 10],
  ]);

  const out = model.forward(x);

  assert.deepEqual(out._shape, [baseConfig.vocabSize, baseConfig.seqLen * 2]);
});

test("Transformers inference path returns last-token logits", () => {
  const model = new Transformers(baseConfig);
  model.eval();
  const x = mj.matrix([
    [0, 0],
    [5, 8],
    [6, 9],
    [7, 10],
  ]);

  const out = model.forwardNextToken(x);
  const pred = model.predict(x);

  assert.deepEqual(out._shape, [baseConfig.vocabSize, 2]);
  assert.deepEqual(pred._shape, [baseConfig.vocabSize, 2]);
});

test("Transformers multi-block training and inference paths keep expected shapes", () => {
  const model = new Transformers(multiBlockConfig);
  const x = mj.matrix([
    [0, 0],
    [5, 8],
    [6, 9],
    [7, 10],
  ]);

  model.train();
  const trainOut = model.forward(x);
  model.eval();
  const inferOut = model.forwardNextToken(x);

  assert.deepEqual(trainOut._shape, [multiBlockConfig.vocabSize, multiBlockConfig.seqLen * 2]);
  assert.deepEqual(inferOut._shape, [multiBlockConfig.vocabSize, 2]);
});

test("Transformers forwardNextToken matches last position of full-sequence logits in eval mode", () => {
  const model = new Transformers({ ...baseConfig, dropoutRate: 0 });
  model.eval();
  const x = mj.matrix([
    [0, 0],
    [5, 8],
    [6, 9],
    [7, 10],
  ]);

  const nextToken = model.forwardNextToken(x);
  const full = model.forwardFullSequence(x);
  const totalCols = baseConfig.seqLen * 2;

  for (let b = 0; b < 2; b++) {
    const lastTokenCol = (b + 1) * baseConfig.seqLen - 1;
    for (let vocab = 0; vocab < baseConfig.vocabSize; vocab++) {
      const fullIndex = vocab * totalCols + lastTokenCol;
      const nextIndex = vocab * 2 + b;
      assert.ok(Math.abs(full._data[fullIndex] - nextToken._data[nextIndex]) <= 1e-6);
    }
  }
});

test("Transformers predict uses eval path temporarily and restores train mode", () => {
  const model = new Transformers(baseConfig);
  const x = mj.matrix([
    [0],
    [5],
    [6],
    [7],
  ]);

  model.train();
  const pred = model.predict(x);
  const trainOut = model.forward(x);

  assert.deepEqual(pred._shape, [baseConfig.vocabSize, 1]);
  assert.deepEqual(trainOut._shape, [baseConfig.vocabSize, baseConfig.seqLen]);
});

test("Transformers causal mask blocks future-token influence on earlier logits", () => {
  const model = new Transformers({ ...baseConfig, dropoutRate: 0 });
  model.eval();
  const xA = mj.matrix([[1], [2], [3], [4]]);
  const xB = mj.matrix([[1], [2], [3], [9]]);

  const logitsA = Array.from(model.forwardFullSequence(xA)._data);
  const logitsB = Array.from(model.forwardFullSequence(xB)._data);

  for (let vocab = 0; vocab < baseConfig.vocabSize; vocab++) {
    for (let pos = 0; pos < 3; pos++) {
      const idx = vocab * 4 + pos;
      assert.ok(
        Math.abs(logitsA[idx] - logitsB[idx]) <= 1e-6,
        `future token leaked into causal position vocab=${vocab} pos=${pos}`
      );
    }
  }
});

test("Transformers full-sequence loss ignores pad positions", () => {
  const seed = new Transformers({ ...baseConfig, dropoutRate: 0 });
  const modelA = cloneModel(seed, { ...baseConfig, dropoutRate: 0 });
  const modelB = cloneModel(seed, { ...baseConfig, dropoutRate: 0 });
  modelA.train();
  modelB.train();

  const xRows = [
    [0, 0],
    [5, 8],
    [6, 9],
    [7, 10],
  ];
  const x = mj.matrix(xRows);
  const yA = shiftedTargets(xRows, 0);
  const yB = mj.matrix([
    [11, 12],
    [6, 9],
    [7, 10],
    [0, 0],
  ]);

  modelA.forward(x);
  modelA.backward(yA);
  modelB.forward(x);
  modelB.backward(yB);

  const weightA = ((modelA as any).dense.weight as { _data: Float32Array })._data;
  const weightB = ((modelB as any).dense.weight as { _data: Float32Array })._data;
  assert.equal(weightA.length, weightB.length);
  for (let i = 0; i < weightA.length; i++) {
    assert.ok(Math.abs(weightA[i] - weightB[i]) <= 1e-6, `pad position changed dense weight at ${i}`);
  }
});

test("Transformers dropout is stochastic in train mode and deterministic in eval mode", () => {
  const model = new Transformers(baseConfig);
  const x = mj.matrix([
    [1],
    [2],
    [3],
    [4],
  ]);

  model.train();
  const trainA = Array.from(model.forward(x)._data);
  const trainB = Array.from(model.forward(x)._data);
  let trainDiffers = false;
  for (let i = 0; i < trainA.length; i++) {
    if (Math.abs(trainA[i] - trainB[i]) > 1e-6) {
      trainDiffers = true;
      break;
    }
  }
  assert.equal(trainDiffers, true);

  model.eval();
  const evalA = Array.from(model.forwardFullSequence(x)._data);
  const evalB = Array.from(model.forwardFullSequence(x)._data);
  for (let i = 0; i < evalA.length; i++) {
    assert.ok(Math.abs(evalA[i] - evalB[i]) <= 1e-6, `eval dropout changed value at ${i}`);
  }
});

test("Transformers backward keeps gradients and parameters finite on full-sequence targets", () => {
  const model = new Transformers({ ...baseConfig, dropoutRate: 0 });
  model.train();
  const xRows = [
    [0, 0],
    [5, 8],
    [6, 9],
    [7, 10],
  ];
  const x = mj.matrix(xRows);
  const y = shiftedTargets(xRows, 0);

  const logits = model.forward(x);
  model.backward(y);

  assertFiniteMatrix("forward logits", logits);
  assert.ok(Number.isFinite(model.loss), `loss is not finite: ${model.loss}`);
  assertFiniteMatrix("dense weight", (model as any).dense.weight);
  assertFiniteMatrix("embedding weight", (model as any).embedding.weight);
  assertFiniteMatrix("attention q", (model as any).mha.q);
});

test("Transformers multi-block save/load roundtrip preserves eval logits", () => {
  const seed = new Transformers({ ...multiBlockConfig, dropoutRate: 0 });
  seed.eval();
  const clone = cloneModel(seed, { ...multiBlockConfig, dropoutRate: 0 });
  clone.eval();
  const x = mj.matrix([
    [0, 0],
    [5, 8],
    [6, 9],
    [7, 10],
  ]);

  const seedLogits = seed.forwardFullSequence(x);
  const cloneLogits = clone.forwardFullSequence(x);

  assert.deepEqual(seedLogits._shape, cloneLogits._shape);
  for (let i = 0; i < seedLogits._data.length; i++) {
    assert.ok(Math.abs(seedLogits._data[i] - cloneLogits._data[i]) <= 1e-6, `multi-block logits mismatch at ${i}`);
  }
});

test("Transformers native masked sparse loss matches JS fallback on small full-sequence batch", () => {
  if (!isNativeAvailable()) {
    console.log("SKIP Transformers native masked sparse loss matches JS fallback on small full-sequence batch");
    return;
  }

  const seed = new Transformers({ ...baseConfig, dropoutRate: 0, alpha: 1e-4 });
  const nativeModel = cloneModel(seed, { ...baseConfig, dropoutRate: 0, alpha: 1e-4 });
  const jsModel = cloneModel(seed, { ...baseConfig, dropoutRate: 0, alpha: 1e-4 });
  const xRows = [
    [0, 0],
    [5, 8],
    [6, 9],
    [7, 10],
  ];
  const x = mj.matrix(xRows);
  const y = shiftedTargets(xRows, 0);

  nativeModel.train();
  jsModel.train();

  setForceDisableNative(false);
  nativeModel.forward(x);
  nativeModel.backward(y);
  const nativeLoss = nativeModel.loss;
  const nativeDenseWeight = Array.from(((nativeModel as any).dense.weight as { _data: Float32Array })._data);

  setForceDisableNative(true);
  jsModel.forward(x);
  jsModel.backward(y);
  const jsLoss = jsModel.loss;
  const jsDenseWeight = Array.from(((jsModel as any).dense.weight as { _data: Float32Array })._data);
  setForceDisableNative(false);

  assert.ok(Math.abs(nativeLoss - jsLoss) <= 1e-5, `native loss ${nativeLoss} != js loss ${jsLoss}`);
  assert.equal(nativeDenseWeight.length, jsDenseWeight.length);
  for (let i = 0; i < nativeDenseWeight.length; i++) {
    assert.ok(Math.abs(nativeDenseWeight[i] - jsDenseWeight[i]) <= 1e-5, `dense weight mismatch at ${i}`);
  }
});
