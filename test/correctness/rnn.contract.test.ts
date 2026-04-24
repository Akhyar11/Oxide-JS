import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import RNN from "../../src/layers/rnn";
import LSTM from "../../src/layers/lstm";
import GRU from "../../src/layers/gru";
import Embedding from "../../src/layers/embedding";
import Dense from "../../src/layers/dense";
import Sequential from "../../src/models/sequential";
import mj from "../../src/math";

function test(name: string, fn: () => void) {
  try {
    fn();
    console.log(`PASS ${name}`);
  } catch (error) {
    console.error(`FAIL ${name}`);
    throw error;
  }
}

function approxEqual(actual: number, expected: number, epsilon = 1e-6) {
  assert.ok(
    Math.abs(actual - expected) <= epsilon,
    `expected ${actual} to be within ${epsilon} of ${expected}`
  );
}

function assertMatrixShape(matrix: { _shape: [number, number] }, expected: [number, number]) {
  assert.deepEqual(matrix._shape, expected);
}

function assertThrowsWithMessage(fn: () => void, pattern: RegExp) {
  assert.throws(fn, pattern);
}

type RecurrentFamily = "RNN" | "LSTM" | "GRU";
type RecurrentLayer = RNN | LSTM | GRU;

type RecurrentFactory = {
  family: RecurrentFamily;
  create: (config?: Record<string, any>) => RecurrentLayer;
  createDeterministicStateful: () => RecurrentLayer;
  getPrimaryStateValue: (layer: RecurrentLayer) => number;
  assertSerializedWeights: (layer: RecurrentLayer) => void;
};

function cloneRecurrentLayer(layer: RecurrentLayer): RecurrentLayer {
  if (layer instanceof RNN) {
    const cloned = new RNN({
      units: layer.units,
      hiddenUnits: layer.hiddenUnits,
      activation: layer.activation,
      returnSequences: layer.returnSequences,
      returnState: layer.returnState,
      stateful: layer.stateful,
      alpha: layer.alpha,
      status: layer.status,
      clipGradient: layer.clipGradient,
    });
    cloned.load(layer.save() as any);
    return cloned;
  }
  if (layer instanceof LSTM) {
    const cloned = new LSTM({
      units: layer.units,
      hiddenUnits: layer.hiddenUnits,
      returnSequences: layer.returnSequences,
      returnState: layer.returnState,
      stateful: layer.stateful,
      alpha: layer.alpha,
      status: layer.status,
      clipGradient: layer.clipGradient,
    });
    cloned.load(layer.save() as any);
    return cloned;
  }
  if (layer instanceof GRU) {
    const cloned = new GRU({
      units: layer.units,
      hiddenUnits: layer.hiddenUnits,
      returnSequences: layer.returnSequences,
      returnState: layer.returnState,
      stateful: layer.stateful,
      bidirectional: layer.bidirectional,
      alpha: layer.alpha,
      status: layer.status,
      clipGradient: layer.clipGradient,
    });
    cloned.load(layer.save() as any);
    return cloned;
  }
  throw new Error("Unsupported recurrent layer type.");
}

function assertMatricesClose(actual: { _data: Float32Array; _shape: [number, number] }, expected: { _data: Float32Array; _shape: [number, number] }, epsilon = 1e-6) {
  assert.deepEqual(actual._shape, expected._shape);
  assert.equal(actual._data.length, expected._data.length);
  for (let i = 0; i < actual._data.length; i++) {
    approxEqual(actual._data[i], expected._data[i], epsilon);
  }
}

function createDeterministicRNN(config: Partial<ConstructorParameters<typeof RNN>[0]> = {}) {
  const rnn = new RNN({
    units: 1,
    hiddenUnits: 1,
    activation: "tanh",
    optimizer: "sgd",
    alpha: 0,
    ...config,
  });
  rnn.load({
    Wxh: [[1]],
    Whh: [[1]],
    bh: [[0]],
    hStateful: [[0]],
    clipGradient: false,
  });
  return rnn;
}

function createDeterministicLSTM(config: Partial<ConstructorParameters<typeof LSTM>[0]> = {}) {
  const lstm = new LSTM({
    units: 1,
    hiddenUnits: 1,
    optimizer: "sgd",
    alpha: 0,
    ...config,
  });
  lstm.load({
    Wxi: [[0]],
    Whi: [[0]],
    bi: [[10]],
    Wxf: [[0]],
    Whf: [[0]],
    bf: [[10]],
    Wxo: [[0]],
    Who: [[0]],
    bo: [[10]],
    Wxg: [[1]],
    Whg: [[1]],
    bg: [[0]],
    hStateful: [[0]],
    cStateful: [[0]],
    clipGradient: false,
  });
  return lstm;
}

function createDeterministicGRU(config: Partial<ConstructorParameters<typeof GRU>[0]> = {}) {
  const gru = new GRU({
    units: 1,
    hiddenUnits: 1,
    optimizer: "sgd",
    alpha: 0,
    ...config,
  });
  gru.load({
    forward: {
      Wxr: [[0]],
      Whr: [[0]],
      br: [[0]],
      Wxz: [[0]],
      Whz: [[0]],
      bz: [[0]],
      Wxh: [[1]],
      Whh: [[1]],
      bh: [[0]],
      hStateful: [[0]],
    },
    clipGradient: false,
  });
  return gru;
}

const recurrentFactories: RecurrentFactory[] = [
  {
    family: "RNN",
    create: (config = {}) => new RNN({ units: 2, hiddenUnits: 3, optimizer: "sgd", alpha: 0, ...config }),
    createDeterministicStateful: () => createDeterministicRNN({ stateful: true }),
    getPrimaryStateValue: (layer) => (layer as RNN).getState().get(0, 0),
    assertSerializedWeights: (layer) => {
      const rnn = layer as RNN;
      assert.deepEqual(rnn.Wxh._value, [[1]]);
      assert.deepEqual(rnn.Whh._value, [[1]]);
      assert.deepEqual(rnn.bh._value, [[0]]);
    },
  },
  {
    family: "LSTM",
    create: (config = {}) => new LSTM({ units: 2, hiddenUnits: 3, optimizer: "sgd", alpha: 0, ...config }),
    createDeterministicStateful: () => createDeterministicLSTM({ stateful: true }),
    getPrimaryStateValue: (layer) => (layer as LSTM).getState().h.get(0, 0),
    assertSerializedWeights: (layer) => {
      const lstm = layer as LSTM;
      assert.deepEqual(lstm.Wxi._value, [[0]]);
      assert.deepEqual(lstm.Whi._value, [[0]]);
      assert.deepEqual(lstm.bi._value, [[10]]);
      assert.deepEqual(lstm.Wxg._value, [[1]]);
      assert.deepEqual(lstm.Whg._value, [[1]]);
    },
  },
  {
    family: "GRU",
    create: (config = {}) => new GRU({ units: 2, hiddenUnits: 3, optimizer: "sgd", alpha: 0, ...config }),
    createDeterministicStateful: () => createDeterministicGRU({ stateful: true }),
    getPrimaryStateValue: (layer) => (layer as GRU).getState().forward.get(0, 0),
    assertSerializedWeights: (layer) => {
      const state = (layer as GRU).save();
      assert.deepEqual(state.forward.Wxh, [[1]]);
      assert.deepEqual(state.forward.Whh, [[1]]);
      assert.deepEqual(state.forward.bh, [[0]]);
    },
  },
];

for (const factory of recurrentFactories) {
  test(`${factory.family} forward shape with returnSequences=false`, () => {
    const layer = factory.create({ returnSequences: false });
    const x = mj.matrix([
      [0.1, 0.2, 0.3],
      [0.4, 0.5, 0.6],
    ]);

    const out = layer.forward(x);

    assertMatrixShape(out, [3, 1]);
    assert.deepEqual(layer.inputShape, [2, 3]);
    assert.deepEqual(layer.outputShape, [3, 1]);
  });

  test(`${factory.family} forward shape with returnSequences=true`, () => {
    const layer = factory.create({ returnSequences: true });
    const x = mj.matrix([
      [0.1, 0.2, 0.3, 0.4],
      [0.5, 0.6, 0.7, 0.8],
    ]);

    const out = layer.forward(x);

    assertMatrixShape(out, [3, 4]);
    assert.deepEqual(layer.inputShape, [2, 4]);
    assert.deepEqual(layer.outputShape, [3, 4]);
  });

  test(`${factory.family} backward returns dx with original single-sequence shape`, () => {
    const layer = factory.create({ returnSequences: false });
    const x = mj.matrix([
      [0.5, -0.2, 0.1],
      [0.3, 0.4, -0.1],
    ]);

    layer.forward(x);
    const dx = layer.backward(mj.zeros([3, 1]), mj.zeros([3, 1]));

    assertMatrixShape(dx, [2, 3]);
    for (const value of dx._data) {
      assert.ok(Number.isFinite(value), `${factory.family} backward produced non-finite gradient`);
    }
  });

  test(`${factory.family} reuses recurrent buffers safely across varying single-sequence lengths`, () => {
    const layer = factory.create({ returnSequences: true, status: "hidden" });
    const inputs = [
      mj.matrix([
        [0.1, -0.2, 0.3, -0.4],
        [0.5, 0.6, -0.7, 0.8],
      ]),
      mj.matrix([
        [0.9, -1.0],
        [0.2, 0.4],
      ]),
      mj.matrix([
        [-0.3, 0.1, 0.7],
        [0.6, -0.5, 0.2],
      ]),
    ];

    for (const x of inputs) {
      const expectedLayer = cloneRecurrentLayer(layer);
      const out = layer.forward(x);
      const expectedOut = expectedLayer.forward(x);
      assertMatricesClose(out, expectedOut);

      const zeroErr = mj.zeros([out._shape[0], out._shape[1]]);
      const dx = layer.backward(mj.zeros([out._shape[0], out._shape[1]]), zeroErr);
      const expectedDx = expectedLayer.backward(mj.zeros([expectedOut._shape[0], expectedOut._shape[1]]), zeroErr);
      assertMatricesClose(dx, expectedDx);
    }
  });

  test(`${factory.family} reuses recurrent buffers safely across varying batched sequence shapes`, () => {
    if (factory.family === "GRU") {
      return;
    }
    const layer = factory.create({ returnSequences: true, status: "hidden" }) as any;
    const cases = [
      {
        batchSize: 2,
        x: mj.matrix([
          [0.1, 0.05, 0.08, 0.04, 0.06, 0.02],
          [0.06, 0.03, 0.05, 0.02, 0.04, 0.01],
        ]),
      },
      {
        batchSize: 1,
        x: mj.matrix([
          [0.07, -0.01, 0.08],
          [0.02, 0.04, -0.06],
        ]),
      },
      {
        batchSize: 3,
        x: mj.matrix([
          [0.09, 0.08, 0.07, -0.01, -0.02, -0.03],
          [0.03, 0.01, -0.01, 0.05, 0.07, 0.09],
        ]),
      },
    ];

    for (const { batchSize, x } of cases) {
      const expectedLayer = cloneRecurrentLayer(layer) as any;
      const out = layer.forwardBatch(x, batchSize);
      const expectedOut = expectedLayer.forwardBatch(x, batchSize);
      assertMatricesClose(out, expectedOut);

      const zeroErr = mj.zeros([out._shape[0], out._shape[1]]);
      const dx = layer.backwardBatch(mj.zeros([out._shape[0], out._shape[1]]), zeroErr, batchSize);
      const expectedDx = expectedLayer.backwardBatch(
        mj.zeros([expectedOut._shape[0], expectedOut._shape[1]]),
        zeroErr,
        batchSize
      );
      assertMatricesClose(dx, expectedDx);
    }
  });

  test(`${factory.family} stateful carries state across calls and resetState clears it`, () => {
    const layer = factory.createDeterministicStateful();
    const first = layer.forward(mj.matrix([[1]]));
    const firstValue = first.get(0, 0);
    const carried = layer.forward(mj.matrix([[0]]));
    const carriedValue = carried.get(0, 0);

    assert.ok(Math.abs(firstValue) > 1e-6, `${factory.family} first stateful output should be non-zero`);
    assert.ok(Math.abs(carriedValue) > 1e-6, `${factory.family} carried output should be non-zero`);
    approxEqual(factory.getPrimaryStateValue(layer), carriedValue);

    layer.resetState();
    approxEqual(factory.getPrimaryStateValue(layer), 0);

    const afterReset = layer.forward(mj.matrix([[0]]));
    approxEqual(afterReset.get(0, 0), 0);
  });

  test(`Sequential save/load roundtrip preserves ${factory.family} weights and state`, () => {
    const tempDir = mkdtempSync(join(tmpdir(), `ml-v2-${factory.family.toLowerCase()}-`));
    const modelPath = join(tempDir, `${factory.family.toLowerCase()}.json`);

    try {
      const model = new Sequential({
        layers: [factory.createDeterministicStateful()],
      });
      model.forward(mj.matrix([[1]]));
      model.save(modelPath);

      const loaded = new Sequential();
      loaded.load(modelPath);

      const loadedLayer = loaded.layers[0] as RecurrentLayer;
      factory.assertSerializedWeights(loadedLayer);
      assert.ok(Math.abs(factory.getPrimaryStateValue(loadedLayer)) > 1e-6);
    } finally {
      rmSync(tempDir, { recursive: true, force: true });
    }
  });

  test(`${factory.family} returnState throws instead of being silently ignored`, () => {
    const layer = factory.create({ units: 1, hiddenUnits: 1, returnState: true });

    assert.throws(
      () => layer.forward(mj.matrix([[1]])),
      /returnState=true is not supported yet/
    );
  });

  test(`${factory.family} rejects empty sequence input explicitly`, () => {
    const layer = factory.create();
    assert.throws(
      () => layer.forward(mj.zeros([2, 0])),
      /non-empty sequence input/
    );
  });

  test(`Sequential.fit rejects ${factory.family} batchSize > 1 because generic batching is invalid for sequences`, () => {
    const RecurrentCtor = factory.create;
    const model = new Sequential({
      layers: [RecurrentCtor({ units: 1, hiddenUnits: 1, status: "output" })],
    });
    const X = [mj.matrix([[1, 2]]), mj.matrix([[3, 4]])];
    const y = [mj.matrix([[0]]), mj.matrix([[0]])];

    assert.throws(
      () => model.fit(X, y, 1, { batchSize: 2, shuffle: false }),
      /batchSize=1/
    );
  });

  test(`Sequential.fit rejects ${factory.family} stateful recurrent training with shuffle=true`, () => {
    const model = new Sequential({
      layers: [factory.create({ units: 1, hiddenUnits: 1, stateful: true, status: "output" })],
    });
    const X = [mj.matrix([[1]]), mj.matrix([[2]])];
    const y = [mj.matrix([[0]]), mj.matrix([[0]])];

    assert.throws(
      () => model.fit(X, y, 1, { batchSize: 1, shuffle: true }),
      /stateful=true.*shuffle=true/
    );
  });

  test(`Sequential.fit rejects ${factory.family} stateful recurrent training with validationSplit > 0`, () => {
    const model = new Sequential({
      layers: [factory.create({ units: 1, hiddenUnits: 1, stateful: true, status: "output" })],
    });
    const X = [mj.matrix([[1]]), mj.matrix([[2]])];
    const y = [mj.matrix([[0]]), mj.matrix([[0]])];

    assert.throws(
      () => model.fit(X, y, 1, { batchSize: 1, shuffle: false, validationSplit: 0.5 }),
      /validationSplit > 0/
    );
  });

  test(`Embedding -> ${factory.family} -> Dense basic pipeline trains with finite values`, () => {
    const embeddingDim = 2;
    const hiddenUnits = 3;
    const vocabSize = 8;
    const seqLen = 3;
    const model = new Sequential({
      layers: [
        new Embedding({ vocabSize, embeddingDim, optimizer: "sgd", alpha: 0.01 }),
        factory.create({ units: embeddingDim, hiddenUnits, returnSequences: false }),
        new Dense({
          units: hiddenUnits,
          outputUnits: vocabSize,
          activation: "linear",
          optimizer: "sgd",
          alpha: 0.01,
          status: "output",
          loss: "softmaxCrossEntropy",
        }),
      ],
    });
    const x = mj.matrix([[1], [2], [3]]);
    const y = mj.matrix([[4]]);

    const out = model.forward(x);
    assertMatrixShape(out, [vocabSize, 1]);
    for (const value of out._data) {
      assert.ok(Number.isFinite(value), `${factory.family} pipeline produced non-finite output`);
    }

    model.backward(y);

    const fitResult = model.fit([x], [y], 1, { batchSize: 1, shuffle: false });
    assert.equal(fitResult.history.loss.length, 1);
    assert.ok(Number.isFinite(fitResult.history.loss[0]));
    assert.equal((model.layers[1] as RecurrentLayer).inputShape[1], seqLen);
  });
}

test("Sequential.fit rejects shuffle=true when a later recurrent layer is stateful", () => {
  const model = new Sequential({
    layers: [
      new RNN({ units: 1, hiddenUnits: 1, stateful: false }),
      new GRU({ units: 1, hiddenUnits: 1, stateful: true, status: "output" }),
    ],
  });
  const X = [mj.matrix([[1]]), mj.matrix([[2]])];
  const y = [mj.matrix([[0]]), mj.matrix([[0]])];

  assertThrowsWithMessage(
    () => model.fit(X, y, 1, { batchSize: 1, shuffle: true }),
    /stateful=true.*shuffle=true/
  );
});

test("Sequential.fit rejects validationSplit > 0 when a later recurrent layer is stateful", () => {
  const model = new Sequential({
    layers: [
      new RNN({ units: 1, hiddenUnits: 1, stateful: false }),
      new GRU({ units: 1, hiddenUnits: 1, stateful: true, status: "output" }),
    ],
  });
  const X = [mj.matrix([[1]]), mj.matrix([[2]])];
  const y = [mj.matrix([[0]]), mj.matrix([[0]])];

  assertThrowsWithMessage(
    () => model.fit(X, y, 1, { batchSize: 1, shuffle: false, validationSplit: 0.5 }),
    /validationSplit > 0/
  );
});

test("LSTM.load rejects malformed serialized payload with explicit error", () => {
  const lstm = new LSTM({ units: 1, hiddenUnits: 1 });
  assertThrowsWithMessage(
    () => lstm.load({}),
    /LSTM\.load: expected serialized matrix 'Wxi'/
  );
});

test("GRU.load rejects malformed serialized payload with explicit error", () => {
  const gru = new GRU({ units: 1, hiddenUnits: 1 });
  assertThrowsWithMessage(
    () => gru.load({}),
    /GRU\.load: expected serialized 'forward' direction/
  );
});

test("Bidirectional GRU forward, backward, and save/load remain consistent", () => {
  const gru = new GRU({
    units: 2,
    hiddenUnits: 3,
    bidirectional: true,
    returnSequences: true,
    optimizer: "sgd",
    alpha: 0,
  });
  const x = mj.matrix([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
  ]);

  const out = gru.forward(x);
  assertMatrixShape(out, [6, 3]);

  const dx = gru.backward(mj.zeros([6, 3]), mj.zeros([6, 3]));
  assertMatrixShape(dx, [2, 3]);
  for (const value of dx._data) {
    assert.ok(Number.isFinite(value), "Bidirectional GRU backward produced non-finite gradient");
  }

  const serialized = gru.save();
  const loaded = new GRU({
    units: 2,
    hiddenUnits: 3,
    bidirectional: true,
    returnSequences: true,
    optimizer: "sgd",
    alpha: 0,
  });
  loaded.load(serialized);
  const reloadedOut = loaded.forward(x);
  assertMatrixShape(reloadedOut, [6, 3]);
});
