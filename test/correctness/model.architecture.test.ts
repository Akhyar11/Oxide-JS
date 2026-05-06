import { mj } from "@oxide-js/core";
import { Matrix } from "@oxide-js/core";
import { Dense, MemoryBank, RNN } from "@oxide-js/layers";
import { Sequential, Transformers } from "@oxide-js/models";

function assert(condition: boolean, message: string): void {
  if (!condition) {
    throw new Error(message);
  }
}

function assertClose(actual: number, expected: number, tolerance: number, message: string): void {
  if (!Number.isFinite(actual) || Math.abs(actual - expected) > tolerance) {
    throw new Error(`${message}: expected ${expected}, got ${actual}`);
  }
}

class InspectableTransformers extends Transformers {
  public inspectLossAndWeight(yTrue: Matrix, yPred: Matrix): { loss: number; weight: number } {
    return this.computeLossAndWeight(yTrue, yPred);
  }

  public inspectBatch(
    X: Matrix[],
    y: Matrix[],
    indices: number[],
    start: number,
    currentBatchSize: number,
    trimPadding: boolean,
    paddingSide: "left" | "right"
  ): { x: Matrix; y: Matrix } {
    return (this as any).buildTransformerBatch(X, y, indices, start, currentBatchSize, trimPadding, paddingSide);
  }
}

function buildFeedForwardSample(a: number, b: number): Matrix {
  return mj.matrix([[a], [b]]);
}

function runSequentialFeedForwardTest(): void {
  const X = [
    buildFeedForwardSample(0, 0),
    buildFeedForwardSample(0, 1),
    buildFeedForwardSample(1, 0),
    buildFeedForwardSample(1, 1),
  ];
  const y = [
    mj.matrix([[0]]),
    mj.matrix([[1]]),
    mj.matrix([[1]]),
    mj.matrix([[0]]),
  ];

  const model = new Sequential({
    layers: [
      new Dense({ units: 2, outputUnits: 6, activation: "relu", status: "input" }),
      new Dense({ units: 6, outputUnits: 4, activation: "relu", status: "train" }),
      new Dense({ units: 4, outputUnits: 2, activation: "linear", status: "output", loss: "softmaxCrossEntropy" }),
    ],
  });

  model.compile({
    alpha: 0.01,
    optimizer: "adam",
    error: "softmaxCrossEntropy",
  });

  const result = model.fit(X, y, 5, {
    batchSize: 2,
    shuffle: false,
    verbose: false,
  });
  const pred = model.predict(X[0]);

  assert(result.history.loss.length === 5, `Sequential feed-forward: expected 5 loss entries, got ${result.history.loss.length}`);
  assert(Number.isFinite(result.history.loss[0]), "Sequential feed-forward: history.loss[0] must be finite");
  assert(pred._shape[0] === 2 && pred._shape[1] === 1, `Sequential feed-forward: unexpected predict shape [${pred._shape[0]}, ${pred._shape[1]}]`);
}

function runSequentialSequenceGuardTest(): void {
  const X = [mj.matrix([[1, 0, 1], [0, 1, 0]])];
  const y = [mj.matrix([[1]])];
  const model = new Sequential({
    layers: [
      new RNN({ units: 2, hiddenUnits: 4, returnSequences: false, status: "input" }),
      new Dense({ units: 4, outputUnits: 2, activation: "linear", status: "output", loss: "softmaxCrossEntropy" }),
    ],
  });

  let threw = false;
  try {
    model.fit(X, y, 1, { batchSize: 1, shuffle: false, verbose: false });
  } catch (error: any) {
    threw = error?.message?.includes("Sequential.fit only supports per-sample supervised loss");
  }

  assert(threw, "Sequential sequence guard: expected clear redirect to Transformers/RecurrentModel");
}

function runSequentialMemoryBankGuardTest(): void {
  let ctorThrew = false;
  try {
    new Sequential({
      layers: [
        new Dense({ units: 2, outputUnits: 2, activation: "relu", status: "input" }),
        new MemoryBank({ units: 2, memorySlots: 2, outputUnits: 2 }),
      ],
    });
  } catch (error: any) {
    ctorThrew = error?.message?.includes("MemoryBank tidak didukung di arsitektur Sequential");
  }
  assert(ctorThrew, "Sequential constructor should reject MemoryBank");

  const model = new Sequential();
  let addThrew = false;
  try {
    model.add(new MemoryBank({ units: 2, memorySlots: 2, outputUnits: 2 }));
  } catch (error: any) {
    addThrew = error?.message?.includes("MemoryBank tidak didukung di arsitektur Sequential");
  }
  assert(addThrew, "Sequential.add should reject MemoryBank");
}

function buildTokenSample(tokens: number[]): Matrix {
  return mj.matrix(tokens.map((token) => [token]));
}

function buildShiftedTarget(tokens: number[], padTokenId: number): Matrix {
  const shifted = tokens.slice(1).concat([padTokenId]);
  return mj.matrix(shifted.map((token) => [token]));
}

function buildBatch(samples: Matrix[]): Matrix {
  const rows = samples[0]._shape[0];
  const batch = mj.zeros([rows, samples.length]);
  for (let i = 0; i < samples.length; i++) {
    batch.setCol(i, samples[i]._data);
  }
  return batch;
}

function runTransformerTokenWeightedFitTest(): void {
  const pad = 0;
  const model = new InspectableTransformers({
    units: 12,
    seqLen: 5,
    vocabSize: 10,
    heads: 2,
    numBlocks: 1,
    dropoutRate: 0,
    alpha: 0,
    padTokenId: pad,
  });

  const X = [
    buildTokenSample([1, 2, 3, 4, 5]),
    buildTokenSample([1, 2, 3, 0, 0]),
    buildTokenSample([2, 3, 4, 5, 6]),
    buildTokenSample([2, 3, 0, 0, 0]),
  ];
  const y = [
    buildShiftedTarget([1, 2, 3, 4, 5], pad),
    buildShiftedTarget([1, 2, 3, 0, 0], pad),
    buildShiftedTarget([2, 3, 4, 5, 6], pad),
    buildShiftedTarget([2, 3, 0, 0, 0], pad),
  ];

  const trainX = X.slice(0, 2);
  const trainY = y.slice(0, 2);
  const valX = X.slice(2);
  const valY = y.slice(2);

  const result = model.fit(X, y, 1, {
    batchSize: 2,
    validationSplit: 0.5,
    shuffle: false,
    verbose: false,
    trimPadding: false,
  });

  const expectedTrain = computeExpectedTokenWeightedLoss(model, trainX, trainY, 2, "train");
  const expectedVal = computeExpectedTokenWeightedLoss(model, valX, valY, 2, "eval");
  assertClose(result.history.loss[0], expectedTrain, 5e-2, "Transformers token-weighted train loss mismatch");
  assertClose((result.history.valLoss as number[])[0], expectedVal, 5e-2, "Transformers token-weighted val loss mismatch");
}

function computeExpectedTokenWeightedLoss(
  model: InspectableTransformers,
  X: Matrix[],
  y: Matrix[],
  batchSize: number,
  phase: "train" | "eval"
): number {
  if (phase === "train") {
    model.train();
  } else {
    model.eval();
  }
  let totalLoss = 0;
  let totalWeight = 0;
  const indices = Array.from({ length: X.length }, (_, i) => i);

  for (let start = 0; start < X.length; start += batchSize) {
    const currentBatchSize = Math.min(batchSize, X.length - start);
    const batch = model.inspectBatch(X, y, indices, start, currentBatchSize, false, "right");
    const pred = model.forwardFullSequence(batch.x);
    const state = model.inspectLossAndWeight(batch.y, pred);
    totalLoss += state.loss * state.weight;
    totalWeight += state.weight;
    model.resetPositionOffset();
  }

  model.eval();
  return totalLoss / totalWeight;
}

export function runModelArchitectureCorrectnessSuite(): void {
  runSequentialFeedForwardTest();
  runSequentialSequenceGuardTest();
  runSequentialMemoryBankGuardTest();
  runTransformerTokenWeightedFitTest();

  console.log("=== Model Architecture Correctness ===");
  console.table([
    { check: "sequential multi-layer feed-forward fit", status: "pass" },
    { check: "sequential rejects recurrent fit path", status: "pass" },
    { check: "sequential rejects memory bank path", status: "pass" },
    { check: "transformers token-weighted fit aggregation", status: "pass" },
  ]);
}
