import mj from "../src/math";
import { Dense } from "../src/layers";
import { DimentionalityReduction, Sequential, Transformers } from "../src/models";

function assert(condition: boolean, message: string): void {
  if (!condition) throw new Error(`Assertion failed: ${message}`);
}

function createSequentialModel(): Sequential {
  const model = new Sequential({
    layers: [
      new Dense({ units: 2, outputUnits: 4, activation: "relu", status: "input" }),
      new Dense({ units: 4, outputUnits: 1, activation: "linear", status: "output", loss: "mse" }),
    ],
  });
  model.compile({ alpha: 0.03, optimizer: "adam", error: "mse" });
  return model;
}

function createDataset() {
  const X = [
    mj.matrix([[0], [0]]),
    mj.matrix([[0], [1]]),
    mj.matrix([[1], [0]]),
    mj.matrix([[1], [1]]),
    mj.matrix([[2], [1]]),
    mj.matrix([[1], [2]]),
  ];
  const y = X.map((x) => mj.matrix([[x.get(0, 0) + x.get(1, 0)]]));
  return { X, y };
}

function runSequentialFitTests() {
  const { X, y } = createDataset();

  // 1. Basic fit dengan default config
  const basicModel = createSequentialModel();
  const basicResult = basicModel.fit(X, y, 8);
  assert(basicResult.history.loss.length === 8, "basic fit history length");

  // 2. Fit dengan custom batchSize
  const batchModel = createSequentialModel();
  const batchResult = batchModel.fit(X, y, 6, { batchSize: 2 });
  assert(batchResult.history.loss.length === 6, "custom batch size history length");

  // 3. Fit dengan validationSplit
  const valModel = createSequentialModel();
  const valResult = valModel.fit(X, y, 5, { validationSplit: 0.33 });
  assert(Array.isArray(valResult.history.valLoss), "validation history exists");
  assert((valResult.history.valLoss as number[]).length === 5, "validation history length");

  // 4. Fit dengan earlyStoppingPatience
  const earlyModel = createSequentialModel();
  const earlyResult = earlyModel.fit(X, y, 12, {
    earlyStoppingPatience: 0,
    minDelta: 1000,
    monitorMetric: "loss",
  });
  assert(earlyResult.stoppedEarly, "early stopping flag true");
  assert(typeof earlyResult.stoppingEpoch === "number", "stopping epoch exists");

  // 5. Fit dengan shuffle=false
  const noShuffleModel = createSequentialModel();
  const noShuffleResult = noShuffleModel.fit(X, y, 4, { shuffle: false, batchSize: 3 });
  assert(noShuffleResult.history.loss.length === 4, "shuffle false runs");

  // 6. Fit dengan verbose=true (check console output)
  const verboseModel = createSequentialModel();
  const originalLog = console.log;
  let logCalls = 0;
  console.log = (..._args: unknown[]) => { logCalls++; };
  try {
    verboseModel.fit(X, y, 3, { verbose: true });
  } finally {
    console.log = originalLog;
  }
  assert(logCalls > 0, "verbose logs emitted");

  // 7. Callback onEpochEnd dipanggil per epoch
  const cbModel = createSequentialModel();
  let callbackCalls = 0;
  cbModel.fit(X, y, 5, {
    onEpochEnd: () => {
      callbackCalls++;
    },
  });
  assert(callbackCalls === 5, "onEpochEnd called each epoch");

  // 8. FitResult.history valid dan lengkap
  const historyModel = createSequentialModel();
  const historyResult = historyModel.fit(X, y, 4, { validationSplit: 0.2 });
  assert(typeof historyResult.bestEpoch === "number", "bestEpoch is number");
  assert(typeof historyResult.bestLoss === "number", "bestLoss is number");
  assert(Array.isArray(historyResult.history.loss), "history.loss is array");

  // 9. Early stopping bekerja dengan patience
  const patienceModel = createSequentialModel();
  const patienceResult = patienceModel.fit(X, y, 20, {
    earlyStoppingPatience: 1,
    minDelta: 1000,
    monitorMetric: "loss",
  });
  assert(patienceResult.stoppedEarly, "patience-based early stop works");

  // 10. Loss decreasing trend
  const trendModel = createSequentialModel();
  const trendResult = trendModel.fit(X, y, 20, { shuffle: false });
  const firstLoss = trendResult.history.loss[0];
  const lastLoss = trendResult.history.loss[trendResult.history.loss.length - 1];
  assert(lastLoss <= firstLoss, "loss trend decreases");

  // Backward compatibility callback
  const legacyModel = createSequentialModel();
  let legacyCalls = 0;
  legacyModel.fit(X, y, 3, (_loss: number) => {
    legacyCalls++;
  });
  assert(legacyCalls === 3, "legacy callback still works");
}

function runTransformersAndDimensionalityTests() {
  const transformer = new Transformers({
    units: 4,
    seqLen: 3,
    vocabSize: 12,
    heads: 2,
    dropoutRate: 0,
    alpha: 0.01,
    padTokenId: 0,
  });

  const tx = [mj.matrix([[1], [2], [3]]), mj.matrix([[2], [3], [4]])];
  const ty = [mj.matrix([[4]]), mj.matrix([[5]])];
  const transformerResult = transformer.fit(tx, ty, 2, { batchSize: 1 });
  assert(transformerResult.history.loss.length === 2, "transformer fit uses unified API");

  const dim = new DimentionalityReduction({
    layers: [
      new Dense({ units: 2, outputUnits: 1, activation: "relu", status: "input" }),
      new Dense({ units: 1, outputUnits: 1, activation: "linear", status: "outputReduction" }),
      new Dense({ units: 1, outputUnits: 2, activation: "linear", status: "output", loss: "mse" }),
    ],
  });
  dim.compile({ alpha: 0.01, optimizer: "adam", error: "mse" });
  const autoX = [mj.matrix([[1], [2]]), mj.matrix([[2], [3]])];
  const autoResult = dim.fit(autoX, 2, { batchSize: 1 });
  assert(autoResult.history.loss.length === 2, "dimensionality reduction autoencoder mode works");
}

runSequentialFitTests();
runTransformersAndDimensionalityTests();
console.log("✅ fit-methods.test.ts passed");
