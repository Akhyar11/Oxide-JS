import { mj } from "@oxide-js/core";
import { Matrix } from "@oxide-js/core";
import { RecurrentModel } from "@oxide-js/models";

type RecurrentFamily = "rnn" | "lstm" | "gru";

type LearningResult = {
  family: RecurrentFamily;
  initialLoss: number;
  finalLoss: number;
  history: number[];
};

function assert(condition: boolean, message: string): void {
  if (!condition) {
    throw new Error(message);
  }
}

function buildSequence(bits: number[]): Matrix {
  const rows = [
    bits.map((bit) => (bit === 0 ? 1 : 0)),
    bits.map((bit) => (bit === 1 ? 1 : 0)),
  ];
  return mj.matrix(rows);
}

function createLearningDataset(): { X: Matrix[]; y: Matrix[] } {
  const patterns = [
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [1, 0, 1, 0],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 1, 1, 1],
    [1, 0, 0, 1],
    [1, 1, 1, 1],
  ];

  const X: Matrix[] = [];
  const y: Matrix[] = [];

  for (let repeat = 0; repeat < 8; repeat++) {
    for (const pattern of patterns) {
      X.push(buildSequence(pattern));
      y.push(mj.matrix([[pattern[pattern.length - 1]]]));
    }
  }

  return { X, y };
}

function createManyToManyDataset(): { X: Matrix[]; y: Matrix[] } {
  const patterns = [
    [0, 0, 0, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 1, 1, 1],
    [1, 0, 0, 1],
    [1, 1, 1, 1],
  ];

  const X: Matrix[] = [];
  const y: Matrix[] = [];

  for (let repeat = 0; repeat < 6; repeat++) {
    for (const pattern of patterns) {
      X.push(buildSequence(pattern));
      y.push(mj.matrix([pattern]));
    }
  }

  return { X, y };
}

function createModel(family: RecurrentFamily): RecurrentModel {
  if (family === "lstm") {
    return new RecurrentModel({
      kind: "lstm",
      inputSize: 2,
      hiddenSizes: [8, 6],
      outputSize: 2,
      seqLen: 4,
      mode: "many-to-one",
      loss: "softmaxCrossEntropy",
      alpha: 0.01,
      optimizer: "adam",
    });
  }

  return new RecurrentModel({
    kind: family,
    inputSize: 2,
    hiddenSize: 8,
    numLayers: 2,
    outputSize: 2,
    seqLen: 4,
    mode: "many-to-one",
    loss: "softmaxCrossEntropy",
    alpha: 0.01,
    optimizer: "adam",
  });
}

function createManyToManyModel(family: RecurrentFamily): RecurrentModel {
  if (family === "lstm") {
    return new RecurrentModel({
      kind: "lstm",
      inputSize: 2,
      hiddenSizes: [8, 6],
      outputSize: 2,
      seqLen: 4,
      mode: "many-to-many",
      loss: "softmaxCrossEntropy",
      alpha: 0.01,
      optimizer: "adam",
    });
  }

  return new RecurrentModel({
    kind: family,
    inputSize: 2,
    hiddenSize: 8,
    numLayers: 2,
    outputSize: 2,
    seqLen: 4,
    mode: "many-to-many",
    loss: "softmaxCrossEntropy",
    alpha: 0.01,
    optimizer: "adam",
  });
}

function runLearningTest(family: RecurrentFamily): LearningResult {
  const { X, y } = createLearningDataset();
  const model = createModel(family);
  const history: number[] = [];

  model.fit(X, y, 10, {
    batchSize: 1,
    shuffle: false,
    verbose: false,
    onEpochEnd: (_epoch, loss) => {
      history.push(loss);
    },
  });

  assert(history.length === 10, `${family}: expected 10 loss entries, got ${history.length}`);

  const initialLoss = history[0];
  const finalLoss = history[history.length - 1];
  const bestLoss = Math.min(...history);
  const pred = model.predict(X[0]);

  assert(Number.isFinite(initialLoss), `${family}: initial loss must be finite`);
  assert(Number.isFinite(finalLoss), `${family}: final loss must be finite`);
  assert(pred._shape[0] === 2 && pred._shape[1] === 1, `${family}: unexpected predict shape [${pred._shape[0]}, ${pred._shape[1]}]`);
  assert(
    finalLoss < initialLoss,
    `${family}: expected final loss to be lower than initial loss (${finalLoss} >= ${initialLoss})`
  );
  assert(
    bestLoss <= initialLoss - 0.01,
    `${family}: expected at least one meaningful loss improvement, history=${history.map((v) => v.toFixed(6)).join(", ")}`
  );

  if (family === "lstm") {
    const outputLayer = model.getDenseOutputLayer();
    assert(outputLayer.units === 6, `lstm: expected output Dense input units 6, got ${outputLayer.units}`);
  }

  return {
    family,
    initialLoss,
    finalLoss,
    history,
  };
}

function runManyToManyLearningTest(family: RecurrentFamily): void {
  const { X, y } = createManyToManyDataset();
  const model = createManyToManyModel(family);
  const history: number[] = [];

  model.fit(X, y, 5, {
    batchSize: 1,
    shuffle: false,
    verbose: false,
    onEpochEnd: (_epoch, loss) => {
      history.push(loss);
    },
  });

  const pred = model.predict(X[0]);
  assert(history.length === 5, `${family} many-to-many: expected 5 loss entries, got ${history.length}`);
  assert(Number.isFinite(history[0]), `${family} many-to-many: initial loss must be finite`);
  assert(Number.isFinite(history[history.length - 1]), `${family} many-to-many: final loss must be finite`);
  assert(
    pred._shape[0] === 2 && pred._shape[1] === 4,
    `${family} many-to-many: unexpected predict shape [${pred._shape[0]}, ${pred._shape[1]}]`
  );
}

function runStatefulGuardTest(): void {
  const { X, y } = createLearningDataset();
  const model = new RecurrentModel({
    kind: "gru",
    inputSize: 2,
    hiddenSize: 8,
    numLayers: 2,
    outputSize: 2,
    seqLen: 4,
    mode: "many-to-one",
    stateful: true,
    loss: "softmaxCrossEntropy",
  });

  let batchGuard = false;
  try {
    model.fit(X, y, 1, { batchSize: 2, shuffle: false, verbose: false });
  } catch (error: any) {
    batchGuard = error?.message?.includes("stateful=true hanya mendukung batchSize=1");
  }
  assert(batchGuard, "RecurrentModel: expected stateful + batchSize > 1 guard");

  let shuffleGuard = false;
  try {
    model.fit(X, y, 1, { batchSize: 1, shuffle: true, verbose: false });
  } catch (error: any) {
    shuffleGuard = error?.message?.includes("stateful=true tidak boleh dipakai bersama shuffle=true");
  }
  assert(shuffleGuard, "RecurrentModel: expected stateful + shuffle=true guard");
}

export function runRecurrentLearningCorrectnessSuite(): LearningResult[] {
  const families: RecurrentFamily[] = ["rnn", "lstm", "gru"];
  const results = families.map((family) => runLearningTest(family));
  families.forEach((family) => runManyToManyLearningTest(family));
  runStatefulGuardTest();

  console.log("=== Recurrent Learning Correctness ===");
  console.table(
    results.map((result) => ({
      family: result.family,
      initialLoss: Number(result.initialLoss.toFixed(6)),
      finalLoss: Number(result.finalLoss.toFixed(6)),
      improvement: Number((result.initialLoss - result.finalLoss).toFixed(6)),
    }))
  );

  return results;
}
