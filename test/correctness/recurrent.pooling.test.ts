import { mj } from "@oxide-js/core";
import { Matrix } from "@oxide-js/core";
import { RecurrentModel } from "@oxide-js/models";

function assert(condition: boolean, message: string): void {
  if (!condition) throw new Error(message);
}

function assertMatrixValue(matrix: Matrix, expected: number[][], message: string): void {
  assert(matrix._shape[0] === expected.length, `${message}: row mismatch`);
  assert(matrix._shape[1] === expected[0].length, `${message}: col mismatch`);
  for (let i = 0; i < expected.length; i++) {
    for (let j = 0; j < expected[i].length; j++) {
      const actual = matrix._value[i][j];
      if (Math.abs(actual - expected[i][j]) > 1e-6) {
        throw new Error(`${message}: mismatch at [${i}, ${j}] expected ${expected[i][j]} got ${actual}`);
      }
    }
  }
}

function createPoolingModel(pooling: "last" | "mean" | "max"): RecurrentModel {
  return new RecurrentModel({
    kind: "rnn",
    vocabSize: 8,
    embeddingDim: 3,
    hiddenSize: 2,
    outputSize: 2,
    seqLen: 3,
    mode: "many-to-one",
    pooling,
    padTokenId: 0,
    alpha: 0.01,
    optimizer: "sgd",
  });
}

function runMeanPoolingForwardShapeAndMaskTest(): void {
  const model = createPoolingModel("mean") as any;
  const sequence = mj.matrix([
    [1, 10, 2, 20, 3, 30],
    [4, 40, 5, 50, 6, 60],
  ]);
  const tokens = mj.matrix([
    [5, 7],
    [6, 0],
    [0, 0],
  ]);
  const pooled = model.poolSequenceOutput(sequence, tokens, 2) as Matrix;

  assertMatrixValue(pooled, [
    [1.5, 10],
    [4.5, 40],
  ], "masked mean pooling harus abaikan PAD dan output shape [hidden, batch]");
}

function runMeanPoolingBackwardDistributionTest(): void {
  const model = createPoolingModel("mean") as any;
  const sequence = mj.matrix([
    [1, 10, 100],
    [2, 20, 200],
  ]);
  const tokens = mj.matrix([[4], [5], [0]]);
  model.poolSequenceOutput(sequence, tokens, 1);
  const expanded = model.expandPooledErrorToSequence(mj.matrix([[6], [8]])) as Matrix;

  assertMatrixValue(expanded, [
    [3, 3, 0],
    [4, 4, 0],
  ], "mean pooling backward harus membagi grad ke semua timestep valid");
}

function runMaxPoolingBackwardArgmaxTest(): void {
  const model = createPoolingModel("max") as any;
  const sequence = mj.matrix([
    [1, 5, 3],
    [7, 2, 9],
  ]);
  const tokens = mj.matrix([[9], [8], [7]]);
  model.poolSequenceOutput(sequence, tokens, 1);
  const expanded = model.expandPooledErrorToSequence(mj.matrix([[4], [6]])) as Matrix;

  assertMatrixValue(expanded, [
    [0, 4, 0],
    [0, 0, 6],
  ], "max pooling backward harus kirim grad hanya ke argmax");
}

function runMaskedMaxIgnoresPadTest(): void {
  const model = createPoolingModel("max") as any;
  const sequence = mj.matrix([
    [1, 99, 3],
    [4, 77, 6],
  ]);
  const tokens = mj.matrix([[5], [0], [6]]);
  const pooled = model.poolSequenceOutput(sequence, tokens, 1) as Matrix;

  assertMatrixValue(pooled, [
    [3],
    [6],
  ], "masked max harus abaikan timestep PAD");
}

function runAllPadThrowsTest(): void {
  const model = createPoolingModel("mean") as any;
  let threw = false;
  try {
    model.poolSequenceOutput(
      mj.matrix([
        [1, 2, 3],
        [4, 5, 6],
      ]),
      mj.matrix([[0], [0], [0]]),
      1
    );
  } catch (error: any) {
    threw = error?.message?.includes("sample has no valid non-pad tokens");
  }
  assert(threw, "all-PAD sample harus throw jelas");
}

function runLastPoolingBackwardCompatibilityTest(): void {
  const model = createPoolingModel("last");
  const recurrent = model.layers[1] as any;
  assert(recurrent.returnSequences === false, "pooling=last harus menjaga behavior lama pada recurrent terakhir");
}

export function runRecurrentPoolingCorrectnessSuite(): void {
  runMeanPoolingForwardShapeAndMaskTest();
  runMeanPoolingBackwardDistributionTest();
  runMaxPoolingBackwardArgmaxTest();
  runMaskedMaxIgnoresPadTest();
  runAllPadThrowsTest();
  runLastPoolingBackwardCompatibilityTest();

  console.log("=== Recurrent Pooling Correctness ===");
  console.table([
    { check: "masked mean forward shape", status: "pass" },
    { check: "mean pooling backward", status: "pass" },
    { check: "max pooling backward", status: "pass" },
    { check: "masked max ignores PAD", status: "pass" },
    { check: "all-PAD throws", status: "pass" },
    { check: "last pooling compatibility", status: "pass" },
  ]);
}

