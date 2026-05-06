import { mj } from "@oxide-js/core";
import { Matrix } from "@oxide-js/core";
import { Transformers } from "@oxide-js/models";

function assert(condition: boolean, message: string): void {
  if (!condition) {
    throw new Error(message);
  }
}

function buildSequenceMatrix(tokens: number[]): Matrix {
  return mj.matrix(tokens.map((token) => [token]));
}

function createTransformerModel(predictMode: "next-token" | "full-sequence" = "next-token"): Transformers {
  return new Transformers({
    units: 24,
    seqLen: 6,
    vocabSize: 16,
    heads: 4,
    numBlocks: 1,
    dropoutRate: 0,
    alpha: 0.01,
    padTokenId: 0,
    predictMode,
  });
}

export function runTransformerApiCorrectnessSuite(): void {
  const x = buildSequenceMatrix([1, 2, 3, 4, 5, 6]);

  const nextTokenModel = createTransformerModel();
  const nextTokenPred = nextTokenModel.predict(x);
  assert(
    nextTokenPred._shape[0] === 16 && nextTokenPred._shape[1] === 1,
    `transformers predict(next-token): expected [16,1], got [${nextTokenPred._shape[0]}, ${nextTokenPred._shape[1]}]`
  );
  assert(
    nextTokenModel.getPredictMode() === "next-token",
    `transformers predict(next-token): expected mode next-token, got ${nextTokenModel.getPredictMode()}`
  );

  const fullSequenceModel = createTransformerModel("full-sequence");
  const fullSequencePred = fullSequenceModel.predict(x);
  assert(
    fullSequencePred._shape[0] === 16 && fullSequencePred._shape[1] === 6,
    `transformers predict(full-sequence): expected [16,6], got [${fullSequencePred._shape[0]}, ${fullSequencePred._shape[1]}]`
  );
  assert(
    fullSequenceModel.getPredictMode() === "full-sequence",
    `transformers predict(full-sequence): expected mode full-sequence, got ${fullSequenceModel.getPredictMode()}`
  );

  fullSequenceModel.train();
  const predWhileTraining = fullSequenceModel.predict(x);
  assert(
    predWhileTraining._shape[0] === 16 && predWhileTraining._shape[1] === 6,
    `transformers predict(full-sequence while training): expected [16,6], got [${predWhileTraining._shape[0]}, ${predWhileTraining._shape[1]}]`
  );
  const forwardAfterPredict = fullSequenceModel.forward(x);
  assert(
    forwardAfterPredict._shape[0] === 16 && forwardAfterPredict._shape[1] === 6,
    `transformers predict(full-sequence while training): expected training mode restored so forward() returns [16,6], got [${forwardAfterPredict._shape[0]}, ${forwardAfterPredict._shape[1]}]`
  );
  assert(
    fullSequenceModel.getPredictMode() === "full-sequence",
    `transformers predict(full-sequence while training): expected mode full-sequence after predict, got ${fullSequenceModel.getPredictMode()}`
  );

  const switchedPred = fullSequenceModel.setPredictMode("next-token").predict(x);
  assert(
    switchedPred._shape[0] === 16 && switchedPred._shape[1] === 1,
    `transformers setPredictMode(next-token): expected [16,1], got [${switchedPred._shape[0]}, ${switchedPred._shape[1]}]`
  );

  console.log("=== Transformer API Correctness ===");
  console.table([
    { check: "predict default next-token", status: "pass" },
    { check: "predict full-sequence", status: "pass" },
    { check: "set/get predictMode", status: "pass" },
    { check: "predict restores training mode", status: "pass" },
  ]);
}
