import { performance } from "perf_hooks";
import mj from "../src/math";
import { Transformers } from "../src/models";

async function benchmarkTrainingStep() {
  const benchmark = "training_step";
  const iterations = 20;
  const batchSize = 64;
  const seqLen = 128;
  const units = 64;
  const vocabSize = 20000;
  const heads = 8;
  const alpha = 1e-5;
  const padTokenId = 0;

  const model = new Transformers({
    units,
    seqLen,
    vocabSize,
    heads,
    alpha,
    padTokenId,
  });

  model.compile({ alpha, optimizer: "adam", error: "softmaxCrossEntropy" });
  for (const layer of model.layers) {
    if (layer.name === "dropout layer") {
      layer.status = "train";
    }
  }

  const x = mj.zeros([seqLen, batchSize]);
  const y = mj.zeros([1, batchSize]);
  for (let i = 0; i < batchSize; i++) {
    y._data[i] = i % 100;
  }

  model.forward(x);
  model.backward(y);

  const start = performance.now();
  for (let i = 0; i < iterations; i++) {
    model.forward(x);
    model.backward(y);
  }
  const elapsed = performance.now() - start;

  const msPerBatch = elapsed / iterations;
  const batchesPerSec = 1000 / msPerBatch;
  const samplesPerSec = batchesPerSec * batchSize;

  console.log(
    JSON.stringify({
      benchmark,
      iterations,
      msPerBatch,
      batchesPerSec,
      samplesPerSec,
    })
  );
}

benchmarkTrainingStep().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
