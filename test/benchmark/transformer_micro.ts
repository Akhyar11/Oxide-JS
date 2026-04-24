import { performance } from "perf_hooks";
import mj from "../../src/math";
import { Transformers } from "../../src/models";

export async function runTransformerMicroBenchmark() {
  const seqLen = 64;
  const batchSize = 8;
  const units = 64;
  const heads = 8;
  const numBlocks = 1;
  const vocabSize = 2000;
  const warmupIterations = 2;
  const iterations = 5;

  const model = new Transformers({
    units,
    seqLen,
    vocabSize,
    heads,
    numBlocks,
    alpha: 1e-5,
    padTokenId: 0,
  });
  model.compile({ alpha: 1e-5, optimizer: "adam", error: "softmaxCrossEntropy" });

  const x = mj.zeros([seqLen, batchSize]);
  const y = mj.zeros([seqLen, batchSize]);
  for (let i = 0; i < x._data.length; i++) {
    x._data[i] = (i % 97) + 1;
    y._data[i] = (i % 89) + 1;
  }

  for (let i = 0; i < warmupIterations; i++) {
    model.forward(x);
    model.backward(y);
  }

  const start = performance.now();
  for (let i = 0; i < iterations; i++) {
    model.forward(x);
    model.backward(y);
  }
  const totalMs = performance.now() - start;

  const result = {
    benchmark: "transformer_micro",
    seqLen,
    batchSize,
    units,
    heads,
    numBlocks,
    warmupIterations,
    iterations,
    totalMs,
    msPerIter: totalMs / iterations,
  };

  console.log(JSON.stringify(result, null, 2));
  return result;
}

if (require.main === module) {
  runTransformerMicroBenchmark().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
