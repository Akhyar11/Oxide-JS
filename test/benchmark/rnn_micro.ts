import { performance } from "perf_hooks";
import mj from "../../src/math";
import RNN from "../../src/layers/rnn";

export async function runRNNMicroBenchmark() {
  const seqLen = 64;
  const batchSize = 8;
  const units = 64;
  const hiddenUnits = 64;
  const warmupIterations = 2;
  const iterations = 20;

  const layer = new RNN({
    units,
    hiddenUnits,
    returnSequences: true,
    optimizer: "sgd",
    alpha: 0,
    status: "input",
    clipGradient: false,
  });

  const totalCols = seqLen * batchSize;
  const x = mj.zeros([units, totalCols]);
  const y = mj.zeros([hiddenUnits, totalCols]);
  const err = mj.zeros([hiddenUnits, totalCols]);
  for (let i = 0; i < x._data.length; i++) {
    x._data[i] = Math.sin(i * 0.01) * 0.25;
  }

  for (let i = 0; i < warmupIterations; i++) {
    layer.forwardBatch(x, batchSize);
    layer.backwardBatch(y, err, batchSize);
  }

  const start = performance.now();
  for (let i = 0; i < iterations; i++) {
    layer.forwardBatch(x, batchSize);
    layer.backwardBatch(y, err, batchSize);
  }
  const totalMs = performance.now() - start;

  const result = {
    benchmark: "rnn_micro",
    seqLen,
    batchSize,
    units,
    hiddenUnits,
    warmupIterations,
    iterations,
    totalMs,
    msPerIter: totalMs / iterations,
  };

  console.log(JSON.stringify(result, null, 2));
  return result;
}

if (require.main === module) {
  runRNNMicroBenchmark().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
