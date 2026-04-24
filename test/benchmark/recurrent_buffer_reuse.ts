import { performance } from "perf_hooks";
import mj from "../../src/math";
import Matrix from "../../src/matrix";
import RNN from "../../src/layers/rnn";
import LSTM from "../../src/layers/lstm";
import GRU from "../../src/layers/gru";

type RecurrentKind = "rnn" | "lstm" | "gru";
type RecurrentLayer = RNN | LSTM | GRU;

type BenchmarkCase = {
  seqLen: number;
  batchSize: number;
};

type BenchmarkResult = {
  model: RecurrentKind;
  mode: "fixed-shape" | "shape-churn";
  iterations: number;
  totalMs: number;
  msPerIter: number;
};

const FIXED_CASE: BenchmarkCase = { seqLen: 64, batchSize: 8 };
const CHURN_CASES: BenchmarkCase[] = [
  { seqLen: 128, batchSize: 4 },
  { seqLen: 64, batchSize: 8 },
  { seqLen: 96, batchSize: 6 },
  { seqLen: 32, batchSize: 16 },
  { seqLen: 48, batchSize: 10 },
];
const WARMUP_ITERS = 8;
const BENCH_ITERS = 40;
const UNITS = 32;
const HIDDEN_UNITS = 32;

function createLayer(kind: RecurrentKind): RecurrentLayer {
  const common = {
    units: UNITS,
    hiddenUnits: HIDDEN_UNITS,
    returnSequences: true,
    optimizer: "sgd" as const,
    alpha: 0,
    status: "input" as const,
    clipGradient: false,
  };

  switch (kind) {
    case "rnn":
      return new RNN(common);
    case "lstm":
      return new LSTM(common);
    case "gru":
      return new GRU(common);
  }
}

function fillInput(matrix: Matrix, seqLen: number, batchSize: number): void {
  const totalCols = seqLen * batchSize;
  for (let row = 0; row < matrix._shape[0]; row++) {
    for (let col = 0; col < totalCols; col++) {
      const sampleIdx = Math.floor(col / batchSize);
      const batchIdx = col % batchSize;
      matrix._data[row * totalCols + col] =
        Math.sin((row + 1) * 0.17 + sampleIdx * 0.11 + batchIdx * 0.07) * 0.25;
    }
  }
}

function buildInput(seqLen: number, batchSize: number): Matrix {
  const totalCols = seqLen * batchSize;
  const x = mj.zeros([UNITS, totalCols]);
  fillInput(x, seqLen, batchSize);
  return x;
}

function createCaseArtifacts(seqLen: number, batchSize: number) {
  const x = buildInput(seqLen, batchSize);
  const err = mj.zeros([HIDDEN_UNITS, seqLen * batchSize]);
  const y = mj.zeros([HIDDEN_UNITS, seqLen * batchSize]);
  return { x, y, err };
}

function createSingleCaseArtifacts(seqLen: number) {
  const x = buildInput(seqLen, 1);
  const err = mj.zeros([HIDDEN_UNITS, seqLen]);
  const y = mj.zeros([HIDDEN_UNITS, seqLen]);
  return { x, y, err };
}

function forwardBatch(layer: RecurrentLayer, x: Matrix, batchSize: number): Matrix {
  if (layer instanceof RNN) return layer.forwardBatch(x, batchSize);
  if (layer instanceof LSTM) return layer.forwardBatch(x, batchSize);
  return layer.forwardBatch(x, batchSize);
}

function backwardBatch(layer: RecurrentLayer, y: Matrix, err: Matrix, batchSize: number): Matrix {
  if (layer instanceof RNN) return layer.backwardBatch(y, err, batchSize);
  if (layer instanceof LSTM) return layer.backwardBatch(y, err, batchSize);
  return layer.backwardBatch(y, err, batchSize);
}

function forwardSingle(layer: RecurrentLayer, x: Matrix): Matrix {
  return layer.forward(x);
}

function backwardSingle(layer: RecurrentLayer, y: Matrix, err: Matrix): Matrix {
  return layer.backward(y, err);
}

function runScenario(kind: RecurrentKind, mode: "fixed-shape" | "shape-churn"): BenchmarkResult {
  const layer = createLayer(kind);
  const cases =
    mode === "fixed-shape"
      ? Array.from({ length: WARMUP_ITERS + BENCH_ITERS }, () => FIXED_CASE)
      : Array.from({ length: WARMUP_ITERS + BENCH_ITERS }, (_, idx) => CHURN_CASES[idx % CHURN_CASES.length]);

  const useBatchPath = kind !== "gru";
  const prepared = useBatchPath
    ? cases.map(({ seqLen, batchSize }) => ({
        seqLen,
        batchSize,
        ...createCaseArtifacts(seqLen, batchSize),
      }))
    : cases.map(({ seqLen }) => ({
        seqLen,
        batchSize: 1,
        ...createSingleCaseArtifacts(seqLen),
      }));

  for (let i = 0; i < WARMUP_ITERS; i++) {
    const item = prepared[i];
    if (useBatchPath) {
      forwardBatch(layer, item.x, item.batchSize);
      backwardBatch(layer, item.y, item.err, item.batchSize);
    } else {
      forwardSingle(layer, item.x);
      backwardSingle(layer, item.y, item.err);
    }
  }

  const start = performance.now();
  for (let i = WARMUP_ITERS; i < prepared.length; i++) {
    const item = prepared[i];
    if (useBatchPath) {
      forwardBatch(layer, item.x, item.batchSize);
      backwardBatch(layer, item.y, item.err, item.batchSize);
    } else {
      forwardSingle(layer, item.x);
      backwardSingle(layer, item.y, item.err);
    }
  }
  const totalMs = performance.now() - start;

  return {
    model: kind,
    mode,
    iterations: BENCH_ITERS,
    totalMs,
    msPerIter: totalMs / BENCH_ITERS,
  };
}

export async function runRecurrentBufferReuseBenchmarks(): Promise<BenchmarkResult[]> {
  const results: BenchmarkResult[] = [];
  const models: RecurrentKind[] = ["rnn", "lstm", "gru"];
  const modes: Array<"fixed-shape" | "shape-churn"> = ["fixed-shape", "shape-churn"];

  console.log("\nRunning recurrent buffer reuse benchmarks...");
  console.log(
    `Config: units=${UNITS}, hiddenUnits=${HIDDEN_UNITS}, warmup=${WARMUP_ITERS}, iterations=${BENCH_ITERS}`
  );

  for (const model of models) {
    for (const mode of modes) {
      const result = runScenario(model, mode);
      results.push(result);
      console.log(`- ${model} ${mode}: ${result.msPerIter.toFixed(3)} ms/iter`);
    }
  }

  console.log("\nRecurrent Buffer Reuse Results:");
  console.table(
    results.map((result) => ({
      Model: result.model,
      Mode: result.mode,
      Iterations: result.iterations,
      "Total ms": result.totalMs.toFixed(2),
      "ms/iter": result.msPerIter.toFixed(3),
    }))
  );

  return results;
}

if (require.main === module) {
  runRecurrentBufferReuseBenchmarks().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
