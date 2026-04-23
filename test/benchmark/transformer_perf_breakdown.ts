import { performance } from "perf_hooks";
import mj from "../../src/math";
import Transformers from "../../src/models/transformers";
import { runSyntheticBaselineBenchmark } from "./synthetic_baseline_benchmark";

type TransformerPerfConfig = {
  runs: number;
  seqLen: number;
  batchSize: number;
  units: number;
  heads: number;
  numBlocks: number;
  vocabSize: number;
  alpha: number;
  padTokenId: number;
  warmupIters: number;
  measureIters: number;
  subsetRecords: number;
  warmupBatches: number;
};

type StageStats = Record<string, { totalMs: number; avgMs: number; count: number }>;

type PerfRun = {
  run: number;
  inferenceOnlyMsPerIter: number;
  forwardOnlyMsPerIter: number;
  backwardOnlyMsPerIter: number;
  trainingStepMsPerIter: number;
  logitsShape: [number, number];
  nextTokenShape: [number, number];
  profile: StageStats;
};

const DEFAULT_CONFIG: TransformerPerfConfig = {
  runs: 3,
  seqLen: 128,
  batchSize: 64,
  units: 64,
  heads: 8,
  numBlocks: 1,
  vocabSize: 2000,
  alpha: 1e-5,
  padTokenId: 0,
  warmupIters: 2,
  measureIters: 5,
  subsetRecords: 256,
  warmupBatches: 1,
};

function createSyntheticBatch(config: TransformerPerfConfig) {
  const x = mj.zeros([config.seqLen, config.batchSize]);
  const y = mj.zeros([config.seqLen, config.batchSize]);

  for (let b = 0; b < config.batchSize; b++) {
    for (let pos = 0; pos < config.seqLen; pos++) {
      const token = pos === 0 ? config.padTokenId : 1 + ((pos + b) % (config.vocabSize - 1));
      x._data[pos * config.batchSize + b] = token;
      y._data[pos * config.batchSize + b] =
        pos < config.seqLen - 1 ? 1 + ((pos + b + 1) % (config.vocabSize - 1)) : config.padTokenId;
    }
  }

  return { x, y };
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

async function runPerfBreakdown(configOverrides: Partial<TransformerPerfConfig> = {}) {
  const config = { ...DEFAULT_CONFIG, ...configOverrides };
  const { x, y } = createSyntheticBatch(config);
  const perfRuns: PerfRun[] = [];

  for (let run = 0; run < config.runs; run++) {
    const model = new Transformers({
      units: config.units,
      seqLen: config.seqLen,
      vocabSize: config.vocabSize,
      heads: config.heads,
      numBlocks: config.numBlocks,
      alpha: config.alpha,
      padTokenId: config.padTokenId,
    });
    model.compile({ alpha: config.alpha, optimizer: "adam", error: "softmaxCrossEntropy" });
    model.train();
    model.enableProfiling(true);

    for (let i = 0; i < config.warmupIters; i++) {
      model.forward(x);
      model.backward(y);
    }

    let nextToken = model.forwardNextToken(x);
    const inferenceStart = performance.now();
    for (let i = 0; i < config.measureIters; i++) {
      nextToken = model.forwardNextToken(x);
    }
    const inferenceEnd = performance.now();

    let logits = model.forward(x);
    const forwardStart = performance.now();
    for (let i = 0; i < config.measureIters; i++) {
      logits = model.forward(x);
    }
    const forwardEnd = performance.now();

    model.forward(x);
    const backwardStart = performance.now();
    for (let i = 0; i < config.measureIters; i++) {
      model.backward(y);
      if (i < config.measureIters - 1) model.forward(x);
    }
    const backwardEnd = performance.now();

    const trainingStart = performance.now();
    model.resetProfiling();
    for (let i = 0; i < config.measureIters; i++) {
      model.forward(x);
      model.backward(y);
    }
    const trainingEnd = performance.now();

    perfRuns.push({
      run: run + 1,
      inferenceOnlyMsPerIter: (inferenceEnd - inferenceStart) / config.measureIters,
      forwardOnlyMsPerIter: (forwardEnd - forwardStart) / config.measureIters,
      backwardOnlyMsPerIter: (backwardEnd - backwardStart) / config.measureIters,
      trainingStepMsPerIter: (trainingEnd - trainingStart) / config.measureIters,
      logitsShape: logits._shape,
      nextTokenShape: nextToken._shape,
      profile: model.getProfilingReport(true),
    });
  }

  const endToEndRuns = [];
  for (let run = 0; run < config.runs; run++) {
    endToEndRuns.push(
      await runSyntheticBaselineBenchmark({
        modelType: "transformers",
        seqLen: config.seqLen,
        batchSize: config.batchSize,
        units: config.units,
        heads: config.heads,
        numBlocks: config.numBlocks,
        alpha: config.alpha,
        subsetRecords: config.subsetRecords,
        warmupBatches: config.warmupBatches,
      })
    );
  }

  const summary = {
    config: {
      seqLen: config.seqLen,
      batchSize: config.batchSize,
      units: config.units,
      heads: config.heads,
      numBlocks: config.numBlocks,
      vocabSize: config.vocabSize,
      alpha: config.alpha,
      subsetRecords: config.subsetRecords,
      warmupBatches: config.warmupBatches,
      warmupIters: config.warmupIters,
      measureIters: config.measureIters,
      runs: config.runs,
    },
    perfRuns,
    perfMedian: {
      inferenceOnlyMsPerIter: median(perfRuns.map((run) => run.inferenceOnlyMsPerIter)),
      forwardOnlyMsPerIter: median(perfRuns.map((run) => run.forwardOnlyMsPerIter)),
      backwardOnlyMsPerIter: median(perfRuns.map((run) => run.backwardOnlyMsPerIter)),
      trainingStepMsPerIter: median(perfRuns.map((run) => run.trainingStepMsPerIter)),
    },
    endToEndRuns,
    endToEndMedian: {
      msPerBatch: median(endToEndRuns.map((run) => run.msPerBatch)),
      msPerSample: median(endToEndRuns.map((run) => run.msPerSample)),
      samplesPerSec: median(endToEndRuns.map((run) => run.samplesPerSec)),
    },
  };

  console.log(JSON.stringify(summary, null, 2));
  return summary;
}

if (require.main === module) {
  runPerfBreakdown().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}

export { runPerfBreakdown };
