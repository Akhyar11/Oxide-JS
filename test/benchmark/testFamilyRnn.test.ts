import { performance } from "perf_hooks";
import { Matrix } from "@oxide-js/core";
import { mj } from "@oxide-js/core";
import { RecurrentModel } from "@oxide-js/models";

type RecurrentFamily = "rnn" | "lstm" | "gru";

type FamilyBenchmarkConfig = {
  epochs?: number;
  sampleCount?: number;
  seqLen?: number;
  units?: number;
  hiddenUnits?: number;
  numClasses?: number;
  alpha?: number;
};

type FamilyBenchmarkResult = {
  model: RecurrentFamily;
  epochs: number;
  samples: number;
  totalMs: number;
  avgEpochMs: number;
  samplesPerSecond: number;
  epochMs: number[];
  epochLoss: number[];
  finalLoss: number;
};

const DEFAULT_CONFIG: Required<FamilyBenchmarkConfig> = {
  epochs: 3,
  sampleCount: 128,
  seqLen: 24,
  units: 16,
  hiddenUnits: 32,
  numClasses: 6,
  alpha: 0.001,
};

function createSyntheticSequenceSample(
  sampleIndex: number,
  seqLen: number,
  units: number
): Matrix {
  const rows: number[][] = [];

  for (let feature = 0; feature < units; feature++) {
    const row: number[] = [];
    for (let t = 0; t < seqLen; t++) {
      const base = ((sampleIndex + 1) * (feature + 3) + (t + 5) * 7) % 23;
      const wave = Math.sin((sampleIndex + 1) * 0.13 + feature * 0.17 + t * 0.11);
      row.push(base / 22 + wave * 0.25);
    }
    rows.push(row);
  }

  return mj.matrix(rows);
}

function createSyntheticClassificationDataset(
  sampleCount: number,
  seqLen: number,
  units: number,
  numClasses: number
): { X: Matrix[]; y: Matrix[] } {
  const X: Matrix[] = [];
  const y: Matrix[] = [];

  for (let i = 0; i < sampleCount; i++) {
    X.push(createSyntheticSequenceSample(i, seqLen, units));
    y.push(mj.matrix([[(i * 3 + seqLen + units) % numClasses]]));
  }

  return { X, y };
}

function createModel(
  family: RecurrentFamily,
  units: number,
  hiddenUnits: number,
  numClasses: number,
  seqLen: number,
  alpha: number
): RecurrentModel {
  return new RecurrentModel({
    kind: family,
    inputSize: units,
    hiddenSize: hiddenUnits,
    numLayers: 1,
    outputSize: numClasses,
    seqLen,
    mode: "many-to-one",
    loss: "softmaxCrossEntropy",
    alpha,
    optimizer: "adam",
  });
}

function benchmarkFamilyTraining(
  family: RecurrentFamily,
  config: Required<FamilyBenchmarkConfig>,
  X: Matrix[],
  y: Matrix[]
): FamilyBenchmarkResult {
  const model = createModel(
    family,
    config.units,
    config.hiddenUnits,
    config.numClasses,
    config.seqLen,
    config.alpha
  );

  const epochMs: number[] = [];
  const epochLoss: number[] = [];
  let epochStart = performance.now();
  const totalStart = performance.now();

  model.fit(X, y, config.epochs, {
    batchSize: 1,
    shuffle: false,
    verbose: false,
    onEpochEnd: (_epoch, loss) => {
      const now = performance.now();
      epochMs.push(now - epochStart);
      epochLoss.push(loss);
      epochStart = now;
    },
  });

  const totalMs = performance.now() - totalStart;
  const totalSamples = config.sampleCount * config.epochs;

  return {
    model: family,
    epochs: config.epochs,
    samples: config.sampleCount,
    totalMs,
    avgEpochMs: totalMs / config.epochs,
    samplesPerSecond: totalSamples / (totalMs / 1000),
    epochMs,
    epochLoss,
    finalLoss: epochLoss[epochLoss.length - 1] ?? Number.NaN,
  };
}

export function runFamilyRnnTrainingBenchmark(
  overrideConfig: FamilyBenchmarkConfig = {}
): FamilyBenchmarkResult[] {
  const config: Required<FamilyBenchmarkConfig> = {
    ...DEFAULT_CONFIG,
    ...overrideConfig,
  };

  const { X, y } = createSyntheticClassificationDataset(
    config.sampleCount,
    config.seqLen,
    config.units,
    config.numClasses
  );

  const families: RecurrentFamily[] = ["rnn", "lstm", "gru"];
  const results = families.map((family) => benchmarkFamilyTraining(family, config, X, y));

  console.log(`=== Family RNN Training Benchmark (${config.epochs} Epoch) ===`);
  console.log(
    `config: samples=${config.sampleCount}, seqLen=${config.seqLen}, units=${config.units}, hiddenUnits=${config.hiddenUnits}, classes=${config.numClasses}, epochs=${config.epochs}`
  );
  console.table(
    results.map((result) => ({
      model: result.model,
      epochs: result.epochs,
      samples: result.samples,
      totalMs: Number(result.totalMs.toFixed(2)),
      avgEpochMs: Number(result.avgEpochMs.toFixed(2)),
      samplesPerSecond: Number(result.samplesPerSecond.toFixed(2)),
      finalLoss: Number(result.finalLoss.toFixed(6)),
    }))
  );

  for (const result of results) {
    console.log(
      `${result.model} epochMs=${result.epochMs.map((ms) => ms.toFixed(2)).join(", ")} | epochLoss=${result.epochLoss
        .map((loss) => loss.toFixed(6))
        .join(", ")}`
    );
  }

  return results;
}

