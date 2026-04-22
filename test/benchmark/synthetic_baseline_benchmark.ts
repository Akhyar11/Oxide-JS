import * as fs from "fs";
import * as path from "path";
import { performance } from "perf_hooks";
import mj from "../../src/math";
import Matrix from "../../src/matrix";
import { Transformers, Sequential } from "../../src/models";
import { RNN, LSTM, GRU, Embedding, Dense } from "../../src/layers";
import { BPETokenizer } from "../../src/tokenizer";

type Sample = {
  x: number[];
  y: number;
  ySeq: number[];
};

type SyntheticBaselineConfig = {
  benchmark: string;
  modelType: "transformers" | "rnn" | "lstm" | "gru";
  seqLen: number;
  batchSize: number;
  units: number;
  heads: number;
  alpha: number;
  subsetRecords: number;
  warmupBatches: number;
};

type SyntheticBaselineResult = {
  benchmark: string;
  modelType: string;
  dataset: string;
  records: number;
  samples: number;
  warmupBatches: number;
  batches: number;
  msPerBatch: number;
  msPerSample: number;
  samplesPerSec: number;
  seqLen: number;
  batchSize: number;
  units: number;
  heads: number;
  vocabSize: number;
};

const DEFAULT_CONFIG: SyntheticBaselineConfig = {
  benchmark: "math_synthetic_baseline",
  modelType: "transformers",
  seqLen: 128,
  batchSize: 64,
  units: 64,
  heads: 8,
  alpha: 1e-5,
  subsetRecords: 256,
  warmupBatches: 1,
};

class RecurrentBenchmarkModel {
  private embedding: Embedding;
  private recurrent: RNN | LSTM | GRU;
  private dense: Dense;
  private emptyErr: Matrix = mj.matrix([[]]);

  constructor(type: "rnn" | "lstm" | "gru", units: number, vocabSize: number, alpha: number) {
    const embedding = new Embedding({ vocabSize, embeddingDim: units, alpha });
    let recurrent: RNN | LSTM | GRU;

    switch (type) {
      case "rnn":
        recurrent = new RNN({ units, hiddenUnits: units, returnSequences: false, alpha });
        break;
      case "lstm":
        recurrent = new LSTM({ units, hiddenUnits: units, returnSequences: false, alpha });
        break;
      case "gru":
        recurrent = new GRU({ units, hiddenUnits: units, returnSequences: false, alpha });
        break;
    }

    const dense = new Dense({
      units: units,
      outputUnits: vocabSize,
      activation: "linear",
      alpha,
      status: "output",
      loss: "softmaxCrossEntropy",
    });

    this.embedding = embedding;
    this.recurrent = recurrent;
    this.dense = dense;
  }

  compile(config: { alpha: number; optimizer: "adam"; error: "softmaxCrossEntropy" }) {
    this.embedding.compile(config);
    this.recurrent.compile(config);
    this.dense.compile(config);
  }

  trainBatch(xBatch: Matrix, yBatch: Matrix) {
    const [, batchSize] = xBatch._shape;
    const xEmb = this.embedding.forwardTimeMajor(xBatch);
    const h = this.forwardRecurrentBatch(xEmb, batchSize);
    this.dense.forward(h);
    const recurrentErr = this.backwardRecurrentBatch(this.dense.backward(yBatch, this.emptyErr), batchSize);
    this.embedding.backward(this.emptyErr, recurrentErr);
  }

  private forwardRecurrentBatch(xEmb: Matrix, batchSize: number): Matrix {
    if (this.recurrent instanceof RNN) return this.recurrent.forwardBatch(xEmb, batchSize);
    if (this.recurrent instanceof LSTM) return this.recurrent.forwardBatch(xEmb, batchSize);
    return this.recurrent.forwardBatch(xEmb, batchSize);
  }

  private backwardRecurrentBatch(err: Matrix, batchSize: number): Matrix {
    if (this.recurrent instanceof RNN) return this.recurrent.backwardBatch(this.emptyErr, err, batchSize);
    if (this.recurrent instanceof LSTM) return this.recurrent.backwardBatch(this.emptyErr, err, batchSize);
    return this.recurrent.backwardBatch(this.emptyErr, err, batchSize);
  }
}

function loadTokenizerSilently(vocabPath: string): BPETokenizer {
  const originalLog = console.log;
  console.log = () => {};
  try {
    return BPETokenizer.load(vocabPath);
  } finally {
    console.log = originalLog;
  }
}

function loadSyntheticCorpus(datasetPath: string, subsetRecords: number): string[] {
  const rawData = fs.readFileSync(datasetPath, "utf-8");
  const rows = JSON.parse(rawData) as Array<{ prompt: string; response: string }>;

  return rows
    .slice(0, subsetRecords)
    .map((item) => `${item.prompt} ${item.response}`.toLowerCase().trim())
    .filter((line) => line.length > 0);
}

function buildSamples(lines: string[], tokenizer: BPETokenizer, seqLen: number): Sample[] {
  const padId = tokenizer.getPadId();
  const samples: Sample[] = [];

  for (const line of lines) {
    const tokens = tokenizer.encode(line);
    if (tokens.length < 2) continue;

    for (let i = 1; i < tokens.length; i++) {
      const start = Math.max(0, i - seqLen);
      const context = tokens.slice(start, i);
      const x = new Array<number>(seqLen).fill(padId);
      const offset = seqLen - context.length;

      for (let j = 0; j < context.length; j++) {
        x[offset + j] = context[j];
      }

      const ySeq = new Array<number>(seqLen).fill(padId);
      for (let j = 0; j < seqLen - 1; j++) {
        ySeq[j] = x[j + 1];
      }
      ySeq[seqLen - 1] = tokens[i];

      samples.push({ x, y: tokens[i], ySeq });
    }
  }

  return samples;
}

function fillBatch(samples: Sample[], startIndex: number, actualBatchSize: number, seqLen: number) {
  const x = mj.zeros([seqLen, actualBatchSize]);
  const y = mj.zeros([1, actualBatchSize]);
  const ySeq = mj.zeros([seqLen, actualBatchSize]);

  for (let batchIndex = 0; batchIndex < actualBatchSize; batchIndex++) {
    const sample = samples[startIndex + batchIndex];
    for (let row = 0; row < seqLen; row++) {
      x._data[row * actualBatchSize + batchIndex] = sample.x[row];
      ySeq._data[row * actualBatchSize + batchIndex] = sample.ySeq[row];
    }
    y._data[batchIndex] = sample.y;
  }

  return { x, y, ySeq };
}

function enableTrainingDropout(model: Sequential): void {
  for (const layer of model.layers) {
    if (typeof (layer as any).setTrainingMode === "function") {
      (layer as any).setTrainingMode(true);
    }
  }
}

export async function runSyntheticBaselineBenchmark(
  overrides: Partial<SyntheticBaselineConfig> = {}
): Promise<SyntheticBaselineResult> {
  const config = { ...DEFAULT_CONFIG, ...overrides };
  const vocabPath = path.join(__dirname, "..", "..", "dataset", "math_vocab.json");
  const datasetPath = path.join(__dirname, "..", "..", "dataset", "dataset_matematika_1000.json");

  if (!fs.existsSync(vocabPath)) {
    throw new Error(`Vocabulary not found: ${vocabPath}`);
  }
  if (!fs.existsSync(datasetPath)) {
    throw new Error(`Dataset not found: ${datasetPath}`);
  }

  const tokenizer = loadTokenizerSilently(vocabPath);
  const lines = loadSyntheticCorpus(datasetPath, config.subsetRecords);
  const samples = buildSamples(lines, tokenizer, config.seqLen);

  if (samples.length === 0) {
    throw new Error("No samples were generated from the synthetic baseline dataset.");
  }

  let model: Sequential | RecurrentBenchmarkModel;
  if (config.modelType === "transformers") {
    model = new Transformers({
      units: config.units,
      seqLen: config.seqLen,
      vocabSize: tokenizer.getVocabSize(),
      heads: config.heads,
      alpha: config.alpha,
      padTokenId: tokenizer.getPadId(),
    });
  } else {
    model = new RecurrentBenchmarkModel(config.modelType, config.units, tokenizer.getVocabSize(), config.alpha);
  }

  model.compile({ alpha: config.alpha, optimizer: "adam", error: "softmaxCrossEntropy" });
  if (model instanceof Sequential) {
    enableTrainingDropout(model);
  }

  const warmupBatches = Math.max(1, config.warmupBatches);
  for (let batchNumber = 0; batchNumber < warmupBatches; batchNumber++) {
    const startIndex = (batchNumber * config.batchSize) % samples.length;
    const actualBatchSize = Math.min(config.batchSize, samples.length - startIndex);
    const { x, y, ySeq } = fillBatch(samples, startIndex, actualBatchSize, config.seqLen);
    if (config.modelType === "transformers") {
      (model as Sequential).forward(x);
      (model as Sequential).backward(ySeq);
    } else {
      (model as RecurrentBenchmarkModel).trainBatch(x, y);
    }
  }

  const start = performance.now();
  let batchCount = 0;
  for (let sampleIndex = 0; sampleIndex < samples.length; sampleIndex += config.batchSize) {
    const actualBatchSize = Math.min(config.batchSize, samples.length - sampleIndex);
    const { x, y, ySeq } = fillBatch(samples, sampleIndex, actualBatchSize, config.seqLen);
    if (config.modelType === "transformers") {
      (model as Sequential).forward(x);
      (model as Sequential).backward(ySeq);
    } else {
      (model as RecurrentBenchmarkModel).trainBatch(x, y);
    }
    batchCount++;
  }
  const elapsed = performance.now() - start;

  const msPerBatch = elapsed / batchCount;
  const msPerSample = elapsed / samples.length;
  const samplesPerSec = 1000 / msPerSample;

  return {
    benchmark: config.benchmark,
    modelType: config.modelType,
    dataset: "dataset_matematika_1000.json",
    records: lines.length,
    samples: samples.length,
    warmupBatches,
    batches: batchCount,
    msPerBatch,
    msPerSample,
    samplesPerSec,
    seqLen: config.seqLen,
    batchSize: config.batchSize,
    units: config.units,
    heads: config.heads,
    vocabSize: tokenizer.getVocabSize(),
  };
}

export async function runAllSyntheticBaselineBenchmarks() {
  const models: Array<"transformers" | "rnn" | "lstm" | "gru"> = ["transformers", "rnn", "lstm", "gru"];
  const results = [];

  console.log("Running synthetic baseline benchmarks...");
  for (const modelType of models) {
    try {
      const result = await runSyntheticBaselineBenchmark({ modelType });
      results.push(result);
      console.log(`- ${modelType}: ${result.samplesPerSec.toFixed(2)} samples/s`);
    } catch (error) {
      console.error(`- ${modelType}: Failed - ${(error as Error).message}`);
    }
  }

  console.log("\nFull Results:");
  console.table(
    results.map((r) => ({
      Model: r.modelType,
      "Samples/s": r.samplesPerSec.toFixed(2),
      "ms/Batch": r.msPerBatch.toFixed(2),
      "ms/Sample": r.msPerSample.toFixed(4),
    }))
  );
}

if (require.main === module) {
  runAllSyntheticBaselineBenchmarks().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
