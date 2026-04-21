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

/**
 * Generic model wrapper for RNN, LSTM, and GRU to match Transformers benchmark structure.
 */
class RecurrentModel extends Sequential {
  private embedding: Embedding;
  private recurrent: RNN | LSTM | GRU;
  private dense: Dense;
  private lastTokenBuffer: Matrix;
  private emptyErr: Matrix = mj.matrix([[]]);

  constructor(type: "rnn" | "lstm" | "gru", units: number, seqLen: number, vocabSize: number, alpha: number) {
    const embedding = new Embedding({ vocabSize, embeddingDim: units, alpha });
    let recurrent: RNN | LSTM | GRU;

    switch (type) {
      case "rnn":
        recurrent = new RNN({ units, hiddenUnits: units, returnSequences: true, alpha });
        break;
      case "lstm":
        recurrent = new LSTM({ units, hiddenUnits: units, returnSequences: true, alpha });
        break;
      case "gru":
        recurrent = new GRU({ units, hiddenUnits: units, returnSequences: true, alpha });
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

    super({ layers: [embedding, recurrent, dense] });
    this.embedding = embedding;
    this.recurrent = recurrent;
    this.dense = dense;
    this.lastTokenBuffer = mj.zeros([units, 1]);
  }

  forward(x: Matrix): Matrix {
    const [seqLen, batchSize] = x._shape;
    const units = this.embedding.embeddingDim;

    // 1. Embedding
    const xEmb = this.embedding.forward(x);

    // 2. Recurrent Layer
    const h = this.recurrent.forward(xEmb);

    // 3. Extract Last Token for each sequence in batch
    if (this.lastTokenBuffer._shape[1] !== batchSize) {
      this.lastTokenBuffer = mj.zeros([units, batchSize]);
    }

    const hData = h._data;
    const lastTokenData = this.lastTokenBuffer._data;
    const totalCols = h._shape[1];

    for (let b = 0; b < batchSize; b++) {
      const lastTokenCol = (b + 1) * seqLen - 1;
      for (let i = 0; i < units; i++) {
        lastTokenData[i * batchSize + b] = hData[i * totalCols + lastTokenCol];
      }
    }

    // 4. Output Dense
    return this.dense.forward(this.lastTokenBuffer);
  }

  backward(y: Matrix) {
    // 1. Output Dense Backward
    const errDense = this.dense.backward(y, this.emptyErr);
    const batchSize = errDense._shape[1];
    const seqLen = (this.embedding as any).inputShape[0]; // approximated from embedding state
    const units = this.embedding.embeddingDim;
    const totalTokens = seqLen * batchSize;

    // 2. Map Dense Error back to full sequence
    const recurrentErrBuf = mj.zeros([units, totalTokens]);
    const recurrentErrData = recurrentErrBuf._data;
    const errDenseData = errDense._data;

    for (let b = 0; b < batchSize; b++) {
      const lastTokenCol = (b + 1) * seqLen - 1;
      for (let i = 0; i < units; i++) {
        recurrentErrData[i * totalTokens + lastTokenCol] = errDenseData[i * batchSize + b];
      }
    }

    // 3. Recurrent & Embedding Backward
    const recurrentErr = this.recurrent.backward(this.emptyErr, recurrentErrBuf);
    this.embedding.backward(this.emptyErr, recurrentErr);
  }

  getSeqLen(): number {
    return (this.embedding as any).inputShape[0];
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

      samples.push({ x, y: tokens[i] });
    }
  }

  return samples;
}

function fillBatch(samples: Sample[], startIndex: number, actualBatchSize: number, seqLen: number) {
  const x = mj.zeros([seqLen, actualBatchSize]);
  const y = mj.zeros([1, actualBatchSize]);

  for (let batchIndex = 0; batchIndex < actualBatchSize; batchIndex++) {
    const sample = samples[startIndex + batchIndex];
    for (let row = 0; row < seqLen; row++) {
      x._data[row * actualBatchSize + batchIndex] = sample.x[row];
    }
    y._data[batchIndex] = sample.y;
  }

  return { x, y };
}

function enableTrainingDropout(model: Sequential): void {
  for (const layer of model.layers) {
    if (layer.name === "dropout layer") {
      (layer as any).status = "train";
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

  let model: Sequential;
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
    model = new RecurrentModel(config.modelType, config.units, config.seqLen, tokenizer.getVocabSize(), config.alpha);
  }

  model.compile({ alpha: config.alpha, optimizer: "adam", error: "softmaxCrossEntropy" });
  enableTrainingDropout(model);

  const warmupBatches = Math.max(1, config.warmupBatches);
  for (let batchNumber = 0; batchNumber < warmupBatches; batchNumber++) {
    const startIndex = (batchNumber * config.batchSize) % samples.length;
    const actualBatchSize = Math.min(config.batchSize, samples.length - startIndex);
    const { x, y } = fillBatch(samples, startIndex, actualBatchSize, config.seqLen);
    model.forward(x);
    model.backward(y);
  }

  const start = performance.now();
  let batchCount = 0;
  for (let sampleIndex = 0; sampleIndex < samples.length; sampleIndex += config.batchSize) {
    const actualBatchSize = Math.min(config.batchSize, samples.length - sampleIndex);
    const { x, y } = fillBatch(samples, sampleIndex, actualBatchSize, config.seqLen);
    model.forward(x);
    model.backward(y);
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

async function main() {
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
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
