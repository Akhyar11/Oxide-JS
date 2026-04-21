import * as fs from "fs";
import * as path from "path";
import { performance } from "perf_hooks";
import mj from "../../src/math";
import { Transformers } from "../../src/models";
import { BPETokenizer } from "../../src/tokenizer";

type Sample = {
  x: number[];
  y: number;
};

type SyntheticBaselineConfig = {
  benchmark: string;
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
  seqLen: 128,
  batchSize: 64,
  units: 64,
  heads: 8,
  alpha: 1e-5,
  subsetRecords: 256,
  warmupBatches: 1,
};

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

function enableTrainingDropout(model: Transformers): void {
  for (const layer of model.layers) {
    if (layer.name === "dropout layer") {
      layer.status = "train";
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

  const model = new Transformers({
    units: config.units,
    seqLen: config.seqLen,
    vocabSize: tokenizer.getVocabSize(),
    heads: config.heads,
    alpha: config.alpha,
    padTokenId: tokenizer.getPadId(),
  });

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
  const result = await runSyntheticBaselineBenchmark();
  console.log(JSON.stringify(result));
}

if (require.main === module) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
