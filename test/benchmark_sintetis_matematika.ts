import * as fs from "fs";
import * as path from "path";
import { performance } from "perf_hooks";
import { BPETokenizer } from "../src/tokenizer";
import { Transformers } from "../src/models";
import mj from "../src/math";

type Sample = {
  x: number[];
  y: number;
};

const seqLen = 128;
const batchSize = 64;
const units = 64;
const heads = 8;
const alpha = 1e-5;
const benchmark = "math_synthetic_v1_1_0";

function loadTokenizerSilently(vocabPath: string): BPETokenizer {
  const originalLog = console.log;
  console.log = () => {};
  try {
    return BPETokenizer.load(vocabPath);
  } finally {
    console.log = originalLog;
  }
}

function buildSamples(lines: string[], tokenizer: BPETokenizer): Sample[] {
  const padId = tokenizer.getPadId();
  const samples: Sample[] = [];

  for (const line of lines) {
    const tokens = tokenizer.encode(line);
    // Kita minimal butuh 2 token untuk (input, target)
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

  for (let b = 0; b < actualBatchSize; b++) {
    const sample = samples[startIndex + b];
    for (let row = 0; row < seqLen; row++) {
      x._data[row * actualBatchSize + b] = sample.x[row];
    }
    y._data[b] = sample.y;
  }

  return { x, y };
}

async function benchmarkMathSynthetic() {
  const vocabPath = path.join(__dirname, "..", "dataset", "math_vocab.json");
  const datasetPath = path.join(__dirname, "..", "dataset", "dataset_matematika_1000.json");

  if (!fs.existsSync(vocabPath)) {
    throw new Error(`Vocabulary not found: ${vocabPath}. Pastikan project math-bot sudah terpasang.`);
  }

  const tokenizer = loadTokenizerSilently(vocabPath);
  const rawData = fs.readFileSync(datasetPath, "utf-8");
  const json = JSON.parse(rawData) as { prompt: string; response: string }[];

  // Ambil subset 256 data untuk benchmark
  const lines = json
    .slice(0, 256)
    .map((item) => `${item.prompt} ${item.response}`.toLowerCase())
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  const samples = buildSamples(lines, tokenizer);
  if (samples.length === 0) {
    throw new Error("No samples were generated from the math database.");
  }

  const model = new Transformers({
    units,
    seqLen,
    vocabSize: tokenizer.getVocabSize(),
    heads,
    alpha,
    padTokenId: tokenizer.getPadId(),
  });

  model.compile({ alpha, optimizer: "adam", error: "softmaxCrossEntropy" });
  for (const layer of model.layers) {
    if (layer.name === "dropout layer") {
      layer.status = "train";
    }
  }

  // Warmup
  const firstBatchSize = Math.min(batchSize, samples.length);
  const firstBatch = fillBatch(samples, 0, firstBatchSize, seqLen);
  model.forward(firstBatch.x);
  model.backward(firstBatch.y);

  // Core benchmark
  const start = performance.now();
  for (let i = 0; i < samples.length; i += batchSize) {
    const actualBatchSize = Math.min(batchSize, samples.length - i);
    const { x, y } = fillBatch(samples, i, actualBatchSize, seqLen);
    model.forward(x);
    model.backward(y);
  }
  const elapsed = performance.now() - start;

  const msPerSample = elapsed / samples.length;
  const samplesPerSec = 1000 / msPerSample;

  console.log(
    JSON.stringify({
      benchmark,
      samples: samples.length,
      msPerSample,
      samplesPerSec,
    })
  );
}

benchmarkMathSynthetic().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
