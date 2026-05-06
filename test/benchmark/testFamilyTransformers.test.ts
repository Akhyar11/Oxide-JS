import { performance } from "perf_hooks";
import { Matrix } from "@oxide-js/core";
import { mj } from "@oxide-js/core";
import { Transformers } from "@oxide-js/models";

type TransformerBenchmarkConfig = {
  epochs?: number;
  sampleCount?: number;
  seqLen?: number;
  vocabSize?: number;
  units?: number;
  heads?: number;
  numBlocks?: number;
  batchSize?: number;
  alpha?: number;
  padTokenId?: number;
  dropoutRate?: number;
};

type FullTokenBenchmarkResult = {
  mode: "full-token" | "full-token-padded-no-trim" | "full-token-trimPad";
  epochs: number;
  samples: number;
  totalMs: number;
  avgEpochMs: number;
  samplesPerSecond: number;
  epochMs: number[];
  epochLoss: number[];
  finalLoss: number;
};

type NextTokenBenchmarkResult = {
  mode: "next-token";
  epochs: number;
  samples: number;
  totalMs: number;
  avgEpochMs: number;
  samplesPerSecond: number;
  epochMs: number[];
  epochLoss: number[];
  finalLoss: number;
};

type TransformerBenchmarkResult = FullTokenBenchmarkResult | NextTokenBenchmarkResult;

const DEFAULT_CONFIG: Required<TransformerBenchmarkConfig> = {
  epochs: 3,
  sampleCount: 96,
  seqLen: 24,
  vocabSize: 128,
  units: 32,
  heads: 4,
  numBlocks: 2,
  batchSize: 8,
  alpha: 0.001,
  padTokenId: 0,
  dropoutRate: 0,
};

function createTokenSequence(
  sampleIndex: number,
  seqLen: number,
  vocabSize: number,
  padTokenId: number
): number[] {
  const tokens: number[] = [];
  const usableVocab = Math.max(2, vocabSize - 1);

  for (let pos = 0; pos < seqLen; pos++) {
    const token = ((sampleIndex + 1) * 11 + (pos + 3) * 7) % usableVocab;
    tokens.push(token + 1);
  }

  if (tokens.length > 0) {
    tokens[seqLen - 1] = ((sampleIndex + seqLen) % usableVocab) + 1;
  }

  if (padTokenId !== 0) {
    for (let i = 0; i < tokens.length; i++) {
      if (tokens[i] === 0) {
        tokens[i] = 1;
      }
    }
  }

  return tokens;
}

function createSequenceMatrix(tokens: number[]): Matrix {
  return mj.matrix(tokens.map((token) => [token]));
}

function createFullTokenTarget(tokens: number[], padTokenId: number): Matrix {
  const shifted: number[][] = [];
  for (let pos = 0; pos < tokens.length - 1; pos++) {
    shifted.push([tokens[pos + 1]]);
  }
  shifted.push([padTokenId]);
  return mj.matrix(shifted);
}

function createNextTokenTarget(tokens: number[], vocabSize: number): Matrix {
  const usableVocab = Math.max(2, vocabSize - 1);
  const nextToken = ((tokens[tokens.length - 1] * 5 + tokens.length * 3) % usableVocab) + 1;
  return mj.matrix([[nextToken]]);
}

function createSyntheticTransformerDataset(
  sampleCount: number,
  seqLen: number,
  vocabSize: number,
  padTokenId: number
): { X: Matrix[]; yFullToken: Matrix[]; yNextToken: Matrix[] } {
  const X: Matrix[] = [];
  const yFullToken: Matrix[] = [];
  const yNextToken: Matrix[] = [];

  for (let i = 0; i < sampleCount; i++) {
    const tokens = createTokenSequence(i, seqLen, vocabSize, padTokenId);
    X.push(createSequenceMatrix(tokens));
    yFullToken.push(createFullTokenTarget(tokens, padTokenId));
    yNextToken.push(createNextTokenTarget(tokens, vocabSize));
  }

  return { X, yFullToken, yNextToken };
}

function createRightPaddedTokenSequence(
  sampleIndex: number,
  seqLen: number,
  vocabSize: number,
  padTokenId: number
): number[] {
  const usableVocab = Math.max(2, vocabSize - 1);
  const minEffectiveLen = Math.min(seqLen, 4);
  const variableSpan = Math.max(1, seqLen - minEffectiveLen + 1);
  const effectiveLen = Math.min(seqLen, minEffectiveLen + (sampleIndex % variableSpan));
  const tokens = new Array<number>(seqLen).fill(padTokenId);

  for (let pos = 0; pos < effectiveLen; pos++) {
    tokens[pos] = (((sampleIndex + 5) * 13 + (pos + 1) * 17) % usableVocab) + 1;
  }

  return tokens;
}

function createRightPaddedFullTokenTarget(tokens: number[], padTokenId: number): Matrix {
  const shifted: number[][] = [];
  for (let pos = 0; pos < tokens.length - 1; pos++) {
    const current = tokens[pos];
    const next = tokens[pos + 1];
    shifted.push([current === padTokenId || next === padTokenId ? padTokenId : next]);
  }
  shifted.push([padTokenId]);
  return mj.matrix(shifted);
}

function createPaddedTransformerDataset(
  sampleCount: number,
  seqLen: number,
  vocabSize: number,
  padTokenId: number
): { X: Matrix[]; yFullToken: Matrix[] } {
  const X: Matrix[] = [];
  const yFullToken: Matrix[] = [];

  for (let i = 0; i < sampleCount; i++) {
    const tokens = createRightPaddedTokenSequence(i, seqLen, vocabSize, padTokenId);
    X.push(createSequenceMatrix(tokens));
    yFullToken.push(createRightPaddedFullTokenTarget(tokens, padTokenId));
  }

  return { X, yFullToken };
}

function resetModelLossAccumulators(model: Transformers): void {
  for (const layer of model.layers) {
    if (typeof (layer as any).resetLoss === "function") {
      (layer as any).resetLoss();
    }
  }
}

function createTransformerModel(config: Required<TransformerBenchmarkConfig>): Transformers {
  return new Transformers({
    units: config.units,
    seqLen: config.seqLen,
    vocabSize: config.vocabSize,
    heads: config.heads,
    numBlocks: config.numBlocks,
    dropoutRate: config.dropoutRate,
    alpha: config.alpha,
    padTokenId: config.padTokenId,
  });
}

function buildBatch(samples: Matrix[], start: number, batchSize: number): Matrix {
  const end = Math.min(start + batchSize, samples.length);
  const currentBatchSize = end - start;
  const rows = samples[0]._shape[0];
  const batch = mj.zeros([rows, currentBatchSize]);

  for (let j = 0; j < currentBatchSize; j++) {
    batch.setCol(j, samples[start + j]._data);
  }

  return batch;
}

function benchmarkFullTokenTraining(
  config: Required<TransformerBenchmarkConfig>,
  X: Matrix[],
  yFullToken: Matrix[],
  options: {
    mode: FullTokenBenchmarkResult["mode"];
    trimPadding: boolean;
    paddingSide?: "left" | "right";
  }
): FullTokenBenchmarkResult {
  const model = createTransformerModel(config);
  const epochMs: number[] = [];
  const epochLoss: number[] = [];
  let epochStart = performance.now();
  const totalStart = performance.now();

  model.fit(X, yFullToken, config.epochs, {
    batchSize: config.batchSize,
    shuffle: false,
    verbose: false,
    trimPadding: options.trimPadding,
    paddingSide: options.paddingSide ?? "right",
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
    mode: options.mode,
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

function benchmarkNextTokenPath(
  config: Required<TransformerBenchmarkConfig>,
  X: Matrix[],
  yNextToken: Matrix[]
): NextTokenBenchmarkResult {
  const model = createTransformerModel(config);
  const epochMs: number[] = [];
  const epochLoss: number[] = [];
  const totalStart = performance.now();
  model.train();

  for (let epoch = 0; epoch < config.epochs; epoch++) {
    const epochStart = performance.now();
    resetModelLossAccumulators(model);

    for (let start = 0; start < X.length; start += config.batchSize) {
      const batchX = buildBatch(X, start, config.batchSize);
      const batchY = buildBatch(yNextToken, start, config.batchSize);
      const logits = model.forwardNextToken(batchX);

      if (logits._shape[1] !== batchY._shape[1]) {
        throw new Error(
          `Transformer next-token benchmark: shape mismatch logits batch=${logits._shape[1]} target batch=${batchY._shape[1]}`
        );
      }

      model.backward(batchY);
    }

    epochMs.push(performance.now() - epochStart);
    epochLoss.push(model.loss);
  }

  const totalMs = performance.now() - totalStart;
  const totalSamples = config.sampleCount * config.epochs;

  return {
    mode: "next-token",
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

export function runTransformerModeBenchmark(
  overrideConfig: TransformerBenchmarkConfig = {}
): TransformerBenchmarkResult[] {
  const config: Required<TransformerBenchmarkConfig> = {
    ...DEFAULT_CONFIG,
    ...overrideConfig,
  };

  const { X, yFullToken, yNextToken } = createSyntheticTransformerDataset(
    config.sampleCount,
    config.seqLen,
    config.vocabSize,
    config.padTokenId
  );
  const paddedDataset = createPaddedTransformerDataset(
    config.sampleCount,
    config.seqLen,
    config.vocabSize,
    config.padTokenId
  );

  const results: TransformerBenchmarkResult[] = [
    benchmarkFullTokenTraining(config, X, yFullToken, {
      mode: "full-token",
      trimPadding: false,
    }),
    benchmarkFullTokenTraining(config, paddedDataset.X, paddedDataset.yFullToken, {
      mode: "full-token-padded-no-trim",
      trimPadding: false,
    }),
    benchmarkFullTokenTraining(config, paddedDataset.X, paddedDataset.yFullToken, {
      mode: "full-token-trimPad",
      trimPadding: true,
      paddingSide: "right",
    }),
    benchmarkNextTokenPath(config, X, yNextToken),
  ];

  console.log(`=== Transformer Benchmark (${config.epochs} Epoch) ===`);
  console.log(
    `config: samples=${config.sampleCount}, seqLen=${config.seqLen}, vocabSize=${config.vocabSize}, units=${config.units}, heads=${config.heads}, blocks=${config.numBlocks}, batchSize=${config.batchSize}`
  );
  console.table(
    results.map((result) => ({
      mode: result.mode,
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
      `${result.mode} epochMs=${result.epochMs.map((ms) => ms.toFixed(2)).join(", ")} | epochLoss=${result.epochLoss
        .map((loss) => loss.toFixed(6))
        .join(", ")}`
    );
  }

  return results;
}

