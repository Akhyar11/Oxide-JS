import { mj } from "@oxide-js/core";
import { Matrix } from "@oxide-js/core";
import { Transformers } from "@oxide-js/models";

type TransformerLearningMode = "full-token" | "full-token-trimPad" | "next-token";

type TransformerLearningResult = {
  mode: TransformerLearningMode;
  initialLoss: number;
  finalLoss: number;
  history: number[];
};

function assert(condition: boolean, message: string): void {
  if (!condition) {
    throw new Error(message);
  }
}

function buildSequenceMatrix(tokens: number[]): Matrix {
  return mj.matrix(tokens.map((token) => [token]));
}

function createFullTokenTarget(tokens: number[], padTokenId: number): Matrix {
  const shifted: number[][] = [];
  for (let i = 0; i < tokens.length - 1; i++) {
    shifted.push([tokens[i + 1]]);
  }
  shifted.push([padTokenId]);
  return mj.matrix(shifted);
}

function createTransformerModel(): Transformers {
  return new Transformers({
    units: 24,
    seqLen: 6,
    vocabSize: 16,
    heads: 4,
    numBlocks: 1,
    dropoutRate: 0,
    alpha: 0.01,
    padTokenId: 0,
  });
}

function createRightPaddedTokenSequence(sampleIndex: number, seqLen: number, padTokenId: number): number[] {
  const minEffectiveLen = Math.min(seqLen, 4);
  const variableSpan = Math.max(1, seqLen - minEffectiveLen + 1);
  const effectiveLen = Math.min(seqLen, minEffectiveLen + (sampleIndex % variableSpan));
  const tokens = new Array<number>(seqLen).fill(padTokenId);

  for (let pos = 0; pos < effectiveLen; pos++) {
    tokens[pos] = ((sampleIndex * 3 + pos) % 9) + 1;
  }

  return tokens;
}

function createRightPaddedFullTokenTarget(tokens: number[], padTokenId: number): Matrix {
  const shifted: number[][] = [];
  for (let i = 0; i < tokens.length - 1; i++) {
    const current = tokens[i];
    const next = tokens[i + 1];
    shifted.push([current === padTokenId || next === padTokenId ? padTokenId : next]);
  }
  shifted.push([padTokenId]);
  return mj.matrix(shifted);
}

function createFullTokenDataset(): { X: Matrix[]; y: Matrix[] } {
  const basePatterns = [
    [1, 2, 3, 4, 5, 6],
    [2, 3, 4, 5, 6, 7],
    [3, 4, 5, 6, 7, 8],
    [4, 5, 6, 7, 8, 9],
  ];

  const X: Matrix[] = [];
  const y: Matrix[] = [];

  for (let repeat = 0; repeat < 8; repeat++) {
    for (const tokens of basePatterns) {
      X.push(buildSequenceMatrix(tokens));
      y.push(createFullTokenTarget(tokens, 0));
    }
  }

  return { X, y };
}

function createNextTokenDataset(): { X: Matrix[]; y: Matrix[] } {
  const basePatterns = [
    [1, 2, 3, 4, 5, 6],
    [2, 3, 4, 5, 6, 7],
    [3, 4, 5, 6, 7, 8],
    [4, 5, 6, 7, 8, 9],
  ];

  const X: Matrix[] = [];
  const y: Matrix[] = [];

  for (let repeat = 0; repeat < 8; repeat++) {
    for (const tokens of basePatterns) {
      const targetToken = (tokens[tokens.length - 1] % 9) + 1;
      X.push(buildSequenceMatrix(tokens));
      y.push(mj.matrix([[targetToken]]));
    }
  }

  return { X, y };
}

function createTrimPadDataset(): { X: Matrix[]; y: Matrix[] } {
  const X: Matrix[] = [];
  const y: Matrix[] = [];

  for (let repeat = 0; repeat < 8; repeat++) {
    for (let sampleIndex = 0; sampleIndex < 4; sampleIndex++) {
      const tokens = createRightPaddedTokenSequence(repeat * 4 + sampleIndex, 6, 0);
      X.push(buildSequenceMatrix(tokens));
      y.push(createRightPaddedFullTokenTarget(tokens, 0));
    }
  }

  return { X, y };
}

function resetModelLossAccumulators(model: Transformers): void {
  for (const layer of model.layers) {
    if (typeof (layer as any).resetLoss === "function") {
      (layer as any).resetLoss();
    }
  }
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

function runFullTokenLearningTest(): TransformerLearningResult {
  const { X, y } = createFullTokenDataset();
  const model = createTransformerModel();
  const history: number[] = [];

  model.fit(X, y, 10, {
    batchSize: 4,
    shuffle: false,
    verbose: false,
    trimPadding: false,
    onEpochEnd: (_epoch, loss) => {
      history.push(loss);
    },
  });

  assert(history.length === 10, `transformers full-token: expected 10 loss entries, got ${history.length}`);

  const initialLoss = history[0];
  const finalLoss = history[history.length - 1];
  const bestLoss = Math.min(...history);

  assert(Number.isFinite(initialLoss), "transformers full-token: initial loss must be finite");
  assert(Number.isFinite(finalLoss), "transformers full-token: final loss must be finite");
  assert(
    finalLoss < initialLoss,
    `transformers full-token: expected final loss to be lower than initial loss (${finalLoss} >= ${initialLoss})`
  );
  assert(
    bestLoss <= initialLoss - 0.01,
    `transformers full-token: expected at least one meaningful loss improvement, history=${history.map((v) => v.toFixed(6)).join(", ")}`
  );

  return {
    mode: "full-token",
    initialLoss,
    finalLoss,
    history,
  };
}

function runNextTokenLearningTest(): TransformerLearningResult {
  const { X, y } = createNextTokenDataset();
  const model = createTransformerModel();
  const history: number[] = [];
  const batchSize = 4;

  model.train();

  for (let epoch = 0; epoch < 10; epoch++) {
    resetModelLossAccumulators(model);

    for (let start = 0; start < X.length; start += batchSize) {
      const batchX = buildBatch(X, start, batchSize);
      const batchY = buildBatch(y, start, batchSize);
      const logits = model.forwardNextToken(batchX);

      assert(
        logits._shape[0] === 16 && logits._shape[1] === batchY._shape[1],
        `transformers next-token: unexpected logits shape [${logits._shape[0]}x${logits._shape[1]}]`
      );

      model.backward(batchY);
    }

    history.push(model.loss);
  }

  assert(history.length === 10, `transformers next-token: expected 10 loss entries, got ${history.length}`);

  const initialLoss = history[0];
  const finalLoss = history[history.length - 1];
  const bestLoss = Math.min(...history);

  assert(Number.isFinite(initialLoss), "transformers next-token: initial loss must be finite");
  assert(Number.isFinite(finalLoss), "transformers next-token: final loss must be finite");
  assert(
    finalLoss < initialLoss,
    `transformers next-token: expected final loss to be lower than initial loss (${finalLoss} >= ${initialLoss})`
  );
  assert(
    bestLoss <= initialLoss - 0.01,
    `transformers next-token: expected at least one meaningful loss improvement, history=${history.map((v) => v.toFixed(6)).join(", ")}`
  );

  return {
    mode: "next-token",
    initialLoss,
    finalLoss,
    history,
  };
}

function runTrimPadLearningTest(): TransformerLearningResult {
  const { X, y } = createTrimPadDataset();
  const model = createTransformerModel();
  const history: number[] = [];

  model.fit(X, y, 10, {
    batchSize: 4,
    shuffle: false,
    verbose: false,
    trimPadding: true,
    paddingSide: "right",
    onEpochEnd: (_epoch, loss) => {
      history.push(loss);
    },
  });

  assert(history.length === 10, `transformers full-token-trimPad: expected 10 loss entries, got ${history.length}`);

  const initialLoss = history[0];
  const finalLoss = history[history.length - 1];
  const bestLoss = Math.min(...history);

  assert(Number.isFinite(initialLoss), "transformers full-token-trimPad: initial loss must be finite");
  assert(Number.isFinite(finalLoss), "transformers full-token-trimPad: final loss must be finite");
  assert(
    finalLoss < initialLoss,
    `transformers full-token-trimPad: expected final loss to be lower than initial loss (${finalLoss} >= ${initialLoss})`
  );
  assert(
    bestLoss <= initialLoss - 0.01,
    `transformers full-token-trimPad: expected at least one meaningful loss improvement, history=${history.map((v) => v.toFixed(6)).join(", ")}`
  );

  return {
    mode: "full-token-trimPad",
    initialLoss,
    finalLoss,
    history,
  };
}

export function runTransformerLearningCorrectnessSuite(): TransformerLearningResult[] {
  const results = [runFullTokenLearningTest(), runTrimPadLearningTest(), runNextTokenLearningTest()];

  console.log("=== Transformer Learning Correctness ===");
  console.table(
    results.map((result) => ({
      mode: result.mode,
      initialLoss: Number(result.initialLoss.toFixed(6)),
      finalLoss: Number(result.finalLoss.toFixed(6)),
      improvement: Number((result.initialLoss - result.finalLoss).toFixed(6)),
    }))
  );

  return results;
}
