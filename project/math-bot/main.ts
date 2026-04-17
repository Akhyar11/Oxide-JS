import * as fs from "fs";
import * as path from "path";
import { Transformers } from "../../src/models";
import { BPETokenizer } from "../../src/tokenizer";
import mj from "../../src/math";
import Matrix from "../../src/matrix";
import { loadMathTrainingCorpus } from "./data";

const CONTEXT_LEN = 64;
const EMBEDDING_DIM = 64;
const HEADS = 8;
const LEARNING_RATE = 1e-3;
const EPOCHS = 500;
const VOCAB_SIZE = 1000;
const BATCH_SIZE = 32;

interface TrainPair {
  xData: Float32Array;
  target: number;
}

interface ModelConfig {
  units: number;
  seqLen: number;
  heads: number;
  padTokenId: number;
  vocabSize: number;
}

export function shouldSaveBestCheckpoint(currentLoss: number, bestLoss: number): boolean {
  return currentLoss < bestLoss;
}

function shuffleInPlace<T>(items: T[]): void {
  for (let i = items.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [items[i], items[j]] = [items[j], items[i]];
  }
}

function readModelConfig(modelPath: string): ModelConfig {
  const layers = JSON.parse(fs.readFileSync(modelPath, "utf-8"));
  const embedding = layers.find((layer: any) => layer.name === "embedding layer");
  const pe = layers.find((layer: any) => layer.name === "positional encoding");
  const mha = layers.find((layer: any) => layer.name === "multi head attention layer");

  if (!embedding) {
    throw new Error(`Cannot infer model config from ${modelPath}: embedding layer not found`);
  }

  return {
    units: embedding.embeddingDim ?? EMBEDDING_DIM,
    seqLen: pe?.maxSeqLen ?? mha?.seqLen ?? CONTEXT_LEN,
    heads: mha?.heads ?? HEADS,
    padTokenId: embedding.padTokenId ?? 0,
    vocabSize: embedding.vocabSize ?? 0,
  };
}

export function loadOrCreateMathTokenizer(corpus: string[], vocabPath: string): BPETokenizer {
  let tokenizer: BPETokenizer;

  if (fs.existsSync(vocabPath)) {
    console.log("\n=== Loading Existing Tokenizer ===\n");
    tokenizer = BPETokenizer.load(vocabPath);
    console.log(`[BPE] Vocab saat ini: ${tokenizer.getVocabSize()}. Mencoba update corpus...`);
    tokenizer.update(corpus, VOCAB_SIZE);
  } else {
    console.log("\n=== Training New Tokenizer ===\n");
    tokenizer = new BPETokenizer({
      vocabSize: VOCAB_SIZE,
      minFrequency: 1,
      specialTokens: ["<SEP>", ...Array.from({ length: 50 }, (_, i) => `<RESERVED_${i}>`)],
    });
    tokenizer.train(corpus);
  }

  tokenizer.save(vocabPath);
  return tokenizer;
}

async function main() {
  const datasetPath = path.join(__dirname, "..", "..", "dataset", "dataset_matematika_1000.json");
  const outputDir = path.join(__dirname, "dataset");
  const modelPath = path.join(outputDir, "math_model.json");
  const vocabPath = path.join(outputDir, "math_vocab.json");

  console.log("=== Math Bot Training ===\n");
  console.log(`Dataset: ${datasetPath}`);
  console.log(`Output : ${outputDir}\n`);

  const corpus = loadMathTrainingCorpus(datasetPath);
  console.log(`Loaded ${corpus.length} normalized training records.`);

  fs.mkdirSync(outputDir, { recursive: true });
  const hasExistingModel = fs.existsSync(modelPath);
  const tokenizer = loadOrCreateMathTokenizer(corpus, vocabPath);

  const padId = tokenizer.getPadId();
  const finalVocabSize = tokenizer.getVocabSize();

  console.log("\n=== Preparing Training Pairs ===\n");
  const trainPairs: TrainPair[] = [];
  for (const text of corpus) {
    const tokens = tokenizer.encode(text);
    for (let i = 0; i < tokens.length - 1; i++) {
      const start = Math.max(0, i - CONTEXT_LEN + 1);
      const contextWindow = new Float32Array(CONTEXT_LEN);
      contextWindow.fill(padId);

      const ctxLen = i - start + 1;
      const offset = CONTEXT_LEN - ctxLen;
      for (let j = 0; j < ctxLen; j++) {
        contextWindow[offset + j] = tokens[start + j];
      }

      trainPairs.push({
        xData: contextWindow,
        target: tokens[i + 1],
      });
    }
  }

  if (trainPairs.length === 0) {
    throw new Error("Training pairs kosong setelah tokenisasi.");
  }

  console.log(`Training pairs: ${trainPairs.length}`);

  const runtimeConfig = hasExistingModel
    ? readModelConfig(modelPath)
    : {
      units: EMBEDDING_DIM,
      seqLen: CONTEXT_LEN,
      heads: HEADS,
      padTokenId: padId,
      vocabSize: finalVocabSize,
    };

  console.log(`\n=== ${hasExistingModel ? "Loading Existing Model" : "Building Model From Scratch"} ===\n`);
  const model = new Transformers({
    units: runtimeConfig.units,
    seqLen: runtimeConfig.seqLen,
    vocabSize: finalVocabSize,
    heads: runtimeConfig.heads,
    alpha: LEARNING_RATE,
    padTokenId: padId,
  });

  if (hasExistingModel) {
    model.load(modelPath);
    if (finalVocabSize > runtimeConfig.vocabSize) {
      console.log(`Expanding model vocabulary: ${runtimeConfig.vocabSize} -> ${finalVocabSize}`);
      model.resizeVocab(finalVocabSize);
    }
  }

  model.compile({ alpha: LEARNING_RATE, optimizer: "adam", error: "softmaxCrossEntropy" });

  function saveArtifacts(epoch?: number): void {
    model.save(modelPath);
    tokenizer.save(vocabPath);

    if (epoch !== undefined) {
      console.log(`Checkpoint terbaik disimpan pada epoch ${epoch}.`);
    } else {
      console.log("Model terakhir disimpan.");
    }
    console.log(`Model: ${modelPath}`);
    console.log(`Vocab: ${vocabPath}`);
  }

  const fullBatchX = mj.zeros([runtimeConfig.seqLen, BATCH_SIZE]);
  const fullBatchY = mj.zeros([1, BATCH_SIZE]);
  const tailBatchBuffers = new Map<number, { x: Matrix; y: Matrix }>();
  let bestLoss = Number.POSITIVE_INFINITY;

  console.log(`\n=== Training ${EPOCHS} Epochs ===\n`);
  for (let ep = 0; ep < EPOCHS; ep++) {
    for (const layer of model.layers) {
      if (typeof (layer as { resetLoss?: () => void }).resetLoss === "function") {
        (layer as { resetLoss: () => void }).resetLoss();
      }
      if (layer.name === "dropout layer") {
        (layer as { status?: string }).status = "train";
      }
    }

    shuffleInPlace(trainPairs);

    for (let i = 0; i < trainPairs.length; i += BATCH_SIZE) {
      const actualBatchSize = Math.min(BATCH_SIZE, trainPairs.length - i);
      let currentBatchX = fullBatchX;
      let currentBatchY = fullBatchY;

      if (actualBatchSize !== BATCH_SIZE) {
        let tailBuffers = tailBatchBuffers.get(actualBatchSize);
        if (!tailBuffers) {
          tailBuffers = {
            x: mj.zeros([runtimeConfig.seqLen, actualBatchSize]),
            y: mj.zeros([1, actualBatchSize]),
          };
          tailBatchBuffers.set(actualBatchSize, tailBuffers);
        }
        currentBatchX = tailBuffers.x;
        currentBatchY = tailBuffers.y;
      }

      currentBatchX._data.fill(padId);
      currentBatchY._data.fill(0);

      for (let batchIndex = 0; batchIndex < actualBatchSize; batchIndex++) {
        const pair = trainPairs[i + batchIndex];
        let dst = batchIndex;
        for (let row = 0; row < runtimeConfig.seqLen; row++, dst += actualBatchSize) {
          currentBatchX._data[dst] = pair.xData[row];
        }
        currentBatchY._data[batchIndex] = pair.target;
      }

      model.forward(currentBatchX);
      model.backward(currentBatchY);
    }

    console.log(`Epoch ${ep + 1}/${EPOCHS} - Loss: ${model.loss.toFixed(6)}`);

    if (shouldSaveBestCheckpoint(model.loss, bestLoss)) {
      bestLoss = model.loss;
      saveArtifacts(ep + 1);
      console.log(`Loss terbaik baru: ${bestLoss.toFixed(6)}`);
    }
  }

  if (!fs.existsSync(modelPath)) {
    saveArtifacts();
  }

  console.log("\nTraining selesai.");
  console.log(`Model: ${modelPath}`);
  console.log(`Vocab: ${vocabPath}`);
}

if (require.main === module) {
  main().catch((error) => {
    console.error("Math bot training failed:", error);
    process.exit(1);
  });
}
