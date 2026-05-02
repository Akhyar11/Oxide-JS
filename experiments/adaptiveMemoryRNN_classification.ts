import fs from "fs";
import path from "path";
import { AdaptiveMemoryRNN, Dense, Embedding } from "../src/layers";
import mj from "../src/math";
import Matrix from "../src/matrix";
import { Sequential } from "../src/models";
import { BPETokenizer } from "../src/tokenizer";

type RawSample = {
  text: string;
  label: string;
};

type Sample = {
  x: Matrix;
  y: Matrix;
};

const PROJECT_ROOT = path.resolve(__dirname, "..");
const EXAMPLE_ROOT = path.join(PROJECT_ROOT, "example/clasification_family_rnn");
const DATASET_TRAIN_PATH = path.join(EXAMPLE_ROOT, "dataset/train_preprocess.tsv");
const DATASET_VALID_PATH = path.join(EXAMPLE_ROOT, "dataset/valid_preprocess.tsv");
const TOKENIZER_PATH = path.join(EXAMPLE_ROOT, "tokenizer.json");
const LOG_DIR = path.join(PROJECT_ROOT, "experiments/log");

const MAX_SEQ_LEN = Number(process.env.MAX_SEQ_LEN ?? 128);
const EMBEDDING_DIM = Number(process.env.EMBEDDING_DIM ?? 128);
const HIDDEN_UNITS = Number(process.env.HIDDEN_UNITS ?? 32);
const MEMORY_SLOTS = Number(process.env.MEMORY_SLOTS ?? 16);
const MEMORY_DIM = Number(process.env.MEMORY_DIM ?? HIDDEN_UNITS);
const OUTPUT_CLASSES = 3;
const EPOCHS = Number(process.env.EPOCHS ?? 30);
const ALPHA = Number(process.env.ALPHA ?? 0.001);
const TRAIN_LIMIT = Number(process.env.TRAIN_LIMIT ?? 0);
const BATCH_SIZE = Number(process.env.BATCH_SIZE ?? 16);
const VALIDATION_RATIO = 0.2;
const VAL_SAMPLES_PER_EPOCH = Number(process.env.VAL_SAMPLES_PER_EPOCH ?? 500);
const FULL_VAL_EVERY = Number(process.env.FULL_VAL_EVERY ?? 5);
const VERBOSE = process.env.VERBOSE !== "false";

function readDataset(filePath: string): RawSample[] {
  if (!fs.existsSync(filePath)) throw new Error(`Dataset not found at: ${filePath}`);
  const content = fs.readFileSync(filePath, "utf-8");
  const lines = content.split("\n").filter((line) => line.trim() !== "");
  if (lines.length <= 1) return [];

  const header = lines[0] ?? "";
  const separator = header.includes("\t") ? "\t" : ",";
  return lines.slice(1).map((line) => {
    const parts = line.split(separator);
    if (separator === "\t") {
      return { text: (parts[0] ?? "").trim(), label: (parts[1] ?? "").trim() };
    }
    if (parts.length > 2) {
      return { label: (parts[1] ?? "").trim(), text: parts.slice(2).join(",").trim() };
    }
    return { text: (parts[0] ?? "").trim(), label: (parts[1] ?? "").trim() };
  });
}

function labelToIndex(label: string): number | null {
  if (label === "negative") return 0;
  if (label === "positive") return 1;
  if (label === "neutral") return 2;
  return null;
}

function toSample(item: RawSample, tokenizer: BPETokenizer, padId: number): Sample | null {
  const text = String(item.text ?? "").trim();
  const label = String(item.label ?? "").trim();
  const classIndex = labelToIndex(label);
  if (!text || classIndex === null) return null;

  const tokenIds = tokenizer.encode(text);
  if (tokenIds.length === 0) return null;

  const slicedTokens = tokenIds.slice(0, MAX_SEQ_LEN);
  const paddedTokenIds = new Array(MAX_SEQ_LEN - slicedTokens.length).fill(padId).concat(slicedTokens);
  return {
    x: mj.matrix(paddedTokenIds.map((id: number) => [id])),
    y: mj.matrix([[classIndex]]),
  };
}

function shuffleArray<T>(array: T[]): T[] {
  const copied = [...array];
  for (let i = copied.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [copied[i], copied[j]] = [copied[j]!, copied[i]!];
  }
  return copied;
}

function argmax(matrix: Matrix): number {
  let maxIndex = 0;
  let maxValue = matrix._data[0] ?? 0;
  for (let i = 1; i < matrix._data.length; i++) {
    if (matrix._data[i] > maxValue) {
      maxValue = matrix._data[i];
      maxIndex = i;
    }
  }
  return maxIndex;
}

function evaluate(model: Sequential, samples: Sample[]): number {
  if (samples.length === 0) return 0;
  let correct = 0;
  model.eval();
  for (const sample of samples) {
    const pred = argmax(model.forward(sample.x));
    if (pred === sample.y._data[0]) correct++;
  }
  model.train();
  return correct / samples.length;
}

function sampleValidationIndices(total: number, limit: number, epoch: number): number[] {
  if (total <= 0) return [];
  if (limit >= total) return Array.from({ length: total }, (_, i) => i);
  const indices: number[] = [];
  const start = (epoch * limit) % total;
  for (let i = 0; i < limit; i++) indices.push((start + i) % total);
  return indices;
}

function validateModel(model: Sequential, valX: Matrix[], valY: Matrix[], epoch: number, full: boolean): number {
  const total = valX.length;
  if (total === 0) return 0;
  const limit = full ? total : Math.min(VAL_SAMPLES_PER_EPOCH, total);
  const indices = sampleValidationIndices(total, limit, epoch);
  const samples = indices.map((idx) => ({ x: valX[idx]!, y: valY[idx]! }));
  return evaluate(model, samples);
}

type EvaluationResult = {
  accuracy: number;
  macroF1: number;
  weightedF1: number;
  processed: number;
  skipped: number;
  confusion: [number[], number[], number[]];
};

function runEvaluation(model: Sequential, tokenizer: BPETokenizer, rawDataset: RawSample[]): EvaluationResult {
  model.eval();
  const classNames = ["negative", "positive", "neutral"];
  const confusion: [number[], number[], number[]] = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  let processed = 0;
  let skipped = 0;

  for (let i = 0; i < rawDataset.length; i++) {
    const item = rawDataset[i];
    const text = String(item?.text ?? "").trim();
    const labelStr = String(item?.label ?? "").trim();
    const actualIndex = labelToIndex(labelStr);

    if (!text || actualIndex === null) {
      skipped++;
      continue;
    }

    const tokenIds = tokenizer.encode(text);
    if (tokenIds.length === 0) {
      skipped++;
      continue;
    }

    const x = mj.matrix(tokenIds.map((id: number) => [id]));
    const predIndex = argmax(model.predict(x));
    const row = confusion[actualIndex];
    if (row && row[predIndex] !== undefined) row[predIndex]++;

    processed++;
    if (processed % 500 === 0) process.stdout.write(`Processed ${processed}/${rawDataset.length} samples...\n`);
  }

  console.log("\n--- Evaluation Results ---");
  console.log(`Processed: ${processed}, Skipped: ${skipped}`);
  console.log("\nConfusion Matrix (Actual \\ Predicted):");
  console.log("          Neg   Pos   Neu");
  for (let i = 0; i < 3; i++) {
    const row = confusion[i]!;
    console.log(
      `${classNames[i]!.padEnd(8)}: ${String(row[0]).padStart(5)} ${String(row[1]).padStart(5)} ${String(row[2]).padStart(5)}`
    );
  }

  let totalF1 = 0;
  let totalPrecision = 0;
  let totalRecall = 0;
  let weightedF1 = 0;

  console.log("\nPer-class Metrics:");
  for (let i = 0; i < 3; i++) {
    const tp = confusion[i]![i]!;
    const fp = confusion[0]![i]! + confusion[1]![i]! + confusion[2]![i]! - tp;
    const fn = confusion[i]![0]! + confusion[i]![1]! + confusion[i]![2]! - tp;
    const precision = tp / (tp + fp) || 0;
    const recall = tp / (tp + fn) || 0;
    const f1 = (2 * precision * recall) / (precision + recall) || 0;

    totalPrecision += precision;
    totalRecall += recall;
    totalF1 += f1;
    const classCount = confusion[i]![0]! + confusion[i]![1]! + confusion[i]![2]!;
    weightedF1 += f1 * (classCount / processed);

    console.log(
      `${classNames[i]!.padEnd(8)} -> Precision: ${(precision * 100).toFixed(2)}%, Recall: ${(recall * 100).toFixed(2)}%, F1: ${(f1 * 100).toFixed(2)}%`
    );
  }

  const macroPrecision = totalPrecision / 3;
  const macroRecall = totalRecall / 3;
  const macroF1 = totalF1 / 3;
  const totalCorrect = confusion[0]![0]! + confusion[1]![1]! + confusion[2]![2]!;
  const accuracy = totalCorrect / processed;

  console.log("--------------------------");
  console.log(`Accuracy:         ${(accuracy * 100).toFixed(2)}%`);
  console.log(`Macro Precision:  ${(macroPrecision * 100).toFixed(2)}%`);
  console.log(`Macro Recall:     ${(macroRecall * 100).toFixed(2)}%`);
  console.log(`Macro F1 Score:   ${(macroF1 * 100).toFixed(2)}%`);
  console.log(`Weighted F1 Score:${(weightedF1 * 100).toFixed(2)}%`);
  console.log("--------------------------");

  return { accuracy, macroF1, weightedF1, processed, skipped, confusion };
}

function logMemory(prefix: string): void {
  const mem = process.memoryUsage();
  console.log(
    `${prefix} | rss=${(mem.rss / 1024 / 1024).toFixed(1)}MB heapUsed=${(mem.heapUsed / 1024 / 1024).toFixed(1)}MB external=${(mem.external / 1024 / 1024).toFixed(1)}MB`
  );
}

function main(): void {
  fs.mkdirSync(LOG_DIR, { recursive: true });
  const tokenizer = BPETokenizer.load(TOKENIZER_PATH);
  const vocabSize = tokenizer.getVocabSize();
  const padId = tokenizer.getPadId();

  const rawTrain = readDataset(DATASET_TRAIN_PATH);
  const rawValid = readDataset(DATASET_VALID_PATH);
  const trainRawLimited = TRAIN_LIMIT > 0 ? rawTrain.slice(0, TRAIN_LIMIT) : rawTrain;

  const samples = trainRawLimited
    .map((item) => toSample(item, tokenizer, padId))
    .filter((item): item is Sample => item !== null);

  const shuffled = shuffleArray(samples);
  const validationSize = Math.floor(shuffled.length * VALIDATION_RATIO);
  const validationSamples = shuffled.slice(0, validationSize);
  const trainSamples = shuffled.slice(validationSize);

  const trainX = trainSamples.map((sample) => sample.x);
  const trainY = trainSamples.map((sample) => sample.y);
  const valX = validationSamples.map((sample) => sample.x);
  const valY = validationSamples.map((sample) => sample.y);

  console.log("--- AdaptiveMemoryRNN Experimental Training ---");
  console.log(`Dataset source: ${DATASET_TRAIN_PATH}`);
  console.log(`F1 test source: ${DATASET_VALID_PATH}`);
  console.log(`Raw train size=${rawTrain.length}, raw valid size=${rawValid.length}`);
  console.log(`Processed train TSV samples=${samples.length}`);
  console.log(`Train split=${trainX.length}, inner-validation split=${valX.length} (${Math.round((1 - VALIDATION_RATIO) * 100)}:${Math.round(VALIDATION_RATIO * 100)})`);
  if (TRAIN_LIMIT > 0) console.log(`TRAIN_LIMIT is active: ${TRAIN_LIMIT}`);
  console.log(`VOCAB_SIZE=${vocabSize}, PAD_ID=${padId}`);
  console.log(
    `Config: seqLen=${MAX_SEQ_LEN}, embeddingDim=${EMBEDDING_DIM}, hiddenUnits=${HIDDEN_UNITS}, memorySlots=${MEMORY_SLOTS}, memoryDim=${MEMORY_DIM}, epochs=${EPOCHS}, batchSize=${BATCH_SIZE}`
  );
  logMemory("After dataset preparation");

  const model = new Sequential({
    layers: [
      new Embedding({
        vocabSize,
        embeddingDim: EMBEDDING_DIM,
        padTokenId: padId,
        alpha: ALPHA,
      }),
      new AdaptiveMemoryRNN({
        units: EMBEDDING_DIM,
        hiddenUnits: HIDDEN_UNITS,
        memorySlots: MEMORY_SLOTS,
        memoryDim: MEMORY_DIM,
        activation: "tanh",
        returnSequences: false,
        stateful: false,
        alpha: ALPHA,
      }),
      new Dense({
        units: HIDDEN_UNITS,
        outputUnits: OUTPUT_CLASSES,
        activation: "linear",
        status: "output",
        loss: "softmaxCrossEntropy",
        alpha: ALPHA,
      }),
    ],
  });

  model.compile({
    alpha: ALPHA,
    error: "softmaxCrossEntropy",
    optimizer: "adam",
    clipGradient: false,
  });

  let totalParams = 0;
  for (const layer of model.layers) totalParams += (layer as any).params ?? 0;
  console.log(`Total parameters: ${totalParams}`);
  model.summary();

  const lossHistory: number[] = [];
  const valHistory: number[] = [];
  let bestValAcc = 0;

  model.fit(trainX, trainY, EPOCHS, {
    batchSize: BATCH_SIZE,
    shuffle: true,
    verbose: VERBOSE,
    onEpochEnd: (epoch, loss) => {
      const isLastEpoch = epoch === EPOCHS - 1;
      const shouldFullValidate = isLastEpoch || (epoch + 1) % FULL_VAL_EVERY === 0;
      const valAcc = validateModel(model, valX, valY, epoch, shouldFullValidate);
      if (valAcc > bestValAcc) bestValAcc = valAcc;
      lossHistory.push(loss);
      valHistory.push(valAcc);
      const valMode = shouldFullValidate ? `full=${valX.length}` : `sample=${Math.min(VAL_SAMPLES_PER_EPOCH, valX.length)}`;
      console.log(
        `[AdaptiveMemoryRNN] Epoch ${epoch + 1}/${EPOCHS} | Loss: ${loss.toFixed(4)} | Val Acc (${valMode}): ${(valAcc * 100).toFixed(2)}% | Best: ${(bestValAcc * 100).toFixed(2)}%`
      );
      logMemory(`After epoch ${epoch + 1}`);
    },
  });

  console.log("Evaluating F1 Score for AdaptiveMemoryRNN...");
  const evalResult = runEvaluation(model, tokenizer, rawValid);

  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const modelPath = path.join(LOG_DIR, `adaptive_memory_rnn_${timestamp}.json`);
  const metadataPath = path.join(LOG_DIR, `adaptive_memory_rnn_${timestamp}_metadata.json`);
  model.save(modelPath);
  fs.writeFileSync(
    metadataPath,
    JSON.stringify(
      {
        timestamp,
        dataset: {
          train: DATASET_TRAIN_PATH,
          valid: DATASET_VALID_PATH,
          rawTrainSize: rawTrain.length,
          rawValidSize: rawValid.length,
          processedTrainTsvSize: samples.length,
          usedTrainSize: trainX.length,
          innerValidationSize: valX.length,
          f1TestSize: rawValid.length,
        },
        config: {
          maxSeqLen: MAX_SEQ_LEN,
          embeddingDim: EMBEDDING_DIM,
          hiddenUnits: HIDDEN_UNITS,
          memorySlots: MEMORY_SLOTS,
          memoryDim: MEMORY_DIM,
          epochs: EPOCHS,
          alpha: ALPHA,
          batchSize: BATCH_SIZE,
          validationRatio: VALIDATION_RATIO,
          valSamplesPerEpoch: VAL_SAMPLES_PER_EPOCH,
          fullValEvery: FULL_VAL_EVERY,
          trainLimit: TRAIN_LIMIT,
        },
        result: {
          finalLoss: lossHistory[lossHistory.length - 1] ?? null,
          finalValAcc: valHistory[valHistory.length - 1] ?? null,
          bestValAcc,
          f1: evalResult.weightedF1,
          accuracy: evalResult.accuracy,
          macroF1: evalResult.macroF1,
          evaluation: evalResult,
          lossHistory,
          valHistory,
          totalParams,
          modelPath,
        },
      },
      null,
      2
    )
  );

  model.dispose();
  console.log(`Saved model: ${modelPath}`);
  console.log(`Saved metadata: ${metadataPath}`);
}

main();
