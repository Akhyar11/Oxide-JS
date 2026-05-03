import fs from "fs";
import path from "path";

import { MemoryBank, Dense, Embedding } from "../src/layers";
import mj from "../src/math";
import Matrix from "../src/matrix";
import { Sequential } from "../src/models";

import {
  loadBpeMemoryEpisodes,
  trainMemoryBpeTokenizer,
  getQueryForTurn,
  BpeMemoryEpisode,
  BpeMemoryTurn,
  BpeMemoryQuery,
} from "./memorybank_bpe_dataset/bpe_memory_dataset_loader";

const PROJECT_ROOT = path.resolve(__dirname, "..");
const DATASET_ROOT = path.join(PROJECT_ROOT, "experiments/memorybank_bpe_dataset");

const TRAIN_PATH = path.join(DATASET_ROOT, "train.jsonl");
const VAL_PATH = path.join(DATASET_ROOT, "val.jsonl");
const TEST_PATH = path.join(DATASET_ROOT, "test.jsonl");
const SMOKE_PATH = path.join(DATASET_ROOT, "smoke.jsonl");
const CORPUS_PATH = path.join(DATASET_ROOT, "bpe_corpus.txt");

const LOG_DIR = path.join(PROJECT_ROOT, "experiments/log");

// ------------------------------
// Hyperparameters
// ------------------------------

const USE_SMOKE = process.argv.includes("--smoke");
const USE_FULL = process.argv.includes("--full");
const USE_DIAGNOSTIC = process.argv.includes("--diagnostic");
const USE_AUX_STORE_LOSS = !process.argv.includes("--no-aux");
const USE_WRITE_PROBE = !process.argv.includes("--no-probe");

const BPE_TARGET_VOCAB_SIZE = Number(process.env.VOCAB_SIZE ?? 256);
const MAX_TURN_TOKENS = Number(process.env.MAX_TURN_TOKENS ?? 16);

const EMBEDDING_DIM = Number(process.env.EMBEDDING_DIM ?? 64);
const MEMORY_SLOTS = Number(process.env.MEMORY_SLOTS ?? 20);
const MEMORY_DIM = Number(process.env.MEMORY_DIM ?? 64);
const OUTPUT_CLASSES = 24;

const EPOCHS = Number(process.env.EPOCHS ?? 30);
const ALPHA = Number(process.env.ALPHA ?? 0.001);

const TRAIN_LIMIT = Number(process.env.TRAIN_LIMIT ?? 0); // 0 means all
const VAL_LIMIT = Number(process.env.VAL_LIMIT ?? 0);
const TEST_LIMIT = Number(process.env.TEST_LIMIT ?? 0);

const PRINT_EVERY = Number(process.env.PRINT_EVERY ?? 100);

// For early debugging, make MemoryBank write on every STORE/UPDATE.
// Query/noop writes are manually frozen.
const MEMORY_WRITE_THRESHOLD = Number(process.env.MEMORY_WRITE_THRESHOLD ?? 0.0);

// How many train episodes to use for per-epoch active/frozen eval.
// smoke: all; full: 512 capped.
const TRAIN_EVAL_N = USE_SMOKE ? 9999 : Number(process.env.TRAIN_EVAL_N ?? 512);
// How many episodes to audit per call.
const AUDIT_N = USE_SMOKE ? 9999 : Number(process.env.AUDIT_N ?? 200);
const AUDIT_PRINT = Number(process.env.AUDIT_PRINT ?? 3);

type EncodedTurn = {
  x: Matrix;
  validLength: number;
  tokenIds: number[];
};

type EvalResult = {
  accuracy: number;
  totalQueries: number;
  correct: number;
  updateAccuracy: number;
  updateQueries: number;
  updateCorrect: number;
  noUpdateAccuracy: number;
  noUpdateQueries: number;
  noUpdateCorrect: number;
  avgMemoryFilled: number;
};

type TrainEpochResult = {
  avgLoss: number;
  queryAccuracy: number;
  auxAccuracy: number;
  totalQueries: number;
  correctQueries: number;
  totalAux: number;
  correctAux: number;
  avgMemoryFilled: number;
  writeProbeAccuracy: number;
  totalProbe: number;
  correctProbe: number;
};

// ------------------------------
// PART 2 — Memory Audit types
// ------------------------------

type SlotFact = {
  keyText: string;
  valueText: string;
  valueClass: number;
};

type MemoryAuditStats = {
  episodes: number;
  writes: number;
  writeMiss: number;
  queries: number;
  topSlotCorrect: number;
  topValueCorrect: number;
  predCorrect: number;
  unexpectedNoopWrites: number;
  unexpectedQueryWrites: number;
  avgTopAttention: number;
};

// ------------------------------
// Local pooling layer
// ------------------------------
//
// This layer converts token-level embedding output:
//   [embeddingDim, maxTurnTokens]
// into one turn-level vector:
//   [embeddingDim, 1]
//
// It distributes gradient back equally to valid non-PAD token positions.

class TurnMaskedMeanPooling {
  name = "turn masked mean pooling layer";
  status = "train" as const;
  params = 0;

  inputShape: [number, number];
  outputShape: [number, number];

  private units: number;
  private maxTokens: number;
  private validLength = 1;

  constructor({ units, maxTokens }: { units: number; maxTokens: number }) {
    this.units = units;
    this.maxTokens = maxTokens;
    this.inputShape = [units, maxTokens];
    this.outputShape = [units, 1];
  }

  setValidLength(validLength: number): void {
    if (!Number.isInteger(validLength) || validLength < 1) {
      throw new Error(`TurnMaskedMeanPooling.setValidLength: invalid validLength=${validLength}`);
    }
    this.validLength = Math.min(validLength, this.maxTokens);
  }

  forward(x: Matrix): Matrix {
    if (x._shape[0] !== this.units || x._shape[1] !== this.maxTokens) {
      throw new Error(
        `TurnMaskedMeanPooling.forward: expected [${this.units}, ${this.maxTokens}], got [${x._shape[0]}, ${x._shape[1]}]`
      );
    }

    const out = mj.zeros([this.units, 1]);
    const denom = Math.max(1, this.validLength);

    for (let d = 0; d < this.units; d++) {
      let sum = 0;
      const rowOffset = d * this.maxTokens;
      for (let t = 0; t < denom; t++) {
        sum += x._data[rowOffset + t];
      }
      out._data[d] = sum / denom;
    }

    return out;
  }

  backward(_y: Matrix, err: Matrix): Matrix {
    if (err._shape[0] !== this.units || err._shape[1] !== 1) {
      throw new Error(
        `TurnMaskedMeanPooling.backward: expected err [${this.units}, 1], got [${err._shape[0]}, ${err._shape[1]}]`
      );
    }

    const dx = mj.zeros([this.units, this.maxTokens]);
    const denom = Math.max(1, this.validLength);

    for (let d = 0; d < this.units; d++) {
      const grad = err._data[d] / denom;
      const rowOffset = d * this.maxTokens;
      for (let t = 0; t < denom; t++) {
        dx._data[rowOffset + t] = grad;
      }
    }

    return dx;
  }

  compile(): void {
    // no-op
  }

  save(): Record<string, unknown> {
    return {
      name: this.name,
      units: this.units,
      maxTokens: this.maxTokens,
      note: "Experiment-local layer. Register this layer in setLayers before loading full Sequential artifacts.",
    };
  }
}

// ------------------------------
// Utility
// ------------------------------

function assertFileExists(filePath: string): void {
  if (!fs.existsSync(filePath)) {
    throw new Error(`File tidak ditemukan: ${filePath}`);
  }
}

function formatPct(v: number): string {
  return `${(v * 100).toFixed(2)}%`;
}

function formatNum(v: number, digits = 4): string {
  if (!Number.isFinite(v)) return "NaN";
  return v.toFixed(digits);
}

function logMemory(prefix: string): void {
  const mem = process.memoryUsage();
  console.log(
    `${prefix} | rss=${(mem.rss / 1024 / 1024).toFixed(1)}MB heap=${(mem.heapUsed / 1024 / 1024).toFixed(1)}MB`
  );
}

function argmax(matrix: Matrix): number {
  let maxIndex = 0;
  let maxValue = matrix._data[0];

  for (let i = 1; i < matrix._data.length; i++) {
    if (matrix._data[i] > maxValue) {
      maxValue = matrix._data[i];
      maxIndex = i;
    }
  }

  return maxIndex;
}

function parseValueClass(valueText?: string): number | null {
  if (!valueText) return null;
  const match = valueText.match(/^value_(\d+)$/);
  if (!match) return null;
  const value = Number(match[1]);
  if (!Number.isInteger(value) || value < 0 || value >= OUTPUT_CLASSES) return null;
  return value;
}

function makeTarget(targetClass: number): Matrix {
  return mj.matrix([[targetClass]]);
}

function takeLimit<T>(arr: T[], limit: number): T[] {
  if (!limit || limit <= 0) return arr;
  return arr.slice(0, Math.min(limit, arr.length));
}

function shuffleInPlace<T>(arr: T[]): void {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

function encodeTurn(tokenizer: any, text: string): EncodedTurn {
  const rawIds = tokenizer.encode(text);
  const validLength = Math.max(1, Math.min(rawIds.length, MAX_TURN_TOKENS));
  const padded = tokenizer.padSequence(rawIds, MAX_TURN_TOKENS);

  return {
    x: Matrix.fromFlat(Float32Array.from(padded), [MAX_TURN_TOKENS, 1]),
    validLength,
    tokenIds: padded,
  };
}

// PART 2 — find MemoryBank layer from Sequential
function getMemoryBankLayer(model: Sequential): MemoryBank {
  const layer = (model.layers as any[]).find((l) => l.name === "memory bank layer");
  if (!layer) throw new Error("MemoryBank layer not found in model.");
  return layer as MemoryBank;
}

function configureMemoryWrites(model: Sequential, op: string, freezeAllWrites = false): void {
  if (freezeAllWrites) {
    model.freezeMemoryWrites();
    return;
  }

  if (op === "STORE" || op === "UPDATE") {
    model.enableMemoryWrites();
  } else {
    // QUERY should read only.
    // NOOP should not pollute memory.
    model.freezeMemoryWrites();
  }
}

function getMemoryFilled(model: Sequential): number {
  let totalFilled = 0;
  let totalSlots = 0;

  for (const layer of model.layers as any[]) {
    if (typeof layer.getMemoryState === "function") {
      const state = layer.getMemoryState();
      totalFilled += state.memoryFilled.reduce((a: number, b: number) => a + b, 0);
      totalSlots += state.memorySlots;
    }
  }

  return totalSlots > 0 ? totalFilled / totalSlots : 0;
}

function forwardTurn(
  model: Sequential,
  pooling: TurnMaskedMeanPooling,
  tokenizer: any,
  turn: BpeMemoryTurn,
  freezeAllWrites = false
): Matrix {
  configureMemoryWrites(model, turn.op, freezeAllWrites);

  const encoded = encodeTurn(tokenizer, turn.text);
  pooling.setValidLength(encoded.validLength);

  return model.forward(encoded.x);
}

// ------------------------------
// Evaluation
// ------------------------------

function evaluateModel(
  model: Sequential,
  pooling: TurnMaskedMeanPooling,
  tokenizer: any,
  episodes: BpeMemoryEpisode[],
  label: string,
  freezeAllWrites = false
): EvalResult {
  let totalQueries = 0;
  let correct = 0;

  let updateQueries = 0;
  let updateCorrect = 0;

  let noUpdateQueries = 0;
  let noUpdateCorrect = 0;

  let memoryFilledSum = 0;

  model.eval();

  const start = Date.now();

  for (let i = 0; i < episodes.length; i++) {
    const episode = episodes[i];

    model.resetMemory();

    for (let t = 0; t < episode.turns.length; t++) {
      const pred = forwardTurn(model, pooling, tokenizer, episode.turns[t], freezeAllWrites);
      const q = getQueryForTurn(episode, t);

      if (!q) continue;

      totalQueries++;

      const predClass = argmax(pred);
      const ok = predClass === q.target_class;

      if (ok) correct++;

      if (episode.has_update) {
        updateQueries++;
        if (ok) updateCorrect++;
      } else {
        noUpdateQueries++;
        if (ok) noUpdateCorrect++;
      }
    }

    memoryFilledSum += getMemoryFilled(model);
  }

  model.train();
  model.enableMemoryWrites();

  const elapsed = (Date.now() - start) / 1000;

  const result: EvalResult = {
    accuracy: totalQueries > 0 ? correct / totalQueries : 0,
    totalQueries,
    correct,
    updateAccuracy: updateQueries > 0 ? updateCorrect / updateQueries : 0,
    updateQueries,
    updateCorrect,
    noUpdateAccuracy: noUpdateQueries > 0 ? noUpdateCorrect / noUpdateQueries : 0,
    noUpdateQueries,
    noUpdateCorrect,
    avgMemoryFilled: episodes.length > 0 ? memoryFilledSum / episodes.length : 0,
  };

  console.log(
    [
      `[eval:${label}]`,
      `acc=${formatPct(result.accuracy)} (${correct}/${totalQueries})`,
      `update=${formatPct(result.updateAccuracy)} (${updateCorrect}/${updateQueries})`,
      `noUpdate=${formatPct(result.noUpdateAccuracy)} (${noUpdateCorrect}/${noUpdateQueries})`,
      `avgMemFilled=${formatPct(result.avgMemoryFilled)}`,
      `time=${elapsed.toFixed(1)}s`,
      freezeAllWrites ? `writes=frozen` : `writes=controlled`,
    ].join(" | ")
  );

  return result;
}

// ------------------------------
// PART 2 — Memory Audit
// ------------------------------

function auditMemoryEpisodes(
  model: Sequential,
  pooling: TurnMaskedMeanPooling,
  tokenizer: any,
  episodes: BpeMemoryEpisode[],
  label: string,
  maxEpisodes = 200,
  printExamples = 3
): MemoryAuditStats {
  const mb = getMemoryBankLayer(model);
  model.eval();

  const stats: MemoryAuditStats = {
    episodes: 0,
    writes: 0,
    writeMiss: 0,
    queries: 0,
    topSlotCorrect: 0,
    topValueCorrect: 0,
    predCorrect: 0,
    unexpectedNoopWrites: 0,
    unexpectedQueryWrites: 0,
    avgTopAttention: 0,
  };

  let printCount = 0;
  let attnSum = 0;

  const maxEp = Math.min(episodes.length, maxEpisodes);

  for (let i = 0; i < maxEp; i++) {
    const episode = episodes[i];
    model.resetMemory();

    const slotFacts = new Map<number, SlotFact>();
    const latestKeyToSlot = new Map<string, number>();

    stats.episodes++;

    for (let t = 0; t < episode.turns.length; t++) {
      const turn = episode.turns[t];
      const encoded = encodeTurn(tokenizer, turn.text);
      pooling.setValidLength(encoded.validLength);

      if (turn.op === "STORE" || turn.op === "UPDATE") {
        model.enableMemoryWrites();
        const pred = model.forward(encoded.x);
        const trace = mb.getDebugTrace();
        const entry = trace[0];

        if (entry?.writeCommitted) {
          const valueClass = parseValueClass(turn.value_text);
          if (turn.key_text && valueClass !== null) {
            slotFacts.set(entry.writeSlot, {
              keyText: turn.key_text,
              valueText: turn.value_text ?? "",
              valueClass,
            });
            latestKeyToSlot.set(turn.key_text, entry.writeSlot);
            stats.writes++;
          }
        } else {
          stats.writeMiss++;
        }
      } else if (turn.op === "QUERY") {
        model.freezeMemoryWrites();
        const pred = model.forward(encoded.x);
        const trace = mb.getDebugTrace();
        const entry = trace[0];

        // Warn if QUERY triggered a write
        if (entry?.writeCommitted) stats.unexpectedQueryWrites++;

        const q = getQueryForTurn(episode, t);
        if (!q) continue;

        stats.queries++;

        const topReadSlot = entry?.readSlots[0]?.slot ?? -1;
        const topAttn = entry?.readSlots[0]?.attn ?? 0;
        attnSum += topAttn;

        const expectedSlot = latestKeyToSlot.get(q.key_text) ?? -1;
        const topFact = topReadSlot >= 0 ? slotFacts.get(topReadSlot) : undefined;
        const predClass = argmax(pred);

        const slotOk = topReadSlot === expectedSlot && expectedSlot >= 0;
        const valueOk = topFact?.valueClass === q.target_class;
        const predOk = predClass === q.target_class;

        if (slotOk) stats.topSlotCorrect++;
        if (valueOk) stats.topValueCorrect++;
        if (predOk) stats.predCorrect++;

        if (printCount < printExamples) {
          console.log(
            `  [audit-example] ep=${i} t=${t}: key="${q.key_text}" ` +
              `expectedSlot=${expectedSlot} topSlot=${topReadSlot} ` +
              `topAttn=${topAttn.toFixed(3)} ` +
              `topFact=${JSON.stringify(topFact)} ` +
              `pred=${predClass} target=${q.target_class} ` +
              `slotOk=${slotOk} valueOk=${valueOk} predOk=${predOk}`
          );
          printCount++;
        }
      } else {
        // NOOP
        model.freezeMemoryWrites();
        const pred = model.forward(encoded.x);
        const trace = mb.getDebugTrace();
        const entry = trace[0];
        if (entry?.writeCommitted) stats.unexpectedNoopWrites++;
      }
    }
  }

  stats.avgTopAttention = stats.queries > 0 ? attnSum / stats.queries : 0;

  const topSlotAcc = stats.queries > 0 ? stats.topSlotCorrect / stats.queries : 0;
  const topValueAcc = stats.queries > 0 ? stats.topValueCorrect / stats.queries : 0;
  const predAcc = stats.queries > 0 ? stats.predCorrect / stats.queries : 0;

  console.log(
    [
      `[memory-audit:${label}]`,
      `queries=${stats.queries}`,
      `writes=${stats.writes}`,
      `writeMiss=${stats.writeMiss}`,
      `topSlotAcc=${formatPct(topSlotAcc)}`,
      `topValueAcc=${formatPct(topValueAcc)}`,
      `predAcc=${formatPct(predAcc)}`,
      `noopWrites=${stats.unexpectedNoopWrites}`,
      `queryWrites=${stats.unexpectedQueryWrites}`,
      `avgTopAttn=${stats.avgTopAttention.toFixed(3)}`,
    ].join(" | ")
  );

  // Interpretive diagnostics
  if (topSlotAcc < 0.3) {
    console.log("  => DIAGNOSIS: topSlotAcc low — query is NOT reading the correct slot. Check similarity/queryKernel.");
  } else if (topValueAcc < 0.3) {
    console.log("  => DIAGNOSIS: topSlotAcc OK but topValueAcc low — value stored in correct slot is WRONG. Check writeValueKernel/probe.");
  } else if (predAcc < 0.1) {
    console.log("  => DIAGNOSIS: topValueAcc OK but predAcc low — output head is NOT using memory values. Check outputKernel.");
  }

  model.train();
  model.enableMemoryWrites();

  return stats;
}

// ------------------------------
// Training
// ------------------------------

function trainOneEpoch(
  model: Sequential,
  pooling: TurnMaskedMeanPooling,
  tokenizer: any,
  episodes: BpeMemoryEpisode[],
  epoch: number,
  writeProbe?: Dense
): TrainEpochResult {
  let totalLoss = 0;
  let lossCount = 0;

  let totalQueries = 0;
  let correctQueries = 0;

  let totalAux = 0;
  let correctAux = 0;

  let totalProbe = 0;
  let correctProbe = 0;

  let memoryFilledSum = 0;

  const indices = Array.from({ length: episodes.length }, (_, i) => i);
  shuffleInPlace(indices);

  const start = Date.now();

  model.train();

  for (let idx = 0; idx < indices.length; idx++) {
    const episode = episodes[indices[idx]];

    model.resetMemory();

    for (let t = 0; t < episode.turns.length; t++) {
      const turn = episode.turns[t];

      const pred = forwardTurn(model, pooling, tokenizer, turn, false);

      const q = getQueryForTurn(episode, t);
      if (q) {
        totalQueries++;

        const predClass = argmax(pred);
        if (predClass === q.target_class) correctQueries++;

        const y = makeTarget(q.target_class);
        model.backward(y);

        totalLoss += model.loss;
        lossCount++;
        continue;
      }

      // Auxiliary objective:
      // STORE / UPDATE turns: train main output on current turn value.
      // NOTE: auxAcc is current-turn value reading accuracy, NOT memory retrieval proof.
      // It does NOT prove that memoryValues encode the target class for future QUERY reads.
      if (USE_AUX_STORE_LOSS && (turn.op === "STORE" || turn.op === "UPDATE")) {
        const target = parseValueClass(turn.value_text);
        if (target !== null) {
          totalAux++;

          const predClass = argmax(pred);
          if (predClass === target) correctAux++;

          const y = makeTarget(target);
          model.backward(y);

          totalLoss += model.loss;
          lossCount++;

          // PART 4A — Write probe: train writeValueKernel to encode target class directly.
          // This bypasses the BPTT gap: query loss does NOT flow back to past STORE writes.
          if (USE_WRITE_PROBE && writeProbe) {
            const mb = getMemoryBankLayer(model);
            const probeLoss = mb.trainLastWriteValue(target, writeProbe);
            if (probeLoss !== null) {
              totalProbe++;
              // Compute probe accuracy by checking probe prediction
              const probeValMat = mb.getLastWriteValueMatrix();
              if (probeValMat) {
                const probePred = writeProbe.forward(probeValMat);
                if (argmax(probePred) === target) correctProbe++;
              }
            }
          }
        }
      }
    }

    memoryFilledSum += getMemoryFilled(model);

    if ((idx + 1) % PRINT_EVERY === 0 || idx + 1 === indices.length) {
      const elapsed = (Date.now() - start) / 1000;
      const speed = (idx + 1) / Math.max(0.001, elapsed);
      const eta = (indices.length - (idx + 1)) / Math.max(0.001, speed);

      const avgLoss = lossCount > 0 ? totalLoss / lossCount : 0;
      const qAcc = totalQueries > 0 ? correctQueries / totalQueries : 0;
      const auxAcc = totalAux > 0 ? correctAux / totalAux : 0;
      const probeAcc = totalProbe > 0 ? correctProbe / totalProbe : 0;
      const memFill = memoryFilledSum / (idx + 1);

      process.stdout.write(
        [
          `\rEpoch ${epoch}`,
          `episodes=${idx + 1}/${indices.length}`,
          `loss=${formatNum(avgLoss)}`,
          `queryAcc=${formatPct(qAcc)} (${correctQueries}/${totalQueries})`,
          `auxAcc(notRetrieval)=${formatPct(auxAcc)}`,
          `writeProbeAcc=${formatPct(probeAcc)} (${correctProbe}/${totalProbe})`,
          `memFilled=${formatPct(memFill)}`,
          `speed=${speed.toFixed(1)} ep/s`,
          `eta=${eta.toFixed(0)}s`,
        ].join(" | ")
      );
    }
  }

  process.stdout.write("\n");

  return {
    avgLoss: lossCount > 0 ? totalLoss / lossCount : 0,
    queryAccuracy: totalQueries > 0 ? correctQueries / totalQueries : 0,
    auxAccuracy: totalAux > 0 ? correctAux / totalAux : 0,
    totalQueries,
    correctQueries,
    totalAux,
    correctAux,
    avgMemoryFilled: episodes.length > 0 ? memoryFilledSum / episodes.length : 0,
    writeProbeAccuracy: totalProbe > 0 ? correctProbe / totalProbe : 0,
    totalProbe,
    correctProbe,
  };
}

// ------------------------------
// Main
// ------------------------------

async function main(): Promise<void> {
  fs.mkdirSync(LOG_DIR, { recursive: true });

  for (const p of [CORPUS_PATH, TRAIN_PATH, VAL_PATH, TEST_PATH, SMOKE_PATH]) {
    assertFileExists(p);
  }

  console.log("=".repeat(96));
  console.log("MemoryBank BPE Experiment");
  console.log("=".repeat(96));
  console.log(`datasetRoot        : ${DATASET_ROOT}`);
  console.log(`mode               : ${USE_SMOKE ? "SMOKE" : USE_DIAGNOSTIC ? "DIAGNOSTIC" : "FULL"}`);
  console.log(`auxStoreLoss        : ${USE_AUX_STORE_LOSS}`);
  console.log(`writeProbe (PART4A) : ${USE_WRITE_PROBE}`);
  console.log(`bpeTargetVocabSize  : ${BPE_TARGET_VOCAB_SIZE}`);
  console.log(`maxTurnTokens       : ${MAX_TURN_TOKENS}`);
  console.log(`embeddingDim        : ${EMBEDDING_DIM}`);
  console.log(`memorySlots         : ${MEMORY_SLOTS}`);
  console.log(`memoryDim           : ${MEMORY_DIM}`);
  console.log(`outputClasses       : ${OUTPUT_CLASSES}`);
  console.log(`epochs              : ${EPOCHS}`);
  console.log(`alpha               : ${ALPHA}`);
  console.log(`writeThreshold      : ${MEMORY_WRITE_THRESHOLD}`);
  console.log("=".repeat(96));

  console.log("[1/5] Training BPE tokenizer...");
  const tokenizer = trainMemoryBpeTokenizer(CORPUS_PATH, BPE_TARGET_VOCAB_SIZE);
  const vocabSize = tokenizer.getVocabSize();
  const vocabCapacity =
    typeof tokenizer.getVocabularyCapacity === "function"
      ? tokenizer.getVocabularyCapacity()
      : vocabSize;

  console.log(`Tokenizer vocabSize=${vocabSize}, capacity=${vocabCapacity}, padId=${tokenizer.getPadId()}`);

  console.log("[2/5] Loading dataset...");
  const trainPath = USE_SMOKE ? SMOKE_PATH : TRAIN_PATH;

  const trainEpisodes = takeLimit(loadBpeMemoryEpisodes(trainPath), TRAIN_LIMIT);
  const valEpisodes = takeLimit(loadBpeMemoryEpisodes(VAL_PATH), VAL_LIMIT || (USE_SMOKE ? 256 : 0));
  const testEpisodes = takeLimit(loadBpeMemoryEpisodes(TEST_PATH), TEST_LIMIT || (USE_SMOKE ? 256 : 0));

  console.log(`train episodes=${trainEpisodes.length}`);
  console.log(`val episodes  =${valEpisodes.length}`);
  console.log(`test episodes =${testEpisodes.length}`);
  console.log(`random baseline ≈ ${formatPct(1 / OUTPUT_CLASSES)}`);

  console.log("[3/5] Building model...");

  const pooling = new TurnMaskedMeanPooling({
    units: EMBEDDING_DIM,
    maxTokens: MAX_TURN_TOKENS,
  });

  const model = new Sequential({
    layers: [
      new Embedding({
        vocabSize: vocabCapacity,
        embeddingDim: EMBEDDING_DIM,
        alpha: ALPHA,
        trainable: true,
      }) as any,

      pooling as any,

      new MemoryBank({
        memorySlots: MEMORY_SLOTS,
        memoryDim: MEMORY_DIM,
        mode: "project",
        outputUnits: EMBEDDING_DIM,
        writeThreshold: MEMORY_WRITE_THRESHOLD,
        updateMode: "gated-merge",
        writePolicy: "empty-first",
        similarity: "cosine",
        readTopK: Math.min(4, MEMORY_SLOTS),
        alpha: ALPHA,
        optimizer: "adam",
      }) as any,

      new Dense({
        units: EMBEDDING_DIM,
        outputUnits: OUTPUT_CLASSES,
        activation: "linear",
        status: "output",
        loss: "softmaxCrossEntropy",
        alpha: ALPHA,
      }) as any,
    ],
  });

  model.compile({
    alpha: ALPHA,
    error: "softmaxCrossEntropy",
    optimizer: "adam",
  });

  console.log("Model architecture:");
  model.summary();

  // PART 4A — Write probe classifier (trains writeValueKernel to encode value class)
  const writeProbe = new Dense({
    units: MEMORY_DIM,
    outputUnits: OUTPUT_CLASSES,
    activation: "linear",
    status: "output",
    loss: "softmaxCrossEntropy",
    alpha: ALPHA,
  });

  console.log("[4/5] Initial evaluation...");
  evaluateModel(model, pooling, tokenizer, valEpisodes, "val-before", false);
  evaluateModel(model, pooling, tokenizer, valEpisodes, "val-before-freezeWrites", true);

  // PART 8 — Diagnostic gate: run manual-read diagnostic before training
  if (!USE_SMOKE || USE_DIAGNOSTIC) {
    console.log("\n[diagnostic] Running manual-read read-path gate...");
    const { runMemoryBankRetrievalSuite } = require("../test/correctness/memoryBank.retrieval.test");
    try {
      runMemoryBankRetrievalSuite();
      console.log("[diagnostic] PASS: Read path is correct. Proceeding to training.");
    } catch (err: any) {
      console.error(`[diagnostic] FAIL: ${err.message}`);
      console.error("Training aborted. Fix MemoryBank read path before running full training.");
      process.exitCode = 1;
      return;
    }
  }

  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const metricsPath = path.join(LOG_DIR, `memorybank_bpe_metrics_${timestamp}.jsonl`);
  const artifactPath = path.join(LOG_DIR, `memorybank_bpe_artifact_${timestamp}.json`);

  console.log("[5/5] Training...");
  console.log(`metrics log: ${metricsPath}`);

  for (let epoch = 1; epoch <= EPOCHS; epoch++) {
    const epochStart = Date.now();

    const train = trainOneEpoch(model, pooling, tokenizer, trainEpisodes, epoch, writeProbe);

    // PART 3 — active vs frozen on a train subset
    const trainN = Math.min(trainEpisodes.length, TRAIN_EVAL_N);
    const trainEval = evaluateModel(model, pooling, tokenizer, trainEpisodes.slice(0, trainN), `train-epoch-${epoch}`, false);
    const trainFrozen = evaluateModel(model, pooling, tokenizer, trainEpisodes.slice(0, trainN), `train-epoch-${epoch}-freezeWrites`, true);
    const trainMemoryGain = trainEval.accuracy - trainFrozen.accuracy;

    const val = evaluateModel(model, pooling, tokenizer, valEpisodes, `val-epoch-${epoch}`, false);
    const valFrozen = evaluateModel(model, pooling, tokenizer, valEpisodes, `val-epoch-${epoch}-freezeWrites`, true);
    const valMemoryGain = val.accuracy - valFrozen.accuracy;

    // PART 2 — Memory audit on val
    auditMemoryEpisodes(model, pooling, tokenizer, valEpisodes, `val-epoch-${epoch}`, Math.min(valEpisodes.length, AUDIT_N), AUDIT_PRINT);

    const elapsed = (Date.now() - epochStart) / 1000;

    const row = {
      epoch,
      train,
      trainEval,
      trainFrozen,
      trainMemoryGain,
      val,
      valFrozen,
      valMemoryGain,
      elapsedSec: elapsed,
      createdAt: new Date().toISOString(),
    };

    fs.appendFileSync(metricsPath, JSON.stringify(row) + "\n", "utf-8");

    // PART 7 — Clarified summary logging
    console.log(
      [
        `Epoch ${epoch} summary`,
        `trainLoss=${formatNum(train.avgLoss)}`,
        `trainQueryAcc=${formatPct(train.queryAccuracy)}`,
        `auxAcc(notRetrieval)=${formatPct(train.auxAccuracy)}`,
        `writeProbeAcc=${formatPct(train.writeProbeAccuracy)}`,
        `trainEvalAcc=${formatPct(trainEval.accuracy)}`,
        `trainFreezeAcc=${formatPct(trainFrozen.accuracy)}`,
        `trainMemGain=${formatPct(trainMemoryGain)}`,
        `valAcc=${formatPct(val.accuracy)}`,
        `valFreezeAcc=${formatPct(valFrozen.accuracy)}`,
        `valMemGain=${formatPct(valMemoryGain)}`,
        `time=${elapsed.toFixed(1)}s`,
      ].join(" | ")
    );
    console.log("  NOTE: auxAcc is NOT proof of memory retrieval. It measures current-turn value encoding.");

    // PART 7 + 3 — Warnings
    if (trainMemoryGain <= 0) {
      console.log("  WARNING: MemoryBank active == freezeWrites on train. Memory is not contributing to query accuracy.");
    }
    if (valMemoryGain <= 0) {
      console.log("  WARNING: MemoryBank active == freezeWrites on val. Memory is not contributing to query accuracy.");
    }
    if (train.auxAccuracy > 0.5 && train.queryAccuracy < 0.1) {
      console.log("  WARNING: auxAcc is high but queryAcc is near random. auxAcc is NOT memory retrieval evidence.");
    }

    logMemory(`After epoch ${epoch}`);
    console.log("-".repeat(96));
  }

  console.log("Final test evaluation...");
  const test = evaluateModel(model, pooling, tokenizer, testEpisodes, "test", false);
  const testFrozen = evaluateModel(model, pooling, tokenizer, testEpisodes, "test-freezeWrites", true);

  const artifact = {
    note:
      "Experiment artifact. Sequential.save() is not used because TurnMaskedMeanPooling is an experiment-local layer not registered in setLayers.",
    config: {
      USE_SMOKE,
      USE_AUX_STORE_LOSS,
      BPE_TARGET_VOCAB_SIZE,
      vocabSize,
      vocabCapacity,
      MAX_TURN_TOKENS,
      EMBEDDING_DIM,
      MEMORY_SLOTS,
      MEMORY_DIM,
      OUTPUT_CLASSES,
      EPOCHS,
      ALPHA,
      MEMORY_WRITE_THRESHOLD,
    },
    final: {
      test,
      testFrozen,
    },
    layers: (model.layers as any[]).map((layer) => {
      if (typeof layer.save === "function") return layer.save();
      return { name: layer.name ?? "unknown local layer" };
    }),
  };

  fs.writeFileSync(artifactPath, JSON.stringify(artifact, null, 2), "utf-8");

  console.log("=".repeat(96));
  console.log("DONE");
  console.log(`metricsPath : ${metricsPath}`);
  console.log(`artifactPath: ${artifactPath}`);
  console.log(`testAcc     : ${formatPct(test.accuracy)}`);
  console.log(`freezeAcc   : ${formatPct(testFrozen.accuracy)}`);
  const finalMemoryGain = test.accuracy - testFrozen.accuracy;
  console.log(`memoryGain  : ${formatPct(finalMemoryGain)}`);
  if (finalMemoryGain <= 0) {
    console.log("WARNING: memoryGain <= 0. MemoryBank is not contributing to test accuracy.");
    console.log("Run: npx ts-node experiments/memorybank_diagnostic.ts to diagnose the read/write path.");
  }
  console.log("=".repeat(96));

  model.dispose();
}

main().catch((err) => {
  console.error("\n[FATAL] Experiment failed.");
  console.error(err);
  process.exitCode = 1;
});