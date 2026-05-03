/**
 * PART 6 — MemoryBank Episodic Retrieval Diagnostic
 *
 * Three modes that progressively isolate where the retrieval pipeline breaks.
 *
 * 1. manual-read
 *    Memory set manually. Query must read correct slot.
 *    If this fails: fix similarity / query / output head.
 *
 * 2. deterministic-write
 *    STORE writes key/value manually (bypasses learned write path).
 *    QUERY reads via normal MemoryBank read path.
 *    If this passes but learned-write fails: fix write training path.
 *
 * 3. learned-write
 *    Current normal MemoryBank flow (no manual intervention).
 *    Expected to fail before write supervision is added.
 *    After write probe fix, this should start improving.
 *
 * Usage:
 *   npx ts-node experiments/memorybank_diagnostic.ts --mode manual-read
 *   npx ts-node experiments/memorybank_diagnostic.ts --mode deterministic-write
 *   npx ts-node experiments/memorybank_diagnostic.ts --mode learned-write
 *   npx ts-node experiments/memorybank_diagnostic.ts          # runs all three
 */

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
} from "./memorybank_bpe_dataset/bpe_memory_dataset_loader";

const PROJECT_ROOT = path.resolve(__dirname, "..");
const DATASET_ROOT = path.join(PROJECT_ROOT, "experiments/memorybank_bpe_dataset");
const SMOKE_PATH = path.join(DATASET_ROOT, "smoke.jsonl");
const CORPUS_PATH = path.join(DATASET_ROOT, "bpe_corpus.txt");

const EMBEDDING_DIM = 32;
const MEMORY_SLOTS = 8;
const MEMORY_DIM = 32;
const OUTPUT_CLASSES = 24;
const MAX_TURN_TOKENS = 12;
const ALPHA = 0.001;
const DIAGNOSTIC_EPISODES = 100;

// ─── helpers ────────────────────────────────────────────────────────────────

function formatPct(v: number): string {
  return `${(v * 100).toFixed(2)}%`;
}

function argmax(m: Matrix): number {
  let maxIdx = 0;
  let maxVal = m._data[0];
  for (let i = 1; i < m._data.length; i++) {
    if (m._data[i] > maxVal) { maxVal = m._data[i]; maxIdx = i; }
  }
  return maxIdx;
}

function parseValueClass(text?: string): number | null {
  if (!text) return null;
  const m = text.match(/^value_(\d+)$/);
  if (!m) return null;
  const v = Number(m[1]);
  return Number.isInteger(v) && v >= 0 && v < OUTPUT_CLASSES ? v : null;
}

// Simple pooling layer (local, no pooling layer from repo needed here)
function meanPool(tokenIds: number[], maxLen: number, embeddingDim: number, embedding: any): Matrix {
  const padded = tokenIds.slice(0, maxLen);
  while (padded.length < maxLen) padded.push(0);
  const x = Matrix.fromFlat(Float32Array.from(padded), [maxLen, 1]);
  const emb: Matrix = (embedding as any).forward(x); // [embeddingDim, maxLen]
  // mean across columns
  const out = mj.zeros([embeddingDim, 1]);
  const validLen = Math.min(tokenIds.length, maxLen);
  for (let d = 0; d < embeddingDim; d++) {
    let s = 0;
    for (let t = 0; t < validLen; t++) s += emb._data[d * maxLen + t];
    out._data[d] = s / Math.max(1, validLen);
  }
  return out;
}

function getMemoryBankLayer(mb: MemoryBank): MemoryBank {
  return mb;
}

// ─── Result types ────────────────────────────────────────────────────────────

interface DiagResult {
  mode: string;
  episodes: number;
  queries: number;
  topSlotCorrect: number;
  topSlotAcc: number;
  topValueCorrect: number;
  topValueAcc: number;
  predCorrect: number;
  predAcc: number;
  activeAcc: number;
  frozenAcc: number;
  memoryGain: number;
}

// ─── mode 1: manual-read ─────────────────────────────────────────────────────

function diagManualRead(): DiagResult {
  console.log("\n" + "=".repeat(72));
  console.log("MODE: manual-read");
  console.log("Proves: MemoryBank read path (similarity + slot selection) is correct");
  console.log("Expected: topSlotAcc=100%, predAcc meaningful");
  console.log("=".repeat(72));

  const N = 4; // small

  const mb = new MemoryBank({
    units: N,
    memorySlots: N,
    memoryDim: N,
    outputUnits: N,
    mode: "project",
    similarity: "cosine",
    readTopK: N,
    writeThreshold: 99.0, // disable real writes
    trainablePolicy: false,
  });

  // Init
  mb.forward(mj.zeros([N, 1]));
  mb.resetMemory();

  // Set queryKernel = identity
  (mb as any).queryKernel = mj.zeros([N, N]);
  for (let i = 0; i < N; i++) (mb as any).queryKernel._data[i * N + i] = 1;

  // Slot 0: key=[1,0,0,0]  slot 1: key=[0,1,0,0]  slot 2: key=[0,0,1,0]
  const keyVectors = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
  ];
  const valVectors = [
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8],
    [0.9, 0.8, 0.7, 0.6],
  ];
  for (let s = 0; s < 3; s++) {
    mb.writeMemoryForDebug(keyVectors[s], valVectors[s], s);
  }

  // Queries: each standard basis vector should retrieve the corresponding slot
  let topSlotCorrect = 0;
  let queries = 0;

  for (let s = 0; s < 3; s++) {
    const qVec = keyVectors[s];
    const input = mj.zeros([N, 1]);
    for (let d = 0; d < N; d++) input._data[d] = qVec[d];

    mb.forward(input);
    const trace = mb.getDebugTrace();
    const topSlot = trace[0]?.readSlots[0]?.slot ?? -1;
    const correct = topSlot === s;
    topSlotCorrect += correct ? 1 : 0;
    queries++;

    if (!correct) {
      console.log(
        `  [manual-read] FAIL: query slot=${s} expected topSlot=${s}, got topSlot=${topSlot}. ` +
          `readSlots=${JSON.stringify(trace[0]?.readSlots)}`
      );
    }
  }

  const topSlotAcc = queries > 0 ? topSlotCorrect / queries : 0;
  console.log(`[manual-read] topSlotAcc=${formatPct(topSlotAcc)} (${topSlotCorrect}/${queries})`);

  if (topSlotAcc < 1.0) {
    console.error(
      "FAIL: manual-read topSlotAcc < 100%. The read path (cosine similarity + slot selection) is broken. " +
        "Fix similarity/query/output path before proceeding."
    );
  } else {
    console.log("PASS: manual-read. Read path selects slots correctly.");
  }

  return {
    mode: "manual-read",
    episodes: 1,
    queries,
    topSlotCorrect,
    topSlotAcc,
    topValueCorrect: 0,
    topValueAcc: 0,
    predCorrect: 0,
    predAcc: 0,
    activeAcc: topSlotAcc,
    frozenAcc: 0,
    memoryGain: 0,
  };
}

// ─── mode 2: deterministic-write ────────────────────────────────────────────

function diagDeterministicWrite(tokenizer: any, episodes: BpeMemoryEpisode[]): DiagResult {
  console.log("\n" + "=".repeat(72));
  console.log("MODE: deterministic-write");
  console.log("Proves: When write state is correct, can query path + output head solve it?");
  console.log("Expected: topSlotAcc high, predAcc above random if output head is reasonable");
  console.log("=".repeat(72));

  const embedding = new (require("../src/layers").Embedding)({
    vocabSize: tokenizer.getVocabSize(),
    embeddingDim: EMBEDDING_DIM,
    alpha: ALPHA,
    trainable: false,
  });

  const mb = new MemoryBank({
    units: EMBEDDING_DIM,
    memorySlots: MEMORY_SLOTS,
    memoryDim: MEMORY_DIM,
    outputUnits: EMBEDDING_DIM,
    mode: "project",
    similarity: "cosine",
    readTopK: Math.min(4, MEMORY_SLOTS),
    writeThreshold: 99.0, // prevent learned writes
    trainablePolicy: false,
  });

  // Create a simple output head
  const outputHead = new Dense({
    units: EMBEDDING_DIM,
    outputUnits: OUTPUT_CLASSES,
    activation: "linear",
    status: "output",
    loss: "softmaxCrossEntropy",
    alpha: ALPHA,
  });

  // Force mb init
  const dummyIn = mj.zeros([EMBEDDING_DIM, 1]);
  mb.forward(dummyIn);

  let totalQueries = 0;
  let topSlotCorrect = 0;
  let topValueCorrect = 0;
  let predCorrect = 0;

  const maxEp = Math.min(episodes.length, DIAGNOSTIC_EPISODES);
  let printCount = 0;

  for (let i = 0; i < maxEp; i++) {
    const episode = episodes[i];
    mb.resetMemory();

    // Shadow tracker
    const slotFacts = new Map<number, { keyText: string; valueText: string; valueClass: number }>();
    const keyToSlot = new Map<string, number>();

    for (let t = 0; t < episode.turns.length; t++) {
      const turn = episode.turns[t];

      // Encode turn text
      const ids = tokenizer.encode(turn.text);
      const validLen = Math.min(ids.length, MAX_TURN_TOKENS);
      const paddedIds = tokenizer.padSequence(ids, MAX_TURN_TOKENS);
      const xTokens = Matrix.fromFlat(Float32Array.from(paddedIds), [MAX_TURN_TOKENS, 1]);
      const embOut: Matrix = (embedding as any).forward(xTokens);

      // Mean pool
      const pooled = mj.zeros([EMBEDDING_DIM, 1]);
      for (let d = 0; d < EMBEDDING_DIM; d++) {
        let s = 0;
        for (let c = 0; c < validLen; c++) s += embOut._data[d * MAX_TURN_TOKENS + c];
        pooled._data[d] = s / Math.max(1, validLen);
      }

      if (turn.op === "STORE" || turn.op === "UPDATE") {
        // Deterministic write: encode key and value text and write manually
        const valueClass = parseValueClass(turn.value_text);
        if (turn.key_text && turn.value_text && valueClass !== null) {
          // Use embedding of key_text as key vector (mean pooled)
          const kIds = tokenizer.encode(turn.key_text);
          const kPadded = tokenizer.padSequence(kIds, MAX_TURN_TOKENS);
          const kX = Matrix.fromFlat(Float32Array.from(kPadded), [MAX_TURN_TOKENS, 1]);
          const kEmb: Matrix = (embedding as any).forward(kX);
          const keyVec: number[] = [];
          const kValidLen = Math.min(kIds.length, MAX_TURN_TOKENS);
          for (let d = 0; d < EMBEDDING_DIM; d++) {
            let s = 0;
            for (let c = 0; c < kValidLen; c++) s += kEmb._data[d * MAX_TURN_TOKENS + c];
            keyVec.push(s / Math.max(1, kValidLen));
          }

          // Value vector: one-hot for valueClass, padded to MEMORY_DIM
          const valVec = new Array<number>(MEMORY_DIM).fill(0);
          if (valueClass < MEMORY_DIM) valVec[valueClass] = 1.0;

          // Find an empty or least-used slot
          const state = mb.getMemoryState();
          let writeSlot = -1;
          for (let s = 0; s < MEMORY_SLOTS; s++) {
            if (!state.memoryFilled[s]) { writeSlot = s; break; }
          }
          if (writeSlot === -1) {
            let minUsage = Infinity;
            for (let s = 0; s < MEMORY_SLOTS; s++) {
              if (state.memoryUsage[s] < minUsage) { minUsage = state.memoryUsage[s]; writeSlot = s; }
            }
          }

          mb.writeMemoryForDebug(keyVec, valVec, writeSlot);
          slotFacts.set(writeSlot, { keyText: turn.key_text, valueText: turn.value_text, valueClass });
          keyToSlot.set(turn.key_text, writeSlot);
        }

        // Also forward through MB (for read side) with frozen writes
        mb.forward(pooled); // reads are computed but writes disabled
      } else if (turn.op === "QUERY") {
        // Forward through mb (writes frozen)
        const mbOut = mb.forward(pooled); // [EMBEDDING_DIM, 1]
        const trace = mb.getDebugTrace();
        const pred = outputHead.forward(mbOut);

        const q = getQueryForTurn(episode, t);
        if (!q) continue;

        totalQueries++;
        const topReadSlot = trace[0]?.readSlots[0]?.slot ?? -1;
        const expectedSlot = keyToSlot.get(q.key_text) ?? -1;
        const topFact = topReadSlot >= 0 ? slotFacts.get(topReadSlot) : undefined;
        const predClass = argmax(pred);

        const slotOk = topReadSlot === expectedSlot && expectedSlot >= 0;
        const valueOk = topFact?.valueClass === q.target_class;
        const predOk = predClass === q.target_class;

        topSlotCorrect += slotOk ? 1 : 0;
        topValueCorrect += valueOk ? 1 : 0;
        predCorrect += predOk ? 1 : 0;

        if (printCount < 3) {
          console.log(
            `  [det-write] ep=${i} t=${t}: key="${q.key_text}" ` +
              `expectedSlot=${expectedSlot} topSlot=${topReadSlot} ` +
              `topFact=${JSON.stringify(topFact)} ` +
              `pred=${predClass} target=${q.target_class} ` +
              `slotOk=${slotOk} valueOk=${valueOk} predOk=${predOk}`
          );
          printCount++;
        }
      } else {
        // NOOP: just forward but ignore
        mb.forward(pooled);
      }
    }
  }

  const topSlotAcc = totalQueries > 0 ? topSlotCorrect / totalQueries : 0;
  const topValueAcc = totalQueries > 0 ? topValueCorrect / totalQueries : 0;
  const predAcc = totalQueries > 0 ? predCorrect / totalQueries : 0;
  const random = 1 / OUTPUT_CLASSES;

  console.log(`[deterministic-write] queries=${totalQueries}`);
  console.log(`  topSlotAcc  = ${formatPct(topSlotAcc)}`);
  console.log(`  topValueAcc = ${formatPct(topValueAcc)}`);
  console.log(`  predAcc     = ${formatPct(predAcc)}  (random baseline=${formatPct(random)})`);

  if (topSlotAcc < 0.5) {
    console.error(
      "WARNING: deterministic-write topSlotAcc < 50%. Even with manually correct writes, " +
        "query cannot find the right slot. Read path or embedding similarity is misaligned."
    );
  }
  if (topSlotAcc >= 0.5 && predAcc <= random + 0.02) {
    console.error(
      "WARNING: deterministic-write topSlotAcc OK but predAcc near random. " +
        "Output head is not using memory values. Check outputKernel or memory read integration."
    );
  }
  if (topSlotAcc >= 0.5 && predAcc > random + 0.05) {
    console.log(
      "PASS: deterministic-write. When writes are correct, model can read and produce above-random predictions."
    );
  }

  return {
    mode: "deterministic-write",
    episodes: maxEp,
    queries: totalQueries,
    topSlotCorrect,
    topSlotAcc,
    topValueCorrect,
    topValueAcc,
    predCorrect,
    predAcc,
    activeAcc: predAcc,
    frozenAcc: 0,
    memoryGain: 0,
  };
}

// ─── mode 3: learned-write ───────────────────────────────────────────────────

function diagLearnedWrite(tokenizer: any, episodes: BpeMemoryEpisode[]): DiagResult {
  console.log("\n" + "=".repeat(72));
  console.log("MODE: learned-write");
  console.log("Current normal MemoryBank flow. Expected to show memoryGain ≈ 0 before write-supervision fix.");
  console.log("=".repeat(72));

  // Build a tiny model similar to the main experiment (no training — just single-pass diagnostic)
  const vocabCapacity =
    typeof tokenizer.getVocabularyCapacity === "function"
      ? tokenizer.getVocabularyCapacity()
      : tokenizer.getVocabSize();

  const embedding = new (require("../src/layers").Embedding)({
    vocabSize: vocabCapacity,
    embeddingDim: EMBEDDING_DIM,
    alpha: ALPHA,
    trainable: false,
  });

  const mb = new MemoryBank({
    units: EMBEDDING_DIM,
    memorySlots: MEMORY_SLOTS,
    memoryDim: MEMORY_DIM,
    outputUnits: EMBEDDING_DIM,
    mode: "project",
    similarity: "cosine",
    readTopK: Math.min(4, MEMORY_SLOTS),
    writeThreshold: 0.0,
    updateMode: "gated-merge",
    writePolicy: "empty-first",
    trainablePolicy: true,
    alpha: ALPHA,
    optimizer: "adam",
  });

  const outputHead = new Dense({
    units: EMBEDDING_DIM,
    outputUnits: OUTPUT_CLASSES,
    activation: "linear",
    status: "output",
    loss: "softmaxCrossEntropy",
    alpha: ALPHA,
  });

  // Init
  mb.forward(mj.zeros([EMBEDDING_DIM, 1]));

  let totalQueries = 0;
  let activeCorrect = 0;
  let frozenCorrect = 0;

  const maxEp = Math.min(episodes.length, DIAGNOSTIC_EPISODES);

  for (let i = 0; i < maxEp; i++) {
    const episode = episodes[i];
    mb.resetMemory();

    for (let t = 0; t < episode.turns.length; t++) {
      const turn = episode.turns[t];

      const ids = tokenizer.encode(turn.text);
      const validLen = Math.min(ids.length, MAX_TURN_TOKENS);
      const paddedIds = tokenizer.padSequence(ids, MAX_TURN_TOKENS);
      const xTokens = Matrix.fromFlat(Float32Array.from(paddedIds), [MAX_TURN_TOKENS, 1]);
      const embOut: Matrix = (embedding as any).forward(xTokens);
      const pooled = mj.zeros([EMBEDDING_DIM, 1]);
      for (let d = 0; d < EMBEDDING_DIM; d++) {
        let s = 0;
        for (let c = 0; c < validLen; c++) s += embOut._data[d * MAX_TURN_TOKENS + c];
        pooled._data[d] = s / Math.max(1, validLen);
      }

      if (turn.op === "STORE" || turn.op === "UPDATE") {
        (mb as any).writeFrozen = false;
        mb.forward(pooled);
      } else if (turn.op === "QUERY") {
        const q = getQueryForTurn(episode, t);
        if (!q) continue;

        // Active pass
        (mb as any).writeFrozen = true;
        const activeOut = mb.forward(pooled);
        const activePred = outputHead.forward(activeOut);
        const activePredClass = argmax(activePred);
        if (activePredClass === q.target_class) activeCorrect++;

        // Frozen pass (same memory state since we didn't write during active)
        const frozenOut = mb.forward(pooled);
        const frozenPred = outputHead.forward(frozenOut);
        const frozenPredClass = argmax(frozenPred);
        if (frozenPredClass === q.target_class) frozenCorrect++;

        totalQueries++;
      } else {
        (mb as any).writeFrozen = true;
        mb.forward(pooled);
      }
    }
  }

  const activeAcc = totalQueries > 0 ? activeCorrect / totalQueries : 0;
  const frozenAcc = totalQueries > 0 ? frozenCorrect / totalQueries : 0;
  const memoryGain = activeAcc - frozenAcc;
  const random = 1 / OUTPUT_CLASSES;

  console.log(`[learned-write] queries=${totalQueries}`);
  console.log(`  activeAcc   = ${formatPct(activeAcc)}`);
  console.log(`  frozenAcc   = ${formatPct(frozenAcc)}`);
  console.log(`  memoryGain  = ${formatPct(memoryGain)}`);
  console.log(`  random      = ${formatPct(random)}`);

  if (Math.abs(memoryGain) < 0.01) {
    console.log(
      "WARNING: MemoryBank active == freezeWrites. Memory is not contributing to query accuracy. " +
        "Write path needs direct supervision (trainLastWriteValue probe). Run with write supervision next."
    );
  }
  if (activeAcc <= random + 0.02) {
    console.log(
      "INFO: activeAcc near random. This is expected before write-path training is fixed. " +
        "Ensure deterministic-write mode passes first."
    );
  }

  return {
    mode: "learned-write",
    episodes: maxEp,
    queries: totalQueries,
    topSlotCorrect: 0,
    topSlotAcc: 0,
    topValueCorrect: 0,
    topValueAcc: 0,
    predCorrect: activeCorrect,
    predAcc: activeAcc,
    activeAcc,
    frozenAcc,
    memoryGain,
  };
}

// ─── Main ─────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const modeArg = args.find((a) => a.startsWith("--mode="))?.split("=")[1] ??
    (args.includes("--mode") ? args[args.indexOf("--mode") + 1] : null);
  const modes = modeArg
    ? [modeArg]
    : ["manual-read", "deterministic-write", "learned-write"];

  console.log("=".repeat(72));
  console.log("MemoryBank Episodic Retrieval Diagnostic");
  console.log("=".repeat(72));
  console.log(`Modes to run: ${modes.join(", ")}`);

  // Always train tokenizer and load episodes for modes 2/3
  let tokenizer: any = null;
  let episodes: BpeMemoryEpisode[] = [];
  const needsData = modes.some((m) => m !== "manual-read");

  if (needsData) {
    console.log("\nLoading tokenizer and smoke episodes...");
    tokenizer = trainMemoryBpeTokenizer(CORPUS_PATH, 256);
    episodes = loadBpeMemoryEpisodes(SMOKE_PATH).slice(0, DIAGNOSTIC_EPISODES);
    console.log(`Loaded ${episodes.length} episodes`);
  }

  const results: DiagResult[] = [];
  let manualReadFailed = false;

  if (modes.includes("manual-read")) {
    const r = diagManualRead();
    results.push(r);
    if (r.topSlotAcc < 1.0) {
      manualReadFailed = true;
      console.error(
        "\n[DIAGNOSTIC GATE] manual-read FAILED. Read path is broken. " +
          "Do NOT proceed to training. Fix cosine similarity / slot selection first."
      );
    }
  }

  if (modes.includes("deterministic-write")) {
    if (manualReadFailed && !modeArg) {
      console.log("\nSkipping deterministic-write because manual-read failed.");
    } else {
      const r = diagDeterministicWrite(tokenizer, episodes);
      results.push(r);
    }
  }

  if (modes.includes("learned-write")) {
    const r = diagLearnedWrite(tokenizer, episodes);
    results.push(r);
    if (Math.abs(r.memoryGain) < 0.01) {
      console.log(
        "\nINFO: learned-write memoryGain ≈ 0. " +
          "Before full training, add write supervision (trainLastWriteValue probe). " +
          "Run full experiment only after diagnostic passes."
      );
    }
  }

  // ─── Summary ──────────────────────────────────────────────────────────────
  console.log("\n" + "=".repeat(72));
  console.log("DIAGNOSTIC SUMMARY");
  console.log("=".repeat(72));
  for (const r of results) {
    if (r.mode === "manual-read") {
      console.log(
        `  manual-read       topSlotAcc=${formatPct(r.topSlotAcc)} ` +
          `[${r.topSlotAcc >= 1.0 ? "PASS" : "FAIL"}]`
      );
    } else if (r.mode === "deterministic-write") {
      console.log(
        `  deterministic-write  topSlotAcc=${formatPct(r.topSlotAcc)} ` +
          `topValueAcc=${formatPct(r.topValueAcc)} ` +
          `predAcc=${formatPct(r.predAcc)}`
      );
    } else if (r.mode === "learned-write") {
      console.log(
        `  learned-write     activeAcc=${formatPct(r.activeAcc)} ` +
          `frozenAcc=${formatPct(r.frozenAcc)} ` +
          `memGain=${formatPct(r.memoryGain)}`
      );
    }
  }

  // PART 8: Gate check — throw if manual-read failed
  const manualResult = results.find((r) => r.mode === "manual-read");
  if (manualResult && manualResult.topSlotAcc < 1.0) {
    throw new Error(
      "DIAGNOSTIC GATE FAILED: manual-read topSlotAcc < 100%. " +
        "MemoryBank read path is broken. Full training aborted."
    );
  }

  const learnedResult = results.find((r) => r.mode === "learned-write");
  if (learnedResult && Math.abs(learnedResult.memoryGain) < 0.01) {
    console.log(
      "\nWARNING: MemoryBank active == freezeWrites in learned-write mode. " +
        "Memory is not contributing to query accuracy. " +
        "Add trainLastWriteValue probe before running full training."
    );
  }
}

main().catch((err) => {
  console.error("\n[FATAL] Diagnostic failed.");
  console.error(err);
  process.exitCode = 1;
});
