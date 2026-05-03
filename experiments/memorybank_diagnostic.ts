/**
 * MemoryBank Episodic Retrieval Diagnostic
 *
 * Modes:
 * 1. manual-read
 *    Proves raw MemoryBank read path works with known keys.
 *
 * 2. deterministic-write
 *    Writes memory manually using canonical query-space keys:
 *      memoryKey = normalize(queryKernel * pooled("query key_xx"))
 *    Then QUERY should retrieve the correct slot.
 *    This is the critical fix: key-space must match between STORE and QUERY.
 *
 * 3. learned-write
 *    Uses normal MemoryBank write path. Expected to need writeKey/writeValue supervision.
 *
 * Usage:
 *   npx ts-node experiments/memorybank_diagnostic.ts
 *   npx ts-node experiments/memorybank_diagnostic.ts --mode manual-read
 *   npx ts-node experiments/memorybank_diagnostic.ts --mode deterministic-write
 *   npx ts-node experiments/memorybank_diagnostic.ts --mode learned-write
 */

import path from "path";

import { MemoryBank, Dense, Embedding } from "../src/layers";
import mj from "../src/math";
import Matrix from "../src/matrix";

import {
  loadBpeMemoryEpisodes,
  trainMemoryBpeTokenizer,
  getQueryForTurn,
  BpeMemoryEpisode,
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

function formatPct(v: number): string {
  return `${(v * 100).toFixed(2)}%`;
}

function argmax(m: Matrix): number {
  let maxIdx = 0;
  let maxVal = m._data[0];
  for (let i = 1; i < m._data.length; i++) {
    if (m._data[i] > maxVal) {
      maxVal = m._data[i];
      maxIdx = i;
    }
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

function matrixToArray(m: Matrix): number[] {
  return Array.from(m._data);
}

/**
 * Mean-pool tokenized text into a [embeddingDim, 1] Matrix.
 */
function pooledText(
  tokenizer: any,
  embedding: Embedding,
  text: string,
  maxLen = MAX_TURN_TOKENS,
  embeddingDim = EMBEDDING_DIM
): Matrix {
  const ids = tokenizer.encode(text);
  const validLen = Math.max(1, Math.min(ids.length, maxLen));
  const padded = tokenizer.padSequence(ids, maxLen);
  const x = Matrix.fromFlat(Float32Array.from(padded), [maxLen, 1]);
  const emb: Matrix = (embedding as any).forward(x);
  const pooled = mj.zeros([embeddingDim, 1]);
  for (let d = 0; d < embeddingDim; d++) {
    let sum = 0;
    const rowOffset = d * maxLen;
    for (let t = 0; t < validLen; t++) sum += emb._data[rowOffset + t];
    pooled._data[d] = sum / validLen;
  }
  return pooled;
}

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

// ─── Mode 1: manual-read ────────────────────────────────────────────────────

function diagManualRead(): DiagResult {
  console.log("\n" + "=".repeat(72));
  console.log("MODE: manual-read");
  console.log("Proves: raw MemoryBank read path works with known keys");
  console.log("Expected: topSlotAcc=100%");
  console.log("=".repeat(72));

  const N = 4;
  const mb = new MemoryBank({
    units: N,
    memorySlots: N,
    memoryDim: N,
    outputUnits: N,
    mode: "project",
    similarity: "cosine",
    readTopK: N,
    writeThreshold: 99.0,
    trainablePolicy: false,
  });

  mb.forward(mj.zeros([N, 1]));
  mb.resetMemory();

  // queryKernel = identity so q = x directly
  (mb as any).queryKernel = mj.zeros([N, N]);
  for (let i = 0; i < N; i++) (mb as any).queryKernel._data[i * N + i] = 1;

  const keyVectors = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
  ];
  const valueVectors = [
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8],
    [0.9, 0.8, 0.7, 0.6],
  ];

  for (let s = 0; s < 3; s++) {
    mb.writeMemoryForDebug(keyVectors[s], valueVectors[s], s);
  }

  let queries = 0;
  let topSlotCorrect = 0;

  for (let s = 0; s < 3; s++) {
    const input = mj.zeros([N, 1]);
    for (let d = 0; d < N; d++) input._data[d] = keyVectors[s][d];
    mb.forward(input);
    const trace = mb.getDebugTrace();
    const topSlot = trace[0]?.readSlots[0]?.slot ?? -1;
    const ok = topSlot === s;
    if (ok) topSlotCorrect++;
    queries++;
    if (!ok) {
      console.log(
        `[manual-read] FAIL querySlot=${s} expected=${s} got=${topSlot} readSlots=${JSON.stringify(trace[0]?.readSlots)}`
      );
    }
  }

  const topSlotAcc = queries ? topSlotCorrect / queries : 0;
  console.log(`[manual-read] topSlotAcc=${formatPct(topSlotAcc)} (${topSlotCorrect}/${queries})`);
  if (topSlotAcc === 1) {
    console.log("PASS: manual-read.");
  } else {
    console.error("FAIL: manual-read. Fix similarity/query/slot selection first.");
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

// ─── Mode 2: deterministic-write (canonical key-space) ──────────────────────

function diagDeterministicWrite(tokenizer: any, episodes: BpeMemoryEpisode[]): DiagResult {
  console.log("\n" + "=".repeat(72));
  console.log("MODE: deterministic-write");
  console.log("Proves: if memory written with canonical query-space keys, QUERY retrieves correct slot.");
  console.log('Key fix: memoryKey = normalize(queryKernel * pooled("query key_xx"))');
  console.log("Expected: topSlotAcc > 80%");
  console.log("=".repeat(72));

  const vocabCapacity =
    typeof tokenizer.getVocabularyCapacity === "function"
      ? tokenizer.getVocabularyCapacity()
      : tokenizer.getVocabSize();

  const embedding = new Embedding({
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
    writeThreshold: 99.0,
    trainablePolicy: false,
  });

  // Output head is intentionally untrained — predAcc is not the main metric here.
  // Main metrics: topSlotAcc and topValueAcc.
  const outputHead = new Dense({
    units: EMBEDDING_DIM,
    outputUnits: OUTPUT_CLASSES,
    activation: "linear",
    status: "output",
    loss: "softmaxCrossEntropy",
    alpha: ALPHA,
  });

  mb.forward(mj.zeros([EMBEDDING_DIM, 1]));
  mb.resetMemory();

  let totalQueries = 0;
  let topSlotCorrect = 0;
  let topValueCorrect = 0;
  let predCorrect = 0;
  let printCount = 0;

  const maxEp = Math.min(episodes.length, DIAGNOSTIC_EPISODES);

  for (let i = 0; i < maxEp; i++) {
    const episode = episodes[i];
    mb.resetMemory();

    const slotFacts = new Map<number, { keyText: string; valueText: string; valueClass: number }>();
    const keyToSlot = new Map<string, number>();

    for (let t = 0; t < episode.turns.length; t++) {
      const turn = episode.turns[t];

      if (turn.op === "STORE" || turn.op === "UPDATE") {
        const valueClass = parseValueClass(turn.value_text);
        if (turn.key_text && turn.value_text && valueClass !== null) {
          // CRITICAL FIX: key must live in query-kernel space.
          // A future QUERY will project "query key_xx" via queryKernel.
          // So write key = normalize(queryKernel * pooled("query key_xx")).
          const canonicalQueryText = `query ${turn.key_text}`;
          const canonicalQueryPooled = pooledText(tokenizer, embedding, canonicalQueryText);
          const keyMatrix = mb.getQueryVectorForInput(canonicalQueryPooled, true);
          const keyVec = matrixToArray(keyMatrix);

          // Store value as one-hot of valueClass in memoryDim space (deterministic diagnosis).
          const valueVec = new Array<number>(MEMORY_DIM).fill(0);
          if (valueClass < MEMORY_DIM) valueVec[valueClass] = 1;

          const state = mb.getMemoryState();
          let writeSlot = -1;

          // UPDATE: reuse same slot if key already written
          if (turn.op === "UPDATE" && keyToSlot.has(turn.key_text)) {
            writeSlot = keyToSlot.get(turn.key_text)!;
          } else {
            for (let s = 0; s < MEMORY_SLOTS; s++) {
              if (!state.memoryFilled[s]) { writeSlot = s; break; }
            }
            if (writeSlot === -1) {
              let minUsage = Infinity;
              for (let s = 0; s < MEMORY_SLOTS; s++) {
                if (state.memoryUsage[s] < minUsage) { minUsage = state.memoryUsage[s]; writeSlot = s; }
              }
            }
          }

          mb.writeMemoryForDebug(keyVec, valueVec, writeSlot);
          slotFacts.set(writeSlot, { keyText: turn.key_text, valueText: turn.value_text, valueClass });
          keyToSlot.set(turn.key_text, writeSlot);
        }
        continue;
      }

      if (turn.op === "QUERY") {
        const q = getQueryForTurn(episode, t);
        if (!q) continue;

        const pooled = pooledText(tokenizer, embedding, turn.text);
        const mbOut = mb.forward(pooled);
        const trace = mb.getDebugTrace();
        const pred = outputHead.forward(mbOut);
        const predClass = argmax(pred);

        const topReadSlot = trace[0]?.readSlots[0]?.slot ?? -1;
        const expectedSlot = keyToSlot.get(q.key_text) ?? -1;
        const topFact = topReadSlot >= 0 ? slotFacts.get(topReadSlot) : undefined;

        const slotOk = topReadSlot === expectedSlot && expectedSlot >= 0;
        const valueOk = topFact?.valueClass === q.target_class;
        const predOk = predClass === q.target_class;

        if (slotOk) topSlotCorrect++;
        if (valueOk) topValueCorrect++;
        if (predOk) predCorrect++;
        totalQueries++;

        if (printCount < 5) {
          console.log(
            [
              `[det-write] ep=${i} t=${t}`,
              `key="${q.key_text}"`,
              `expectedSlot=${expectedSlot}`,
              `topSlot=${topReadSlot}`,
              `topAttn=${(trace[0]?.readSlots[0]?.attn ?? 0).toFixed(3)}`,
              `topFact=${JSON.stringify(topFact)}`,
              `pred=${predClass}`,
              `target=${q.target_class}`,
              `slotOk=${slotOk}`,
              `valueOk=${valueOk}`,
              `predOk=${predOk}`,
            ].join(" | ")
          );
          printCount++;
        }
      }
    }
  }

  const topSlotAcc = totalQueries ? topSlotCorrect / totalQueries : 0;
  const topValueAcc = totalQueries ? topValueCorrect / totalQueries : 0;
  const predAcc = totalQueries ? predCorrect / totalQueries : 0;

  console.log(`[deterministic-write] queries=${totalQueries}`);
  console.log(`  topSlotAcc  = ${formatPct(topSlotAcc)} (${topSlotCorrect}/${totalQueries})`);
  console.log(`  topValueAcc = ${formatPct(topValueAcc)} (${topValueCorrect}/${totalQueries})`);
  console.log(`  predAcc     = ${formatPct(predAcc)} (${predCorrect}/${totalQueries})`);
  console.log(`  random      = ${formatPct(1 / OUTPUT_CLASSES)}`);

  if (topSlotAcc >= 0.8) {
    console.log("PASS: deterministic-write. Canonical key-space retrieval is healthy.");
    if (topValueAcc < 0.8) {
      console.error("WARNING: topSlotAcc OK but topValueAcc low. Check slot shadow tracking in diagnostic.");
    }
  } else {
    console.error(
      "WARNING: deterministic-write topSlotAcc < 80%. " +
        "Even with canonical query-space keys, retrieval is failing. " +
        "Check pooledText encoding consistency between STORE and QUERY."
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

// ─── Mode 3: learned-write ───────────────────────────────────────────────────

function diagLearnedWrite(tokenizer: any, episodes: BpeMemoryEpisode[]): DiagResult {
  console.log("\n" + "=".repeat(72));
  console.log("MODE: learned-write");
  console.log("Proves: normal learned writes contribute beyond no-write baseline");
  console.log("Expected before write-key/value supervision: memoryGain near 0");
  console.log("=".repeat(72));

  const vocabCapacity =
    typeof tokenizer.getVocabularyCapacity === "function"
      ? tokenizer.getVocabularyCapacity()
      : tokenizer.getVocabSize();

  function buildStack() {
    const embedding = new Embedding({
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
    mb.forward(mj.zeros([EMBEDDING_DIM, 1]));
    mb.resetMemory();
    return { embedding, mb, outputHead };
  }

  function runPass(freezeWritesFromStart: boolean): { correct: number; total: number } {
    const { embedding, mb, outputHead } = buildStack();
    let correct = 0;
    let total = 0;
    const maxEp = Math.min(episodes.length, DIAGNOSTIC_EPISODES);

    for (let i = 0; i < maxEp; i++) {
      const episode = episodes[i];
      mb.resetMemory();

      for (let t = 0; t < episode.turns.length; t++) {
        const turn = episode.turns[t];
        const pooled = pooledText(tokenizer, embedding, turn.text);

        if (turn.op === "STORE" || turn.op === "UPDATE") {
          if (freezeWritesFromStart) mb.freezeWrites();
          else mb.enableWrites();
          mb.forward(pooled);
          continue;
        }

        if (turn.op === "QUERY") {
          const q = getQueryForTurn(episode, t);
          if (!q) continue;
          mb.freezeWrites();
          const out = mb.forward(pooled);
          const pred = outputHead.forward(out);
          if (argmax(pred) === q.target_class) correct++;
          total++;
          continue;
        }

        mb.freezeWrites();
        mb.forward(pooled);
      }
    }
    return { correct, total };
  }

  const active = runPass(false);
  const frozen = runPass(true);

  const activeAcc = active.total ? active.correct / active.total : 0;
  const frozenAcc = frozen.total ? frozen.correct / frozen.total : 0;
  const memoryGain = activeAcc - frozenAcc;

  console.log(`[learned-write] queries=${active.total}`);
  console.log(`  activeAcc   = ${formatPct(activeAcc)} (${active.correct}/${active.total})`);
  console.log(`  frozenAcc   = ${formatPct(frozenAcc)} (${frozen.correct}/${frozen.total})`);
  console.log(`  memoryGain  = ${formatPct(memoryGain)}`);
  console.log(`  random      = ${formatPct(1 / OUTPUT_CLASSES)}`);

  if (Math.abs(memoryGain) < 0.01) {
    console.log(
      "WARNING: MemoryBank active == no-write baseline. " +
        "Learned writes are not contributing yet. " +
        "Run experiment with --smoke to train writeKeyKernel + writeValueKernel supervision."
    );
  } else if (memoryGain > 0.01) {
    console.log(`INFO: memoryGain=${formatPct(memoryGain)}. Memory is starting to contribute.`);
  }

  return {
    mode: "learned-write",
    episodes: Math.min(episodes.length, DIAGNOSTIC_EPISODES),
    queries: active.total,
    topSlotCorrect: 0,
    topSlotAcc: 0,
    topValueCorrect: 0,
    topValueAcc: 0,
    predCorrect: active.correct,
    predAcc: activeAcc,
    activeAcc,
    frozenAcc,
    memoryGain,
  };
}

// ─── Main ─────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const modeArg =
    args.find((a) => a.startsWith("--mode="))?.split("=")[1] ??
    (args.includes("--mode") ? args[args.indexOf("--mode") + 1] : null);
  const modes = modeArg ? [modeArg] : ["manual-read", "deterministic-write", "learned-write"];

  console.log("=".repeat(72));
  console.log("MemoryBank Episodic Retrieval Diagnostic");
  console.log("=".repeat(72));
  console.log(`Modes to run: ${modes.join(", ")}`);

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

  if (modes.includes("manual-read")) {
    results.push(diagManualRead());
  }

  const manual = results.find((r) => r.mode === "manual-read");
  const manualFailed = manual ? manual.topSlotAcc < 1.0 : false;

  if (modes.includes("deterministic-write")) {
    if (manualFailed && !modeArg) {
      console.log("\nSkipping deterministic-write because manual-read failed.");
    } else {
      results.push(diagDeterministicWrite(tokenizer, episodes));
    }
  }

  if (modes.includes("learned-write")) {
    results.push(diagLearnedWrite(tokenizer, episodes));
  }

  // ─── Summary ──────────────────────────────────────────────────────────────
  console.log("\n" + "=".repeat(72));
  console.log("DIAGNOSTIC SUMMARY");
  console.log("=".repeat(72));

  for (const r of results) {
    if (r.mode === "manual-read") {
      console.log(`  manual-read          topSlotAcc=${formatPct(r.topSlotAcc)} [${r.topSlotAcc >= 1.0 ? "PASS" : "FAIL"}]`);
    } else if (r.mode === "deterministic-write") {
      console.log(
        `  deterministic-write  topSlotAcc=${formatPct(r.topSlotAcc)} topValueAcc=${formatPct(r.topValueAcc)} predAcc=${formatPct(r.predAcc)}`
      );
    } else if (r.mode === "learned-write") {
      console.log(
        `  learned-write        activeAcc=${formatPct(r.activeAcc)} frozenAcc=${formatPct(r.frozenAcc)} memGain=${formatPct(r.memoryGain)}`
      );
    }
  }

  // Gate checks
  if (manual && manual.topSlotAcc < 1.0) {
    throw new Error("DIAGNOSTIC GATE FAILED: manual-read failed. Fix read path before training.");
  }

  const det = results.find((r) => r.mode === "deterministic-write");
  if (det && det.topSlotAcc < 0.8) {
    console.log(
      "\nWARNING: deterministic-write topSlotAcc < 80%." +
        "\nThe canonical key-space fix is not yet achieving reliable retrieval." +
        "\nDo not trust learned-write training results until this passes."
    );
  }
}

main().catch((err) => {
  console.error("\n[FATAL] Diagnostic failed.");
  console.error(err);
  process.exitCode = 1;
});
