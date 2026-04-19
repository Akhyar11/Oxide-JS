import { performance } from "perf_hooks";
import { writeFileSync } from "fs";
import mj from "../../src/math";
import Matrix from "../../src/matrix";
import { Transformers } from "../../src/models";
import MultiHeadAttention from "../../src/layers/multiHeadAttention";
import {
  isNativeAvailable,
  multiHeadAttentionBackwardNative,
  multiHeadAttentionForwardNative,
  setForceDisableNative,
} from "../../src/math/rust_backend";

type ScenarioResult = {
  name: string;
  group: string;
  warmup: number;
  iterations: number;
  totalMs: number;
  avgMs: number;
  minMs: number;
  maxMs: number;
  iterPerSec: number;
  memDeltaHeapMB: number;
  memDeltaRssMB: number;
  notes?: string;
};

type BenchConfig = {
  warmup: number;
  iterations: number;
};

const OUTPUT_JSON = "benchmark-results/synthetic_patch_benchmark.latest.json";

function rng(seed = 1337): () => number {
  let s = seed >>> 0;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return ((s >>> 0) % 1_000_000) / 1_000_000;
  };
}

function makeMatrix(rows: number, cols: number, seed: number): Matrix {
  const rand = rng(seed);
  const data = new Float32Array(rows * cols);
  for (let i = 0; i < data.length; i++) {
    data[i] = (rand() - 0.5) * 2;
  }
  return Matrix.fromFlat(data, [rows, cols]);
}

function makeTokenMatrix(seqLen: number, batchSize: number, vocabSize: number, seed: number): Matrix {
  const rand = rng(seed);
  const data = new Float32Array(seqLen * batchSize);
  for (let i = 0; i < data.length; i++) {
    data[i] = Math.floor(rand() * vocabSize);
  }
  return Matrix.fromFlat(data, [seqLen, batchSize]);
}

function makeLabelMatrix(batchSize: number, vocabSize: number, seed: number): Matrix {
  const rand = rng(seed);
  const data = new Float32Array(batchSize);
  for (let i = 0; i < batchSize; i++) {
    data[i] = Math.floor(rand() * vocabSize);
  }
  return Matrix.fromFlat(data, [1, batchSize]);
}

function checksum(m: Matrix): number {
  let acc = 0;
  const stride = Math.max(1, Math.floor(m._data.length / 97));
  for (let i = 0; i < m._data.length; i += stride) {
    acc += m._data[i] * 0.0001;
  }
  return Number(acc.toFixed(6));
}

function forceGcIfPossible(): void {
  const gc = (globalThis as { gc?: () => void }).gc;
  if (gc) gc();
}

function runScenario(
  name: string,
  group: string,
  config: BenchConfig,
  fn: () => void,
  notes?: string
): ScenarioResult {
  for (let i = 0; i < config.warmup; i++) fn();
  forceGcIfPossible();
  const memStart = process.memoryUsage();

  const samples: number[] = [];
  const totalStart = performance.now();
  for (let i = 0; i < config.iterations; i++) {
    const t0 = performance.now();
    fn();
    const t1 = performance.now();
    samples.push(t1 - t0);
  }
  const totalEnd = performance.now();

  forceGcIfPossible();
  const memEnd = process.memoryUsage();

  const totalMs = totalEnd - totalStart;
  const minMs = Math.min(...samples);
  const maxMs = Math.max(...samples);
  const avgMs = totalMs / config.iterations;

  return {
    name,
    group,
    warmup: config.warmup,
    iterations: config.iterations,
    totalMs,
    avgMs,
    minMs,
    maxMs,
    iterPerSec: 1000 / avgMs,
    memDeltaHeapMB: (memEnd.heapUsed - memStart.heapUsed) / (1024 * 1024),
    memDeltaRssMB: (memEnd.rss - memStart.rss) / (1024 * 1024),
    notes,
  };
}

function formatRows(results: ScenarioResult[]): string {
  const header = "| Group | Scenario | Avg (ms) | Min (ms) | Max (ms) | Iter/s | ΔHeap (MB) | ΔRSS (MB) | Notes |\n|---|---|---:|---:|---:|---:|---:|---:|---|";
  const body = results
    .map((r) => {
      return `| ${r.group} | ${r.name} | ${r.avgMs.toFixed(3)} | ${r.minMs.toFixed(3)} | ${r.maxMs.toFixed(3)} | ${r.iterPerSec.toFixed(2)} | ${r.memDeltaHeapMB.toFixed(3)} | ${r.memDeltaRssMB.toFixed(3)} | ${r.notes ?? ""} |`;
    })
    .join("\n");
  return `${header}\n${body}`;
}

function benchmarkAddSub(results: ScenarioResult[]) {
  const shapes: Array<[string, number, number]> = [
    ["small", 64, 64],
    ["medium", 128, 1024],
    ["large", 256, 4096],
    ["transformer-like", 128, 4096],
  ];

  for (const [label, rows, cols] of shapes) {
    const a = makeMatrix(rows, cols, 10 + rows);
    const b = makeMatrix(rows, cols, 20 + cols);
    const out = mj.zeros([rows, cols]);

    const cfgAlloc: BenchConfig = { warmup: 40, iterations: 140 };
    const cfgReuse: BenchConfig = { warmup: 40, iterations: 180 };

    results.push(
      runScenario(`add alloc ${label} [${rows}x${cols}]`, "add/sub", cfgAlloc, () => {
        const r = mj.add(a, b);
        if (r._data[0] === Number.MAX_SAFE_INTEGER) throw new Error("dead-code");
      })
    );
    results.push(
      runScenario(`addInto reuse ${label} [${rows}x${cols}]`, "add/sub", cfgReuse, () => {
        const r = mj.addInto(a, b, out);
        if (r._data[0] === Number.MAX_SAFE_INTEGER) throw new Error("dead-code");
      })
    );

    results.push(
      runScenario(`sub alloc ${label} [${rows}x${cols}]`, "add/sub", cfgAlloc, () => {
        const r = mj.sub(a, b);
        if (r._data[0] === Number.MAX_SAFE_INTEGER) throw new Error("dead-code");
      })
    );
    results.push(
      runScenario(`subInto reuse ${label} [${rows}x${cols}]`, "add/sub", cfgReuse, () => {
        const r = mj.subInto(a, b, out);
        if (r._data[0] === Number.MAX_SAFE_INTEGER) throw new Error("dead-code");
      })
    );
  }
}

function benchmarkResidual(results: ScenarioResult[]) {
  const units = 128;
  const seqLen = 256;
  const batch = 16;
  const cols = seqLen * batch;

  const h = makeMatrix(units, cols, 111);
  const attn = makeMatrix(units, cols, 112);
  const ffn = makeMatrix(units, cols, 113);
  const res2Err = makeMatrix(units, cols, 114);
  const errLn2 = makeMatrix(units, cols, 115);
  const errLn1 = makeMatrix(units, cols, 116);

  const res1 = mj.zeros([units, cols]);
  const res2 = mj.zeros([units, cols]);
  const res1Err = mj.zeros([units, cols]);
  const peErr = mj.zeros([units, cols]);

  const cfg = { warmup: 30, iterations: 100 };

  results.push(
    runScenario("residual allocation-heavy", "residual", cfg, () => {
      const r1 = mj.add(h, attn);
      const r2 = mj.add(r1, ffn);
      const e1 = mj.add(res2Err, errLn2);
      const e2 = mj.add(e1, errLn1);
      if (checksum(r2) + checksum(e2) === Number.MIN_VALUE) throw new Error("dead-code");
    })
  );

  results.push(
    runScenario("residual reusable-buffer", "residual", cfg, () => {
      mj.addInto(h, attn, res1);
      mj.addInto(res1, ffn, res2);
      mj.addInto(res2Err, errLn2, res1Err);
      mj.addInto(res1Err, errLn1, peErr);
      if (checksum(res2) + checksum(peErr) === Number.MIN_VALUE) throw new Error("dead-code");
    })
  );
}

function benchmarkTransformers(results: ScenarioResult[]) {
  const configs = [
    { seqLen: 128, batchSize: 8, units: 128, heads: 8, vocabSize: 2048 },
    { seqLen: 256, batchSize: 16, units: 128, heads: 8, vocabSize: 2048 },
    { seqLen: 512, batchSize: 8, units: 128, heads: 8, vocabSize: 2048 },
  ];

  for (const cfg of configs) {
    const model = new Transformers({
      units: cfg.units,
      seqLen: cfg.seqLen,
      vocabSize: cfg.vocabSize,
      heads: cfg.heads,
      alpha: 1e-4,
      dropoutRate: 0,
      padTokenId: 0,
    });

    model.compile({ alpha: 1e-4, optimizer: "adam", error: "softmaxCrossEntropy" });

    const x = makeTokenMatrix(cfg.seqLen, cfg.batchSize, cfg.vocabSize, cfg.seqLen + cfg.batchSize);
    const y = makeLabelMatrix(cfg.batchSize, cfg.vocabSize, cfg.seqLen + cfg.batchSize + 7);

    results.push(
      runScenario(
        `transformer forward [seq=${cfg.seqLen},batch=${cfg.batchSize}]`,
        "transformer",
        { warmup: 3, iterations: 12 },
        () => {
          const out = model.forward(x);
          if (out._shape[1] !== cfg.batchSize) throw new Error("forward shape mismatch");
        }
      )
    );

    results.push(
      runScenario(
        `transformer backward [seq=${cfg.seqLen},batch=${cfg.batchSize}]`,
        "transformer",
        { warmup: 2, iterations: 8 },
        () => {
          model.forward(x);
          model.backward(y);
          if (model.loss < 0) throw new Error("invalid loss");
        }
      )
    );

    results.push(
      runScenario(
        `training-step forward+backward [seq=${cfg.seqLen},batch=${cfg.batchSize}]`,
        "training-step",
        { warmup: 2, iterations: 8 },
        () => {
          model.forward(x);
          model.backward(y);
        }
      )
    );
  }
}

function benchmarkMhaNativeImpact(results: ScenarioResult[]) {
  if (!isNativeAvailable()) {
    results.push({
      name: "native MHA not available",
      group: "mha-native",
      warmup: 0,
      iterations: 0,
      totalMs: 0,
      avgMs: 0,
      minMs: 0,
      maxMs: 0,
      iterPerSec: 0,
      memDeltaHeapMB: 0,
      memDeltaRssMB: 0,
      notes: "skip: native backend unavailable",
    });
    return;
  }

  const units = 128;
  const heads = 8;
  const headUnits = units / heads;
  const seqLen = 128;
  const batchSize = 8;
  const cols = seqLen * batchSize;
  const scale = 1 / Math.sqrt(headUnits);

  const q = makeMatrix(units, cols, 900)._data;
  const k = makeMatrix(units, cols, 901)._data;
  const v = makeMatrix(units, cols, 902)._data;
  const dOut = makeMatrix(units, cols, 903)._data;
  const padMask = new Array<boolean>(cols).fill(false);

  const out = new Float32Array(units * cols);
  const attention = new Float32Array(heads * batchSize * seqLen * seqLen);
  const dQ = new Float32Array(units * cols);
  const dK = new Float32Array(units * cols);
  const dV = new Float32Array(units * cols);

  const cfg = { warmup: 10, iterations: 60 };

  results.push(
    runScenario(
      "native MHA forward/backward (current no pre-zero)",
      "mha-native",
      cfg,
      () => {
        multiHeadAttentionForwardNative(q, k, v, padMask, heads, headUnits, seqLen, batchSize, scale, out, attention);
        multiHeadAttentionBackwardNative(
          q,
          k,
          v,
          attention,
          dOut,
          padMask,
          heads,
          headUnits,
          seqLen,
          batchSize,
          scale,
          dQ,
          dK,
          dV
        );
      }
    )
  );

  results.push(
    runScenario(
      "native MHA emulated old zero-fill",
      "mha-native",
      cfg,
      () => {
        out.fill(0);
        attention.fill(0);
        dQ.fill(0);
        dK.fill(0);
        dV.fill(0);
        multiHeadAttentionForwardNative(q, k, v, padMask, heads, headUnits, seqLen, batchSize, scale, out, attention);
        multiHeadAttentionBackwardNative(
          q,
          k,
          v,
          attention,
          dOut,
          padMask,
          heads,
          headUnits,
          seqLen,
          batchSize,
          scale,
          dQ,
          dK,
          dV
        );
      },
      "old behavior emulation by zeroing output buffers each iter"
    )
  );

  setForceDisableNative(true);
  const mhaJs = new MultiHeadAttention({ units, heads, seqLen, alpha: 1e-4, status: "train" });
  const x = makeMatrix(units, cols, 911);
  const err = makeMatrix(units, cols, 912);
  mhaJs.compile({ alpha: 1e-4, optimizer: "sgd" });
  results.push(
    runScenario(
      "MHA JS fallback forward/backward",
      "mha-native",
      { warmup: 5, iterations: 20 },
      () => {
        mhaJs.forward(x);
        mhaJs.backward(mj.matrix([[]]), err);
      }
    )
  );
  setForceDisableNative(false);
}

function main() {
  const startedAt = new Date().toISOString();
  const gcExposed = typeof (globalThis as { gc?: () => void }).gc === "function";

  const results: ScenarioResult[] = [];
  benchmarkAddSub(results);
  benchmarkResidual(results);
  benchmarkTransformers(results);
  benchmarkMhaNativeImpact(results);

  const endedAt = new Date().toISOString();
  const markdownTable = formatRows(results);

  console.log("# Synthetic Patch Benchmark");
  console.log(`start=${startedAt}`);
  console.log(`end=${endedAt}`);
  console.log(`nativeAvailable=${isNativeAvailable()}`);
  console.log(`gcExposed=${gcExposed}`);
  console.log(markdownTable);

  const payload = {
    meta: {
      startedAt,
      endedAt,
      nativeAvailable: isNativeAvailable(),
      gcExposed,
      nodeVersion: process.version,
      platform: process.platform,
      arch: process.arch,
    },
    results,
    markdownTable,
  };

  writeFileSync(OUTPUT_JSON, JSON.stringify(payload, null, 2), "utf-8");
  console.log(`Saved JSON result to ${OUTPUT_JSON}`);
}

main();
