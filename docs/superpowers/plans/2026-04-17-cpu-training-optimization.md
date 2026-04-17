# CPU Training Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Menurunkan waktu training epoch CPU-only secara drastis dengan memotong overhead alokasi, menambah profiler granular, dan memindahkan hot kernels transformer ke jalur native Rust yang lebih efisien.

**Architecture:** Fase pertama memasang benchmark dan profiler repeatable lalu membersihkan hot path `Float32` serta buffer reuse. Fase berikutnya memfokuskan rewrite attention, layernorm, softmax-crossentropy, dan optimizer agar compute berat berjalan di Rust dengan crossing N-API minimum dan paralelisme `rayon`.

**Tech Stack:** TypeScript, Node.js, ts-node, Rust, N-API, rayon, matrixmultiply

---

### Task 1: Freeze Baseline Benchmarks

**Files:**
- Modify: `project/generative-bot/main.ts`
- Create: `test/benchmark_training_step.ts`
- Create: `test/benchmark_wikipedia_subset.ts`

- [ ] **Step 1: Write the failing benchmark harness for a repeatable synthetic step**

```ts
// test/benchmark_training_step.ts
import { performance } from "perf_hooks";
import mj from "../src/math";
import Transformers from "../src/models/transformers";

const model = new Transformers({
  units: 64,
  seqLen: 128,
  vocabSize: 20_000,
  heads: 8,
  alpha: 1e-5,
  padTokenId: 0,
});

model.compile({ alpha: 1e-5, optimizer: "adam", error: "softmaxCrossEntropy" });
for (const layer of model.layers) {
  if (layer.name === "dropout layer") layer.status = "train";
  if (typeof (layer as any).compile === "function") (layer as any).compile({ alpha: 1e-5 });
}

const x = mj.zeros([128, 64]);
const y = mj.zeros([1, 64]);
for (let i = 0; i < 64; i++) y._data[i] = i % 100;

const warmup = () => {
  model.forward(x);
  model.backward(y);
};

warmup();
const start = performance.now();
for (let i = 0; i < 20; i++) {
  model.forward(x);
  model.backward(y);
}
const totalMs = performance.now() - start;
console.log(JSON.stringify({
  benchmark: "training-step",
  iterations: 20,
  msPerBatch: totalMs / 20,
  batchesPerSec: 1000 / (totalMs / 20),
  samplesPerSec: 64 * 1000 / (totalMs / 20),
}, null, 2));
```

- [ ] **Step 2: Run benchmark to capture current baseline**

Run: `node -r ts-node/register/transpile-only test/benchmark_training_step.ts`
Expected: JSON output with `msPerBatch`, `batchesPerSec`, and `samplesPerSec`

- [ ] **Step 3: Write the failing Wikipedia subset benchmark**

```ts
// test/benchmark_wikipedia_subset.ts
import fs from "fs";
import path from "path";
import { performance } from "perf_hooks";
import { BPETokenizer } from "../src/tokenizer";
import Transformers from "../src/models/transformers";
import mj from "../src/math";

const tokenizer = BPETokenizer.load(path.join(__dirname, "..", "project", "generative-bot", "dataset", "generative_vocab.json"));
const lines = fs.readFileSync(path.join(__dirname, "..", "dataset", "wikipedia_belum_normalisasi.txt"), "utf-8")
  .toLowerCase()
  .split("\n")
  .filter((line) => line.trim().length > 0)
  .slice(0, 256);

const seqLen = 128;
const batchSize = 64;
const pairs: { xData: Float32Array; target: number }[] = [];
for (const line of lines) {
  const tokens = tokenizer.encode(line);
  for (let i = 0; i < tokens.length - 1; i++) {
    const start = Math.max(0, i - seqLen + 1);
    const ctxLen = i - start + 1;
    const xData = new Float32Array(seqLen);
    xData.fill(tokenizer.getPadId());
    const offset = seqLen - ctxLen;
    for (let j = 0; j < ctxLen; j++) xData[offset + j] = tokens[start + j];
    pairs.push({ xData, target: tokens[i + 1] });
  }
}

const model = new Transformers({
  units: 64,
  seqLen,
  vocabSize: tokenizer.getVocabSize(),
  heads: 8,
  alpha: 1e-5,
  padTokenId: tokenizer.getPadId(),
});
model.compile({ alpha: 1e-5, optimizer: "adam", error: "softmaxCrossEntropy" });

const batchX = mj.zeros([seqLen, batchSize]);
const batchY = mj.zeros([1, batchSize]);
const start = performance.now();
let processed = 0;
for (let i = 0; i < pairs.length; i += batchSize) {
  const actual = Math.min(batchSize, pairs.length - i);
  batchX._data.fill(0);
  batchY._data.fill(0);
  for (let b = 0; b < actual; b++) {
    const pair = pairs[i + b];
    for (let row = 0; row < seqLen; row++) batchX._data[row * batchSize + b] = pair.xData[row];
    batchY._data[b] = pair.target;
  }
  model.forward(batchX);
  model.backward(batchY);
  processed += actual;
}
const elapsedMs = performance.now() - start;
console.log(JSON.stringify({
  benchmark: "wikipedia-subset",
  samples: processed,
  msPerSample: elapsedMs / processed,
  samplesPerSec: processed * 1000 / elapsedMs,
}, null, 2));
```

- [ ] **Step 4: Run subset benchmark and confirm the current throughput**

Run: `node -r ts-node/register/transpile-only test/benchmark_wikipedia_subset.ts`
Expected: JSON output with stable `samplesPerSec`

- [ ] **Step 5: Commit**

```bash
git add test/benchmark_training_step.ts test/benchmark_wikipedia_subset.ts project/generative-bot/main.ts
git commit -m "test: add cpu training baseline benchmarks"
```

### Task 2: Install Granular Profiler

**Files:**
- Modify: `src/utils/profiler.ts`
- Modify: `project/generative-bot/main.ts`
- Modify: `src/models/transformers.ts`
- Modify: `src/layers/dense.ts`
- Modify: `src/layers/multiHeadAttention.ts`
- Modify: `src/layers/layerNormalization.ts`

- [ ] **Step 1: Write the failing profiler API with scoped timing and counters**

```ts
// src/utils/profiler.ts
type ProfilerStat = { total: number; count: number; max: number };

export class Profiler {
  private static stats = new Map<string, ProfilerStat>();
  private static counters = new Map<string, number>();

  static scope<T>(label: string, fn: () => T): T {
    const start = performance.now();
    try {
      return fn();
    } finally {
      const duration = performance.now() - start;
      const stat = this.stats.get(label) ?? { total: 0, count: 0, max: 0 };
      stat.total += duration;
      stat.count += 1;
      stat.max = Math.max(stat.max, duration);
      this.stats.set(label, stat);
    }
  }

  static increment(label: string, delta = 1): void {
    this.counters.set(label, (this.counters.get(label) ?? 0) + delta);
  }

  static reset(): void {
    this.stats.clear();
    this.counters.clear();
  }
}
```

- [ ] **Step 2: Run TypeScript entrypoint and verify current code fails because report output is incomplete**

Run: `node -r ts-node/register/transpile-only project/generative-bot/main.ts`
Expected: output still lacks layer-level timing, proving instrumentation is not wired yet

- [ ] **Step 3: Wire layer-level scopes and batch-level summaries**

```ts
// in src/models/transformers.ts
const xLn1 = Profiler.scope("layernorm1.forward", () => this.ln1.forward(h));
const xMhaOut = Profiler.scope("mha.forward", () => this.mha.forward(xLn1));
const xFfn1Out = Profiler.scope("ffn1.forward", () => this.ffn1.forward(xLn2));
```

```ts
// in project/generative-bot/main.ts
Profiler.scope("batch.forward", () => model.forward(currentBatchX));
Profiler.scope("batch.backward", () => model.backward(currentBatchY));
```

- [ ] **Step 4: Run the synthetic benchmark and verify profiler reports top hotspots**

Run: `node -r ts-node/register/transpile-only test/benchmark_training_step.ts`
Expected: output now includes identifiable layer hotspot summaries

- [ ] **Step 5: Commit**

```bash
git add src/utils/profiler.ts project/generative-bot/main.ts src/models/transformers.ts src/layers/dense.ts src/layers/multiHeadAttention.ts src/layers/layerNormalization.ts
git commit -m "feat: add granular cpu training profiler"
```

### Task 3: Unify Hot Path Dtypes to Float32

**Files:**
- Modify: `src/layers/layerNormalization.ts`
- Modify: `src/math/add.ts`
- Modify: `src/math/dotProduct.ts`
- Modify: `src/math/sumAxis.ts`
- Modify: `src/cost/softmaxCrossEntropy.ts`
- Modify: `src/math/absm.ts`

- [ ] **Step 1: Write the failing dtype consistency check**

```ts
// add to test/benchmark_training_step.ts temporarily or a dedicated check
const output = model.forward(x);
if (!(output._data instanceof Float32Array)) {
  throw new Error("Expected Float32Array output buffer");
}
```

- [ ] **Step 2: Run the check to confirm some hot paths still produce Float64-backed buffers**

Run: `node -r ts-node/register/transpile-only test/benchmark_training_step.ts`
Expected: failure or mixed-type buffers discovered in runtime inspection

- [ ] **Step 3: Replace all hot-path `Float32Array` allocations with `Float32Array`**

```ts
// src/math/add.ts
const resultData = new Float32Array(a._data.length);
```

```ts
// src/math/dotProduct.ts
const resultData = out ? out._data : new Float32Array(aRows * bCols);
```

```ts
// src/cost/softmaxCrossEntropy.ts
const gradData = new Float32Array(pData);
```

- [ ] **Step 4: Run benchmark again and verify the dtype check passes**

Run: `node -r ts-node/register/transpile-only test/benchmark_training_step.ts`
Expected: no dtype failure, benchmark still emits runtime numbers

- [ ] **Step 5: Commit**

```bash
git add src/layers/layerNormalization.ts src/math/add.ts src/math/dotProduct.ts src/math/sumAxis.ts src/cost/softmaxCrossEntropy.ts src/math/absm.ts
git commit -m "perf: unify hot path buffers to float32"
```

### Task 4: Eliminate Per-Batch Allocation in Transformer Core

**Files:**
- Modify: `src/models/transformers.ts`
- Modify: `src/layers/dense.ts`
- Modify: `src/layers/layerNormalization.ts`

- [ ] **Step 1: Write a failing allocation counter for batch-resized buffers**

```ts
// src/models/transformers.ts
private res2ErrBuffer: Matrix = mj.zeros([this.embedding.embeddingDim, this.pe.maxSeqLen]);
private ensureBatchBuffers(batchSize: number): void {
  const units = this.embedding.embeddingDim;
  const totalTokens = this.pe.maxSeqLen * batchSize;
  if (this.res2ErrBuffer._shape[0] !== units || this.res2ErrBuffer._shape[1] !== totalTokens) {
    Profiler.increment("buffer.resize.transformers.res2Err");
    this.res2ErrBuffer = mj.zeros([units, totalTokens]);
  } else {
    this.res2ErrBuffer._data.fill(0);
  }
}
```

- [ ] **Step 2: Run benchmark and observe resize counters incrementing repeatedly**

Run: `node -r ts-node/register/transpile-only test/benchmark_training_step.ts`
Expected: resize counters show avoidable allocation churn

- [ ] **Step 3: Reuse sequence error, layernorm temp buffers, and dense error buffers**

```ts
// in backward()
this.ensureBatchBuffers(batchSize);
const res2Err = this.res2ErrBuffer;
res2Err._data.fill(0);
```

```ts
// in layernorm forward()
if (this.normalized._shape[0] !== rows || this.normalized._shape[1] !== cols) {
  this.normalized = Matrix.fromFlat(new Float32Array(rows * cols), [rows, cols]);
}
```

- [ ] **Step 4: Run baseline benchmarks and verify resize counters flatten after warmup**

Run: `node -r ts-node/register/transpile-only test/benchmark_training_step.ts`
Expected: allocation counters only rise on initial shape setup

- [ ] **Step 5: Commit**

```bash
git add src/models/transformers.ts src/layers/dense.ts src/layers/layerNormalization.ts
git commit -m "perf: reuse transformer and layer buffers across batches"
```

### Task 5: Remove Object Churn from MultiHeadAttention

**Files:**
- Modify: `src/layers/multiHeadAttention.ts`
- Test: `test/benchmark_training_step.ts`

- [ ] **Step 1: Add a failing counter around `Matrix.fromFlat`-driven attention views**

```ts
// src/layers/multiHeadAttention.ts
Profiler.increment("mha.view.alloc");
const qSample = Matrix.fromFlat(/* current subarray logic */);
```

- [ ] **Step 2: Run benchmark and confirm the counter explodes with head × batch iterations**

Run: `node -r ts-node/register/transpile-only test/benchmark_training_step.ts`
Expected: very high `mha.view.alloc` count

- [ ] **Step 3: Replace per-iteration temporary views with cached reusable views or direct index math**

```ts
// direction of change inside MultiHeadAttention
private headSampleViews: {
  q: Matrix[];
  k: Matrix[];
  v: Matrix[];
  out: Matrix[];
  dQ: Matrix[];
  dK: Matrix[];
  dV: Matrix[];
}[] = [];
```

```ts
// create once inside ensureSequenceBuffersForBatch()
const view = Matrix.fromFlat(
  this.Q._data.subarray(start, end),
  [this.headUnits, this.seqLen],
);
```

- [ ] **Step 4: Run benchmark and verify `mha.view.alloc` stops growing in steady state**

Run: `node -r ts-node/register/transpile-only test/benchmark_training_step.ts`
Expected: view allocation count remains near-zero after warmup, `msPerBatch` drops materially

- [ ] **Step 5: Commit**

```bash
git add src/layers/multiHeadAttention.ts test/benchmark_training_step.ts
git commit -m "perf: cache attention views and remove hot path object churn"
```

### Task 6: Add Native Softmax-CrossEntropy Fusion

**Files:**
- Modify: `src-rust/src/lib.rs`
- Modify: `src/math/rust_backend.ts`
- Modify: `src/cost/softmaxCrossEntropy.ts`
- Test: `test/benchmark_training_step.ts`

- [ ] **Step 1: Write a failing native binding contract**

```ts
// src/math/rust_backend.ts
export const softmaxCrossEntropyNative = (
  logits: Float32Array,
  target: Float32Array,
  rows: number,
  cols: number,
  outProb: Float32Array,
  outGrad: Float32Array
): number => {
  if (!native) throw new Error("Native backend not available");
  return native.softmaxCrossEntropyNative(logits, target, rows, cols, outProb, outGrad);
};
```

- [ ] **Step 2: Run build to confirm the binding does not exist yet**

Run: `npm run build:rust`
Expected: TypeScript or runtime still lacks the exported fused function

- [ ] **Step 3: Implement the fused kernel in Rust and route cost computation through it**

```rust
#[napi]
pub fn softmax_cross_entropy_native(
    logits: Float32Array,
    target: Float32Array,
    rows: u32,
    cols: u32,
    mut out_prob: Float32Array,
    mut out_grad: Float32Array,
) -> f64 {
    // compute stable softmax and gradient in one pass
    0.0
}
```

- [ ] **Step 4: Rebuild native module and rerun benchmark**

Run: `npm run build:rust`
Expected: Rust build succeeds and Node benchmark uses the fused path without throwing

- [ ] **Step 5: Commit**

```bash
git add src-rust/src/lib.rs src/math/rust_backend.ts src/cost/softmaxCrossEntropy.ts
git commit -m "perf: fuse softmax cross entropy in native backend"
```

### Task 7: Add Native LayerNorm Forward and Backward

**Files:**
- Modify: `src-rust/src/lib.rs`
- Modify: `src/math/rust_backend.ts`
- Modify: `src/layers/layerNormalization.ts`

- [ ] **Step 1: Write the failing backward binding API**

```ts
export const layerNormBackwardNative = (
  err: Float32Array,
  norm: Float32Array,
  gamma: Float32Array,
  std: Float32Array,
  rows: number,
  cols: number,
  outDx: Float32Array,
  outDGamma: Float32Array,
  outDBeta: Float32Array
): void => {
  if (!native) throw new Error("Native backend not available");
  native.layerNormBackwardNativeInto(err, norm, gamma, std, rows, cols, outDx, outDGamma, outDBeta);
};
```

- [ ] **Step 2: Run benchmark or quick script to confirm backward is still falling back to TypeScript**

Run: `node -r ts-node/register/transpile-only test/benchmark_training_step.ts`
Expected: profiler shows `layernorm.backward` dominated by TS path

- [ ] **Step 3: Implement native backward and route layer normalization fully through reusable Float32 buffers**

```ts
// src/layers/layerNormalization.ts
layerNormBackwardNative(
  errData,
  this.normalized._data,
  this.gamma._data,
  this.std._data,
  rows,
  cols,
  this.dxBuffer._data,
  this.dGammaBuffer._data,
  this.dBetaBuffer._data,
);
```

- [ ] **Step 4: Rebuild native module and validate a short training run**

Run: `npm run build:rust`
Expected: build passes and a short benchmark run completes with finite loss

- [ ] **Step 5: Commit**

```bash
git add src-rust/src/lib.rs src/math/rust_backend.ts src/layers/layerNormalization.ts
git commit -m "perf: move layernorm forward and backward to native path"
```

### Task 8: Optimize Adam Update and Native Counters

**Files:**
- Modify: `src-rust/src/lib.rs`
- Modify: `src/math/rust_backend.ts`
- Modify: `src/optimizer/adam.ts`

- [ ] **Step 1: Add a failing counter for optimizer hot path**

```ts
Profiler.increment("adam.calculate.calls");
```

- [ ] **Step 2: Run synthetic benchmark to confirm optimizer is called per parameter tensor as expected**

Run: `node -r ts-node/register/transpile-only test/benchmark_training_step.ts`
Expected: optimizer counters rise rapidly

- [ ] **Step 3: Extend native Adam path with counter hooks and ensure update buffer is reused**

```rust
#[napi]
pub fn adam_update_native(/* existing args */) {
    // update in place and increment native perf counters if enabled
}
```

- [ ] **Step 4: Run benchmark and verify optimizer stays native-only in steady state**

Run: `node -r ts-node/register/transpile-only test/benchmark_training_step.ts`
Expected: profiler shows lower `adam` share and no extra allocations

- [ ] **Step 5: Commit**

```bash
git add src-rust/src/lib.rs src/math/rust_backend.ts src/optimizer/adam.ts
git commit -m "perf: streamline native adam updates and counters"
```

### Task 9: Introduce Batched Native Attention Path

**Files:**
- Modify: `src-rust/src/lib.rs`
- Modify: `src/math/rust_backend.ts`
- Modify: `src/layers/multiHeadAttention.ts`
- Test: `test/benchmark_training_step.ts`

- [ ] **Step 1: Write the failing binding for batched attention score/value**

```ts
export const attentionForwardNative = (
  q: Float32Array,
  k: Float32Array,
  v: Float32Array,
  padMask: Uint8Array,
  heads: number,
  headUnits: number,
  seqLen: number,
  batchSize: number,
  out: Float32Array
): void => {
  if (!native) throw new Error("Native backend not available");
  native.attentionForwardNativeInto(q, k, v, padMask, heads, headUnits, seqLen, batchSize, out);
};
```

- [ ] **Step 2: Build and confirm the native function is still missing**

Run: `npm run build:rust`
Expected: missing symbol until the Rust side is implemented

- [ ] **Step 3: Implement native batched attention with rayon-parallel loops**

```rust
#[napi]
pub fn attention_forward_native_into(
    q: Float32Array,
    k: Float32Array,
    v: Float32Array,
    pad_mask: Uint8Array,
    heads: u32,
    head_units: u32,
    seq_len: u32,
    batch_size: u32,
    mut out: Float32Array,
) {
    // batch/head parallel attention compute
}
```

- [ ] **Step 4: Route MultiHeadAttention through the native batched path and rerun benchmark**

Run: `npm run build:rust && node -r ts-node/register/transpile-only test/benchmark_training_step.ts`
Expected: significant `msPerBatch` drop and profiler hotspot shifts away from TS attention loops

- [ ] **Step 5: Commit**

```bash
git add src-rust/src/lib.rs src/math/rust_backend.ts src/layers/multiHeadAttention.ts
git commit -m "perf: add batched native attention path"
```

### Task 10: Verify End-to-End Throughput and Epoch Estimate

**Files:**
- Modify: `project/generative-bot/main.ts`
- Modify: `test/benchmark_training_step.ts`
- Modify: `test/benchmark_wikipedia_subset.ts`

- [ ] **Step 1: Add final reporting for epoch estimate and CPU throughput**

```ts
console.log(JSON.stringify({
  msPerBatch,
  samplesPerSec,
  tokensPerSec,
  estimatedEpochMinutes: estimatedPairs / samplesPerSec / 60,
}, null, 2));
```

- [ ] **Step 2: Run the synthetic benchmark fresh**

Run: `node -r ts-node/register/transpile-only test/benchmark_training_step.ts`
Expected: final steady-state throughput JSON

- [ ] **Step 3: Run the Wikipedia subset benchmark fresh**

Run: `node -r ts-node/register/transpile-only test/benchmark_wikipedia_subset.ts`
Expected: final `samplesPerSec`, `tokensPerSec`, and `epoch estimate`

- [ ] **Step 4: Run a short real training smoke test**

Run: `node -r ts-node/register/transpile-only project/generative-bot/main.ts`
Expected: at least one epoch or an intentionally shortened debug run completes with finite loss and no crashes

- [ ] **Step 5: Commit**

```bash
git add project/generative-bot/main.ts test/benchmark_training_step.ts test/benchmark_wikipedia_subset.ts
git commit -m "perf: report final cpu training throughput estimates"
```
