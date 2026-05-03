import { readFileSync, writeFileSync } from "fs";
import { Optimzier, OptimzierType, StatusLayer, matrix2d } from "../@types/type";
import mj from "../math";
import Matrix from "../matrix";
import setOptimizer from "../utils/setOptimizer";
import { dotProductNative, isNativeAvailable } from "../math/rust_backend";

export type MemoryBankMode = "project" | "concat" | "add";
export type MemorySimilarity = "cosine" | "dot";
export type MemoryUpdateMode = "replace" | "merge" | "gated-merge";
export type MemoryWritePolicy = "empty-first" | "least-used" | "oldest" | "least-relevant";
export type MemoryPersistence = "session" | "manual";

export interface MemoryBankDebugReadSlot {
  slot: number;
  score: number;
  attn: number;
}

export interface MemoryBankDebugTrace {
  column: number;

  readSlots: MemoryBankDebugReadSlot[];

  writeCommitted: boolean;
  writeSlot: number;
  writeGate: number;

  memoryFilled: number[];
  memoryUsage: number[];
  memoryAge: number[];
}

export interface MemoryBankConfig {
  units?: number;
  memorySlots: number;
  memoryDim?: number;
  outputUnits?: number;

  mode?: MemoryBankMode;
  similarity?: MemorySimilarity;
  readTopK?: number;

  updateMode?: MemoryUpdateMode;
  writePolicy?: MemoryWritePolicy;
  writeThreshold?: number;

  persistence?: MemoryPersistence;
  resetOnInit?: boolean;
  writeEnabled?: boolean;
  trainablePolicy?: boolean;

  alpha?: number;
  optimizer?: Optimzier;
  clipGradient?: number | boolean;
  status?: StatusLayer;
}

export interface MemoryBankState {
  memoryKeys: number[][];
  memoryValues: number[][];
  memoryFilled: number[];
  memoryUsage: number[];
  memoryAge: number[];
  memoryStep: number;
  units: number;
  memoryDim: number;
  memorySlots: number;
}

interface ReadSlotCache {
  slot: number;
  attn: number;
  score: number;
  key: Float32Array;
  value: Float32Array;
}

interface ForwardCacheItem {
  xCol: Float32Array;
  q: Float32Array;
  read: Float32Array;
  need: number;
  needInput: Float32Array;
  context: Float32Array;
  combined: Float32Array;
  readSlots: ReadSlotCache[];
  writeGatePre: number;
  writeGate: number;
  writeCommitted: boolean;
  writeSlot: number;
  newKeyRaw: Float32Array;
  newKey: Float32Array;
  newValue: Float32Array;
}

export default class MemoryBank {
  name = "memory bank layer";

  units!: number;
  memorySlots: number;
  memoryDim!: number;
  outputUnits!: number;

  mode: MemoryBankMode;
  similarity: MemorySimilarity;
  readTopK: number;

  updateMode: MemoryUpdateMode;
  writePolicy: MemoryWritePolicy;
  writeThreshold: number;

  persistence: MemoryPersistence;
  resetOnInit: boolean;
  writeEnabled: boolean;
  trainablePolicy: boolean;

  alpha: number;
  optimizerName: Optimzier;
  clipGradient: number | boolean;
  status: StatusLayer;

  // trainable params used in exact gradients
  queryKernel!: Matrix; // [memoryDim, units]
  needKernel!: Matrix; // [1, units + memoryDim]
  outputKernel?: Matrix; // [outputUnits, units + memoryDim]
  outputBias?: Matrix; // [outputUnits, 1]

  // trainable params used in forward policy only (no exact BPTT through writes yet)
  writeKeyKernel!: Matrix; // [memoryDim, units]
  writeValueKernel!: Matrix; // [memoryDim, units]
  writeGateKernel!: Matrix; // [1, units + memoryDim]

  private optimizerQuery!: OptimzierType;
  private optimizerNeed!: OptimzierType;
  private optimizerOutput?: OptimzierType;
  private optimizerOutputBias?: OptimzierType;
  private optimizerWriteKey!: OptimzierType;
  private optimizerWriteValue!: OptimzierType;
  private optimizerWriteGate!: OptimzierType;

  // runtime state (not optimizer-trained)
  memoryKeys!: Matrix; // [memoryDim, memorySlots]
  memoryValues!: Matrix; // [memoryDim, memorySlots]
  memoryFilled!: Uint8Array;
  memoryUsage!: Float32Array;
  memoryAge!: Float32Array;
  memoryStep = 0;

  inputShape: [number, number] = [0, 1];
  outputShape: [number, number] = [0, 1];
  params = 0;

  private initialized = false;
  private writeFrozen = false;
  private cache: ForwardCacheItem[] = [];

  private debugTrace: MemoryBankDebugTrace[] = [];
  private lastWriteInfo: {
    committed: boolean;
    slot: number;
    writeGate: number;
    newKeyRaw: Float32Array;
    newKey: Float32Array;
    newValue: Float32Array;
    xCol: Float32Array;
  } | null = null;

  private configuredMemoryDim?: number;
  private configuredOutputUnits?: number;

  constructor(cfg: MemoryBankConfig) {
    this.assertPositiveInt(cfg.memorySlots, "memorySlots");
    if (cfg.units !== undefined) this.assertPositiveInt(cfg.units, "units");
    if (cfg.memoryDim !== undefined) this.assertPositiveInt(cfg.memoryDim, "memoryDim");
    if (cfg.outputUnits !== undefined) this.assertPositiveInt(cfg.outputUnits, "outputUnits");

    this.memorySlots = cfg.memorySlots;
    this.mode = cfg.mode ?? "project";
    this.similarity = cfg.similarity ?? "cosine";

    this.readTopK = cfg.readTopK ?? Math.min(4, this.memorySlots);
    this.assertPositiveInt(this.readTopK, "readTopK");
    if (this.readTopK > this.memorySlots) {
      throw new Error("MemoryBank: readTopK must be <= memorySlots");
    }

    this.updateMode = cfg.updateMode ?? "gated-merge";
    this.writePolicy = cfg.writePolicy ?? "empty-first";
    this.writeThreshold = cfg.writeThreshold ?? 0.5;

    this.persistence = cfg.persistence ?? "session";
    this.resetOnInit = cfg.resetOnInit ?? true;
    this.writeEnabled = cfg.writeEnabled ?? true;
    this.trainablePolicy = cfg.trainablePolicy ?? true;

    this.alpha = cfg.alpha ?? 0.01;
    this.optimizerName = cfg.optimizer ?? "adam";
    this.clipGradient = cfg.clipGradient ?? 5.0;
    this.status = cfg.status ?? "train";

    this.configuredMemoryDim = cfg.memoryDim;
    this.configuredOutputUnits = cfg.outputUnits;

    if (cfg.units !== undefined) {
      this.init(cfg.units, cfg.memoryDim ?? cfg.units, cfg.outputUnits ?? cfg.units);
    }
  }

  private assertPositiveInt(v: number, name: string): void {
    if (!Number.isInteger(v) || v <= 0) {
      throw new Error(`MemoryBank: ${name} must be positive integer`);
    }
  }

  private sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
  }

  private vectorDot(a: Float32Array, b: Float32Array): number {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }

  private l2Norm(v: Float32Array): number {
    let s = 0;
    for (let i = 0; i < v.length; i++) s += v[i] * v[i];
    return Math.sqrt(s);
  }

  private normalizeSafe(v: Float32Array): Float32Array {
    const out = new Float32Array(v.length);
    const n = this.l2Norm(v);
    if (!Number.isFinite(n) || n <= 1e-12) return out;
    const inv = 1 / n;
    for (let i = 0; i < v.length; i++) out[i] = v[i] * inv;
    return out;
  }

  private matVecMul(weight: Matrix, x: Float32Array): Float32Array {
    const [rows, cols] = weight._shape;
    if (cols !== x.length) {
      throw new Error(`MemoryBank: matVecMul shape mismatch [${rows},${cols}] x [${x.length}]`);
    }
    if (isNativeAvailable()) {
      const out = new Float32Array(rows);
      dotProductNative(weight._data, rows, cols, x, cols, 1, false, false, out);
      return out;
    }
    const out = new Float32Array(rows);
    const w = weight._data;
    for (let r = 0; r < rows; r++) {
      let sum = 0;
      const off = r * cols;
      for (let c = 0; c < cols; c++) sum += w[off + c] * x[c];
      out[r] = sum;
    }
    return out;
  }

  private matTVecMul(weight: Matrix, x: Float32Array): Float32Array {
    const [rows, cols] = weight._shape;
    if (rows !== x.length) {
      throw new Error(`MemoryBank: matTVecMul shape mismatch [${rows},${cols}]^T x [${x.length}]`);
    }
    if (isNativeAvailable()) {
      const out = new Float32Array(cols);
      dotProductNative(weight._data, rows, cols, x, rows, 1, true, false, out);
      return out;
    }
    const out = new Float32Array(cols);
    const w = weight._data;
    for (let c = 0; c < cols; c++) {
      let sum = 0;
      for (let r = 0; r < rows; r++) {
        sum += w[r * cols + c] * x[r];
      }
      out[c] = sum;
    }
    return out;
  }

  private addOuter(grad: Matrix, a: Float32Array, b: Float32Array, scale = 1): void {
    const cols = grad._shape[1];
    const g = grad._data;
    for (let i = 0; i < a.length; i++) {
      const ai = a[i] * scale;
      const off = i * cols;
      for (let j = 0; j < b.length; j++) g[off + j] += ai * b[j];
    }
  }

  private rowAsVector(m: Matrix, row: number): Float32Array {
    const cols = m._shape[1];
    const out = new Float32Array(cols);
    const off = row * cols;
    for (let j = 0; j < cols; j++) out[j] = m._data[off + j];
    return out;
  }

  private similarityScore(q: Float32Array, key: Float32Array): number {
    if (this.similarity === "dot") {
      return this.vectorDot(q, key) / Math.sqrt(this.memoryDim);
    }
    const nq = Math.max(this.l2Norm(q), 1e-12);
    const nk = Math.max(this.l2Norm(key), 1e-12);
    return this.vectorDot(q, key) / (nq * nk);
  }

  private cosineGradWrtQ(q: Float32Array, key: Float32Array): Float32Array {
    const out = new Float32Array(q.length);
    const nq = this.l2Norm(q);
    const nk = this.l2Norm(key);
    if (!Number.isFinite(nq) || !Number.isFinite(nk) || nq <= 1e-12 || nk <= 1e-12) {
      return out;
    }
    const dot = this.vectorDot(q, key);
    const invNqNk = 1 / (nq * nk);
    const coeffQ = dot / (nq * nq * nq * nk);
    for (let i = 0; i < q.length; i++) {
      out[i] = key[i] * invNqNk - q[i] * coeffQ;
    }
    return out;
  }

  private normalizeBackward(raw: Float32Array, gradOut: Float32Array): Float32Array {
    const out = new Float32Array(raw.length);
    const n = this.l2Norm(raw);
    if (!Number.isFinite(n) || n <= 1e-12) return out;
    const dot = this.vectorDot(raw, gradOut);
    const invN = 1 / n;
    const invN3 = invN * invN * invN;
    for (let i = 0; i < raw.length; i++) {
      out[i] = gradOut[i] * invN - raw[i] * dot * invN3;
    }
    return out;
  }

  private softmax(scores: number[]): number[] {
    if (scores.length === 0) return [];
    let maxv = -Infinity;
    for (const s of scores) if (s > maxv) maxv = s;
    let sum = 0;
    const exps = new Array(scores.length);
    for (let i = 0; i < scores.length; i++) {
      const e = Math.exp(scores[i] - maxv);
      exps[i] = e;
      sum += e;
    }
    if (!Number.isFinite(sum) || sum <= 0) {
      const u = 1 / scores.length;
      return new Array(scores.length).fill(u);
    }
    for (let i = 0; i < exps.length; i++) exps[i] /= sum;
    return exps;
  }

  private init(units: number, memoryDim: number, outputUnits: number): void {
    if (this.initialized) return;
    this.assertPositiveInt(units, "units");
    this.assertPositiveInt(memoryDim, "memoryDim");
    this.assertPositiveInt(outputUnits, "outputUnits");

    this.units = units;
    this.memoryDim = memoryDim;
    this.outputUnits = outputUnits;

    if (this.mode === "add") {
      if (this.memoryDim !== this.units || this.outputUnits !== this.units) {
        throw new Error("MemoryBank(mode=add) requires memoryDim===units and outputUnits===units");
      }
    }

    this.queryKernel = mj.xavier([this.memoryDim, this.units]);
    this.writeKeyKernel = mj.xavier([this.memoryDim, this.units]);
    this.writeValueKernel = mj.xavier([this.memoryDim, this.units]);
    this.needKernel = mj.xavier([1, this.units + this.memoryDim]);
    this.writeGateKernel = mj.xavier([1, this.units + this.memoryDim]);

    if (this.mode === "project") {
      this.outputKernel = mj.xavier([this.outputUnits, this.units + this.memoryDim]);
      this.outputBias = mj.zeros([this.outputUnits, 1]);
    }

    this.optimizerQuery = setOptimizer(this.optimizerName, this.queryKernel._shape, 1e-5);
    this.optimizerNeed = setOptimizer(this.optimizerName, this.needKernel._shape, 1e-5);
    this.optimizerWriteKey = setOptimizer(this.optimizerName, this.writeKeyKernel._shape, 1e-5);
    this.optimizerWriteValue = setOptimizer(this.optimizerName, this.writeValueKernel._shape, 1e-5);
    this.optimizerWriteGate = setOptimizer(this.optimizerName, this.writeGateKernel._shape, 1e-5);

    if (this.outputKernel) {
      this.optimizerOutput = setOptimizer(this.optimizerName, this.outputKernel._shape, 1e-5);
      this.optimizerOutputBias = setOptimizer(this.optimizerName, this.outputBias!._shape, 1e-5);
    }

    this.memoryKeys = mj.zeros([this.memoryDim, this.memorySlots]);
    this.memoryValues = mj.zeros([this.memoryDim, this.memorySlots]);
    this.memoryFilled = new Uint8Array(this.memorySlots);
    this.memoryUsage = new Float32Array(this.memorySlots);
    this.memoryAge = new Float32Array(this.memorySlots);
    this.memoryStep = 0;

    this.inputShape = [this.units, 1];
    if (this.mode === "project") this.outputShape = [this.outputUnits, 1];
    else if (this.mode === "concat") this.outputShape = [this.units + this.memoryDim, 1];
    else this.outputShape = [this.units, 1];

    const outputParams = this.outputKernel ? this.outputKernel._shape[0] * this.outputKernel._shape[1] + this.outputBias!._shape[0] : 0;
    this.params =
      this.queryKernel._shape[0] * this.queryKernel._shape[1] +
      this.writeKeyKernel._shape[0] * this.writeKeyKernel._shape[1] +
      this.writeValueKernel._shape[0] * this.writeValueKernel._shape[1] +
      this.needKernel._shape[0] * this.needKernel._shape[1] +
      this.writeGateKernel._shape[0] * this.writeGateKernel._shape[1] +
      outputParams;

    this.initialized = true;
    if (this.resetOnInit) this.resetMemory();
  }

  private pickWriteSlot(query: Float32Array): number {
    for (let s = 0; s < this.memorySlots; s++) {
      if (!this.memoryFilled[s]) return s;
    }

    if (this.writePolicy === "least-used" || this.writePolicy === "empty-first") {
      let best = 0;
      let minUsage = this.memoryUsage[0];
      for (let s = 1; s < this.memorySlots; s++) {
        if (this.memoryUsage[s] < minUsage) {
          minUsage = this.memoryUsage[s];
          best = s;
        }
      }
      return best;
    }

    if (this.writePolicy === "oldest") {
      let best = 0;
      let minAge = this.memoryAge[0];
      for (let s = 1; s < this.memorySlots; s++) {
        if (this.memoryAge[s] < minAge) {
          minAge = this.memoryAge[s];
          best = s;
        }
      }
      return best;
    }

    // least-relevant
    let best = -1;
    let minScore = Infinity;
    for (let s = 0; s < this.memorySlots; s++) {
      if (!this.memoryFilled[s]) continue;
      const key = this.getMemoryColumn(this.memoryKeys, s);
      const score = this.similarityScore(query, key);
      if (score < minScore) {
        minScore = score;
        best = s;
      }
    }
    return best >= 0 ? best : 0;
  }

  private getMemoryColumn(m: Matrix, col: number): Float32Array {
    const out = new Float32Array(this.memoryDim);
    const cols = this.memorySlots;
    for (let i = 0; i < this.memoryDim; i++) out[i] = m._data[i * cols + col];
    return out;
  }

  private setMemoryColumn(m: Matrix, col: number, v: Float32Array): void {
    const cols = this.memorySlots;
    for (let i = 0; i < this.memoryDim; i++) m._data[i * cols + col] = v[i];
  }

  private updateMemorySlot(slot: number, newKey: Float32Array, newValue: Float32Array, writeGate: number): void {
    const oldKey = this.getMemoryColumn(this.memoryKeys, slot);
    const oldValue = this.getMemoryColumn(this.memoryValues, slot);

    let nextKey = new Float32Array(this.memoryDim);
    let nextValue = new Float32Array(this.memoryDim);

    if (this.updateMode === "replace") {
      for (let i = 0; i < this.memoryDim; i++) {
        nextKey[i] = newKey[i];
        nextValue[i] = newValue[i];
      }
    } else if (this.updateMode === "merge") {
      for (let i = 0; i < this.memoryDim; i++) {
        nextKey[i] = 0.5 * oldKey[i] + 0.5 * newKey[i];
        nextValue[i] = 0.5 * oldValue[i] + 0.5 * newValue[i];
      }
      const normalized = this.normalizeSafe(nextKey);
      for (let i = 0; i < this.memoryDim; i++) nextKey[i] = normalized[i];
    } else {
      const g = Math.min(1, Math.max(0, writeGate));
      for (let i = 0; i < this.memoryDim; i++) {
        nextKey[i] = (1 - g) * oldKey[i] + g * newKey[i];
        nextValue[i] = (1 - g) * oldValue[i] + g * newValue[i];
      }
      const normalized = this.normalizeSafe(nextKey);
      for (let i = 0; i < this.memoryDim; i++) nextKey[i] = normalized[i];
    }

    this.setMemoryColumn(this.memoryKeys, slot, nextKey);
    this.setMemoryColumn(this.memoryValues, slot, nextValue);

    this.memoryFilled[slot] = 1;
    this.memoryUsage[slot] += 1;
    this.memoryAge[slot] = this.memoryStep;
  }

  private ensureInitializedFromInput(rows: number): void {
    if (this.initialized) return;
    const md = this.configuredMemoryDim ?? rows;
    const ou = this.configuredOutputUnits ?? rows;
    this.init(rows, md, ou);
  }

  /**
   * PART 1 – Returns a deep-copy of the debug trace collected during the last forward() call.
   * Each entry corresponds to one input column processed.
   */
  getDebugTrace(): MemoryBankDebugTrace[] {
    return JSON.parse(JSON.stringify(this.debugTrace));
  }

  /**
   * PART 1 – Clears the debug trace accumulated so far.
   * Safe to call manually at any time.
   */
  clearDebugTrace(): void {
    this.debugTrace = [];
  }

  /**
   * PART 4A – Returns info about the last committed write during the last forward() call.
   * Returns null if no write was committed.
   */
  getLastWriteInfo(): {
    committed: boolean;
    slot: number;
    writeGate: number;
    newKey: number[];
    newValue: number[];
  } | null {
    if (!this.lastWriteInfo || !this.lastWriteInfo.committed) return null;
    return {
      committed: this.lastWriteInfo.committed,
      slot: this.lastWriteInfo.slot,
      writeGate: this.lastWriteInfo.writeGate,
      newKey: Array.from(this.lastWriteInfo.newKey),
      newValue: Array.from(this.lastWriteInfo.newValue),
    };
  }

  /**
   * Returns queryKernel * x as a [memoryDim, 1] Matrix.
   * If normalize=true, the vector is L2-normalized — matching the cosine key space.
   *
   * Use this to generate canonical target keys for trainLastWriteKey():
   *   targetKey = mb.getQueryVectorForInput(pooled("query key_xx"), true)
   */
  getQueryVectorForInput(x: Matrix, normalize = true): Matrix {
    if (!this.initialized) {
      this.ensureInitializedFromInput(x._shape[0]);
    }
    if (x._shape[0] !== this.units || x._shape[1] !== 1) {
      throw new Error(
        `MemoryBank.getQueryVectorForInput: expected [${this.units}, 1], got [${x._shape[0]}, ${x._shape[1]}]`
      );
    }
    const q = this.matVecMul(this.queryKernel, x.getCol(0));
    const outVec = normalize ? this.normalizeSafe(q) : q;
    const out = mj.zeros([this.memoryDim, 1]);
    for (let i = 0; i < this.memoryDim; i++) out._data[i] = outVec[i];
    return out;
  }

  /**
   * Directly train writeKeyKernel so the last written key aligns with a target key.
   *
   * The target key should be: normalize(queryKernel * pooled("query key_xx"))
   * i.e., how a future QUERY turn would project the same key text.
   *
   * This closes the key-space gap between writeKeyKernel and queryKernel projections.
   *
   * Loss: 0.5 * mean((currentKey - targetKey)^2)
   * Gradient: backprop through normalizeBackward -> writeKeyKernel
   *
   * Acceptance: topSlotAcc in memory audit should rise.
   * If topSlotAcc rises but predAcc stays low, check output head / outputKernel.
   */
  trainLastWriteKey(targetKey: Matrix | number[]): number | null {
    if (!this.lastWriteInfo || !this.lastWriteInfo.committed) return null;

    let target: Float32Array;

    if (targetKey instanceof Matrix) {
      if (targetKey._shape[0] !== this.memoryDim || targetKey._shape[1] !== 1) {
        throw new Error(
          `MemoryBank.trainLastWriteKey: target Matrix must be [${this.memoryDim}, 1], got [${targetKey._shape[0]}, ${targetKey._shape[1]}]`
        );
      }
      target = targetKey.getCol(0);
    } else {
      if ((targetKey as number[]).length !== this.memoryDim) {
        throw new Error(
          `MemoryBank.trainLastWriteKey: target array length ${(targetKey as number[]).length} !== memoryDim ${this.memoryDim}`
        );
      }
      target = Float32Array.from(targetKey as number[]);
    }

    // Normalize target so it's in the same cosine key space
    target = this.normalizeSafe(target);

    const current = this.lastWriteInfo.newKey; // already normalized in forward()
    const gradNewKey = new Float32Array(this.memoryDim);

    let loss = 0;
    for (let i = 0; i < this.memoryDim; i++) {
      const diff = current[i] - target[i];
      loss += 0.5 * diff * diff;
      gradNewKey[i] = diff / this.memoryDim;
    }
    loss /= this.memoryDim;

    // Backprop through L2-normalization of newKeyRaw
    const gradRaw = this.normalizeBackward(this.lastWriteInfo.newKeyRaw, gradNewKey);

    // Accumulate gradient for writeKeyKernel: dWK += gradRaw outer xCol
    const gWK = mj.zeros(this.writeKeyKernel._shape);
    this.addOuter(gWK, gradRaw, this.lastWriteInfo.xCol);

    if (this.clipGradient !== false) {
      const limit = typeof this.clipGradient === "number" ? this.clipGradient : 5.0;
      mj.clipGradients(gWK, limit);
    }

    const upWK = this.optimizerWriteKey.calculate(gWK, this.alpha);
    this.writeKeyKernel.subInPlace(upWK);

    return loss;
  }

  /**
   * PART 4A – Returns the last written value vector as a [memoryDim, 1] Matrix copy.
   * Returns null if no write was committed in the last forward() call.
   */
  getLastWriteValueMatrix(): Matrix | null {
    if (!this.lastWriteInfo || !this.lastWriteInfo.committed) return null;
    const v = this.lastWriteInfo.newValue;
    const m = mj.zeros([v.length, 1]);
    for (let i = 0; i < v.length; i++) m._data[i] = v[i];
    return m;
  }

  /**
   * PART 6 (Opsi B) – Manually write a key/value pair into a specific slot.
   * Useful for deterministic-write diagnostic: bypasses learned write path entirely.
   * @param keyVector - memoryDim-length array for the key.
   * @param valueVector - memoryDim-length array for the value.
   * @param slot - slot index to write to (0-indexed). Defaults to first empty slot.
   */
  writeMemoryForDebug(keyVector: number[], valueVector: number[], slot?: number): void {
    if (!this.initialized) throw new Error("MemoryBank.writeMemoryForDebug: layer not initialized");
    if (keyVector.length !== this.memoryDim || valueVector.length !== this.memoryDim) {
      throw new Error(`MemoryBank.writeMemoryForDebug: vectors must be length ${this.memoryDim}`);
    }
    let targetSlot = slot;
    if (targetSlot === undefined) {
      targetSlot = -1;
      for (let s = 0; s < this.memorySlots; s++) {
        if (!this.memoryFilled[s]) { targetSlot = s; break; }
      }
      if (targetSlot === -1) targetSlot = 0;
    }
    if (targetSlot < 0 || targetSlot >= this.memorySlots) {
      throw new Error(`MemoryBank.writeMemoryForDebug: slot ${targetSlot} out of range`);
    }
    this.setMemoryColumn(this.memoryKeys, targetSlot, Float32Array.from(keyVector));
    this.setMemoryColumn(this.memoryValues, targetSlot, Float32Array.from(valueVector));
    this.memoryFilled[targetSlot] = 1;
    this.memoryUsage[targetSlot] += 1;
    this.memoryAge[targetSlot] = this.memoryStep;
  }

  /**
   * PART 4A – Train writeValueKernel using a direct classification probe on the last written value.
   *
   * This provides direct gradient signal to writeValueKernel so that stored memoryValues
   * encode the target class — bypassing the need for full BPTT through future QUERY turns.
   *
   * @param targetClass - Integer class index that the stored value should represent.
   * @param classifier - A Dense layer (outputUnits=OUTPUT_CLASSES, units=memoryDim, activation=linear,
   *                     loss=softmaxCrossEntropy). Must have already been forward-passed externally.
   * @returns Loss value from classifier, or null if no write was committed.
   *
   * Flow:
   *   1. classifier.forward(lastWriteValueMatrix)
   *   2. classifier.backward(y=[targetClass])
   *   3. Backprop the input gradient from classifier through writeValueKernel
   *
   * Acceptance: writeProbeAcc should rise; if it does but queryAcc stays random, issue is in read/query path.
   */
  trainLastWriteValue(
    targetClass: number,
    classifier: { forward(x: Matrix): Matrix; backward(y: Matrix, err: Matrix): Matrix; loss: number }
  ): number | null {
    if (!this.lastWriteInfo || !this.lastWriteInfo.committed) return null;

    const valMatrix = this.getLastWriteValueMatrix()!;

    // Forward through the probe classifier
    classifier.forward(valMatrix);

    // Backward through the probe classifier with sparse target
    const y = mj.matrix([[targetClass]]);
    const dVal = classifier.backward(y, mj.matrix([[]])); // [memoryDim, 1]

    // dVal is [memoryDim, 1]; update writeValueKernel
    // dWriteValueKernel += dVal outer xCol
    const gWV = mj.zeros(this.writeValueKernel._shape);
    const dv = dVal.getCol(0);
    this.addOuter(gWV, dv, this.lastWriteInfo.xCol);

    if (this.clipGradient !== false) {
      const limit = typeof this.clipGradient === "number" ? this.clipGradient : 5.0;
      mj.clipGradients(gWV, limit);
    }

    const upWV = this.optimizerWriteValue.calculate(gWV, this.alpha);
    this.writeValueKernel.subInPlace(upWV);

    return (classifier as any).loss as number;
  }

  forward(x: Matrix): Matrix {
    const [rows, cols] = x._shape;
    this.ensureInitializedFromInput(rows);

    if (rows !== this.units) {
      throw new Error(`MemoryBank: input rows ${rows} does not match units ${this.units}`);
    }

    let out: Matrix;
    if (this.mode === "project") out = mj.zeros([this.outputUnits, cols]);
    else if (this.mode === "concat") out = mj.zeros([this.units + this.memoryDim, cols]);
    else out = mj.zeros([this.units, cols]);

    this.cache = [];
    this.debugTrace = [];
    this.lastWriteInfo = null;

    for (let c = 0; c < cols; c++) {
      const xCol = x.getCol(c);
      const q = this.matVecMul(this.queryKernel, xCol);

      const filledSlots: number[] = [];
      for (let s = 0; s < this.memorySlots; s++) if (this.memoryFilled[s]) filledSlots.push(s);

      const read = new Float32Array(this.memoryDim);
      const readSlots: ReadSlotCache[] = [];

      if (filledSlots.length > 0) {
        const scored = filledSlots.map((slot) => {
          const key = this.getMemoryColumn(this.memoryKeys, slot);
          return { slot, score: this.similarityScore(q, key), key };
        });
        scored.sort((a, b) => b.score - a.score);
        const top = scored.slice(0, Math.min(this.readTopK, scored.length));

        const attn = this.softmax(top.map((t) => t.score));
        for (let i = 0; i < top.length; i++) {
          const slot = top[i].slot;
          const value = this.getMemoryColumn(this.memoryValues, slot);
          for (let d = 0; d < this.memoryDim; d++) {
            read[d] += attn[i] * value[d];
          }
          readSlots.push({
            slot,
            attn: attn[i],
            score: top[i].score,
            key: top[i].key,
            value,
          });
        }
      }

      const needInput = new Float32Array(this.units + this.memoryDim);
      needInput.set(xCol, 0);
      needInput.set(read, this.units);

      const needPre = this.matVecMul(this.needKernel, needInput)[0];
      const need = this.sigmoid(needPre);

      const context = new Float32Array(this.memoryDim);
      for (let i = 0; i < this.memoryDim; i++) context[i] = need * read[i];

      const combined = new Float32Array(this.units + this.memoryDim);
      combined.set(xCol, 0);
      combined.set(context, this.units);

      if (this.mode === "project") {
        const outputKernel = this.outputKernel;
        const outputBias = this.outputBias;
        if (!outputKernel || !outputBias) {
          throw new Error("MemoryBank: outputKernel/outputBias unavailable for mode='project'");
        }
        const o = this.matVecMul(outputKernel, combined);
        for (let r = 0; r < this.outputUnits; r++) {
          out._data[r * cols + c] = o[r] + outputBias._data[r];
        }
      } else if (this.mode === "concat") {
        for (let i = 0; i < this.units + this.memoryDim; i++) {
          out._data[i * cols + c] = combined[i];
        }
      } else {
        for (let i = 0; i < this.units; i++) {
          out._data[i * cols + c] = xCol[i] + context[i];
        }
      }

      let writeGatePre = 0;
      let writeGate = 0;
      let writeCommitted = false;
      let writeSlot = -1;
      let newKeyRaw = new Float32Array(this.memoryDim);
      let newKey = new Float32Array(this.memoryDim);
      let newValue = new Float32Array(this.memoryDim);

      if (this.writeEnabled && !this.writeFrozen) {
        writeGatePre = this.matVecMul(this.writeGateKernel, needInput)[0];
        writeGate = this.sigmoid(writeGatePre);
        if (writeGate >= this.writeThreshold) {
          const computedKeyRaw = this.matVecMul(this.writeKeyKernel, xCol);
          for (let i = 0; i < this.memoryDim; i++) newKeyRaw[i] = computedKeyRaw[i];

          const computedKey = this.normalizeSafe(newKeyRaw);
          for (let i = 0; i < this.memoryDim; i++) newKey[i] = computedKey[i];

          const computedValue = this.matVecMul(this.writeValueKernel, xCol);
          for (let i = 0; i < this.memoryDim; i++) newValue[i] = computedValue[i];
          writeSlot = this.pickWriteSlot(q);
          this.updateMemorySlot(writeSlot, newKey, newValue, writeGate);
          writeCommitted = true;

          // PART 4A: track last committed write for probe training
          this.lastWriteInfo = {
            committed: true,
            slot: writeSlot,
            writeGate,
            newKeyRaw: new Float32Array(newKeyRaw),
            newKey: new Float32Array(newKey),
            newValue: new Float32Array(newValue),
            xCol: new Float32Array(xCol),
          };
        }
      }

      // memoryStep policy: increment once per processed column.
      this.memoryStep += 1;

      // PART 1: push debug trace for this column
      this.debugTrace.push({
        column: c,
        readSlots: readSlots.map((r) => ({
          slot: r.slot,
          score: r.score,
          attn: r.attn,
        })),
        writeCommitted,
        writeSlot,
        writeGate,
        memoryFilled: Array.from(this.memoryFilled),
        memoryUsage: Array.from(this.memoryUsage),
        memoryAge: Array.from(this.memoryAge),
      });

      this.cache.push({
        xCol,
        q,
        read,
        need,
        needInput,
        context,
        combined,
        readSlots,
        writeGatePre,
        writeGate,
        writeCommitted,
        writeSlot,
        newKeyRaw,
        newKey,
        newValue,
      });
    }

    return out;
  }

  backward(_y: Matrix, err: Matrix): Matrix {
    if (!this.initialized) {
      throw new Error("MemoryBank.backward called before forward initialization");
    }

    const cols = err._shape[1];
    if (this.cache.length !== cols) {
      throw new Error("MemoryBank.backward: cache length mismatch, call forward() before backward() with same column count");
    }

    const expectedRows = this.mode === "project" ? this.outputUnits : this.mode === "concat" ? this.units + this.memoryDim : this.units;
    if (err._shape[0] !== expectedRows) {
      throw new Error(`MemoryBank.backward: err rows must be ${expectedRows}, got ${err._shape[0]}`);
    }

    const dx = mj.zeros([this.units, cols]);

    const gQuery = mj.zeros(this.queryKernel._shape);
    const gNeed = mj.zeros(this.needKernel._shape);
    const gWriteKey = mj.zeros(this.writeKeyKernel._shape);
    const gWriteValue = mj.zeros(this.writeValueKernel._shape);
    const gWriteGate = mj.zeros(this.writeGateKernel._shape);

    const gOut = this.outputKernel ? mj.zeros(this.outputKernel._shape) : undefined;
    const gOutBias = this.outputBias ? mj.zeros(this.outputBias._shape) : undefined;

    for (let c = 0; c < cols; c++) {
      const cache = this.cache[c];

      const dCombined = new Float32Array(this.units + this.memoryDim);
      const dxDirect = new Float32Array(this.units);
      const dContext = new Float32Array(this.memoryDim);

      if (this.mode === "project") {
        const outputKernel = this.outputKernel!;
        const e = err.getCol(c); // [outputUnits]

        // dW_out += e * combined^T, db += e, dCombined = W_out^T * e
        this.addOuter(gOut!, e, cache.combined);
        for (let i = 0; i < e.length; i++) gOutBias!._data[i] += e[i];

        const dComb = this.matTVecMul(outputKernel, e);
        for (let j = 0; j < dCombined.length; j++) dCombined[j] = dComb[j];

        for (let i = 0; i < this.units; i++) dxDirect[i] = dCombined[i];
        for (let i = 0; i < this.memoryDim; i++) dContext[i] = dCombined[this.units + i];
      } else if (this.mode === "concat") {
        const e = err.getCol(c);
        for (let i = 0; i < this.units; i++) dxDirect[i] = e[i];
        for (let i = 0; i < this.memoryDim; i++) dContext[i] = e[this.units + i];
      } else {
        // add mode
        const e = err.getCol(c);
        for (let i = 0; i < this.units; i++) dxDirect[i] = e[i];
        for (let i = 0; i < this.memoryDim; i++) dContext[i] = e[i];
      }

      // context = need * read
      // dNeed = sum(dContext * read)
      let dNeed = 0;
      const dReadFromContext = new Float32Array(this.memoryDim);
      for (let i = 0; i < this.memoryDim; i++) {
        dNeed += dContext[i] * cache.read[i];
        dReadFromContext[i] = dContext[i] * cache.need;
      }

      // need = sigmoid(needKernel * needInput)
      const dNeedPre = dNeed * cache.need * (1 - cache.need);
      this.addOuter(gNeed, new Float32Array([dNeedPre]), cache.needInput);

      // dNeedInput = needKernel^T * dNeedPre
      const dNeedInput = this.matTVecMul(this.needKernel, new Float32Array([dNeedPre]));

      const dxNeed = new Float32Array(this.units);
      const dReadNeed = new Float32Array(this.memoryDim);
      for (let i = 0; i < this.units; i++) dxNeed[i] = dNeedInput[i];
      for (let i = 0; i < this.memoryDim; i++) dReadNeed[i] = dNeedInput[this.units + i];

      const dRead = new Float32Array(this.memoryDim);
      for (let i = 0; i < this.memoryDim; i++) dRead[i] = dReadFromContext[i] + dReadNeed[i];

      // read = sum(attn_i * value_i)
      // dAttn_i = dot(dRead, value_i)
      // dScore_i = softmax backward
      const dQ = new Float32Array(this.memoryDim);
      if (cache.readSlots.length > 0) {
        const dAttn = new Array<number>(cache.readSlots.length).fill(0);
        for (let i = 0; i < cache.readSlots.length; i++) {
          dAttn[i] = this.vectorDot(dRead, cache.readSlots[i].value);
        }

        let weighted = 0;
        for (let i = 0; i < cache.readSlots.length; i++) {
          weighted += cache.readSlots[i].attn * dAttn[i];
        }

        const dScore = new Array<number>(cache.readSlots.length).fill(0);
        for (let i = 0; i < cache.readSlots.length; i++) {
          dScore[i] = cache.readSlots[i].attn * (dAttn[i] - weighted);
        }

        if (this.similarity === "dot") {
          const inv = 1 / Math.sqrt(this.memoryDim);
          for (let i = 0; i < cache.readSlots.length; i++) {
            const key = cache.readSlots[i].key;
            const coeff = dScore[i] * inv;
            for (let d = 0; d < this.memoryDim; d++) dQ[d] += coeff * key[d];
          }
        } else {
          for (let i = 0; i < cache.readSlots.length; i++) {
            const gradCos = this.cosineGradWrtQ(cache.q, cache.readSlots[i].key);
            const coeff = dScore[i];
            for (let d = 0; d < this.memoryDim; d++) dQ[d] += coeff * gradCos[d];
          }
        }
      }

      // q = queryKernel * x
      this.addOuter(gQuery, dQ, cache.xCol);

      // Surrogate local write-policy gradients (no gradient through discrete slot selection).
      // Objective surrogate:
      // - value write should align with downstream read usefulness: -writeGate * <dRead, newValue>
      // - key write should align with query pressure: -writeGate * <dQ, newKey>
      // - gate should increase when write seems useful.
      if (cache.writeCommitted) {
        const writeGate = cache.writeGate;

        const dNewValue = new Float32Array(this.memoryDim);
        for (let i = 0; i < this.memoryDim; i++) {
          dNewValue[i] = -writeGate * dRead[i];
        }
        this.addOuter(gWriteValue, dNewValue, cache.xCol);

        const dNewKey = new Float32Array(this.memoryDim);
        for (let i = 0; i < this.memoryDim; i++) {
          dNewKey[i] = -writeGate * dQ[i];
        }
        const dNewKeyRaw = this.normalizeBackward(cache.newKeyRaw, dNewKey);
        this.addOuter(gWriteKey, dNewKeyRaw, cache.xCol);

        const usefulness = this.vectorDot(dRead, cache.newValue) + this.vectorDot(dQ, cache.newKey);
        const dWriteGatePre = -usefulness * writeGate * (1 - writeGate);
        this.addOuter(gWriteGate, new Float32Array([dWriteGatePre]), cache.needInput);
      }

      // dx_query = queryKernel^T * dQ
      const dxQuery = this.matTVecMul(this.queryKernel, dQ);

      for (let i = 0; i < this.units; i++) {
        dx._data[i * cols + c] = dxDirect[i] + dxNeed[i] + dxQuery[i];
      }
    }

    if (this.trainablePolicy) {
      if (this.clipGradient !== false) {
        const limit = typeof this.clipGradient === "number" ? this.clipGradient : 5.0;
        mj.clipGradients(gQuery, limit);
        mj.clipGradients(gNeed, limit);
        mj.clipGradients(gWriteKey, limit);
        mj.clipGradients(gWriteValue, limit);
        mj.clipGradients(gWriteGate, limit);
        if (gOut) mj.clipGradients(gOut, limit);
        if (gOutBias) mj.clipGradients(gOutBias, limit);
      }

      const upQ = this.optimizerQuery.calculate(gQuery, this.alpha);
      const upN = this.optimizerNeed.calculate(gNeed, this.alpha);
      const upWK = this.optimizerWriteKey.calculate(gWriteKey, this.alpha);
      const upWV = this.optimizerWriteValue.calculate(gWriteValue, this.alpha);
      const upWG = this.optimizerWriteGate.calculate(gWriteGate, this.alpha);
      this.queryKernel.subInPlace(upQ);
      this.needKernel.subInPlace(upN);
      this.writeKeyKernel.subInPlace(upWK);
      this.writeValueKernel.subInPlace(upWV);
      this.writeGateKernel.subInPlace(upWG);

      if (gOut && this.outputKernel && this.optimizerOutput) {
        const upO = this.optimizerOutput.calculate(gOut, this.alpha);
        this.outputKernel.subInPlace(upO);
      }
      if (gOutBias && this.outputBias && this.optimizerOutputBias) {
        const upB = this.optimizerOutputBias.calculate(gOutBias, this.alpha);
        this.outputBias.subInPlace(upB);
      }

      // Note: write-policy gradients above are local surrogate updates.
      // They intentionally do not differentiate through discrete slot selection
      // and cross-column memory mutation history.
      //
      // PART 9 — TODO: Full BPTT Design
      // To make MemoryBank fully differentiable for write path:
      // 1. Cache (xCol, writeValueKernel, newValue) for each STORE/UPDATE turn.
      // 2. When a QUERY loss backward happens, propagate dRead through the weighted
      //    attn sum back to selected slot's memoryValues.
      // 3. From dMemoryValues[slot], backprop through updateMemorySlot (gated-merge)
      //    to dNewValue, then to dWriteValueKernel += dNewValue outer xCol (STORE turn).
      // 4. Slot selection (pickWriteSlot) remains hard/non-differentiable — that is OK.
      // 5. Key path: similarly propagate dAttn -> dScore -> dQ -> dQueryKernel (exists),
      //    and dNewKey -> dWriteKeyKernel (cross-turn, not yet implemented).
      // DO NOT claim MemoryBank is fully differentiable until this is implemented.
      // Current write probe (trainLastWriteValue) is a local approximation only.
    }

    return dx;
  }

  compile(cfg: { alpha?: number; optimizer?: Optimzier; clipGradient?: number | boolean }): void {
    if (cfg.alpha !== undefined) this.alpha = cfg.alpha;
    if (cfg.clipGradient !== undefined) this.clipGradient = cfg.clipGradient;

    if (cfg.optimizer !== undefined) {
      this.optimizerName = cfg.optimizer;
      if (this.initialized) {
        this.optimizerQuery = setOptimizer(this.optimizerName, this.queryKernel._shape, 1e-5);
        this.optimizerNeed = setOptimizer(this.optimizerName, this.needKernel._shape, 1e-5);
        this.optimizerWriteKey = setOptimizer(this.optimizerName, this.writeKeyKernel._shape, 1e-5);
        this.optimizerWriteValue = setOptimizer(this.optimizerName, this.writeValueKernel._shape, 1e-5);
        this.optimizerWriteGate = setOptimizer(this.optimizerName, this.writeGateKernel._shape, 1e-5);
        if (this.outputKernel && this.outputBias) {
          this.optimizerOutput = setOptimizer(this.optimizerName, this.outputKernel._shape, 1e-5);
          this.optimizerOutputBias = setOptimizer(this.optimizerName, this.outputBias._shape, 1e-5);
        }
      }
    }
  }

  private toMatrix2D(data: matrix2d, rows: number, cols: number, name: string): Matrix {
    if (!Array.isArray(data) || data.length !== rows) {
      throw new Error(`MemoryBank.load: invalid ${name} rows`);
    }
    for (let r = 0; r < rows; r++) {
      if (!Array.isArray(data[r]) || data[r].length !== cols) {
        throw new Error(`MemoryBank.load: invalid ${name} cols`);
      }
      for (let c = 0; c < cols; c++) {
        const v = data[r][c];
        if (!Number.isFinite(v)) {
          throw new Error(`MemoryBank.load: non-finite value in ${name}`);
        }
      }
    }
    return new Matrix({ array: data });
  }

  save(): any {
    return {
      name: this.name,
      status: this.status,
      units: this.units,
      memorySlots: this.memorySlots,
      memoryDim: this.memoryDim,
      outputUnits: this.outputUnits,
      mode: this.mode,
      similarity: this.similarity,
      readTopK: this.readTopK,
      updateMode: this.updateMode,
      writePolicy: this.writePolicy,
      writeThreshold: this.writeThreshold,
      persistence: this.persistence,
      resetOnInit: this.resetOnInit,
      writeEnabled: this.writeEnabled,
      trainablePolicy: this.trainablePolicy,
      alpha: this.alpha,
      optimizer: this.optimizerName,
      clipGradient: this.clipGradient,
      queryKernel: this.queryKernel?._value,
      writeKeyKernel: this.writeKeyKernel?._value,
      writeValueKernel: this.writeValueKernel?._value,
      needKernel: this.needKernel?._value,
      writeGateKernel: this.writeGateKernel?._value,
      outputKernel: this.outputKernel?._value,
      outputBias: this.outputBias?._value,
    };
  }

  load(data: any): void {
    const units = data.units;
    const memoryDim = data.memoryDim ?? data.units;
    const outputUnits = data.outputUnits ?? data.units;
    this.init(units, memoryDim, outputUnits);

    if (data.queryKernel) this.queryKernel = this.toMatrix2D(data.queryKernel, this.memoryDim, this.units, "queryKernel");
    if (data.writeKeyKernel) this.writeKeyKernel = this.toMatrix2D(data.writeKeyKernel, this.memoryDim, this.units, "writeKeyKernel");
    if (data.writeValueKernel) this.writeValueKernel = this.toMatrix2D(data.writeValueKernel, this.memoryDim, this.units, "writeValueKernel");
    if (data.needKernel) this.needKernel = this.toMatrix2D(data.needKernel, 1, this.units + this.memoryDim, "needKernel");
    if (data.writeGateKernel) this.writeGateKernel = this.toMatrix2D(data.writeGateKernel, 1, this.units + this.memoryDim, "writeGateKernel");

    if (this.mode === "project") {
      if (data.outputKernel) this.outputKernel = this.toMatrix2D(data.outputKernel, this.outputUnits, this.units + this.memoryDim, "outputKernel");
      if (data.outputBias) this.outputBias = this.toMatrix2D(data.outputBias, this.outputUnits, 1, "outputBias");
    }

    // refresh optimizer instances for restored weights
    this.compile({ optimizer: this.optimizerName });
  }

  resetMemory(): void {
    if (!this.initialized) return;
    this.memoryKeys = mj.zeros([this.memoryDim, this.memorySlots]);
    this.memoryValues = mj.zeros([this.memoryDim, this.memorySlots]);
    this.memoryFilled = new Uint8Array(this.memorySlots);
    this.memoryUsage = new Float32Array(this.memorySlots);
    this.memoryAge = new Float32Array(this.memorySlots);
    this.memoryStep = 0;
  }

  clearMemory(): void {
    this.resetMemory();
  }

  hasMemory(): boolean {
    if (!this.initialized) return false;
    for (let i = 0; i < this.memoryFilled.length; i++) if (this.memoryFilled[i] === 1) return true;
    return false;
  }

  getMemoryState(): MemoryBankState {
    if (!this.initialized) {
      throw new Error("MemoryBank.getMemoryState: layer is not initialized yet");
    }
    return {
      memoryKeys: this.memoryKeys._value,
      memoryValues: this.memoryValues._value,
      memoryFilled: Array.from(this.memoryFilled),
      memoryUsage: Array.from(this.memoryUsage),
      memoryAge: Array.from(this.memoryAge),
      memoryStep: this.memoryStep,
      units: this.units,
      memoryDim: this.memoryDim,
      memorySlots: this.memorySlots,
    };
  }

  setMemoryState(state: MemoryBankState): void {
    if (!state || typeof state !== "object") {
      throw new Error("MemoryBank.setMemoryState: invalid state object");
    }

    if (!this.initialized) {
      this.init(state.units, state.memoryDim, this.configuredOutputUnits ?? state.units);
    }

    if (state.units !== this.units || state.memoryDim !== this.memoryDim || state.memorySlots !== this.memorySlots) {
      throw new Error("MemoryBank.setMemoryState: dimensions mismatch with current layer configuration");
    }

    if (state.memoryKeys.length !== this.memoryDim || state.memoryValues.length !== this.memoryDim) {
      throw new Error("MemoryBank.setMemoryState: invalid memory matrix rows");
    }

    for (let r = 0; r < this.memoryDim; r++) {
      if (state.memoryKeys[r].length !== this.memorySlots || state.memoryValues[r].length !== this.memorySlots) {
        throw new Error("MemoryBank.setMemoryState: invalid memory matrix cols");
      }
      for (let c = 0; c < this.memorySlots; c++) {
        if (!Number.isFinite(state.memoryKeys[r][c]) || !Number.isFinite(state.memoryValues[r][c])) {
          throw new Error("MemoryBank.setMemoryState: memory keys/values must be finite");
        }
      }
    }

    if (state.memoryFilled.length !== this.memorySlots || state.memoryUsage.length !== this.memorySlots || state.memoryAge.length !== this.memorySlots) {
      throw new Error("MemoryBank.setMemoryState: invalid vector lengths");
    }

    for (let i = 0; i < this.memorySlots; i++) {
      const f = state.memoryFilled[i];
      if (f !== 0 && f !== 1) {
        throw new Error("MemoryBank.setMemoryState: memoryFilled values must be 0 or 1");
      }
      if (!Number.isFinite(state.memoryUsage[i]) || !Number.isFinite(state.memoryAge[i])) {
        throw new Error("MemoryBank.setMemoryState: memoryUsage/memoryAge must be finite");
      }
    }

    this.memoryKeys = new Matrix({ array: state.memoryKeys });
    this.memoryValues = new Matrix({ array: state.memoryValues });
    this.memoryFilled = Uint8Array.from(state.memoryFilled);
    this.memoryUsage = Float32Array.from(state.memoryUsage);
    this.memoryAge = Float32Array.from(state.memoryAge);
    this.memoryStep = state.memoryStep;
  }

  saveMemory(path: string): void {
    writeFileSync(path, JSON.stringify(this.getMemoryState()), "utf-8");
  }

  loadMemory(path: string): void {
    const raw = readFileSync(path, "utf-8");
    const state = JSON.parse(raw) as MemoryBankState;
    this.setMemoryState(state);
  }

  freezeWrites(): this {
    this.writeFrozen = true;
    return this;
  }

  enableWrites(): this {
    this.writeFrozen = false;
    return this;
  }

  dispose(): void {
    this.cache = [];
  }
}
