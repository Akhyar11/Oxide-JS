import { Optimzier, OptimzierType, StatusLayer, matrix2d } from "../@types/type";
import mj from "../math";
import Matrix from "../matrix";
import { dotProductNative, isNativeAvailable } from "../math/rust_backend";
import setOptimizer from "../utils/setOptimizer";

type Vec = Float32Array<ArrayBufferLike>;

export type MemoryBankMode = "project" | "concat" | "add" | "read-project";
export type MemorySimilarity = "cosine" | "dot";
export type MemoryUpdateMode = "replace" | "merge" | "gated-merge";
export type MemoryWritePolicy = "empty-first" | "least-used" | "oldest" | "least-relevant";
export type MemoryPersistence = "session" | "manual";
export type MemoryValueMode = "identity" | "project";
export type MemoryWriteKeyMode = "shared-query" | "separate-project";
export type MemoryWriteGateMode = "always" | "threshold" | "learned";

export interface MemoryBankDebugReadSlot {
  slot: number;
  score: number;
  attn: number;
}

export interface MemoryBankDebugTrace {
  column: number;
  readSlots: MemoryBankDebugReadSlot[];
  need: number;
  readNorm: number;
  contextNorm: number;
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
  alpha?: number;
  optimizer?: Optimzier;
  clipGradient?: number | boolean;
  status?: StatusLayer;
  forceNeedGate?: number;
  valueMode?: MemoryValueMode;
  writeKeyMode?: MemoryWriteKeyMode;
  writeGateMode?: MemoryWriteGateMode;
  trainablePolicy?: boolean;
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

export interface MemoryBankSequenceConfig {
  maxHistorySteps?: number;
}

interface ReadSlotCache {
  slot: number;
  attn: number;
  score: number;
  key: Vec;
  value: Vec;
}

interface ForwardCacheItem {
  xCol: Vec;
  qRaw: Vec;
  q: Vec;
  read: Vec;
  need: number;
  needInput: Vec;
  context: Vec;
  combined: Vec;
  readSlots: ReadSlotCache[];
  writeGatePre: number;
  writeGate: number;
  writeCommitted: boolean;
  writeSlot: number;
  newKeyRaw: Vec;
  newKey: Vec;
  newValue: Vec;
  writeSlotWasFilled: boolean;
  prevSlotKey: Vec;
  prevSlotValue: Vec;
}

type MemoryBankSaveData = {
  name: string;
  status: StatusLayer;
  config: {
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
    forceNeedGate?: number;
    valueMode: MemoryValueMode;
    writeKeyMode: MemoryWriteKeyMode;
    writeGateMode: MemoryWriteGateMode;
  };
  dimensions: {
    units: number;
    memorySlots: number;
    memoryDim: number;
    outputUnits: number;
  };
  trainableParams: {
    queryKernel: matrix2d;
    needKernel: matrix2d;
    outputKernel?: matrix2d;
    outputBias?: matrix2d;
    writeValueKernel?: matrix2d;
    writeGateKernel?: matrix2d;
    writeKeyKernel?: matrix2d;
  };
  memoryState: MemoryBankState;
  optimizerState: {
    alpha: number;
    optimizer: Optimzier;
    clipGradient: number | boolean;
    status: StatusLayer;
  };
  units: number;
  memorySlots: number;
  memoryDim: number;
  outputUnits: number;
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
  forceNeedGate?: number;
  valueMode: MemoryValueMode;
  writeKeyMode: MemoryWriteKeyMode;
  writeGateMode: MemoryWriteGateMode;
  alpha: number;
  optimizer: Optimzier;
  clipGradient: number | boolean;
  queryKernel: matrix2d;
  needKernel: matrix2d;
  outputKernel?: matrix2d;
  outputBias?: matrix2d;
  writeValueKernel?: matrix2d;
  writeGateKernel?: matrix2d;
  writeKeyKernel?: matrix2d;
  memoryKeys: number[][];
  memoryValues: number[][];
  memoryFilled: number[];
  memoryUsage: number[];
  memoryAge: number[];
  memoryStep: number;
};

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
  alpha: number;
  optimizerName: Optimzier;
  clipGradient: number | boolean;
  status: StatusLayer;
  forceNeedGate?: number;
  valueMode!: MemoryValueMode;
  writeKeyMode: MemoryWriteKeyMode;
  writeGateMode: MemoryWriteGateMode;
  trainablePolicy: boolean;

  queryKernel!: Matrix;
  needKernel!: Matrix;
  outputKernel?: Matrix;
  outputBias?: Matrix;
  writeValueKernel?: Matrix;
  writeGateKernel?: Matrix;
  writeKeyKernel?: Matrix;

  private optimizerQuery!: OptimzierType;
  private optimizerNeed!: OptimzierType;
  private optimizerOutput?: OptimzierType;
  private optimizerOutputBias?: OptimzierType;
  private optimizerWriteValue?: OptimzierType;
  private optimizerWriteGate?: OptimzierType;
  private optimizerWriteKey?: OptimzierType;

  // Runtime memory state. This is mutable session state, not optimizer-trained weights.
  memoryKeys!: Matrix;
  memoryValues!: Matrix;
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
  private configuredMemoryDim?: number;
  private configuredOutputUnits?: number;
  private configuredValueMode?: MemoryValueMode;
  private sequenceActive = false;
  private sequenceMaxHistorySteps: number | null = null;
  private sequenceHistory: ForwardCacheItem[] = [];

  private lastWriteInfo: {
    committed: boolean;
    slot: number;
    writeGate: number;
    newKeyRaw: Vec;
    newKey: Vec;
    newValue: Vec;
    xCol: Vec;
    needInput: Vec;
    need: number;
  } | null = null;

  constructor(cfg: MemoryBankConfig) {
    this.assertPositiveInt(cfg.memorySlots, "memorySlots");
    if (cfg.units !== undefined) this.assertPositiveInt(cfg.units, "units");
    if (cfg.memoryDim !== undefined) this.assertPositiveInt(cfg.memoryDim, "memoryDim");
    if (cfg.outputUnits !== undefined) this.assertPositiveInt(cfg.outputUnits, "outputUnits");

    this.memorySlots = cfg.memorySlots;
    this.mode = cfg.mode ?? "read-project";
    this.similarity = cfg.similarity ?? "cosine";
    this.readTopK = cfg.readTopK ?? Math.min(4, this.memorySlots);
    this.updateMode = cfg.updateMode ?? "replace";
    this.writePolicy = cfg.writePolicy ?? "empty-first";
    this.writeThreshold = cfg.writeThreshold ?? 0.5;
    this.persistence = cfg.persistence ?? "session";
    this.resetOnInit = cfg.resetOnInit ?? true;
    this.writeEnabled = cfg.writeEnabled ?? true;
    this.alpha = cfg.alpha ?? 0.01;
    this.optimizerName = cfg.optimizer ?? "adam";
    this.clipGradient = cfg.clipGradient ?? 5.0;
    this.status = cfg.status ?? "train";
    this.forceNeedGate = cfg.forceNeedGate;
    this.writeKeyMode = cfg.writeKeyMode ?? "shared-query";
    this.writeGateMode = cfg.writeGateMode ?? "always";
    this.trainablePolicy = cfg.trainablePolicy ?? true;
    this.configuredValueMode = cfg.valueMode;

    if (this.readTopK <= 0 || this.readTopK > this.memorySlots) {
      throw new Error("MemoryBank: readTopK must be in [1, memorySlots]");
    }
    if (this.forceNeedGate !== undefined && (!Number.isFinite(this.forceNeedGate) || this.forceNeedGate < 0 || this.forceNeedGate > 1)) {
      throw new Error(`MemoryBank: forceNeedGate must be in [0,1], got ${this.forceNeedGate}`);
    }

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

  private vectorDot(a: Vec, b: Vec): number {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }

  private l2Norm(v: Vec): number {
    let s = 0;
    for (let i = 0; i < v.length; i++) s += v[i] * v[i];
    return Math.sqrt(s);
  }

  private normalizeSafe(v: Vec): Vec {
    const out = new Float32Array(v.length);
    const n = this.l2Norm(v);
    if (!Number.isFinite(n) || n <= 1e-12) return out;
    const inv = 1 / n;
    for (let i = 0; i < v.length; i++) out[i] = v[i] * inv;
    return out;
  }

  private normalizeBackward(raw: Vec, gradOut: Vec): Vec {
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

  private matVecMul(weight: Matrix, x: Vec): Vec {
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
    for (let r = 0; r < rows; r++) {
      let sum = 0;
      const offset = r * cols;
      for (let c = 0; c < cols; c++) sum += weight._data[offset + c] * x[c];
      out[r] = sum;
    }
    return out;
  }

  private matTVecMul(weight: Matrix, x: Vec): Vec {
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
    for (let c = 0; c < cols; c++) {
      let sum = 0;
      for (let r = 0; r < rows; r++) sum += weight._data[r * cols + c] * x[r];
      out[c] = sum;
    }
    return out;
  }

  private addOuter(grad: Matrix, a: Vec, b: Vec, scale = 1): void {
    const cols = grad._shape[1];
    for (let i = 0; i < a.length; i++) {
      const ai = a[i] * scale;
      const offset = i * cols;
      for (let j = 0; j < b.length; j++) {
        grad._data[offset + j] += ai * b[j];
      }
    }
  }

  private cosineGradWrtQ(q: Vec, key: Vec): Vec {
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

  private cosineGradWrtKey(q: Vec, key: Vec): Vec {
    const out = new Float32Array(key.length);
    const nq = this.l2Norm(q);
    const nk = this.l2Norm(key);
    if (!Number.isFinite(nq) || !Number.isFinite(nk) || nq <= 1e-12 || nk <= 1e-12) {
      return out;
    }
    const dot = this.vectorDot(q, key);
    const invNqNk = 1 / (nq * nk);
    const coeffK = dot / (nq * nk * nk * nk);
    for (let i = 0; i < key.length; i++) {
      out[i] = q[i] * invNqNk - key[i] * coeffK;
    }
    return out;
  }

  private addStateGrad(flat: Float32Array, slot: number, grad: Vec, scale = 1): void {
    for (let i = 0; i < this.memoryDim; i++) {
      flat[i * this.memorySlots + slot] += grad[i] * scale;
    }
  }

  private getStateGrad(flat: Float32Array, slot: number): Vec {
    const out = new Float32Array(this.memoryDim);
    for (let i = 0; i < this.memoryDim; i++) {
      out[i] = flat[i * this.memorySlots + slot];
    }
    return out;
  }

  private softmax(scores: number[]): number[] {
    if (scores.length === 0) return [];
    let maxScore = -Infinity;
    for (const score of scores) if (score > maxScore) maxScore = score;
    let sum = 0;
    const exps = new Array(scores.length);
    for (let i = 0; i < scores.length; i++) {
      const e = Math.exp(scores[i] - maxScore);
      exps[i] = e;
      sum += e;
    }
    if (!Number.isFinite(sum) || sum <= 0) {
      return new Array(scores.length).fill(1 / scores.length);
    }
    for (let i = 0; i < exps.length; i++) exps[i] /= sum;
    return exps;
  }

  private similarityScore(q: Float32Array, key: Float32Array): number {
    if (this.similarity === "dot") {
      return this.vectorDot(q, key) / Math.sqrt(this.memoryDim);
    }
    return this.vectorDot(q, key);
  }

  private resolveValueMode(memoryDim: number, units: number): MemoryValueMode {
    if (this.configuredValueMode) return this.configuredValueMode;
    return memoryDim === units ? "identity" : "project";
  }

  private ensureInitializedFromInput(rows: number): void {
    if (this.initialized) return;
    this.init(rows, this.configuredMemoryDim ?? rows, this.configuredOutputUnits ?? rows);
  }

  private init(units: number, memoryDim: number, outputUnits: number): void {
    if (this.initialized) return;

    this.units = units;
    this.memoryDim = memoryDim;
    this.outputUnits = outputUnits;
    this.valueMode = this.resolveValueMode(memoryDim, units);

    if (this.mode === "add" && (this.memoryDim !== this.units || this.outputUnits !== this.units)) {
      throw new Error("MemoryBank(mode=add) requires memoryDim===units and outputUnits===units");
    }
    if (this.valueMode === "identity" && this.memoryDim !== this.units) {
      throw new Error("MemoryBank(valueMode='identity') requires memoryDim===units");
    }

    this.queryKernel = mj.xavier([this.memoryDim, this.units]);
    this.needKernel = mj.xavier([1, this.units + this.memoryDim]);

    if (this.valueMode === "project") {
      this.writeValueKernel = mj.xavier([this.memoryDim, this.units]);
      this.optimizerWriteValue = setOptimizer(this.optimizerName, this.writeValueKernel._shape, 1e-5);
    }
    if (this.writeGateMode === "learned") {
      this.writeGateKernel = mj.xavier([1, this.units + this.memoryDim]);
      this.optimizerWriteGate = setOptimizer(this.optimizerName, this.writeGateKernel._shape, 1e-5);
    }
    if (this.writeKeyMode === "separate-project") {
      this.writeKeyKernel = mj.xavier([this.memoryDim, this.units]);
      this.optimizerWriteKey = setOptimizer(this.optimizerName, this.writeKeyKernel._shape, 1e-5);
    }

    if (this.mode === "project") {
      this.outputKernel = mj.xavier([this.outputUnits, this.units + this.memoryDim]);
      this.outputBias = mj.zeros([this.outputUnits, 1]);
    } else if (this.mode === "read-project") {
      this.outputKernel = mj.xavier([this.outputUnits, this.memoryDim]);
      this.outputBias = mj.zeros([this.outputUnits, 1]);
    }

    this.optimizerQuery = setOptimizer(this.optimizerName, this.queryKernel._shape, 1e-5);
    this.optimizerNeed = setOptimizer(this.optimizerName, this.needKernel._shape, 1e-5);
    if (this.outputKernel && this.outputBias) {
      this.optimizerOutput = setOptimizer(this.optimizerName, this.outputKernel._shape, 1e-5);
      this.optimizerOutputBias = setOptimizer(this.optimizerName, this.outputBias._shape, 1e-5);
    }

    this.memoryKeys = mj.zeros([this.memoryDim, this.memorySlots]);
    this.memoryValues = mj.zeros([this.memoryDim, this.memorySlots]);
    this.memoryFilled = new Uint8Array(this.memorySlots);
    this.memoryUsage = new Float32Array(this.memorySlots);
    this.memoryAge = new Float32Array(this.memorySlots);
    this.memoryStep = 0;

    this.inputShape = [this.units, 1];
    if (this.mode === "project" || this.mode === "read-project") this.outputShape = [this.outputUnits, 1];
    else if (this.mode === "concat") this.outputShape = [this.units + this.memoryDim, 1];
    else this.outputShape = [this.units, 1];

    const outputParams = this.outputKernel
      ? this.outputKernel._shape[0] * this.outputKernel._shape[1] + (this.outputBias ? this.outputBias._data.length : 0)
      : 0;

    this.params =
      this.queryKernel._data.length +
      this.needKernel._data.length +
      outputParams +
      (this.writeValueKernel ? this.writeValueKernel._data.length : 0) +
      (this.writeGateKernel ? this.writeGateKernel._data.length : 0) +
      (this.writeKeyKernel ? this.writeKeyKernel._data.length : 0);

    this.initialized = true;
    if (this.resetOnInit) this.resetMemory();
  }

  private getMemoryColumn(m: Matrix, col: number): Vec {
    const out = new Float32Array(this.memoryDim);
    for (let i = 0; i < this.memoryDim; i++) out[i] = m._data[i * this.memorySlots + col];
    return out;
  }

  private setMemoryColumn(m: Matrix, col: number, v: Vec): void {
    for (let i = 0; i < this.memoryDim; i++) m._data[i * this.memorySlots + col] = v[i];
  }

  private pickWriteSlot(query: Vec): number {
    for (let slot = 0; slot < this.memorySlots; slot++) {
      if (!this.memoryFilled[slot]) return slot;
    }

    if (this.writePolicy === "empty-first" || this.writePolicy === "least-used") {
      let best = 0;
      let minUsage = this.memoryUsage[0];
      for (let slot = 1; slot < this.memorySlots; slot++) {
        if (this.memoryUsage[slot] < minUsage) {
          minUsage = this.memoryUsage[slot];
          best = slot;
        }
      }
      return best;
    }

    if (this.writePolicy === "oldest") {
      let best = 0;
      let minAge = this.memoryAge[0];
      for (let slot = 1; slot < this.memorySlots; slot++) {
        if (this.memoryAge[slot] < minAge) {
          minAge = this.memoryAge[slot];
          best = slot;
        }
      }
      return best;
    }

    let best = 0;
    let minScore = Infinity;
    for (let slot = 0; slot < this.memorySlots; slot++) {
      const key = this.getMemoryColumn(this.memoryKeys, slot);
      const score = this.similarityScore(query, key);
      if (score < minScore) {
        minScore = score;
        best = slot;
      }
    }
    return best;
  }

  private updateMemorySlot(slot: number, newKey: Vec, newValue: Vec, writeGate: number): void {
    if (!this.memoryFilled[slot]) {
      // Empty slots must fully replace zero state so the first write is exact.
      this.setMemoryColumn(this.memoryKeys, slot, newKey);
      this.setMemoryColumn(this.memoryValues, slot, newValue);
      this.memoryFilled[slot] = 1;
      this.memoryUsage[slot] += 1;
      this.memoryAge[slot] = this.memoryStep;
      return;
    }

    const oldKey = this.getMemoryColumn(this.memoryKeys, slot);
    const oldValue = this.getMemoryColumn(this.memoryValues, slot);
    const nextKey = new Float32Array(this.memoryDim);
    const nextValue = new Float32Array(this.memoryDim);

    if (this.updateMode === "replace") {
      nextKey.set(newKey);
      nextValue.set(newValue);
    } else if (this.updateMode === "merge") {
      for (let i = 0; i < this.memoryDim; i++) {
        nextKey[i] = 0.5 * oldKey[i] + 0.5 * newKey[i];
        nextValue[i] = 0.5 * oldValue[i] + 0.5 * newValue[i];
      }
      nextKey.set(this.normalizeSafe(nextKey));
    } else {
      const gate = Math.max(0, Math.min(1, writeGate));
      for (let i = 0; i < this.memoryDim; i++) {
        nextKey[i] = (1 - gate) * oldKey[i] + gate * newKey[i];
        nextValue[i] = (1 - gate) * oldValue[i] + gate * newValue[i];
      }
      nextKey.set(this.normalizeSafe(nextKey));
    }

    this.setMemoryColumn(this.memoryKeys, slot, nextKey);
    this.setMemoryColumn(this.memoryValues, slot, nextValue);
    this.memoryFilled[slot] = 1;
    this.memoryUsage[slot] += 1;
    this.memoryAge[slot] = this.memoryStep;
  }

  private getWriteKeyForInput(xCol: Vec): { raw: Vec; normalized: Vec } {
    if (this.writeKeyMode === "shared-query") {
      const raw = this.matVecMul(this.queryKernel, xCol);
      return { raw, normalized: this.similarity === "cosine" ? this.normalizeSafe(raw) : raw };
    }
    const raw = this.matVecMul(this.writeKeyKernel!, xCol);
    return { raw, normalized: this.similarity === "cosine" ? this.normalizeSafe(raw) : raw };
  }

  private getWriteValueForInput(xCol: Vec): Vec {
    if (this.valueMode === "identity") return new Float32Array(xCol);
    return this.matVecMul(this.writeValueKernel!, xCol);
  }

  private getWriteGate(needInput: Vec, need: number): { pre: number; gate: number } {
    if (this.writeGateMode === "always") return { pre: 1, gate: 1 };
    if (this.writeGateMode === "threshold") return { pre: need, gate: need };
    const pre = this.matVecMul(this.writeGateKernel!, needInput)[0];
    return { pre, gate: this.sigmoid(pre) };
  }

  private maybeClip(...grads: (Matrix | undefined)[]): void {
    if (this.clipGradient === false) return;
    const limit = typeof this.clipGradient === "number" ? this.clipGradient : 5.0;
    for (const grad of grads) {
      if (grad) mj.clipGradients(grad, limit);
    }
  }

  private trimSequenceHistoryIfNeeded(): void {
    if (this.sequenceMaxHistorySteps === null) return;
    if (this.sequenceHistory.length <= this.sequenceMaxHistorySteps) return;
    const drop = this.sequenceHistory.length - this.sequenceMaxHistorySteps;
    this.sequenceHistory.splice(0, drop);
  }

  getDebugTrace(): MemoryBankDebugTrace[] {
    return JSON.parse(JSON.stringify(this.debugTrace));
  }

  clearDebugTrace(): void {
    this.debugTrace = [];
  }

  getLastWriteInfo(): { committed: boolean; slot: number; writeGate: number; newKey: number[]; newValue: number[] } | null {
    if (!this.lastWriteInfo || !this.lastWriteInfo.committed) return null;
    return {
      committed: true,
      slot: this.lastWriteInfo.slot,
      writeGate: this.lastWriteInfo.writeGate,
      newKey: Array.from(this.lastWriteInfo.newKey),
      newValue: Array.from(this.lastWriteInfo.newValue),
    };
  }

  getLastReadValueMatrix(): Matrix | null {
    if (this.cache.length === 0) return null;
    const item = this.cache[this.cache.length - 1];
    const out = mj.zeros([this.memoryDim, 1]);
    out.setCol(0, item.read);
    return out;
  }

  getLastContextMatrix(): Matrix | null {
    if (this.cache.length === 0) return null;
    const item = this.cache[this.cache.length - 1];
    const out = mj.zeros([this.memoryDim, 1]);
    out.setCol(0, item.context);
    return out;
  }

  getLastCombinedMatrix(): Matrix | null {
    if (this.cache.length === 0) return null;
    const item = this.cache[this.cache.length - 1];
    const out = mj.zeros([this.units + this.memoryDim, 1]);
    out.setCol(0, item.combined);
    return out;
  }

  getLastWriteValueMatrix(): Matrix | null {
    if (!this.lastWriteInfo || !this.lastWriteInfo.committed) return null;
    const out = mj.zeros([this.memoryDim, 1]);
    out.setCol(0, this.lastWriteInfo.newValue);
    return out;
  }

  getQueryVectorForInput(x: Matrix, normalize = true): Matrix {
    if (!this.initialized) this.ensureInitializedFromInput(x._shape[0]);
    if (x._shape[0] !== this.units || x._shape[1] !== 1) {
      throw new Error(`MemoryBank.getQueryVectorForInput: expected [${this.units}, 1], got [${x._shape[0]}, ${x._shape[1]}]`);
    }
    const qRaw = this.matVecMul(this.queryKernel, x.getCol(0));
    const q = normalize && this.similarity === "cosine" ? this.normalizeSafe(qRaw) : qRaw;
    const out = mj.zeros([this.memoryDim, 1]);
    out.setCol(0, q);
    return out;
  }

  trainLastWriteKey(targetKey: Matrix | number[]): number | null {
    if (this.writeKeyMode !== "separate-project" || !this.writeKeyKernel || !this.optimizerWriteKey) return null;
    if (!this.lastWriteInfo || !this.lastWriteInfo.committed) return null;

    let target: Float32Array;
    if (targetKey instanceof Matrix) {
      if (targetKey._shape[0] !== this.memoryDim || targetKey._shape[1] !== 1) {
        throw new Error(`MemoryBank.trainLastWriteKey: target Matrix must be [${this.memoryDim}, 1]`);
      }
      target = targetKey.getCol(0);
    } else {
      if (targetKey.length !== this.memoryDim) {
        throw new Error(`MemoryBank.trainLastWriteKey: target array length ${targetKey.length} !== memoryDim ${this.memoryDim}`);
      }
      target = Float32Array.from(targetKey);
    }

    if (this.similarity === "cosine") target = this.normalizeSafe(target);
    const gradKey = new Float32Array(this.memoryDim);
    let loss = 0;
    for (let i = 0; i < this.memoryDim; i++) {
      const diff = this.lastWriteInfo.newKey[i] - target[i];
      gradKey[i] = diff / this.memoryDim;
      loss += 0.5 * diff * diff;
    }
    loss /= this.memoryDim;

    const gradRaw = this.similarity === "cosine"
      ? this.normalizeBackward(this.lastWriteInfo.newKeyRaw, gradKey)
      : gradKey;

    const grad = mj.zeros(this.writeKeyKernel._shape);
    this.addOuter(grad, gradRaw, this.lastWriteInfo.xCol);
    this.maybeClip(grad);
    this.writeKeyKernel.subInPlace(this.optimizerWriteKey.calculate(grad, this.alpha));
    return loss;
  }

  trainLastWriteValue(targetValue: Matrix | number[]): number | null {
    if (this.valueMode !== "project" || !this.writeValueKernel || !this.optimizerWriteValue) return null;
    if (!this.lastWriteInfo || !this.lastWriteInfo.committed) return null;

    let target: Float32Array;
    if (targetValue instanceof Matrix) {
      if (targetValue._shape[0] !== this.memoryDim || targetValue._shape[1] !== 1) {
        throw new Error(`MemoryBank.trainLastWriteValue: target Matrix must be [${this.memoryDim}, 1]`);
      }
      target = targetValue.getCol(0);
    } else {
      if (targetValue.length !== this.memoryDim) {
        throw new Error(`MemoryBank.trainLastWriteValue: target array length ${targetValue.length} !== memoryDim ${this.memoryDim}`);
      }
      target = Float32Array.from(targetValue);
    }

    const gradValue = new Float32Array(this.memoryDim);
    let loss = 0;
    for (let i = 0; i < this.memoryDim; i++) {
      const diff = this.lastWriteInfo.newValue[i] - target[i];
      gradValue[i] = diff / this.memoryDim;
      loss += 0.5 * diff * diff;
    }
    loss /= this.memoryDim;

    const grad = mj.zeros(this.writeValueKernel._shape);
    this.addOuter(grad, gradValue, this.lastWriteInfo.xCol);
    this.maybeClip(grad);
    this.writeValueKernel.subInPlace(this.optimizerWriteValue.calculate(grad, this.alpha));
    return loss;
  }

  trainLastWriteGate(targetGate: number): number | null {
    if (this.writeGateMode !== "learned" || !this.writeGateKernel || !this.optimizerWriteGate) return null;
    if (!this.lastWriteInfo || !this.lastWriteInfo.committed) return null;
    if (!Number.isFinite(targetGate) || targetGate < 0 || targetGate > 1) {
      throw new Error(`MemoryBank.trainLastWriteGate: targetGate must be in [0,1], got ${targetGate}`);
    }

    const gate = this.lastWriteInfo.writeGate;
    const diff = gate - targetGate;
    const gradPre = diff * gate * (1 - gate);
    const grad = mj.zeros(this.writeGateKernel._shape);
    this.addOuter(grad, new Float32Array([gradPre]), this.lastWriteInfo.needInput);
    this.maybeClip(grad);
    this.writeGateKernel.subInPlace(this.optimizerWriteGate.calculate(grad, this.alpha));
    return 0.5 * diff * diff;
  }

  writeMemoryForDebug(keyVector: number[], valueVector: number[], slot?: number): void {
    if (!this.initialized) throw new Error("MemoryBank.writeMemoryForDebug: layer not initialized");
    if (keyVector.length !== this.memoryDim || valueVector.length !== this.memoryDim) {
      throw new Error(`MemoryBank.writeMemoryForDebug: vectors must be length ${this.memoryDim}`);
    }
    let targetSlot = slot;
    if (targetSlot === undefined) {
      targetSlot = 0;
      for (let i = 0; i < this.memorySlots; i++) {
        if (!this.memoryFilled[i]) {
          targetSlot = i;
          break;
        }
      }
    }
    if (targetSlot < 0 || targetSlot >= this.memorySlots) {
      throw new Error(`MemoryBank.writeMemoryForDebug: slot ${targetSlot} out of range`);
    }
    const key = Float32Array.from(keyVector);
    const normalizedKey = this.similarity === "cosine" ? this.normalizeSafe(key) : key;
    this.updateMemorySlot(targetSlot, normalizedKey, Float32Array.from(valueVector), 1);
  }

  forward(x: Matrix): Matrix {
    const [rows, cols] = x._shape;
    this.ensureInitializedFromInput(rows);
    if (rows !== this.units) {
      throw new Error(`MemoryBank: input rows ${rows} does not match units ${this.units}`);
    }

    let out: Matrix;
    if (this.mode === "project" || this.mode === "read-project") out = mj.zeros([this.outputUnits, cols]);
    else if (this.mode === "concat") out = mj.zeros([this.units + this.memoryDim, cols]);
    else out = mj.zeros([this.units, cols]);

    this.cache = [];
    this.debugTrace = [];
    this.lastWriteInfo = null;

    for (let c = 0; c < cols; c++) {
      const xCol = x.getCol(c);
      const qRaw = this.matVecMul(this.queryKernel, xCol);
      const q = this.similarity === "cosine" ? this.normalizeSafe(qRaw) : qRaw;

      const scored: Array<{ slot: number; score: number; key: Float32Array }> = [];
      for (let slot = 0; slot < this.memorySlots; slot++) {
        if (!this.memoryFilled[slot]) continue;
        const key = this.getMemoryColumn(this.memoryKeys, slot);
        scored.push({ slot, score: this.similarityScore(q, key), key });
      }
      scored.sort((a, b) => b.score - a.score);

      const top = scored.slice(0, Math.min(this.readTopK, scored.length));
      const attn = this.softmax(top.map((item) => item.score));
      const read = new Float32Array(this.memoryDim);
      const readSlots: ReadSlotCache[] = [];
      for (let i = 0; i < top.length; i++) {
        const value = this.getMemoryColumn(this.memoryValues, top[i].slot);
        for (let d = 0; d < this.memoryDim; d++) read[d] += attn[i] * value[d];
        readSlots.push({
          slot: top[i].slot,
          score: top[i].score,
          attn: attn[i],
          key: top[i].key,
          value,
        });
      }

      const needInput = new Float32Array(this.units + this.memoryDim);
      needInput.set(xCol, 0);
      needInput.set(read, this.units);
      const needPre = this.matVecMul(this.needKernel, needInput)[0];
      const learnedNeed = this.sigmoid(needPre);
      const need = this.forceNeedGate === undefined ? learnedNeed : this.forceNeedGate;

      const context = new Float32Array(this.memoryDim);
      for (let i = 0; i < this.memoryDim; i++) context[i] = need * read[i];

      const combined = new Float32Array(this.units + this.memoryDim);
      combined.set(xCol, 0);
      combined.set(context, this.units);

      if (this.mode === "project") {
        const projected = this.matVecMul(this.outputKernel!, combined);
        for (let r = 0; r < this.outputUnits; r++) out._data[r * cols + c] = projected[r] + this.outputBias!._data[r];
      } else if (this.mode === "read-project") {
        const projected = this.matVecMul(this.outputKernel!, read);
        for (let r = 0; r < this.outputUnits; r++) out._data[r * cols + c] = projected[r] + this.outputBias!._data[r];
      } else if (this.mode === "concat") {
        for (let r = 0; r < this.units + this.memoryDim; r++) out._data[r * cols + c] = combined[r];
      } else {
        for (let r = 0; r < this.units; r++) out._data[r * cols + c] = xCol[r] + context[r];
      }

      let writeGatePre = 0;
      let writeGate = 0;
      let writeCommitted = false;
      let writeSlot = -1;
      let newKeyRaw: Vec = new Float32Array(this.memoryDim);
      let newKey: Vec = new Float32Array(this.memoryDim);
      let newValue: Vec = new Float32Array(this.memoryDim);
      let writeSlotWasFilled = false;
      let prevSlotKey: Vec = new Float32Array(this.memoryDim);
      let prevSlotValue: Vec = new Float32Array(this.memoryDim);

      if (this.writeEnabled && !this.writeFrozen) {
        const gateInfo = this.getWriteGate(needInput, need);
        writeGatePre = gateInfo.pre;
        writeGate = gateInfo.gate;
        if (writeGate >= this.writeThreshold) {
          const keyInfo = this.getWriteKeyForInput(xCol);
          newKeyRaw = keyInfo.raw;
          newKey = keyInfo.normalized;
          newValue = this.getWriteValueForInput(xCol);
          writeSlot = this.pickWriteSlot(q);
          writeSlotWasFilled = this.memoryFilled[writeSlot] === 1;
          if (writeSlotWasFilled) {
            prevSlotKey = this.getMemoryColumn(this.memoryKeys, writeSlot);
            prevSlotValue = this.getMemoryColumn(this.memoryValues, writeSlot);
          }
          this.updateMemorySlot(writeSlot, newKey, newValue, writeGate);
          writeCommitted = true;
          this.lastWriteInfo = {
            committed: true,
            slot: writeSlot,
            writeGate,
            newKeyRaw: new Float32Array(newKeyRaw),
            newKey: new Float32Array(newKey),
            newValue: new Float32Array(newValue),
            xCol: new Float32Array(xCol),
            needInput: new Float32Array(needInput),
            need,
          };
        }
      }

      this.memoryStep += 1;
      this.debugTrace.push({
        column: c,
        readSlots: readSlots.map((slot) => ({ slot: slot.slot, score: slot.score, attn: slot.attn })),
        need,
        readNorm: this.l2Norm(read),
        contextNorm: this.l2Norm(context),
        writeCommitted,
        writeSlot,
        writeGate,
        memoryFilled: Array.from(this.memoryFilled),
        memoryUsage: Array.from(this.memoryUsage),
        memoryAge: Array.from(this.memoryAge),
      });

      this.cache.push({
        xCol,
        qRaw,
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
        writeSlotWasFilled,
        prevSlotKey,
        prevSlotValue,
      });
    }

    if (this.sequenceActive && this.cache.length > 0) {
      this.sequenceHistory.push(...this.cache);
      this.trimSequenceHistoryIfNeeded();
    }

    return out;
  }

  private backwardThroughCaches(caches: ForwardCacheItem[], err: Matrix, callerName: string): Matrix {
    if (!this.initialized) throw new Error(`${callerName} called before forward initialization`);
    if (caches.length !== err._shape[1]) {
      throw new Error(`${callerName}: cache length mismatch, expected ${caches.length} columns, got ${err._shape[1]}`);
    }

    const expectedRows =
      this.mode === "project" || this.mode === "read-project"
        ? this.outputUnits
        : this.mode === "concat"
        ? this.units + this.memoryDim
        : this.units;
    if (err._shape[0] !== expectedRows) {
      throw new Error(`${callerName}: err rows must be ${expectedRows}, got ${err._shape[0]}`);
    }

    const dx = mj.zeros([this.units, err._shape[1]]);
    const gQuery = mj.zeros(this.queryKernel._shape);
    const gNeed = mj.zeros(this.needKernel._shape);
    const gOutput = this.outputKernel ? mj.zeros(this.outputKernel._shape) : undefined;
    const gOutputBias = this.outputBias ? mj.zeros(this.outputBias._shape) : undefined;
    const gWriteValue = this.writeValueKernel ? mj.zeros(this.writeValueKernel._shape) : undefined;
    const gWriteGate = this.writeGateKernel ? mj.zeros(this.writeGateKernel._shape) : undefined;
    const gWriteKey = this.writeKeyKernel ? mj.zeros(this.writeKeyKernel._shape) : undefined;

    let futureKeyStateGrad = new Float32Array(this.memoryDim * this.memorySlots);
    let futureValueStateGrad = new Float32Array(this.memoryDim * this.memorySlots);

    for (let c = err._shape[1] - 1; c >= 0; c--) {
      const cache = caches[c];
      const dxDirect = new Float32Array(this.units);
      const dContext = new Float32Array(this.memoryDim);
      const dRead = new Float32Array(this.memoryDim);
      const dPreKeys = new Float32Array(futureKeyStateGrad);
      const dPreValues = new Float32Array(futureValueStateGrad);
      let dNeedFromWrite = 0;

      if (this.mode === "project") {
        const e = err.getCol(c);
        this.addOuter(gOutput!, e, cache.combined);
        for (let i = 0; i < e.length; i++) gOutputBias!._data[i] += e[i];
        const dCombined = this.matTVecMul(this.outputKernel!, e);
        for (let i = 0; i < this.units; i++) dxDirect[i] += dCombined[i];
        for (let i = 0; i < this.memoryDim; i++) dContext[i] += dCombined[this.units + i];
      } else if (this.mode === "read-project") {
        const e = err.getCol(c);
        this.addOuter(gOutput!, e, cache.read);
        for (let i = 0; i < e.length; i++) gOutputBias!._data[i] += e[i];
        const gradReadOut = this.matTVecMul(this.outputKernel!, e);
        for (let i = 0; i < this.memoryDim; i++) dRead[i] += gradReadOut[i];
      } else if (this.mode === "concat") {
        const e = err.getCol(c);
        for (let i = 0; i < this.units; i++) dxDirect[i] += e[i];
        for (let i = 0; i < this.memoryDim; i++) dContext[i] += e[this.units + i];
      } else {
        const e = err.getCol(c);
        for (let i = 0; i < this.units; i++) {
          dxDirect[i] += e[i];
          dContext[i] += e[i];
        }
      }

      if (cache.writeCommitted) {
        const slot = cache.writeSlot;
        const dPostKeySlot = this.getStateGrad(futureKeyStateGrad, slot);
        const dPostValueSlot = this.getStateGrad(futureValueStateGrad, slot);
        for (let i = 0; i < this.memoryDim; i++) {
          dPreKeys[i * this.memorySlots + slot] = 0;
          dPreValues[i * this.memorySlots + slot] = 0;
        }

        const dNewKey = new Float32Array(this.memoryDim);
        const dNewValue = new Float32Array(this.memoryDim);
        let dGate = 0;

        if (!cache.writeSlotWasFilled || this.updateMode === "replace") {
          dNewKey.set(dPostKeySlot);
          dNewValue.set(dPostValueSlot);
        } else if (this.updateMode === "merge") {
          const mixedKey = new Float32Array(this.memoryDim);
          for (let i = 0; i < this.memoryDim; i++) {
            mixedKey[i] = 0.5 * cache.prevSlotKey[i] + 0.5 * cache.newKey[i];
          }
          const dMixedKey = this.normalizeBackward(mixedKey, dPostKeySlot);
          this.addStateGrad(dPreKeys, slot, dMixedKey, 0.5);
          this.addStateGrad(dPreValues, slot, dPostValueSlot, 0.5);
          for (let i = 0; i < this.memoryDim; i++) {
            dNewKey[i] += 0.5 * dMixedKey[i];
            dNewValue[i] += 0.5 * dPostValueSlot[i];
          }
        } else {
          const gate = cache.writeGate;
          const mixedKey = new Float32Array(this.memoryDim);
          for (let i = 0; i < this.memoryDim; i++) {
            mixedKey[i] = (1 - gate) * cache.prevSlotKey[i] + gate * cache.newKey[i];
          }
          const dMixedKey = this.normalizeBackward(mixedKey, dPostKeySlot);
          this.addStateGrad(dPreKeys, slot, dMixedKey, 1 - gate);
          this.addStateGrad(dPreValues, slot, dPostValueSlot, 1 - gate);
          for (let i = 0; i < this.memoryDim; i++) {
            dNewKey[i] += gate * dMixedKey[i];
            dNewValue[i] += gate * dPostValueSlot[i];
            dGate += (cache.newKey[i] - cache.prevSlotKey[i]) * dMixedKey[i];
            dGate += (cache.newValue[i] - cache.prevSlotValue[i]) * dPostValueSlot[i];
          }
        }

        if (this.valueMode === "identity") {
          for (let i = 0; i < this.units; i++) dxDirect[i] += dNewValue[i];
        } else if (gWriteValue && this.writeValueKernel) {
          this.addOuter(gWriteValue, dNewValue, cache.xCol);
          const dxWriteValue = this.matTVecMul(this.writeValueKernel, dNewValue);
          for (let i = 0; i < this.units; i++) dxDirect[i] += dxWriteValue[i];
        }

        const dNewKeyRaw = this.similarity === "cosine"
          ? this.normalizeBackward(cache.newKeyRaw, dNewKey)
          : dNewKey;
        if (this.writeKeyMode === "shared-query") {
          this.addOuter(gQuery, dNewKeyRaw, cache.xCol);
          const dxWriteKey = this.matTVecMul(this.queryKernel, dNewKeyRaw);
          for (let i = 0; i < this.units; i++) dxDirect[i] += dxWriteKey[i];
        } else if (gWriteKey && this.writeKeyKernel) {
          this.addOuter(gWriteKey, dNewKeyRaw, cache.xCol);
          const dxWriteKey = this.matTVecMul(this.writeKeyKernel, dNewKeyRaw);
          for (let i = 0; i < this.units; i++) dxDirect[i] += dxWriteKey[i];
        }

        if (this.writeGateMode === "threshold") {
          if (this.forceNeedGate === undefined) dNeedFromWrite += dGate;
        } else if (this.writeGateMode === "learned" && gWriteGate && this.writeGateKernel) {
          const dGatePre = dGate * cache.writeGate * (1 - cache.writeGate);
          this.addOuter(gWriteGate, new Float32Array([dGatePre]), cache.needInput);
          const dGateInput = this.matTVecMul(this.writeGateKernel, new Float32Array([dGatePre]));
          for (let i = 0; i < this.units; i++) dxDirect[i] += dGateInput[i];
          for (let i = 0; i < this.memoryDim; i++) dRead[i] += dGateInput[this.units + i];
        }
      }

      let dNeed = dNeedFromWrite;
      for (let i = 0; i < this.memoryDim; i++) {
        dNeed += dContext[i] * cache.read[i];
        dRead[i] += dContext[i] * cache.need;
      }

      const dNeedInput = new Float32Array(this.units + this.memoryDim);
      if (this.forceNeedGate === undefined) {
        const dNeedPre = dNeed * cache.need * (1 - cache.need);
        this.addOuter(gNeed, new Float32Array([dNeedPre]), cache.needInput);
        const back = this.matTVecMul(this.needKernel, new Float32Array([dNeedPre]));
        dNeedInput.set(back);
      }

      for (let i = 0; i < this.units; i++) dxDirect[i] += dNeedInput[i];
      for (let i = 0; i < this.memoryDim; i++) dRead[i] += dNeedInput[this.units + i];

      const dQuery = new Float32Array(this.memoryDim);
      if (cache.readSlots.length > 0) {
        const dAttn = new Array<number>(cache.readSlots.length).fill(0);
        for (let i = 0; i < cache.readSlots.length; i++) {
          dAttn[i] = this.vectorDot(dRead, cache.readSlots[i].value);
        }

        let weighted = 0;
        for (let i = 0; i < cache.readSlots.length; i++) weighted += cache.readSlots[i].attn * dAttn[i];

        for (let i = 0; i < cache.readSlots.length; i++) {
          this.addStateGrad(dPreValues, cache.readSlots[i].slot, dRead, cache.readSlots[i].attn);
          const dScore = cache.readSlots[i].attn * (dAttn[i] - weighted);
          if (this.similarity === "dot") {
            const scale = dScore / Math.sqrt(this.memoryDim);
            const dKey = new Float32Array(this.memoryDim);
            for (let d = 0; d < this.memoryDim; d++) {
              dQuery[d] += scale * cache.readSlots[i].key[d];
              dKey[d] = scale * cache.q[d];
            }
            this.addStateGrad(dPreKeys, cache.readSlots[i].slot, dKey);
          } else {
            const gradCosQ = this.cosineGradWrtQ(cache.q, cache.readSlots[i].key);
            const gradCosKey = this.cosineGradWrtKey(cache.q, cache.readSlots[i].key);
            for (let d = 0; d < this.memoryDim; d++) {
              dQuery[d] += dScore * gradCosQ[d];
              gradCosKey[d] *= dScore;
            }
            this.addStateGrad(dPreKeys, cache.readSlots[i].slot, gradCosKey);
          }
        }
      }

      const gradQueryRaw = this.similarity === "cosine"
        ? this.normalizeBackward(cache.qRaw, dQuery)
        : dQuery;
      this.addOuter(gQuery, gradQueryRaw, cache.xCol);
      const dxQuery = this.matTVecMul(this.queryKernel, gradQueryRaw);

      for (let i = 0; i < this.units; i++) {
        dx._data[i * err._shape[1] + c] = dxDirect[i] + dxQuery[i];
      }

      futureKeyStateGrad = dPreKeys;
      futureValueStateGrad = dPreValues;
    }

    if (this.trainablePolicy) {
      this.maybeClip(gQuery, gNeed, gOutput, gOutputBias, gWriteValue, gWriteGate, gWriteKey);
      this.queryKernel.subInPlace(this.optimizerQuery.calculate(gQuery, this.alpha));
      this.needKernel.subInPlace(this.optimizerNeed.calculate(gNeed, this.alpha));
      if (gOutput && this.outputKernel && this.optimizerOutput) {
        this.outputKernel.subInPlace(this.optimizerOutput.calculate(gOutput, this.alpha));
      }
      if (gOutputBias && this.outputBias && this.optimizerOutputBias) {
        this.outputBias.subInPlace(this.optimizerOutputBias.calculate(gOutputBias, this.alpha));
      }
      if (gWriteValue && this.writeValueKernel && this.optimizerWriteValue) {
        this.writeValueKernel.subInPlace(this.optimizerWriteValue.calculate(gWriteValue, this.alpha));
      }
      if (gWriteGate && this.writeGateKernel && this.optimizerWriteGate) {
        this.writeGateKernel.subInPlace(this.optimizerWriteGate.calculate(gWriteGate, this.alpha));
      }
      if (gWriteKey && this.writeKeyKernel && this.optimizerWriteKey) {
        this.writeKeyKernel.subInPlace(this.optimizerWriteKey.calculate(gWriteKey, this.alpha));
      }
    }

    // Runtime memory state mutates only via forward writes/reset/setMemoryState/load.
    // This backward pass propagates exact gradients through read/output path and
    // through within-sequence write history for the same forward() call.
    return dx;
  }

  backward(_y: Matrix, err: Matrix): Matrix {
    return this.backwardThroughCaches(this.cache, err, "MemoryBank.backward");
  }

  backwardSequence(err: Matrix): Matrix {
    if (!this.sequenceActive) {
      throw new Error("MemoryBank.backwardSequence: sequence mode is not active, call beginSequence() first");
    }
    return this.backwardThroughCaches(this.sequenceHistory, err, "MemoryBank.backwardSequence");
  }

  beginSequence(cfg: MemoryBankSequenceConfig = {}): this {
    if (cfg.maxHistorySteps !== undefined) {
      if (!Number.isInteger(cfg.maxHistorySteps) || cfg.maxHistorySteps <= 0) {
        throw new Error(`MemoryBank.beginSequence: maxHistorySteps must be positive integer, got ${cfg.maxHistorySteps}`);
      }
      this.sequenceMaxHistorySteps = cfg.maxHistorySteps;
    } else {
      this.sequenceMaxHistorySteps = null;
    }
    this.sequenceHistory = [];
    this.sequenceActive = true;
    return this;
  }

  detachSequence(): this {
    this.sequenceHistory = [];
    return this;
  }

  clearSequenceHistory(): this {
    this.sequenceHistory = [];
    return this;
  }

  endSequence(): this {
    this.sequenceHistory = [];
    this.sequenceActive = false;
    this.sequenceMaxHistorySteps = null;
    return this;
  }

  isSequenceActive(): boolean {
    return this.sequenceActive;
  }

  getSequenceLength(): number {
    return this.sequenceHistory.length;
  }

  compile(cfg: { alpha?: number; optimizer?: Optimzier; clipGradient?: number | boolean }): void {
    if (cfg.alpha !== undefined) this.alpha = cfg.alpha;
    if (cfg.clipGradient !== undefined) this.clipGradient = cfg.clipGradient;
    if (cfg.optimizer !== undefined) this.optimizerName = cfg.optimizer;
    if (!this.initialized) return;

    this.optimizerQuery = setOptimizer(this.optimizerName, this.queryKernel._shape, 1e-5);
    this.optimizerNeed = setOptimizer(this.optimizerName, this.needKernel._shape, 1e-5);
    if (this.outputKernel && this.outputBias) {
      this.optimizerOutput = setOptimizer(this.optimizerName, this.outputKernel._shape, 1e-5);
      this.optimizerOutputBias = setOptimizer(this.optimizerName, this.outputBias._shape, 1e-5);
    }
    if (this.writeValueKernel) this.optimizerWriteValue = setOptimizer(this.optimizerName, this.writeValueKernel._shape, 1e-5);
    if (this.writeGateKernel) this.optimizerWriteGate = setOptimizer(this.optimizerName, this.writeGateKernel._shape, 1e-5);
    if (this.writeKeyKernel) this.optimizerWriteKey = setOptimizer(this.optimizerName, this.writeKeyKernel._shape, 1e-5);
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
        if (!Number.isFinite(data[r][c])) throw new Error(`MemoryBank.load: non-finite value in ${name}`);
      }
    }
    return new Matrix({ array: data });
  }

  save(): MemoryBankSaveData {
    const memoryState = this.getMemoryState();
    const outputKernel = this.outputKernel?._value;
    const outputBias = this.outputBias?._value;
    const writeValueKernel = this.writeValueKernel?._value;
    const writeGateKernel = this.writeGateKernel?._value;
    const writeKeyKernel = this.writeKeyKernel?._value;

    return {
      name: this.name,
      status: this.status,
      config: {
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
        forceNeedGate: this.forceNeedGate,
        valueMode: this.valueMode,
        writeKeyMode: this.writeKeyMode,
        writeGateMode: this.writeGateMode,
      },
      dimensions: {
        units: this.units,
        memorySlots: this.memorySlots,
        memoryDim: this.memoryDim,
        outputUnits: this.outputUnits,
      },
      trainableParams: {
        queryKernel: this.queryKernel._value,
        needKernel: this.needKernel._value,
        outputKernel,
        outputBias,
        writeValueKernel,
        writeGateKernel,
        writeKeyKernel,
      },
      memoryState,
      optimizerState: {
        alpha: this.alpha,
        optimizer: this.optimizerName,
        clipGradient: this.clipGradient,
        status: this.status,
      },
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
      forceNeedGate: this.forceNeedGate,
      valueMode: this.valueMode,
      writeKeyMode: this.writeKeyMode,
      writeGateMode: this.writeGateMode,
      alpha: this.alpha,
      optimizer: this.optimizerName,
      clipGradient: this.clipGradient,
      queryKernel: this.queryKernel._value,
      needKernel: this.needKernel._value,
      outputKernel,
      outputBias,
      writeValueKernel,
      writeGateKernel,
      writeKeyKernel,
      memoryKeys: memoryState.memoryKeys,
      memoryValues: memoryState.memoryValues,
      memoryFilled: memoryState.memoryFilled,
      memoryUsage: memoryState.memoryUsage,
      memoryAge: memoryState.memoryAge,
      memoryStep: memoryState.memoryStep,
    };
  }

  load(data: any): void {
    const config = data.config ?? {};
    const dimensions = data.dimensions ?? {};
    const trainableParams = data.trainableParams ?? {};
    const optimizerState = data.optimizerState ?? {};
    const memoryStateData = data.memoryState ?? (
      data.memoryKeys
        ? {
            memoryKeys: data.memoryKeys,
            memoryValues: data.memoryValues,
            memoryFilled: data.memoryFilled,
            memoryUsage: data.memoryUsage,
            memoryAge: data.memoryAge,
            memoryStep: data.memoryStep,
            units: data.units,
            memoryDim: data.memoryDim,
            memorySlots: data.memorySlots,
          }
        : null
    );

    this.mode = config.mode ?? data.mode ?? this.mode;
    this.similarity = config.similarity ?? data.similarity ?? this.similarity;
    this.readTopK = config.readTopK ?? data.readTopK ?? this.readTopK;
    this.updateMode = config.updateMode ?? data.updateMode ?? this.updateMode;
    this.writePolicy = config.writePolicy ?? data.writePolicy ?? this.writePolicy;
    this.writeThreshold = config.writeThreshold ?? data.writeThreshold ?? this.writeThreshold;
    this.persistence = config.persistence ?? data.persistence ?? this.persistence;
    this.resetOnInit = config.resetOnInit ?? data.resetOnInit ?? this.resetOnInit;
    this.writeEnabled = config.writeEnabled ?? data.writeEnabled ?? this.writeEnabled;
    this.trainablePolicy = config.trainablePolicy ?? data.trainablePolicy ?? this.trainablePolicy;
    this.forceNeedGate = config.forceNeedGate ?? data.forceNeedGate ?? this.forceNeedGate;
    this.writeKeyMode = config.writeKeyMode ?? data.writeKeyMode ?? this.writeKeyMode;
    this.writeGateMode = config.writeGateMode ?? data.writeGateMode ?? this.writeGateMode;
    this.configuredValueMode = config.valueMode ?? data.valueMode ?? this.configuredValueMode;
    this.alpha = optimizerState.alpha ?? data.alpha ?? this.alpha;
    this.optimizerName = optimizerState.optimizer ?? data.optimizer ?? this.optimizerName;
    this.clipGradient = optimizerState.clipGradient ?? data.clipGradient ?? this.clipGradient;
    this.status = optimizerState.status ?? data.status ?? this.status;

    const units = dimensions.units ?? data.units;
    const memoryDim = dimensions.memoryDim ?? data.memoryDim ?? units;
    const outputUnits = dimensions.outputUnits ?? data.outputUnits ?? units;
    this.init(units, memoryDim, outputUnits);

    const queryKernel = trainableParams.queryKernel ?? data.queryKernel;
    const needKernel = trainableParams.needKernel ?? data.needKernel;
    const outputKernel = trainableParams.outputKernel ?? data.outputKernel;
    const outputBias = trainableParams.outputBias ?? data.outputBias;
    const writeValueKernel = trainableParams.writeValueKernel ?? data.writeValueKernel;
    const writeGateKernel = trainableParams.writeGateKernel ?? data.writeGateKernel;
    const writeKeyKernel = trainableParams.writeKeyKernel ?? data.writeKeyKernel;

    if (queryKernel) this.queryKernel = this.toMatrix2D(queryKernel, this.memoryDim, this.units, "queryKernel");
    if (needKernel) this.needKernel = this.toMatrix2D(needKernel, 1, this.units + this.memoryDim, "needKernel");
    if (this.outputKernel && outputKernel) {
      const cols = this.mode === "project" ? this.units + this.memoryDim : this.memoryDim;
      this.outputKernel = this.toMatrix2D(outputKernel, this.outputUnits, cols, "outputKernel");
    }
    if (this.outputBias && outputBias) {
      this.outputBias = this.toMatrix2D(outputBias, this.outputUnits, 1, "outputBias");
    }
    if (this.writeValueKernel && writeValueKernel) {
      this.writeValueKernel = this.toMatrix2D(writeValueKernel, this.memoryDim, this.units, "writeValueKernel");
    }
    if (this.writeGateKernel && writeGateKernel) {
      this.writeGateKernel = this.toMatrix2D(writeGateKernel, 1, this.units + this.memoryDim, "writeGateKernel");
    }
    if (this.writeKeyKernel && writeKeyKernel) {
      this.writeKeyKernel = this.toMatrix2D(writeKeyKernel, this.memoryDim, this.units, "writeKeyKernel");
    }

    if (memoryStateData) this.setMemoryState(memoryStateData as MemoryBankState);
    this.compile({ optimizer: this.optimizerName, alpha: this.alpha, clipGradient: this.clipGradient });
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
    if (!this.initialized) throw new Error("MemoryBank.getMemoryState: layer is not initialized yet");
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
    if (!state || typeof state !== "object") throw new Error("MemoryBank.setMemoryState: invalid state object");
    if (!this.initialized) this.init(state.units, state.memoryDim, this.configuredOutputUnits ?? state.units);
    if (state.units !== this.units || state.memoryDim !== this.memoryDim || state.memorySlots !== this.memorySlots) {
      throw new Error("MemoryBank.setMemoryState: dimensions mismatch with current layer configuration");
    }

    this.memoryKeys = this.toMatrix2D(state.memoryKeys, this.memoryDim, this.memorySlots, "memoryKeys");
    this.memoryValues = this.toMatrix2D(state.memoryValues, this.memoryDim, this.memorySlots, "memoryValues");

    if (state.memoryFilled.length !== this.memorySlots || state.memoryUsage.length !== this.memorySlots || state.memoryAge.length !== this.memorySlots) {
      throw new Error("MemoryBank.setMemoryState: invalid vector lengths");
    }

    for (let i = 0; i < this.memorySlots; i++) {
      if (state.memoryFilled[i] !== 0 && state.memoryFilled[i] !== 1) {
        throw new Error("MemoryBank.setMemoryState: memoryFilled values must be 0 or 1");
      }
      if (!Number.isFinite(state.memoryUsage[i]) || !Number.isFinite(state.memoryAge[i])) {
        throw new Error("MemoryBank.setMemoryState: memoryUsage/memoryAge must be finite");
      }
    }

    this.memoryFilled = Uint8Array.from(state.memoryFilled);
    this.memoryUsage = Float32Array.from(state.memoryUsage);
    this.memoryAge = Float32Array.from(state.memoryAge);
    this.memoryStep = state.memoryStep;
  }

  saveMemory(path: string): void {
    const fs = require("fs") as typeof import("fs");
    fs.writeFileSync(path, JSON.stringify(this.getMemoryState()), "utf-8");
  }

  loadMemory(path: string): void {
    const fs = require("fs") as typeof import("fs");
    const raw = fs.readFileSync(path, "utf-8");
    this.setMemoryState(JSON.parse(raw) as MemoryBankState);
  }

  freezeWrites(): this {
    this.writeFrozen = true;
    return this;
  }

  unfreezeWrites(): this {
    this.writeFrozen = false;
    return this;
  }

  setWriteFrozen(value: boolean): this {
    this.writeFrozen = value;
    return this;
  }

  enableWrites(): this {
    return this.unfreezeWrites();
  }

  disableWrites(): this {
    return this.freezeWrites();
  }

  dispose(): void {
    this.cache = [];
    this.debugTrace = [];
    this.lastWriteInfo = null;
  }
}
