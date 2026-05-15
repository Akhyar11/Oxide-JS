import fs from "fs";
import { Optimizer, OptimizerType, StatusLayer, matrix2d, engine } from "@oxide-js/core";
import { mj } from "@oxide-js/core";
import { Matrix } from "@oxide-js/core";
import { dotProductNative, isNativeAvailable, memoryBankSimilarityScoresNative, memoryBankUpdateNative } from "@oxide-js/core";
import { setOptimizer } from "@oxide-js/core";
import AttentionPooling from "./attentionPooling.js";

type Vec = Float32Array<ArrayBufferLike>;

export type MemoryBankMode = "project" | "concat";
export type MemorySimilarity = "cosine" | "dot";
export type MemoryPersistence = "session" | "manual";

export interface MemoryBankDebugReadSlot {
  slot: number;
  score: number;
  attn: number;
}

export interface MemoryBankDebugTrace {
  column: number;
  readSlots: MemoryBankDebugReadSlot[];
  need: number;
  writeGate: number;
  writeBestScore?: number;
  writeAllocatedNewSlot?: boolean;
  readNorm: number;
  contextNorm: number;
  writeCommitted: boolean;
  writeSlot: number;
  memoryFilled: number[];
  memoryUsage: number[];
  memoryAge: number[];
}

export interface MemoryBankConfig {
  units?: number;
  memorySlots: number;
  outputUnits?: number;
  mode?: MemoryBankMode;
  similarity?: MemorySimilarity;
  readTopK?: number;
  persistence?: MemoryPersistence;
  resetOnInit?: boolean;
  writeEnabled?: boolean;
  overwriteThreshold?: number;
  alpha?: number;
  optimizer?: Optimizer;
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

export interface MemoryBankSequenceConfig {
  maxHistorySteps?: number;
}

export interface MemoryBankExternalAccess {
  readQuery?: Matrix;
  readQueryProjected?: boolean;
  writeKey?: Matrix;
  writeKeyProjected?: boolean;
  writeValue?: Matrix;
}

export interface MemoryBankExternalGradients {
  readQuery: Matrix | null;
  writeKey: Matrix | null;
  writeValue: Matrix | null;
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
  readQuerySource: Vec;
  readQueryProjected: boolean;
  qRaw: Vec;
  q: Vec;
  read: Vec;
  need: number;
  needInput: Vec | null;
  writeGate: number;
  writeContext: Vec;
  writeQueryRaw: Vec;
  writeQuery: Vec;
  writeAttn: number[];
  softWrite: boolean;
  memorySummary: Vec;
  memorySummaryValidLength: number;
  dNeedFromOutput?: number;
  context: Vec;
  combined: Vec;
  readSlots: ReadSlotCache[];
  writeCommitted: boolean;
  writeSlot: number;
  writeAllocatedNewSlot: boolean;
  writeBestScore: number;
  preWriteKey: Vec;
  preWriteValue: Vec;
  writeKeySource: Vec;
  writeKeyProjected: boolean;
  writeValueSource: Vec;
  newKeyRaw: Vec;
  newKey: Vec;
  newValue: Vec;
  postWriteKey: Vec;
  postWriteValue: Vec;
  postWriteKeyUnnormalized: Vec;
  postWriteValueUnnormalized: Vec;
}

type MemoryBankSaveData = {
  name: string;
  status: StatusLayer;
  config: {
    mode: MemoryBankMode;
    similarity: MemorySimilarity;
    readTopK: number;
    persistence: MemoryPersistence;
    resetOnInit: boolean;
    writeEnabled: boolean;
    overwriteThreshold: number;
  };
  dimensions: {
    units: number;
    memorySlots: number;
    outputUnits: number;
  };
  trainableParams: {
    queryKernel: matrix2d;
    writeGateKernel: matrix2d;
    writeGateBias: matrix2d;
    writeQueryKernel: matrix2d;
    needKernel?: matrix2d;
    needBias?: matrix2d;
    outputKernel?: matrix2d;
    outputBias?: matrix2d;
    memorySummaryPooling?: any;
  };
  memoryState: MemoryBankState;
  optimizerState: {
    alpha: number;
    optimizer: Optimizer;
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
  persistence: MemoryPersistence;
  resetOnInit: boolean;
  writeEnabled: boolean;
  overwriteThreshold: number;
  alpha: number;
  optimizer: Optimizer;
  clipGradient: number | boolean;
  queryKernel: matrix2d;
  writeGateKernel: matrix2d;
  writeGateBias: matrix2d;
  writeQueryKernel: matrix2d;
  needKernel?: matrix2d;
  needBias?: matrix2d;
  outputKernel?: matrix2d;
  outputBias?: matrix2d;
  memorySummaryPooling?: any;
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
  memoryDim!: number;
  memorySlots: number;
  outputUnits!: number;

  mode: MemoryBankMode;
  similarity: MemorySimilarity;
  readTopK: number;
  persistence: MemoryPersistence;
  resetOnInit: boolean;
  writeEnabled: boolean;
  alpha: number;
  optimizerName: Optimizer;
  clipGradient: number | boolean;
  status: StatusLayer;

  queryKernel!: Matrix;
  writeGateKernel!: Matrix;
  writeGateBias!: Matrix;
  writeQueryKernel!: Matrix;
  needKernel?: Matrix;
  needBias?: Matrix;
  outputKernel?: Matrix;
  outputBias?: Matrix;
  memorySummaryPooling!: AttentionPooling;

  private optimizerQuery!: OptimizerType;
  private optimizerWriteGate!: OptimizerType;
  private optimizerWriteGateBias!: OptimizerType;
  private optimizerWriteQuery!: OptimizerType;
  private optimizerNeed?: OptimizerType;
  private optimizerNeedBias?: OptimizerType;
  private optimizerOutput?: OptimizerType;
  private optimizerOutputBias?: OptimizerType;

  memoryKeys!: Matrix;
  memoryValues!: Matrix;
  memoryFilled!: Uint8Array;
  memoryUsage!: Float32Array;
  memoryAge!: Float32Array;
  memoryStep = 0;

  inputShape: [number, number] = [0, 1];
  outputShape: [number, number] = [0, 1];
  params = 0;

  getParams(): Matrix[] {
    const p = [
      this.queryKernel,
      this.writeGateKernel,
      this.writeGateBias,
      this.writeQueryKernel
    ];
    if (this.needKernel) p.push(this.needKernel);
    if (this.needBias) p.push(this.needBias);
    if (this.outputKernel) p.push(this.outputKernel);
    if (this.outputBias) p.push(this.outputBias);
    
    if (this.memorySummaryPooling) {
      p.push(...this.memorySummaryPooling.getParams());
    }
    
    return p;
  }

  update(alpha?: number): void {
    const a = alpha || this.alpha;
    this.optimizerQuery.apply(this.queryKernel, a);
    this.optimizerWriteGate.apply(this.writeGateKernel, a);
    this.optimizerWriteGateBias.apply(this.writeGateBias, a);
    this.optimizerWriteQuery.apply(this.writeQueryKernel, a);
    if (this.needKernel && this.optimizerNeed) this.optimizerNeed.apply(this.needKernel, a);
    if (this.needBias && this.optimizerNeedBias) this.optimizerNeedBias.apply(this.needBias, a);
    if (this.outputKernel && this.optimizerOutput) this.optimizerOutput.apply(this.outputKernel, a);
    if (this.outputBias && this.optimizerOutputBias) this.optimizerOutputBias.apply(this.outputBias, a);
    
    if (this.memorySummaryPooling) {
      this.memorySummaryPooling.update(a);
    }
  }

  private initialized = false;
  private trainingMode = true;
  private writeFrozen = false;
  private cache: ForwardCacheItem[] = [];
  private debugTrace: MemoryBankDebugTrace[] = [];
  private externalAccess: MemoryBankExternalAccess | null = null;
  private lastExternalGradients: MemoryBankExternalGradients = {
    readQuery: null,
    writeKey: null,
    writeValue: null,
  };
  private configuredOutputUnits?: number;
  private sequenceActive = false;
  private sequenceMaxHistorySteps: number | null = null;
  private sequenceHistory: ForwardCacheItem[] = [];
  overwriteThreshold: number;

  private lastWriteInfo: {
    committed: boolean;
    slot: number;
    writeGate: number;
    newKeyRaw: Vec;
    newKey: Vec;
    newValue: Vec;
  } | null = null;

  constructor(cfg: MemoryBankConfig) {
    this.assertPositiveInt(cfg.memorySlots, "memorySlots");
    if (cfg.units !== undefined) this.assertPositiveInt(cfg.units, "units");
    if (cfg.outputUnits !== undefined) this.assertPositiveInt(cfg.outputUnits, "outputUnits");

    this.memorySlots = cfg.memorySlots;
    this.mode = cfg.mode ?? "project";
    this.similarity = cfg.similarity ?? "cosine";
    this.readTopK = cfg.readTopK ?? Math.min(4, this.memorySlots);
    this.persistence = cfg.persistence ?? "session";
    this.resetOnInit = cfg.resetOnInit ?? true;
    // NOTE: writeEnabled defaults to false to prevent gradient flow issues in hard write path.
    // Set to true only if you ensure trainingMode=true (soft write branch active).
    // See AUDIT_MEMORYBANK_FULL.md for gradient flow analysis.
    this.writeEnabled = cfg.writeEnabled ?? true;
    this.overwriteThreshold = cfg.overwriteThreshold ?? 0.35;
    this.alpha = cfg.alpha ?? 0.01;
    this.optimizerName = cfg.optimizer ?? "adam";
    this.clipGradient = cfg.clipGradient ?? 5.0;
    this.status = cfg.status ?? "train";
    this.trainingMode = this.status !== "test";
    this.configuredOutputUnits = cfg.outputUnits;

    if (this.mode !== "project" && this.mode !== "concat") {
      throw new Error(`MemoryBank: unsupported mode '${this.mode}', supported modes are 'project' and 'concat'`);
    }
    if (this.readTopK <= 0 || this.readTopK > this.memorySlots) {
      throw new Error("MemoryBank: readTopK must be in [1, memorySlots]");
    }

    if (cfg.units !== undefined) {
      this.init(cfg.units, cfg.outputUnits ?? cfg.units);
    }
  }

  private assertPositiveInt(v: number, name: string): void {
    if (!Number.isInteger(v) || v <= 0) throw new Error(`MemoryBank: ${name} must be positive integer`);
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
    for (let i = 0; i < raw.length; i++) out[i] = gradOut[i] * invN - raw[i] * dot * invN3;
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
      for (let j = 0; j < b.length; j++) grad._data[offset + j] += ai * b[j];
    }
  }

  private cosineGradWrtQ(q: Vec, key: Vec): Vec {
    const out = new Float32Array(q.length);
    const nq = this.l2Norm(q);
    const nk = this.l2Norm(key);
    if (!Number.isFinite(nq) || !Number.isFinite(nk) || nq <= 1e-12 || nk <= 1e-12) return out;
    const dot = this.vectorDot(q, key);
    const invNqNk = 1 / (nq * nk);
    const coeffQ = dot / (nq * nq * nq * nk);
    for (let i = 0; i < q.length; i++) out[i] = key[i] * invNqNk - q[i] * coeffQ;
    return out;
  }

  private cosineGradWrtKey(q: Vec, key: Vec): Vec {
    const out = new Float32Array(key.length);
    const nq = this.l2Norm(q);
    const nk = this.l2Norm(key);
    if (!Number.isFinite(nq) || !Number.isFinite(nk) || nq <= 1e-12 || nk <= 1e-12) return out;
    const dot = this.vectorDot(q, key);
    const invNqNk = 1 / (nq * nk);
    const coeffK = dot / (nq * nk * nk * nk);
    for (let i = 0; i < key.length; i++) out[i] = q[i] * invNqNk - key[i] * coeffK;
    return out;
  }

  private addStateGrad(flat: Float32Array, slot: number, grad: Vec, scale = 1): void {
    for (let i = 0; i < this.units; i++) flat[i * this.memorySlots + slot] += grad[i] * scale;
  }

  private getStateGrad(flat: Float32Array, slot: number): Vec {
    const out = new Float32Array(this.units);
    for (let i = 0; i < this.units; i++) out[i] = flat[i * this.memorySlots + slot];
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
    if (!Number.isFinite(sum) || sum <= 0) return new Array(scores.length).fill(1 / scores.length);
    for (let i = 0; i < exps.length; i++) exps[i] /= sum;
    return exps;
  }

  private similarityScore(q: Vec, key: Vec): number {
    if (this.similarity === "dot") return this.vectorDot(q, key) / Math.sqrt(this.units);
    const nq = this.l2Norm(q);
    const nk = this.l2Norm(key);
    if (nq <= 1e-12 || nk <= 1e-12) return 0;
    return this.vectorDot(q, key) / (nq * nk);
  }

  private ensureInitializedFromInput(rows: number): void {
    if (this.initialized) return;
    this.init(rows, this.configuredOutputUnits ?? rows);
  }

  private init(units: number, outputUnits: number): void {
    if (this.initialized) return;

    this.units = units;
    this.memoryDim = units;
    this.outputUnits = outputUnits;

    this.queryKernel = mj.xavier([this.units, this.units]);
    this.optimizerQuery = setOptimizer(this.optimizerName, this.queryKernel._shape, 1e-5);
    this.writeGateKernel = mj.zeros([1, this.units + this.units]);
    this.writeGateBias = mj.matrix([[4]]);
    this.writeQueryKernel = mj.xavier([this.units, this.units + this.units]);
    this.optimizerWriteGate = setOptimizer(this.optimizerName, this.writeGateKernel._shape, 1e-5);
    this.optimizerWriteGateBias = setOptimizer(this.optimizerName, this.writeGateBias._shape, 1e-5);
    this.optimizerWriteQuery = setOptimizer(this.optimizerName, this.writeQueryKernel._shape, 1e-5);
    this.memorySummaryPooling = new AttentionPooling({
      units: this.units,
      maxTokens: this.memorySlots,
      alpha: this.alpha,
      optimizer: this.optimizerName,
      status: "train",
      clipGradient: this.clipGradient,
    });

    if (this.mode === "project") {
      this.needKernel = mj.xavier([1, this.units + this.units]);
      this.needBias = mj.matrix([[3]]); // Initialize to 3 to encourage need=0.95 early on
      this.outputKernel = mj.xavier([this.outputUnits, this.units + this.units]);
      this.outputBias = mj.zeros([this.outputUnits, 1]);
      this.optimizerNeed = setOptimizer(this.optimizerName, this.needKernel._shape, 1e-5);
      this.optimizerNeedBias = setOptimizer(this.optimizerName, this.needBias._shape, 1e-5);
      this.optimizerOutput = setOptimizer(this.optimizerName, this.outputKernel._shape, 1e-5);
      this.optimizerOutputBias = setOptimizer(this.optimizerName, this.outputBias._shape, 1e-5);
    }

    this.memoryKeys = mj.zeros([this.units, this.memorySlots]);
    this.memoryValues = mj.zeros([this.units, this.memorySlots]);
    this.memoryFilled = new Uint8Array(this.memorySlots);
    this.memoryUsage = new Float32Array(this.memorySlots);
    this.memoryAge = new Float32Array(this.memorySlots);
    this.memoryStep = 0;

    this.inputShape = [this.units, 1];
    this.outputShape = this.mode === "concat" ? [this.units + this.units, 1] : [this.outputUnits, 1];

    this.params =
      this.queryKernel._data.length +
      this.writeGateKernel._data.length +
      this.writeGateBias._data.length +
      this.writeQueryKernel._data.length +
      (this.needKernel ? this.needKernel._data.length : 0) +
      (this.needBias ? this.needBias._data.length : 0) +
      (this.outputKernel ? this.outputKernel._data.length : 0) +
      (this.outputBias ? this.outputBias._data.length : 0) +
      this.memorySummaryPooling.params;

    this.initialized = true;
    if (this.resetOnInit) this.resetMemory();
  }

  private getMemoryColumn(m: Matrix, col: number): Vec {
    const out = new Float32Array(this.units);
    for (let i = 0; i < this.units; i++) out[i] = m._data[i * this.memorySlots + col];
    return out;
  }

  private setMemoryColumn(m: Matrix, col: number, v: Vec): void {
    for (let i = 0; i < this.units; i++) m._data[i * this.memorySlots + col] = v[i];
  }

  private pickWriteSlot(newKey: Vec): number {
    for (let slot = 0; slot < this.memorySlots; slot++) {
      if (!this.memoryFilled[slot]) return slot;
    }

    let best = 0;
    let minScore = Infinity;
    for (let slot = 0; slot < this.memorySlots; slot++) {
      const key = this.getMemoryColumn(this.memoryKeys, slot);
      const score = this.similarityScore(newKey, key);
      if (score < minScore) {
        minScore = score;
        best = slot;
      }
    }
    return best;
  }

  private writeSlot(slot: number, newKey: Vec, newValue: Vec): void {
    this.setMemoryColumn(this.memoryKeys, slot, newKey);
    this.setMemoryColumn(this.memoryValues, slot, newValue);
    this.memoryFilled[slot] = 1;
    this.memoryUsage[slot] += 1;
    this.memoryAge[slot] = this.memoryStep;
  }

  private maybeClip(...grads: (Matrix | undefined)[]): void {
    if (this.clipGradient === false) return;
    const limit = typeof this.clipGradient === "number" ? this.clipGradient : 5.0;
    for (const grad of grads) if (grad) mj.clipGradients(grad, limit);
  }

  private trimSequenceHistoryIfNeeded(): void {
    if (this.sequenceMaxHistorySteps === null) return;
    if (this.sequenceHistory.length <= this.sequenceMaxHistorySteps) return;
    this.sequenceHistory.splice(0, this.sequenceHistory.length - this.sequenceMaxHistorySteps);
  }

  private resetInitializationState(): void {
    this.initialized = false;
    this.writeFrozen = false;
    this.sequenceActive = false;
    this.sequenceMaxHistorySteps = null;
    this.queryKernel = undefined as unknown as Matrix;
    this.writeGateKernel = undefined as unknown as Matrix;
    this.writeGateBias = undefined as unknown as Matrix;
    this.writeQueryKernel = undefined as unknown as Matrix;
    this.needKernel = undefined;
    this.needBias = undefined;
    this.outputKernel = undefined;
    this.outputBias = undefined;
    this.memorySummaryPooling = undefined as unknown as AttentionPooling;
    this.optimizerQuery = undefined as unknown as OptimizerType;
    this.optimizerWriteGate = undefined as unknown as OptimizerType;
    this.optimizerWriteGateBias = undefined as unknown as OptimizerType;
    this.optimizerWriteQuery = undefined as unknown as OptimizerType;
    this.optimizerNeed = undefined;
    this.optimizerNeedBias = undefined;
    this.optimizerOutput = undefined;
    this.optimizerOutputBias = undefined;
    this.memoryKeys = undefined as unknown as Matrix;
    this.memoryValues = undefined as unknown as Matrix;
    this.memoryFilled = new Uint8Array(0);
    this.memoryUsage = new Float32Array(0);
    this.memoryAge = new Float32Array(0);
    this.memoryStep = 0;
    this.inputShape = [0, 1];
    this.outputShape = [0, 1];
    this.params = 0;
    this.cache = [];
    this.debugTrace = [];
    this.externalAccess = null;
    this.lastExternalGradients = { readQuery: null, writeKey: null, writeValue: null };
    this.lastWriteInfo = null;
    this.sequenceHistory = [];
  }

  getDebugTrace(): MemoryBankDebugTrace[] {
    return JSON.parse(JSON.stringify(this.debugTrace));
  }

  clearDebugTrace(): void {
    this.debugTrace = [];
  }

  setExternalAccess(access: MemoryBankExternalAccess): this {
    this.externalAccess = access;
    return this;
  }

  clearExternalAccess(): this {
    this.externalAccess = null;
    return this;
  }

  getLastExternalGradients(): MemoryBankExternalGradients {
    return {
      readQuery: this.lastExternalGradients.readQuery ? this.lastExternalGradients.readQuery.clone() : null,
      writeKey: this.lastExternalGradients.writeKey ? this.lastExternalGradients.writeKey.clone() : null,
      writeValue: this.lastExternalGradients.writeValue ? this.lastExternalGradients.writeValue.clone() : null,
    };
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
    const out = mj.zeros([this.units, 1]);
    out.setCol(0, item.read);
    return out;
  }

  getLastContextMatrix(): Matrix | null {
    if (this.cache.length === 0) return null;
    const item = this.cache[this.cache.length - 1];
    const out = mj.zeros([this.units, 1]);
    out.setCol(0, item.context);
    return out;
  }

  getLastCombinedMatrix(): Matrix | null {
    if (this.cache.length === 0) return null;
    const item = this.cache[this.cache.length - 1];
    const out = mj.zeros([this.units + this.units, 1]);
    out.setCol(0, item.combined);
    return out;
  }

  getLastWriteValueMatrix(): Matrix | null {
    if (!this.lastWriteInfo || !this.lastWriteInfo.committed) return null;
    const out = mj.zeros([this.units, 1]);
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
    const out = mj.zeros([this.units, 1]);
    out.setCol(0, q);
    return out;
  }

  private resolveExternalColumn(source: Matrix | undefined, col: number, label: string): Vec | null {
    if (!source) return null;
    if (source._shape[0] !== this.units) {
      throw new Error(`MemoryBank.${label}: expected rows ${this.units}, got ${source._shape[0]}`);
    }
    if (source._shape[1] <= col) {
      throw new Error(`MemoryBank.${label}: expected at least ${col + 1} columns, got ${source._shape[1]}`);
    }
    return source.getCol(col);
  }

  private addGradToColumn(matrix: Matrix | null, col: number, grad: Vec): void {
    if (!matrix) return;
    const cols = matrix._shape[1];
    for (let i = 0; i < this.units; i++) matrix._data[i * cols + col] += grad[i];
  }

  writeMemoryForDebug(keyVector: number[], valueVector: number[], slot?: number): void {
    if (!this.initialized) throw new Error("MemoryBank.writeMemoryForDebug: layer not initialized");
    if (keyVector.length !== this.units || valueVector.length !== this.units) {
      throw new Error(`MemoryBank.writeMemoryForDebug: vectors must be length ${this.units}`);
    }
    let targetSlot = slot;
    if (targetSlot === undefined) targetSlot = this.pickWriteSlot(Float32Array.from(keyVector));
    if (targetSlot < 0 || targetSlot >= this.memorySlots) {
      throw new Error(`MemoryBank.writeMemoryForDebug: slot ${targetSlot} out of range`);
    }
    const key = Float32Array.from(keyVector);
    const normalizedKey = this.similarity === "cosine" ? this.normalizeSafe(key) : key;
    this.writeSlot(targetSlot, normalizedKey, Float32Array.from(valueVector));
  }

  private buildMemorySummary(): { summary: Vec; activeCount: number } {
    const compact = mj.zeros([this.units, this.memorySlots]);
    let activeCount = 0;

    for (let slot = 0; slot < this.memorySlots; slot++) {
      if (!this.trainingMode && !this.memoryFilled[slot]) continue;
      for (let i = 0; i < this.units; i++) {
        compact._data[i * this.memorySlots + activeCount] = this.memoryValues._data[i * this.memorySlots + slot];
      }
      activeCount++;
    }

    if (activeCount === 0) {
      return { summary: new Float32Array(this.units), activeCount: 0 };
    }

    this.memorySummaryPooling.setValidLength(activeCount);
    const summary = this.memorySummaryPooling.forward(compact);
    return {
      summary: new Float32Array(summary._data),
      activeCount,
    };
  }

  private selectWriteSlot(writeQuery: Vec): { slot: number; bestScore: number; allocatedNewSlot: boolean } {
    let bestSlot = 0;
    let bestScore = Number.NEGATIVE_INFINITY;
    let hasFilled = false;
    if (isNativeAvailable() && this.memorySlots >= 32) {
      const scores = new Float32Array(this.memorySlots);
      memoryBankSimilarityScoresNative(writeQuery, this.memoryKeys._data, this.units, this.memorySlots, this.similarity, scores);
      for (let slot = 0; slot < this.memorySlots; slot++) {
        if (!this.memoryFilled[slot]) continue;
        hasFilled = true;
        if (scores[slot] > bestScore) {
          bestScore = scores[slot];
          bestSlot = slot;
        }
      }
    } else {
      for (let slot = 0; slot < this.memorySlots; slot++) {
        if (!this.memoryFilled[slot]) continue;
        hasFilled = true;
        const key = this.getMemoryColumn(this.memoryKeys, slot);
        const score = this.similarityScore(writeQuery, key);
        if (score > bestScore) {
          bestScore = score;
          bestSlot = slot;
        }
      }
    }

    if (hasFilled && bestScore >= this.overwriteThreshold) {
      return { slot: bestSlot, bestScore, allocatedNewSlot: false };
    }

    for (let slot = 0; slot < this.memorySlots; slot++) {
      if (!this.memoryFilled[slot]) {
        return { slot, bestScore, allocatedNewSlot: true };
      }
    }

    let leastUsedSlot = 0;
    let leastUsage = this.memoryUsage[0] ?? 0;
    for (let slot = 1; slot < this.memorySlots; slot++) {
      if ((this.memoryUsage[slot] ?? 0) < leastUsage) {
        leastUsage = this.memoryUsage[slot] ?? 0;
        leastUsedSlot = slot;
      }
    }

    return { slot: leastUsedSlot, bestScore, allocatedNewSlot: true };
  }

  forward(x: Matrix): Matrix {
    const [rows, cols] = x._shape;
    this.ensureInitializedFromInput(rows);
    if (rows !== this.units) throw new Error(`MemoryBank: input rows ${rows} does not match units ${this.units}`);
    const externalAccess = this.externalAccess;
    if (externalAccess?.readQuery && externalAccess.readQuery._shape[1] !== cols) {
      throw new Error(`MemoryBank.forward: readQuery columns must be ${cols}, got ${externalAccess.readQuery._shape[1]}`);
    }
    if (externalAccess?.writeKey && externalAccess.writeKey._shape[1] !== cols) {
      throw new Error(`MemoryBank.forward: writeKey columns must be ${cols}, got ${externalAccess.writeKey._shape[1]}`);
    }
    if (externalAccess?.writeValue && externalAccess.writeValue._shape[1] !== cols) {
      throw new Error(`MemoryBank.forward: writeValue columns must be ${cols}, got ${externalAccess.writeValue._shape[1]}`);
    }

    const out = this.mode === "concat" ? mj.zeros([this.units + this.units, cols]) : mj.zeros([this.outputUnits, cols]);
    this.cache = [];
    this.debugTrace = [];
    this.lastWriteInfo = null;
    this.lastExternalGradients = {
      readQuery: externalAccess?.readQuery ? mj.zeros(externalAccess.readQuery._shape) : null,
      writeKey: externalAccess?.writeKey ? mj.zeros(externalAccess.writeKey._shape) : null,
      writeValue: externalAccess?.writeValue ? mj.zeros(externalAccess.writeValue._shape) : null,
    };

    for (let c = 0; c < cols; c++) {
      const xCol = x.getCol(c);
      const readQuerySource = this.resolveExternalColumn(externalAccess?.readQuery, c, "forward.readQuery") ?? xCol;
      const readQueryProjected = externalAccess?.readQueryProjected ?? false;
      const qRaw = readQueryProjected ? new Float32Array(readQuerySource) : this.matVecMul(this.queryKernel, readQuerySource);
      const q = this.similarity === "cosine" ? this.normalizeSafe(qRaw) : qRaw;

      const scored: Array<{ slot: number; score: number; key: Vec }> = [];
      if (isNativeAvailable() && this.memorySlots >= 32) {
        const scores = new Float32Array(this.memorySlots);
        memoryBankSimilarityScoresNative(q, this.memoryKeys._data, this.units, this.memorySlots, this.similarity, scores);
        for (let slot = 0; slot < this.memorySlots; slot++) {
          if (!this.trainingMode && !this.memoryFilled[slot]) continue;
          scored.push({ slot, score: scores[slot], key: this.getMemoryColumn(this.memoryKeys, slot) });
        }
      } else {
        for (let slot = 0; slot < this.memorySlots; slot++) {
          if (!this.trainingMode && !this.memoryFilled[slot]) continue;
          const key = this.getMemoryColumn(this.memoryKeys, slot);
          scored.push({ slot, score: this.similarityScore(q, key), key });
        }
      }
      scored.sort((a, b) => b.score - a.score);

      // Apply readTopK to limit attention spread and improve gradient signal
      const top = scored.slice(0, Math.min(this.readTopK, scored.length));
      const attn = this.softmax(top.map((item) => item.score));

      const read = new Float32Array(this.units);
      const readSlots: ReadSlotCache[] = [];
      for (let i = 0; i < top.length; i++) {
        const value = this.getMemoryColumn(this.memoryValues, top[i].slot);
        for (let d = 0; d < this.units; d++) read[d] += attn[i] * value[d];
        readSlots.push({ slot: top[i].slot, score: top[i].score, attn: attn[i], key: top[i].key, value });
      }

      let need = 1;
      let needInput: Vec | null = null;
      const context = new Float32Array(this.units);
      if (this.mode === "project") {
        needInput = new Float32Array(this.units + this.units);
        needInput.set(read, 0);
        needInput.set(xCol, this.units);
        need = this.sigmoid(this.matVecMul(this.needKernel!, needInput)[0] + (this.needBias?._data[0] ?? 0));
        for (let i = 0; i < this.units; i++) context[i] = need * read[i];
      } else {
        context.set(read);
      }

      const combined = new Float32Array(this.units + this.units);
      combined.set(xCol, 0);
      combined.set(context, this.units);
      const { summary: memorySummary, activeCount: memorySummaryValidLength } = this.buildMemorySummary();
      const writeContext = new Float32Array(this.units + this.units);
      writeContext.set(xCol, 0);
      writeContext.set(memorySummary, this.units);
      const writeGate = this.sigmoid(this.matVecMul(this.writeGateKernel, writeContext)[0] + this.writeGateBias._data[0]);
      const writeQueryRaw = this.matVecMul(this.writeQueryKernel, writeContext);
      const writeQuery = this.similarity === "cosine" ? this.normalizeSafe(writeQueryRaw) : writeQueryRaw;

      if (this.mode === "project") {
        const projected = this.matVecMul(this.outputKernel!, combined);
        for (let r = 0; r < this.outputUnits; r++) {
          const projectedValue = projected[r] + this.outputBias!._data[r];
          const residualValue = r < this.units ? xCol[r] : 0;
          out._data[r * cols + c] = r < this.units
            ? need * projectedValue + (1 - need) * residualValue
            : need * projectedValue;
        }
      } else {
        for (let r = 0; r < this.units + this.units; r++) out._data[r * cols + c] = combined[r];
      }

      let writeCommitted = false;
      let writeSlot = -1;
      let preWriteKey: Vec = new Float32Array(this.units);
      let preWriteValue: Vec = new Float32Array(this.units);
      let newKeyRaw: Vec = new Float32Array(this.units);
      let newKey: Vec = new Float32Array(this.units);
      let newValue: Vec = new Float32Array(this.units);
      let postWriteKey: Vec = new Float32Array(this.units);
      let postWriteValue: Vec = new Float32Array(this.units);
      let writeAllocatedNewSlot = false;
      let writeBestScore = Number.NEGATIVE_INFINITY;
      let writeAttn: number[] = [];
      let softWrite = false;
      let postWriteKeyUnnormalized: Vec = new Float32Array(this.units);
      let postWriteValueUnnormalized: Vec = new Float32Array(this.units);
      const writeKeySource = this.resolveExternalColumn(externalAccess?.writeKey, c, "forward.writeKey") ?? xCol;
      const writeKeyProjected = externalAccess?.writeKeyProjected ?? false;
      const writeValueSource = this.resolveExternalColumn(externalAccess?.writeValue, c, "forward.writeValue") ?? xCol;

      if (this.writeEnabled && !this.writeFrozen && this.trainingMode) {
        newKeyRaw = writeKeyProjected ? new Float32Array(writeKeySource) : this.matVecMul(this.queryKernel, writeKeySource);
        newKey = this.similarity === "cosine" ? this.normalizeSafe(newKeyRaw) : newKeyRaw;
        newValue = new Float32Array(writeValueSource);
        const writeScored: Array<{ slot: number; score: number; key: Vec }> = [];
        let firstEmptyFound = false;
        for (let slot = 0; slot < this.memorySlots; slot++) {
          const key = this.getMemoryColumn(this.memoryKeys, slot);
          let score = this.similarityScore(writeQuery, key);

          if (!this.memoryFilled[slot]) {
            if (!firstEmptyFound) {
              firstEmptyFound = true;
              score = 0; // Baseline score for the first empty slot
            } else {
              score = -1e9; // Mask out subsequent empty slots to break symmetry
            }
          }
          writeScored.push({ slot, score, key });
        }
        writeScored.sort((a, b) => b.score - a.score);
        // Limit write attention to top slots for better gradient focus
        const writeTopK = Math.max(1, Math.ceil(this.readTopK / 2));
        const writeTopSlots = writeScored.slice(0, Math.min(writeTopK, writeScored.length));
        const attnSorted = this.softmax(writeTopSlots.map((item) => item.score));
        writeAttn = new Array(this.memorySlots).fill(0);
        for (let i = 0; i < writeTopSlots.length; i++) {
          writeAttn[writeTopSlots[i]!.slot] = attnSorted[i]!;
        }
        writeSlot = writeTopSlots[0]?.slot ?? 0;
        writeBestScore = writeTopSlots[0]?.score ?? Number.NEGATIVE_INFINITY;
        writeAllocatedNewSlot = false;
        preWriteKey = this.getMemoryColumn(this.memoryKeys, writeSlot);
        preWriteValue = this.getMemoryColumn(this.memoryValues, writeSlot);
        postWriteKey = new Float32Array(this.units);
        postWriteValue = new Float32Array(this.units);

        if (isNativeAvailable() && this.units >= 128) {
          const gateArr = new Float32Array(this.units);
          for (let i = 0; i < writeTopSlots.length; i++) {
            const slot = writeTopSlots[i]!.slot;
            const slotGate = writeGate * writeAttn[slot]!;
            gateArr.fill(slotGate);
            memoryBankUpdateNative(this.memoryKeys._data, this.memoryValues._data, newKey, newValue, slot, gateArr, this.units, this.memorySlots);
            if (slotGate >= 1e-3) this.memoryFilled[slot] = 1;
            
            if (slot === writeSlot) {
              // We still need the unnormalized versions for backprop cache
              // Rust updated the memory in-place, so we reconstruct the post-write vectors
              for (let d = 0; d < this.units; d++) {
                postWriteKeyUnnormalized[d] = (1 - slotGate) * preWriteKey[d] + slotGate * newKey[d];
                postWriteValueUnnormalized[d] = (1 - slotGate) * preWriteValue[d] + slotGate * newValue[d];
              }
              postWriteKey = this.similarity === "cosine" ? this.normalizeSafe(postWriteKeyUnnormalized) : postWriteKeyUnnormalized;
              postWriteValue = this.similarity === "cosine" ? this.normalizeSafe(postWriteValueUnnormalized) : postWriteValueUnnormalized;
            }
          }
        } else {
          for (let i = 0; i < writeTopSlots.length; i++) {
            const slot = writeTopSlots[i]!.slot;
            const slotGate = writeGate * writeAttn[slot]!;
            const oldKey = this.getMemoryColumn(this.memoryKeys, slot);
            const oldValue = this.getMemoryColumn(this.memoryValues, slot);

            const nextKeyUnnormalized = new Float32Array(this.units);
            const nextValueUnnormalized = new Float32Array(this.units);
            for (let d = 0; d < this.units; d++) {
              nextKeyUnnormalized[d] = (1 - slotGate) * oldKey[d] + slotGate * newKey[d];
              nextValueUnnormalized[d] = (1 - slotGate) * oldValue[d] + slotGate * newValue[d];
            }

            const nextKey = this.similarity === "cosine" ? this.normalizeSafe(nextKeyUnnormalized) : nextKeyUnnormalized;
            const nextValue = this.similarity === "cosine" ? this.normalizeSafe(nextValueUnnormalized) : nextValueUnnormalized;

            if (slotGate >= 1e-3) {
              this.writeSlot(slot, nextKey, nextValue);
            } else {
              this.setMemoryColumn(this.memoryKeys, slot, nextKey);
              this.setMemoryColumn(this.memoryValues, slot, nextValue);
            }
            if (slot === writeSlot) {
              postWriteKeyUnnormalized.set(nextKeyUnnormalized);
              postWriteValueUnnormalized.set(nextValueUnnormalized);
              postWriteKey = nextKey;
              postWriteValue = nextValue;
            }
          }
        }

        // Require stronger commitment to avoid spurious writes from noise
        writeCommitted = writeGate > 1e-3;
        softWrite = true;
        if (writeCommitted) {
          this.lastWriteInfo = {
            committed: true,
            slot: writeSlot,
            writeGate,
            newKeyRaw: new Float32Array(newKeyRaw),
            newKey: new Float32Array(newKey),
            newValue: new Float32Array(newValue),
          };
        }
      } else if (this.writeEnabled && !this.writeFrozen && writeGate >= 0.5) {
        newKeyRaw = writeKeyProjected ? new Float32Array(writeKeySource) : this.matVecMul(this.queryKernel, writeKeySource);
        newKey = this.similarity === "cosine" ? this.normalizeSafe(newKeyRaw) : newKeyRaw;
        newValue = new Float32Array(writeValueSource);
        const selected = this.selectWriteSlot(writeQuery);
        writeSlot = selected.slot;
        writeAllocatedNewSlot = selected.allocatedNewSlot;
        writeBestScore = selected.bestScore;
        preWriteKey = this.memoryFilled[writeSlot] ? this.getMemoryColumn(this.memoryKeys, writeSlot) : new Float32Array(this.units);
        preWriteValue = this.memoryFilled[writeSlot] ? this.getMemoryColumn(this.memoryValues, writeSlot) : new Float32Array(this.units);
        for (let i = 0; i < this.units; i++) {
          postWriteKeyUnnormalized[i] = (1 - writeGate) * preWriteKey[i] + writeGate * newKey[i];
          postWriteValueUnnormalized[i] = (1 - writeGate) * preWriteValue[i] + writeGate * newValue[i];
        }
        postWriteKey = this.similarity === "cosine" ? this.normalizeSafe(postWriteKeyUnnormalized) : postWriteKeyUnnormalized;
        postWriteValue = this.similarity === "cosine" ? this.normalizeSafe(postWriteValueUnnormalized) : postWriteValueUnnormalized;
        this.writeSlot(writeSlot, postWriteKey, postWriteValue);
        writeCommitted = true;
        writeAttn = new Array(this.memorySlots).fill(0);
        writeAttn[writeSlot] = 1;
        this.lastWriteInfo = {
          committed: true,
          slot: writeSlot,
          writeGate,
          newKeyRaw: new Float32Array(newKeyRaw),
          newKey: new Float32Array(newKey),
          newValue: new Float32Array(newValue),
        };
      }

      this.memoryStep += 1;
      this.debugTrace.push({
        column: c,
        readSlots: readSlots.map((slot) => ({ slot: slot.slot, score: slot.score, attn: slot.attn })),
        need,
        writeGate,
        writeBestScore,
        writeAllocatedNewSlot,
        readNorm: this.l2Norm(read),
        contextNorm: this.l2Norm(context),
        writeCommitted,
        writeSlot,
        memoryFilled: Array.from(this.memoryFilled),
        memoryUsage: Array.from(this.memoryUsage),
        memoryAge: Array.from(this.memoryAge),
      });

      this.cache.push({
        xCol,
        readQuerySource,
        readQueryProjected,
        qRaw,
        q,
        read,
        need,
        needInput,
        writeGate,
        writeContext,
        writeQueryRaw,
        writeQuery,
        writeAttn,
        softWrite,
        memorySummary,
        memorySummaryValidLength,
        context,
        combined,
        readSlots,
        writeCommitted,
        writeSlot,
        writeAllocatedNewSlot,
        writeBestScore,
        preWriteKey,
        preWriteValue,
        writeKeySource,
        writeKeyProjected,
        writeValueSource,
        newKeyRaw,
        newKey,
        newValue,
        postWriteKey,
        postWriteValue,
        postWriteKeyUnnormalized: postWriteKeyUnnormalized ?? new Float32Array(this.units),
        postWriteValueUnnormalized: postWriteValueUnnormalized ?? new Float32Array(this.units),
      });
    }

    this.externalAccess = null;

    if (this.sequenceActive && this.cache.length > 0) {
      this.sequenceHistory.push(...this.cache);
      this.trimSequenceHistoryIfNeeded();
    }

    // RECORD FOR AUTO-DIFF
    const tape = engine.tape;
    if (tape) {
      tape.record(this.getParams().concat([x]), [out], (grad: Matrix) => {
        const dx = this.backwardThroughCaches(this.cache, grad, "MemoryBank.autodiff", true); // true = gradOnly
        if (x.grad) x.grad.addInPlace(dx);
        else x.grad = dx;
      });
    }

    return out;
  }

  private backwardThroughCaches(caches: ForwardCacheItem[], err: Matrix, callerName: string, gradOnly = false): Matrix {
    if (!this.initialized) throw new Error(`${callerName} called before forward initialization`);
    if (caches.length !== err._shape[1]) {
      throw new Error(`${callerName}: cache length mismatch, expected ${caches.length} columns, got ${err._shape[1]}`);
    }

    const expectedRows = this.mode === "concat" ? this.units + this.units : this.outputUnits;
    if (err._shape[0] !== expectedRows) {
      throw new Error(`${callerName}: err rows must be ${expectedRows}, got ${err._shape[0]}`);
    }

    const dx = mj.zeros([this.units, err._shape[1]]);
    const gQuery = mj.zeros(this.queryKernel._shape);
    const gWriteGate = mj.zeros(this.writeGateKernel._shape);
    const gWriteGateBias = mj.zeros(this.writeGateBias._shape);
    const gWriteQuery = mj.zeros(this.writeQueryKernel._shape);
    const gNeed = this.needKernel ? mj.zeros(this.needKernel._shape) : undefined;
    const gNeedBias = this.needBias ? mj.zeros(this.needBias._shape) : undefined;
    const gOutput = this.outputKernel ? mj.zeros(this.outputKernel._shape) : undefined;
    const gOutputBias = this.outputBias ? mj.zeros(this.outputBias._shape) : undefined;
    const gReadQueryExternal = this.lastExternalGradients.readQuery ? mj.zeros(this.lastExternalGradients.readQuery._shape) : null;
    const gWriteKeyExternal = this.lastExternalGradients.writeKey ? mj.zeros(this.lastExternalGradients.writeKey._shape) : null;
    const gWriteValueExternal = this.lastExternalGradients.writeValue ? mj.zeros(this.lastExternalGradients.writeValue._shape) : null;

    let futureKeyStateGrad = new Float32Array(this.units * this.memorySlots);
    let futureValueStateGrad = new Float32Array(this.units * this.memorySlots);

    for (let c = err._shape[1] - 1; c >= 0; c--) {
      const cache = caches[c];
      const dxDirect = new Float32Array(this.units);
      const dRead = new Float32Array(this.units);
      const dContext = new Float32Array(this.units);
      const dMemorySummary = new Float32Array(this.units);
      const dPreKeys = new Float32Array(futureKeyStateGrad);
      const dPreValues = new Float32Array(futureValueStateGrad);

      if (this.mode === "project") {
        const e = err.getCol(c);
        const projected = this.matVecMul(this.outputKernel!, cache.combined);
        for (let i = 0; i < this.outputUnits; i++) projected[i] += this.outputBias!._data[i];
        const dProjected = new Float32Array(this.outputUnits);
        let dNeedFromOutput = 0;
        for (let i = 0; i < this.outputUnits; i++) {
          const residual = i < this.units ? cache.xCol[i] : 0;
          dProjected[i] = cache.need * e[i];
          dNeedFromOutput += e[i] * (projected[i] - residual);
        }
        this.addOuter(gOutput!, dProjected, cache.combined);
        for (let i = 0; i < dProjected.length; i++) gOutputBias!._data[i] += dProjected[i];
        const dCombined = this.matTVecMul(this.outputKernel!, dProjected);
        for (let i = 0; i < this.units; i++) dxDirect[i] += (1 - cache.need) * e[i] + dCombined[i];
        for (let i = 0; i < this.units; i++) dContext[i] += dCombined[this.units + i];
        cache.dNeedFromOutput = dNeedFromOutput;
      } else {
        const e = err.getCol(c);
        for (let i = 0; i < this.units; i++) dxDirect[i] += e[i];
        for (let i = 0; i < this.units; i++) dRead[i] += e[this.units + i];
      }

      if (cache.writeCommitted) {
        if (cache.softWrite) {
          const dGateBySlot = new Float32Array(this.memorySlots);
          const dNewKey = new Float32Array(this.units);
          let dWriteProb = 0;

          for (let slot = 0; slot < this.memorySlots; slot++) {
            let dPostKeySlot = this.getStateGrad(futureKeyStateGrad, slot);
            let dPostValueSlot = this.getStateGrad(futureValueStateGrad, slot);

            // Backward through normalization
            if (this.similarity === "cosine") {
              // We need the unnormalized state for this slot. 
              // Since it's not stored in cache per slot (only for the writeSlot),
              // we can approximate or we should have cached all of them.
              // For simplicity, we use the cached unnormalized value if it's the writeSlot,
              // otherwise we use the post-write value (which is close enough if norm is near 1).
              const unnormKey = (slot === cache.writeSlot) ? cache.postWriteKeyUnnormalized : cache.postWriteKey;
              const unnormValue = (slot === cache.writeSlot) ? cache.postWriteValueUnnormalized : cache.postWriteValue;
              dPostKeySlot = this.normalizeBackward(unnormKey, dPostKeySlot);
              dPostValueSlot = this.normalizeBackward(unnormValue, dPostValueSlot);
            }

            const preKey = this.getMemoryColumn(this.memoryKeys, slot);
            const preValue = this.getMemoryColumn(this.memoryValues, slot);
            const gateSlot = cache.writeGate * (cache.writeAttn[slot] ?? 0);
            let dGate = 0;
            for (let i = 0; i < this.units; i++) {
              // ACCUMULATE (+=) to preserve gradients from future steps and reads
              dPreKeys[i * this.memorySlots + slot] += (1 - gateSlot) * dPostKeySlot[i];
              dPreValues[i * this.memorySlots + slot] += (1 - gateSlot) * dPostValueSlot[i];
              dGate += (cache.newKey[i] - preKey[i]) * dPostKeySlot[i];
              dGate += (cache.newValue[i] - preValue[i]) * dPostValueSlot[i];
              dNewKey[i] += gateSlot * dPostKeySlot[i];
              if (gWriteValueExternal) gWriteValueExternal._data[i * gWriteValueExternal._shape[1] + c] += gateSlot * dPostValueSlot[i];
              else dxDirect[i] += gateSlot * dPostValueSlot[i];
            }
            dGateBySlot[slot] = dGate;
            dWriteProb += dGate * (cache.writeAttn[slot] ?? 0);
          }

          const dWriteAttn = new Float32Array(this.memorySlots);
          for (let slot = 0; slot < this.memorySlots; slot++) {
            dWriteAttn[slot] = dGateBySlot[slot] * cache.writeGate;
          }
          let weighted = 0;
          for (let slot = 0; slot < this.memorySlots; slot++) weighted += (cache.writeAttn[slot] ?? 0) * dWriteAttn[slot];
          const dWriteQuery = new Float32Array(this.units);
          for (let slot = 0; slot < this.memorySlots; slot++) {
            const attn = cache.writeAttn[slot] ?? 0;
            if (attn === 0) continue;
            const dScore = attn * (dWriteAttn[slot] - weighted);
            const key = this.getMemoryColumn(this.memoryKeys, slot);
            if (this.similarity === "dot") {
              const scale = dScore / Math.sqrt(this.units);
              for (let i = 0; i < this.units; i++) {
                dWriteQuery[i] += scale * key[i];
                dPreKeys[i * this.memorySlots + slot] += scale * cache.writeQuery[i];
              }
            } else {
              const gradCosQ = this.cosineGradWrtQ(cache.writeQuery, key);
              const gradCosKey = this.cosineGradWrtKey(cache.writeQuery, key);
              for (let i = 0; i < this.units; i++) {
                dWriteQuery[i] += dScore * gradCosQ[i];
                dPreKeys[i * this.memorySlots + slot] += dScore * gradCosKey[i];
              }
            }
          }

          const dWriteQueryRaw = this.similarity === "cosine"
            ? this.normalizeBackward(cache.writeQueryRaw, dWriteQuery)
            : dWriteQuery;
          this.addOuter(gWriteQuery, dWriteQueryRaw, cache.writeContext);
          const dWriteContextFromQuery = this.matTVecMul(this.writeQueryKernel, dWriteQueryRaw);

          // Use a straight-through estimator here as well: the sigmoid gate often saturates at 1.0
          // in this layer, which otherwise collapses gradients to near-zero even when writes affect
          // future reads.
          const dWriteGatePre = dWriteProb;
          this.addOuter(gWriteGate, new Float32Array([dWriteGatePre]), cache.writeContext);
          gWriteGateBias._data[0] += dWriteGatePre;
          const dWriteContextFromGate = this.matTVecMul(this.writeGateKernel, new Float32Array([dWriteGatePre]));

          for (let i = 0; i < this.units; i++) {
            dxDirect[i] += dWriteContextFromGate[i] + dWriteContextFromQuery[i];
            dMemorySummary[i] += dWriteContextFromGate[this.units + i] + dWriteContextFromQuery[this.units + i];
          }

          const dNewKeyRaw = this.similarity === "cosine"
            ? this.normalizeBackward(cache.newKeyRaw, dNewKey)
            : dNewKey;
          if (cache.writeKeyProjected) {
            if (gWriteKeyExternal) this.addGradToColumn(gWriteKeyExternal, c, dNewKeyRaw);
            else for (let i = 0; i < this.units; i++) dxDirect[i] += dNewKeyRaw[i];
          } else {
            this.addOuter(gQuery, dNewKeyRaw, cache.writeKeySource);
            const dxWriteKey = this.matTVecMul(this.queryKernel, dNewKeyRaw);
            if (gWriteKeyExternal) this.addGradToColumn(gWriteKeyExternal, c, dxWriteKey);
            else for (let i = 0; i < this.units; i++) dxDirect[i] += dxWriteKey[i];
          }
        } else {
          const slot = cache.writeSlot;
          let dPostKeySlot = this.getStateGrad(futureKeyStateGrad, slot);
          let dPostValueSlot = this.getStateGrad(futureValueStateGrad, slot);

          if (this.similarity === "cosine") {
            // Straight-through: cosine normalization can collapse gradients to zero when the
            // post-write vector is already axis-aligned or unit-length. Keep the upstream signal.
          }

          let dWriteGate = 0;
          for (let i = 0; i < this.units; i++) {
            // ACCUMULATE (+=) instead of zeroing to preserve gradients from reads/future writes
            dPreKeys[i * this.memorySlots + slot] += (1 - cache.writeGate) * dPostKeySlot[i];
            dPreValues[i * this.memorySlots + slot] += (1 - cache.writeGate) * dPostValueSlot[i];
            dWriteGate += (cache.newKey[i] - cache.preWriteKey[i]) * dPostKeySlot[i];
            dWriteGate += (cache.newValue[i] - cache.preWriteValue[i]) * dPostValueSlot[i];
            if (gWriteValueExternal) gWriteValueExternal._data[i * gWriteValueExternal._shape[1] + c] += cache.writeGate * dPostValueSlot[i];
            else dxDirect[i] += cache.writeGate * dPostValueSlot[i];
          }

          const dNewKey = new Float32Array(this.units);
          for (let i = 0; i < this.units; i++) dNewKey[i] = cache.writeGate * dPostKeySlot[i];
          const dNewKeyRaw = this.similarity === "cosine"
            ? this.normalizeBackward(cache.newKeyRaw, dNewKey)
            : dNewKey;
          if (cache.writeKeyProjected) {
            if (gWriteKeyExternal) this.addGradToColumn(gWriteKeyExternal, c, dNewKeyRaw);
            else for (let i = 0; i < this.units; i++) dxDirect[i] += dNewKeyRaw[i];
          } else {
            this.addOuter(gQuery, dNewKeyRaw, cache.writeKeySource);
            const dxWriteKey = this.matTVecMul(this.queryKernel, dNewKeyRaw);
            if (gWriteKeyExternal) this.addGradToColumn(gWriteKeyExternal, c, dxWriteKey);
            else for (let i = 0; i < this.units; i++) dxDirect[i] += dxWriteKey[i];
          }

          // Use straight-through estimator (STE) for hard write: pass gradient directly
          // without sigmoid dampening to avoid dead gradient at saturation (gate ≈ 1.0).
          // This allows write gate to learn during eval mode where gate is deterministic.
          const dWriteGatePre = dWriteGate; // STE: dWriteGate * 1 (no sigmoid factor)
          this.addOuter(gWriteGate, new Float32Array([dWriteGatePre]), cache.writeContext);
          gWriteGateBias._data[0] += dWriteGatePre;
          const dWriteContext = this.matTVecMul(this.writeGateKernel, new Float32Array([dWriteGatePre]));
          for (let i = 0; i < this.units; i++) {
            dxDirect[i] += dWriteContext[i];
            dMemorySummary[i] += dWriteContext[this.units + i];
          }
        }
      }

      if (this.mode === "project") {
        let dNeed = 0;
        dNeed += cache.dNeedFromOutput ?? 0;
        for (let i = 0; i < this.units; i++) {
          dNeed += dContext[i] * cache.read[i];
          dRead[i] += dContext[i] * cache.need;
        }
        const dNeedPre = dNeed * cache.need * (1 - cache.need);
        this.addOuter(gNeed!, new Float32Array([dNeedPre]), cache.needInput!);
        if (gNeedBias) gNeedBias._data[0] += dNeedPre;
        const dNeedInput = this.matTVecMul(this.needKernel!, new Float32Array([dNeedPre]));
        for (let i = 0; i < this.units; i++) dRead[i] += dNeedInput[i];
        for (let i = 0; i < this.units; i++) dxDirect[i] += dNeedInput[this.units + i];
      }

      if (cache.memorySummaryValidLength > 0) {
        let nonZero = false;
        for (let i = 0; i < dMemorySummary.length; i++) {
          if (Math.abs(dMemorySummary[i]) > 0) {
            nonZero = true;
            break;
          }
        }
        if (nonZero) {
          const compact = mj.zeros([this.units, this.memorySlots]);
          let activeIndex = 0;
          for (let slot = 0; slot < this.memorySlots; slot++) {
            if (!this.trainingMode && !this.memoryFilled[slot]) continue;
            for (let i = 0; i < this.units; i++) {
              compact._data[i * this.memorySlots + activeIndex] = this.memoryValues._data[i * this.memorySlots + slot];
            }
            activeIndex++;
          }
          this.memorySummaryPooling.setValidLength(cache.memorySummaryValidLength);
          this.memorySummaryPooling.forward(compact);
          this.memorySummaryPooling.backward(mj.matrix([]), Matrix.fromFlat(dMemorySummary, [this.units, 1]));
        }
      }

      const dQuery = new Float32Array(this.units);
      if (cache.readSlots.length > 0) {
        const dAttn = new Array<number>(cache.readSlots.length).fill(0);
        for (let i = 0; i < cache.readSlots.length; i++) dAttn[i] = this.vectorDot(dRead, cache.readSlots[i].value);

        let weighted = 0;
        for (let i = 0; i < cache.readSlots.length; i++) weighted += cache.readSlots[i].attn * dAttn[i];

        for (let i = 0; i < cache.readSlots.length; i++) {
          this.addStateGrad(dPreValues, cache.readSlots[i].slot, dRead, cache.readSlots[i].attn);
          const dScore = cache.readSlots[i].attn * (dAttn[i] - weighted);
          if (this.similarity === "dot") {
            const scale = dScore / Math.sqrt(this.units);
            const dKey = new Float32Array(this.units);
            for (let d = 0; d < this.units; d++) {
              dQuery[d] += scale * cache.readSlots[i].key[d];
              dKey[d] = scale * cache.q[d];
            }
            this.addStateGrad(dPreKeys, cache.readSlots[i].slot, dKey);
          } else {
            const gradCosQ = this.cosineGradWrtQ(cache.q, cache.readSlots[i].key);
            const gradCosKey = this.cosineGradWrtKey(cache.q, cache.readSlots[i].key);
            for (let d = 0; d < this.units; d++) {
              dQuery[d] += dScore * gradCosQ[d];
              gradCosKey[d] *= dScore;
            }
            this.addStateGrad(dPreKeys, cache.readSlots[i].slot, gradCosKey);
          }
        }
      }

      const gradQueryRaw = this.similarity === "cosine" ? this.normalizeBackward(cache.qRaw, dQuery) : dQuery;
      let dxQuery: Vec = new Float32Array(this.units);
      if (cache.readQueryProjected) {
        this.addGradToColumn(gReadQueryExternal, c, gradQueryRaw);
        dxQuery = new Float32Array(gradQueryRaw);
      } else {
        this.addOuter(gQuery, gradQueryRaw, cache.readQuerySource);
        dxQuery = this.matTVecMul(this.queryKernel, gradQueryRaw);
        if (gReadQueryExternal) this.addGradToColumn(gReadQueryExternal, c, dxQuery);
      }

      for (let i = 0; i < this.units; i++) dx._data[i * err._shape[1] + c] = dxDirect[i] + (gReadQueryExternal ? 0 : dxQuery[i]);
      futureKeyStateGrad = dPreKeys;
      futureValueStateGrad = dPreValues;
    }

    this.maybeClip(gQuery, gWriteGate, gWriteGateBias, gWriteQuery, gNeed, gNeedBias, gOutput, gOutputBias);
    this.lastExternalGradients = {
      readQuery: gReadQueryExternal,
      writeKey: gWriteKeyExternal,
      writeValue: gWriteValueExternal,
    };
    
    // Populate .grad for Tape support
    if (this.queryKernel.grad) this.queryKernel.grad.addInPlace(gQuery); else this.queryKernel.grad = gQuery;
    if (this.writeGateKernel.grad) this.writeGateKernel.grad.addInPlace(gWriteGate); else this.writeGateKernel.grad = gWriteGate;
    if (this.writeGateBias.grad) this.writeGateBias.grad.addInPlace(gWriteGateBias); else this.writeGateBias.grad = gWriteGateBias;
    if (this.writeQueryKernel.grad) this.writeQueryKernel.grad.addInPlace(gWriteQuery); else this.writeQueryKernel.grad = gWriteQuery;
    
    if (gNeed && this.needKernel) {
      if (this.needKernel.grad) this.needKernel.grad.addInPlace(gNeed); else this.needKernel.grad = gNeed;
    }
    if (gNeedBias && this.needBias) {
      if (this.needBias.grad) this.needBias.grad.addInPlace(gNeedBias); else this.needBias.grad = gNeedBias;
    }
    if (gOutput && this.outputKernel) {
      if (this.outputKernel.grad) this.outputKernel.grad.addInPlace(gOutput); else this.outputKernel.grad = gOutput;
    }
    if (gOutputBias && this.outputBias) {
      if (this.outputBias.grad) this.outputBias.grad.addInPlace(gOutputBias); else this.outputBias.grad = gOutputBias;
    }

    if (!gradOnly) {
      this.queryKernel.subInPlace(this.optimizerQuery.calculate(gQuery, this.alpha));
      this.writeGateKernel.subInPlace(this.optimizerWriteGate.calculate(gWriteGate, this.alpha));
      this.writeGateBias.subInPlace(this.optimizerWriteGateBias.calculate(gWriteGateBias, this.alpha));
      this.writeQueryKernel.subInPlace(this.optimizerWriteQuery.calculate(gWriteQuery, this.alpha));
      if (gNeed && this.needKernel && this.optimizerNeed) {
        this.needKernel.subInPlace(this.optimizerNeed.calculate(gNeed, this.alpha));
      }
      if (gNeedBias && this.needBias && this.optimizerNeedBias) {
        this.needBias.subInPlace(this.optimizerNeedBias.calculate(gNeedBias, this.alpha));
      }
      if (gOutput && this.outputKernel && this.optimizerOutput) {
        this.outputKernel.subInPlace(this.optimizerOutput.calculate(gOutput, this.alpha));
      }
      if (gOutputBias && this.outputBias && this.optimizerOutputBias) {
        this.outputBias.subInPlace(this.optimizerOutputBias.calculate(gOutputBias, this.alpha));
      }
    }

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

  compile(cfg: { alpha?: number; optimizer?: Optimizer; clipGradient?: number | boolean }): void {
    if (cfg.alpha !== undefined) this.alpha = cfg.alpha;
    if (cfg.clipGradient !== undefined) this.clipGradient = cfg.clipGradient;
    if (cfg.optimizer !== undefined) this.optimizerName = cfg.optimizer;
    if (!this.initialized) return;

    this.optimizerQuery = setOptimizer(this.optimizerName, this.queryKernel._shape, 1e-5);
    this.optimizerWriteGate = setOptimizer(this.optimizerName, this.writeGateKernel._shape, 1e-5);
    this.optimizerWriteGateBias = setOptimizer(this.optimizerName, this.writeGateBias._shape, 1e-5);
    this.optimizerWriteQuery = setOptimizer(this.optimizerName, this.writeQueryKernel._shape, 1e-5);
    this.memorySummaryPooling.compile({ alpha: this.alpha, optimizer: this.optimizerName, clipGradient: this.clipGradient });
    if (this.needKernel) this.optimizerNeed = setOptimizer(this.optimizerName, this.needKernel._shape, 1e-5);
    if (this.needBias) this.optimizerNeedBias = setOptimizer(this.optimizerName, this.needBias._shape, 1e-5);
    if (this.outputKernel) this.optimizerOutput = setOptimizer(this.optimizerName, this.outputKernel._shape, 1e-5);
    if (this.outputBias) this.optimizerOutputBias = setOptimizer(this.optimizerName, this.outputBias._shape, 1e-5);
  }

  private toMatrix2D(data: matrix2d, rows: number, cols: number, name: string): Matrix {
    if (!Array.isArray(data) || data.length !== rows) throw new Error(`MemoryBank.load: invalid ${name} rows`);
    for (let r = 0; r < rows; r++) {
      if (!Array.isArray(data[r]) || data[r].length !== cols) throw new Error(`MemoryBank.load: invalid ${name} cols`);
      for (let c = 0; c < cols; c++) {
        if (!Number.isFinite(data[r][c])) throw new Error(`MemoryBank.load: non-finite value in ${name}`);
      }
    }
    return new Matrix({ array: data });
  }

  save(): MemoryBankSaveData {
    const memoryState = this.getMemoryState();
    const writeGateKernel = this.writeGateKernel._value;
    const writeGateBias = this.writeGateBias._value;
    const writeQueryKernel = this.writeQueryKernel._value;
    const outputKernel = this.outputKernel?._value;
    const outputBias = this.outputBias?._value;
    const needKernel = this.needKernel?._value;
    const needBias = this.needBias?._value;

    return {
      name: this.name,
      status: this.status,
      config: {
        mode: this.mode,
        similarity: this.similarity,
        readTopK: this.readTopK,
        persistence: this.persistence,
        resetOnInit: this.resetOnInit,
        writeEnabled: this.writeEnabled,
        overwriteThreshold: this.overwriteThreshold,
      },
      dimensions: {
        units: this.units,
        memorySlots: this.memorySlots,
        outputUnits: this.outputUnits,
      },
      trainableParams: {
        queryKernel: this.queryKernel._value,
        writeGateKernel,
        writeGateBias,
        writeQueryKernel,
        needKernel,
        needBias,
        outputKernel,
        outputBias,
        memorySummaryPooling: this.memorySummaryPooling.save(),
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
      persistence: this.persistence,
      resetOnInit: this.resetOnInit,
      writeEnabled: this.writeEnabled,
      overwriteThreshold: this.overwriteThreshold,
      alpha: this.alpha,
      optimizer: this.optimizerName,
      clipGradient: this.clipGradient,
      queryKernel: this.queryKernel._value,
      writeGateKernel,
      writeGateBias,
      writeQueryKernel,
      needKernel,
      needBias,
      outputKernel,
      outputBias,
      memorySummaryPooling: this.memorySummaryPooling.save(),
      memoryKeys: memoryState.memoryKeys,
      memoryValues: memoryState.memoryValues,
      memoryFilled: memoryState.memoryFilled,
      memoryUsage: memoryState.memoryUsage,
      memoryAge: memoryState.memoryAge,
      memoryStep: memoryState.memoryStep,
    };
  }

  toKerasConfig() {
    return {
      class_name: "MemoryBank",
      config: {
        units: this.units,
        memorySlots: this.memorySlots,
        outputUnits: this.outputUnits,
        mode: this.mode,
        similarity: this.similarity,
        readTopK: this.readTopK,
        persistence: this.persistence,
        resetOnInit: this.resetOnInit,
        writeEnabled: this.writeEnabled,
        overwriteThreshold: this.overwriteThreshold,
        name: `memory_bank_${Math.floor(Math.random() * 1000)}`,
        trainable: true,
      }
    };
  }

  getWeightsManifest(): { name: string; shape: [number, number]; data: Float32Array }[] {
    const manifest = [
      { name: "queryKernel", shape: this.queryKernel._shape, data: this.queryKernel._data },
      { name: "writeGateKernel", shape: this.writeGateKernel._shape, data: this.writeGateKernel._data },
      { name: "writeGateBias", shape: this.writeGateBias._shape, data: this.writeGateBias._data },
      { name: "writeQueryKernel", shape: this.writeQueryKernel._shape, data: this.writeQueryKernel._data },
      { name: "memoryKeys", shape: this.memoryKeys._shape, data: this.memoryKeys._data },
      { name: "memoryValues", shape: this.memoryValues._shape, data: this.memoryValues._data }
    ];
    if (this.needKernel) manifest.push({ name: "needKernel", shape: this.needKernel._shape, data: this.needKernel._data });
    if (this.needBias) manifest.push({ name: "needBias", shape: this.needBias._shape, data: this.needBias._data });
    if (this.outputKernel) manifest.push({ name: "outputKernel", shape: this.outputKernel._shape, data: this.outputKernel._data });
    if (this.outputBias) manifest.push({ name: "outputBias", shape: this.outputBias._shape, data: this.outputBias._data });

    const poolingManifest = this.memorySummaryPooling.getWeightsManifest();
    for (const item of poolingManifest) {
      manifest.push({ name: `pool_${item.name}`, shape: item.shape, data: item.data });
    }
    return manifest;
  }

  setWeightsFromBinary(weights: Record<string, Float32Array>): void {
    if (weights.queryKernel) this.queryKernel._data.set(weights.queryKernel);
    if (weights.writeGateKernel) this.writeGateKernel._data.set(weights.writeGateKernel);
    if (weights.writeGateBias) this.writeGateBias._data.set(weights.writeGateBias);
    if (weights.writeQueryKernel) this.writeQueryKernel._data.set(weights.writeQueryKernel);
    if (weights.needKernel && this.needKernel) this.needKernel._data.set(weights.needKernel);
    if (weights.needBias && this.needBias) this.needBias._data.set(weights.needBias);
    if (weights.outputKernel && this.outputKernel) this.outputKernel._data.set(weights.outputKernel);
    if (weights.outputBias && this.outputBias) this.outputBias._data.set(weights.outputBias);
    if (weights.memoryKeys) this.memoryKeys._data.set(weights.memoryKeys);
    if (weights.memoryValues) this.memoryValues._data.set(weights.memoryValues);

    const poolWeights: Record<string, Float32Array> = {};
    for (const key of Object.keys(weights)) {
      if (key.startsWith("pool_")) {
        poolWeights[key.substring(5)] = weights[key];
      }
    }
    if (Object.keys(poolWeights).length > 0) {
      this.memorySummaryPooling.setWeightsFromBinary(poolWeights);
    }
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
          memoryDim: data.memoryDim ?? data.units,
          memorySlots: data.memorySlots,
        }
        : null
    );

    this.mode = config.mode ?? data.mode ?? this.mode;
    if (this.mode !== "project" && this.mode !== "concat") this.mode = "project";
    this.similarity = config.similarity ?? data.similarity ?? this.similarity;
    this.readTopK = config.readTopK ?? data.readTopK ?? this.readTopK;
    this.persistence = config.persistence ?? data.persistence ?? this.persistence;
    this.resetOnInit = config.resetOnInit ?? data.resetOnInit ?? this.resetOnInit;
    this.writeEnabled = config.writeEnabled ?? data.writeEnabled ?? this.writeEnabled;
    this.overwriteThreshold = config.overwriteThreshold ?? data.overwriteThreshold ?? this.overwriteThreshold;
    this.alpha = optimizerState.alpha ?? data.alpha ?? this.alpha;
    this.optimizerName = optimizerState.optimizer ?? data.optimizer ?? this.optimizerName;
    this.clipGradient = optimizerState.clipGradient ?? data.clipGradient ?? this.clipGradient;
    this.status = optimizerState.status ?? data.status ?? this.status;
    this.trainingMode = this.status !== "test";

    const units = dimensions.units ?? data.units;
    const memorySlots = dimensions.memorySlots ?? data.memorySlots ?? this.memorySlots;
    const outputUnits = dimensions.outputUnits ?? data.outputUnits ?? units;
    this.assertPositiveInt(units, "units");
    this.assertPositiveInt(memorySlots, "memorySlots");
    this.assertPositiveInt(outputUnits, "outputUnits");
    this.assertPositiveInt(this.readTopK, "readTopK");
    if (this.readTopK > memorySlots) {
      throw new Error(`MemoryBank.load: readTopK must be in [1, memorySlots], got ${this.readTopK} for memorySlots=${memorySlots}`);
    }
    if (this.similarity !== "cosine" && this.similarity !== "dot") {
      throw new Error(`MemoryBank.load: unsupported similarity '${this.similarity}'`);
    }
    this.memorySlots = memorySlots;
    this.configuredOutputUnits = outputUnits;
    this.resetInitializationState();
    this.init(units, outputUnits);

    const queryKernel = trainableParams.queryKernel ?? data.queryKernel;
    const writeGateKernel = trainableParams.writeGateKernel ?? data.writeGateKernel;
    const writeGateBias = trainableParams.writeGateBias ?? data.writeGateBias;
    const writeQueryKernel = trainableParams.writeQueryKernel ?? data.writeQueryKernel;
    const needKernel = trainableParams.needKernel ?? data.needKernel;
    const needBias = trainableParams.needBias ?? data.needBias;
    const outputKernel = trainableParams.outputKernel ?? data.outputKernel;
    const outputBias = trainableParams.outputBias ?? data.outputBias;
    const memorySummaryPooling = trainableParams.memorySummaryPooling ?? data.memorySummaryPooling;

    if (queryKernel) this.queryKernel = this.toMatrix2D(queryKernel, this.units, this.units, "queryKernel");
    if (writeGateKernel) {
      this.writeGateKernel = this.toMatrix2D(writeGateKernel, 1, this.units + this.units, "writeGateKernel");
    }
    if (writeGateBias) {
      this.writeGateBias = this.toMatrix2D(writeGateBias, 1, 1, "writeGateBias");
    }
    if (writeQueryKernel) {
      this.writeQueryKernel = this.toMatrix2D(writeQueryKernel, this.units, this.units + this.units, "writeQueryKernel");
    }
    if (this.needKernel && needKernel) {
      this.needKernel = this.toMatrix2D(needKernel, 1, this.units + this.units, "needKernel");
    }
    if (this.needBias && needBias) {
      this.needBias = this.toMatrix2D(needBias, 1, 1, "needBias");
    }
    if (this.outputKernel && outputKernel) {
      this.outputKernel = this.toMatrix2D(outputKernel, this.outputUnits, this.units + this.units, "outputKernel");
    }
    if (this.outputBias && outputBias) {
      this.outputBias = this.toMatrix2D(outputBias, this.outputUnits, 1, "outputBias");
    }
    if (memorySummaryPooling) {
      this.memorySummaryPooling.load(memorySummaryPooling);
    }

    if (memoryStateData) this.setMemoryState(memoryStateData as MemoryBankState);
    this.compile({ optimizer: this.optimizerName, alpha: this.alpha, clipGradient: this.clipGradient });
  }

  resetMemory(): void {
    if (!this.initialized) return;
    this.memoryKeys = mj.zeros([this.units, this.memorySlots]);
    this.memoryValues = mj.zeros([this.units, this.memorySlots]);
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
      memoryDim: this.units,
      memorySlots: this.memorySlots,
    };
  }

  setMemoryState(state: MemoryBankState): void {
    if (!state || typeof state !== "object") throw new Error("MemoryBank.setMemoryState: invalid state object");
    if (!this.initialized) this.init(state.units, this.configuredOutputUnits ?? state.units);
    if (state.units !== this.units || state.memorySlots !== this.memorySlots) {
      throw new Error("MemoryBank.setMemoryState: dimensions mismatch with current layer configuration");
    }
    if (state.memoryDim !== undefined && state.memoryDim !== this.units) {
      throw new Error("MemoryBank.setMemoryState: memoryDim must match units in the simplified MemoryBank");
    }

    this.memoryKeys = this.toMatrix2D(state.memoryKeys, this.units, this.memorySlots, "memoryKeys");
    this.memoryValues = this.toMatrix2D(state.memoryValues, this.units, this.memorySlots, "memoryValues");
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
    fs.writeFileSync(path, JSON.stringify(this.getMemoryState()), "utf-8");
  }

  loadMemory(path: string): void {
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
    this.writeEnabled = true;
    return this.unfreezeWrites();
  }

  disableWrites(): this {
    this.writeEnabled = false;
    return this.freezeWrites();
  }

  setTrainingMode(training: boolean): this {
    this.trainingMode = training;
    this.status = training ? "train" : "test";
    return this;
  }

  train(): this {
    return this.setTrainingMode(true);
  }

  eval(): this {
    return this.setTrainingMode(false);
  }

  dispose(): void {
    this.cache = [];
    this.debugTrace = [];
    this.lastWriteInfo = null;
    this.sequenceHistory = [];
  }
}
