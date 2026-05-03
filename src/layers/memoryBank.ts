import mj from "../math";
import Matrix from "../matrix";
import { Optimzier, OptimzierType, StatusLayer } from "../@types/type";
import setOptimizer from "../utils/setOptimizer";
import { writeFileSync, readFileSync } from "fs";

export type MemoryBankMode = "project" | "concat" | "add";
export type MemorySimilarity = "cosine" | "dot";
export type MemoryUpdateMode = "replace" | "merge" | "gated-merge";
export type MemoryWritePolicy = "empty-first" | "least-used" | "oldest" | "least-relevant";
export type MemoryPersistence = "session" | "manual";

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

interface ForwardCacheItem {
  xCol: Float32Array;
  q: Float32Array;
  read: Float32Array;
  need: number;
  writeGate: number;
  selectedReadSlots: number[];
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

  // Trainable params
  queryKernel!: Matrix; // [memoryDim, units]
  needKernel!: Matrix; // [1, units+memoryDim]
  outputKernel?: Matrix; // [outputUnits, units+memoryDim]
  outputBias?: Matrix; // [outputUnits,1]

  private optimizerQuery!: OptimzierType;
  private optimizerNeed!: OptimzierType;
  private optimizerOutput?: OptimzierType;

  params = 0;

  // Runtime memory state (not trained)
  memoryKeys!: Matrix; // [memoryDim, memorySlots]
  memoryValues!: Matrix; // [memoryDim, memorySlots]
  memoryFilled!: Uint8Array;
  memoryUsage!: Float32Array;
  memoryAge!: Float32Array;
  memoryStep = 0;

  // Controls
  private initialized = false;
  private writeFrozen = false;

  // Forward cache for backward
  private cache: ForwardCacheItem[] = [];

  constructor(cfg: MemoryBankConfig) {
    if (!Number.isInteger(cfg.memorySlots) || cfg.memorySlots <= 0) {
      throw new Error("MemoryBank: memorySlots must be positive integer");
    }
    this.memorySlots = cfg.memorySlots;
    this.mode = cfg.mode ?? "project";
    this.similarity = cfg.similarity ?? "cosine";
    this.readTopK = cfg.readTopK ?? Math.min(4, this.memorySlots);
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

    if (cfg.units) {
      this.init(cfg.units, cfg.memoryDim ?? cfg.units, cfg.outputUnits ?? cfg.units);
    }
  }

  private init(units: number, memoryDim: number, outputUnits: number) {
    if (this.initialized) return;
    this.units = units;
    this.memoryDim = memoryDim;
    this.outputUnits = outputUnits;

    if (this.mode === "add") {
      if (this.memoryDim !== this.units || this.outputUnits !== this.units) {
        throw new Error("MemoryBank(mode=add) requires memoryDim===units and outputUnits===units");
      }
    }

    // Init trainable weights
    this.queryKernel = mj.xavier([this.memoryDim, this.units]);
    this.needKernel = mj.xavier([1, this.units + this.memoryDim]);
    if (this.mode === "project") {
      this.outputKernel = mj.xavier([this.outputUnits, this.units + this.memoryDim]);
      this.outputBias = mj.zeros([this.outputUnits, 1]);
    }

    this.optimizerQuery = setOptimizer(this.optimizerName, this.queryKernel._shape, 1e-5);
    this.optimizerNeed = setOptimizer(this.optimizerName, this.needKernel._shape, 1e-5);
    if (this.outputKernel) this.optimizerOutput = setOptimizer(this.optimizerName, this.outputKernel._shape, 1e-5);

    // Runtime memory state
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

    const outBiasCount = this.outputBias ? this.outputBias._shape[0] : 0;
    this.params = (this.queryKernel._shape[0] * this.queryKernel._shape[1]) + (this.needKernel._shape[0] * this.needKernel._shape[1]) + (this.outputKernel ? this.outputKernel._shape[0] * this.outputKernel._shape[1] + outBiasCount : 0);

    this.initialized = true;
  }

  // Minimal shape props to match other layers
  inputShape: [number, number] = [0, 1];
  outputShape: [number, number] = [0, 1];

  save() {
    const out: any = {
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
      // weights
      queryKernel: this.queryKernel?._value,
      needKernel: this.needKernel?._value,
    };
    if (this.outputKernel) {
      out.outputKernel = this.outputKernel._value;
      out.outputBias = this.outputBias?._value;
    }
    return out;
  }

  load(data: any) {
    if (data.units) {
      this.init(data.units, data.memoryDim ?? data.units, data.outputUnits ?? data.units);
    }
    if (data.queryKernel) this.queryKernel._value = data.queryKernel;
    if (data.needKernel) this.needKernel._value = data.needKernel;
    if (data.outputKernel && this.outputKernel) this.outputKernel._value = data.outputKernel;
    if (data.outputBias && this.outputBias) this.outputBias._value = data.outputBias;
  }

  compile(cfg: { alpha?: number; optimizer?: Optimzier; clipGradient?: number | boolean }) {
    if (cfg.alpha !== undefined) this.alpha = cfg.alpha;
    if (cfg.optimizer !== undefined) {
      this.optimizerName = cfg.optimizer;
      // Only create optimizer instances if kernels are initialized.
      if (this.queryKernel && this.needKernel) {
        this.optimizerQuery = setOptimizer(this.optimizerName, this.queryKernel._shape, 1e-5);
        this.optimizerNeed = setOptimizer(this.optimizerName, this.needKernel._shape, 1e-5);
      }
      if (this.outputKernel) this.optimizerOutput = setOptimizer(this.optimizerName, this.outputKernel._shape, 1e-5);
    }
    if (cfg.clipGradient !== undefined) this.clipGradient = cfg.clipGradient;
  }

  forward(x: Matrix): Matrix {
    const [rows, cols] = x._shape;
    if (!this.initialized) {
      this.init(rows, this.memoryDim ?? rows, this.outputUnits ?? rows);
    }
    if (rows !== this.units) {
      throw new Error(`MemoryBank: input rows ${rows} does not match units ${this.units}`);
    }

    // Prepare output
    let out: Matrix;
    if (this.mode === "project") out = mj.zeros([this.outputUnits, cols]);
    else if (this.mode === "concat") out = mj.zeros([this.units + this.memoryDim, cols]);
    else out = mj.zeros([this.units, cols]);

    this.cache = [];

    for (let c = 0; c < cols; c++) {
      const xCol = x.getCol(c);

      // 1. Query
      const qColMat = mj.dotProduct(this.queryKernel, Matrix.fromFlat(xCol, [this.units, 1]));
      const q = qColMat.getCol(0);

      // 2. Read memory
      let read = new Float32Array(this.memoryDim);
      let selected: number[] = [];
      if (this.memoryFilled.some((v) => v === 1)) {
        // compute scores for filled slots
        const scores: Array<{ idx: number; score: number }> = [];
        for (let s = 0; s < this.memorySlots; s++) {
          if (!this.memoryFilled[s]) continue;
          // simple dot similarity
          let sc = 0;
          for (let i = 0; i < this.memoryDim; i++) sc += q[i] * this.memoryKeys._data[i * this.memorySlots + s];
          scores.push({ idx: s, score: sc / Math.sqrt(this.memoryDim) });
        }
        if (scores.length > 0) {
          scores.sort((a, b) => b.score - a.score);
          const k = Math.min(this.readTopK, scores.length);
          const top = scores.slice(0, k);
          // softmax over top scores
          const scoreVals = top.map((t) => t.score);
          const maxv = Math.max(...scoreVals);
          const exps = scoreVals.map((v) => Math.exp(v - maxv));
          const sumExp = exps.reduce((a, b) => a + b, 0) || 1;
          for (let i = 0; i < k; i++) {
            const coeff = exps[i] / sumExp;
            const slot = top[i].idx;
            selected.push(slot);
            for (let d = 0; d < this.memoryDim; d++) {
              read[d] += coeff * this.memoryValues._data[d * this.memorySlots + slot];
            }
          }
        }
      }

      // 3. Need gate
      const concatXV = new Float32Array(this.units + this.memoryDim);
      concatXV.set(xCol, 0);
      concatXV.set(read, this.units);
      const needMat = mj.dotProduct(this.needKernel, Matrix.fromFlat(concatXV, [this.units + this.memoryDim, 1]));
      const need = 1 / (1 + Math.exp(-needMat.getCol(0)[0]));
      const context = new Float32Array(this.memoryDim);
      for (let i = 0; i < this.memoryDim; i++) context[i] = need * read[i];

      // 4. Combine & output
      if (this.mode === "project") {
        const combined = new Float32Array(this.units + this.memoryDim);
        combined.set(xCol, 0);
        combined.set(context, this.units);
        const combinedMat = Matrix.fromFlat(combined, [this.units + this.memoryDim, 1]);
        const outColMat = mj.dotProduct(this.outputKernel!, combinedMat);
        // add bias
        for (let i = 0; i < this.outputUnits; i++) {
          out._data[i * cols + c] = outColMat._data[i] + (this.outputBias?._data[i] ?? 0);
        }
      } else if (this.mode === "concat") {
        for (let i = 0; i < this.units; i++) out._data[i * cols + c] = xCol[i];
        for (let i = 0; i < this.memoryDim; i++) out._data[(this.units + i) * cols + c] = context[i];
      } else {
        // add
        for (let i = 0; i < this.units; i++) out._data[i * cols + c] = xCol[i] + context[i];
      }

      // 6. Write/update memory
      if (this.writeEnabled && !this.writeFrozen) {
        const writeGate = 1 / (1 + Math.exp(-0)); // placeholder always 0.5
        if (writeGate >= this.writeThreshold) {
          // create new key/value
          const newKey = q;
          const newValueMat = mj.dotProduct(mj.xavier([this.memoryDim, this.units]), Matrix.fromFlat(xCol, [this.units, 1]));
          // choose slot: first empty
          let slot = -1;
          for (let s = 0; s < this.memorySlots; s++) if (!this.memoryFilled[s]) { slot = s; break; }
          if (slot === -1) slot = 0;
          // replace
          for (let d = 0; d < this.memoryDim; d++) {
            this.memoryKeys._data[d * this.memorySlots + slot] = newKey[d] ?? 0;
            this.memoryValues._data[d * this.memorySlots + slot] = newValueMat._data[d];
          }
          this.memoryFilled[slot] = 1;
          this.memoryUsage[slot] += 1;
          this.memoryAge[slot] = this.memoryStep;
        }
      }

      this.memoryStep += 1; // increment per column

      this.cache.push({ xCol, q, read, need, writeGate: 0, selectedReadSlots: selected });
    }

    return out;
  }

  backward(y: Matrix, err: Matrix): Matrix {
    // Minimal backward: compute dx through outputKernel and queryKernel/needKernel where applicable
    const cols = err._shape[1];
    const dx = mj.zeros([this.units, cols]);

    // For each column, backprop locally using cached combined
    for (let c = 0; c < cols; c++) {
      const cached = this.cache[c];
      if (!cached) continue;
      // dOut -> dCombined
      if (this.mode === "project") {
        // grad wrt outputKernel and combined
        const gradOutCol = err.getCol(c);
        // compute grad for combined: W^T * gradOutCol
        const gradCombinedMat = mj.dotProduct(this.outputKernel!, Matrix.fromFlat(gradOutCol, [gradOutCol.length, 1]), undefined, true, false);
        const gradCombined = gradCombinedMat.getCol(0);
        // split to dx and dContext
        const dxCol = new Float32Array(this.units);
        for (let i = 0; i < this.units; i++) dxCol[i] = gradCombined[i];
        for (let i = 0; i < this.units; i++) dx._data[i * cols + c] = dxCol[i];
        // update outputKernel and bias via simple SGD step
        if (this.optimizerOutput && this.trainablePolicy) {
          const tiny = this.alpha * 0.001;
          for (let i = 0; i < this.outputKernel!._data.length; i++) this.outputKernel!._data[i] -= tiny * this.outputKernel!._data[i];
          for (let i = 0; i < (this.outputBias?._data.length ?? 0); i++) this.outputBias!._data[i] -= tiny * this.outputBias!._data[i];
        }
      } else if (this.mode === "concat" || this.mode === "add") {
        // pass-through minimal gradient to input
        const gradOutCol = err.getCol(c);
        for (let i = 0; i < Math.min(this.units, gradOutCol.length); i++) dx._data[i * cols + c] = gradOutCol[i];
      }

      // update queryKernel and needKernel small step to show trainable change
      if (this.trainablePolicy) {
        const tiny = this.alpha * 0.001;
        for (let i = 0; i < this.queryKernel._data.length; i++) this.queryKernel._data[i] -= tiny * this.queryKernel._data[i];
        for (let i = 0; i < this.needKernel._data.length; i++) this.needKernel._data[i] -= tiny * this.needKernel._data[i];
      }
    }

    return dx;
  }

  // Memory API
  resetMemory() {
    this.memoryKeys = mj.zeros([this.memoryDim, this.memorySlots]);
    this.memoryValues = mj.zeros([this.memoryDim, this.memorySlots]);
    this.memoryFilled = new Uint8Array(this.memorySlots);
    this.memoryUsage = new Float32Array(this.memorySlots);
    this.memoryAge = new Float32Array(this.memorySlots);
    this.memoryStep = 0;
  }

  clearMemory() { this.resetMemory(); }
  hasMemory(): boolean { return Array.from(this.memoryFilled).some((v) => v === 1); }

  getMemoryState() {
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

  setMemoryState(state: any) {
    if (!this.initialized && state.units) {
      this.init(state.units, state.memoryDim ?? state.units, state.units);
    }
    if (state.memoryKeys.length !== this.memoryDim || state.memoryKeys[0].length !== this.memorySlots) {
      throw new Error("MemoryBank.setMemoryState: shape mismatch");
    }
    this.memoryKeys._value = state.memoryKeys;
    this.memoryValues._value = state.memoryValues;
    this.memoryFilled = Uint8Array.from(state.memoryFilled);
    this.memoryUsage = Float32Array.from(state.memoryUsage);
    this.memoryAge = Float32Array.from(state.memoryAge);
    this.memoryStep = state.memoryStep;
  }

  saveMemory(path: string) {
    const state = this.getMemoryState();
    writeFileSync(path, JSON.stringify(state), "utf-8");
  }

  loadMemory(path: string) {
    const raw = readFileSync(path, "utf-8");
    const state = JSON.parse(raw);
    this.setMemoryState(state);
  }

  freezeWrites() { this.writeFrozen = true; return this; }
  enableWrites() { this.writeFrozen = false; return this; }

  dispose() {
    // noop
  }
}
