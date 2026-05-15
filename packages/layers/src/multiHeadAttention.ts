import { mj, engine, Cost, Optimizer, OptimizerType, StatusLayer } from "@oxide-js/core";
import { isNativeAvailable, multiHeadAttentionBackwardNative, multiHeadAttentionForwardNative } from "@oxide-js/core";
import { Matrix } from "@oxide-js/core";
import { setOptimizer } from "@oxide-js/core";
import Dense from "./dense.js";

interface MultiHeadAttentionLayer {
  units: number;
  heads: number;
  seqLen: number;
  alpha?: number;
  status?: StatusLayer;
  clipGradient?: number | boolean;
}

export interface MultiHeadAttentionExternalInputs {
  query?: Matrix;
  key?: Matrix;
  value?: Matrix;
  queryProjected?: boolean;
  keyProjected?: boolean;
  valueProjected?: boolean;
  queryPadMask?: boolean[];
  keyPadMask?: boolean[];
  querySeqLen?: number;
  keySeqLen?: number;
  causal?: boolean;
}

export interface MultiHeadAttentionInputGradients {
  query: Matrix | null;
  key: Matrix | null;
  value: Matrix | null;
}

interface ResolvedAttentionContext {
  querySource: Matrix;
  keySource: Matrix;
  valueSource: Matrix;
  queryProjected: boolean;
  keyProjected: boolean;
  valueProjected: boolean;
  queryPadMask: boolean[];
  keyPadMask: boolean[];
  querySeqLen: number;
  keySeqLen: number;
  batchSize: number;
  totalQueryCols: number;
  totalKeyCols: number;
  causal: boolean;
  recordSourceGradients: boolean;
}

export default class MultiHeadAttention {
  name = "multi head attention layer";
  units: number;
  heads: number;
  headUnits: number;
  seqLen: number;
  alpha: number;
  status: StatusLayer;
  clipGradient: number | boolean;

  q: Matrix;
  k: Matrix;
  v: Matrix;
  wo: Dense;

  inputShape: [number, number];
  outputShape: [number, number];
  params: number;
  loss: number = 0;

  private input: Matrix = mj.matrix([]);
  private padMask: boolean[] = [];
  private hasExternalPadMask: boolean = false;
  private padMaskSourceRef: Float32Array | null = null;
  private attentionInputs: MultiHeadAttentionExternalInputs | null = null;
  private lastContext: ResolvedAttentionContext | null = null;
  private lastInputGradients: MultiHeadAttentionInputGradients = {
    query: null,
    key: null,
    value: null,
  };

  private optimizerQ: OptimizerType;
  private optimizerK: OptimizerType;
  private optimizerV: OptimizerType;
  private optimizerName: Optimizer = "sgd";

  private Q: Matrix;
  private K: Matrix;
  private V: Matrix;
  private concatenated: Matrix;
  private qBuffer: Float32Array = new Float32Array(0);
  private kBuffer: Float32Array = new Float32Array(0);
  private vBuffer: Float32Array = new Float32Array(0);
  private concatenatedBuffer: Float32Array = new Float32Array(0);

  private gradInputBuffer: Matrix;
  private gradContributionBuffer: Matrix;
  private dQAll: Matrix;
  private dKAll: Matrix;
  private dVAll: Matrix;
  private gradInputDataBuffer: Float32Array = new Float32Array(0);
  private gradContributionDataBuffer: Float32Array = new Float32Array(0);
  private dQAllBuffer: Float32Array = new Float32Array(0);
  private dKAllBuffer: Float32Array = new Float32Array(0);
  private dVAllBuffer: Float32Array = new Float32Array(0);

  private gradQBuffer: Matrix;
  private gradKBuffer: Matrix;
  private gradVBuffer: Matrix;

  private attentionBuffer: Float32Array = new Float32Array(0);
  private attentionData: Float32Array = new Float32Array(0);
  private errAttentionBuffer: Float32Array = new Float32Array(0);
  private errScoreBuffer: Float32Array = new Float32Array(0);
  private errAttentionScratch: Float32Array;
  private errScoreScratch: Float32Array;
  private _effectiveSeqLen: number | null = null;

  constructor({ units, heads, seqLen, alpha = 0.1, status = "input", clipGradient = 5.0 }: MultiHeadAttentionLayer) {
    this.units = units;
    this.heads = heads;
    this.seqLen = seqLen;
    this.alpha = alpha;
    this.status = status;
    this.clipGradient = clipGradient;

    this.inputShape = [units, seqLen];
    this.outputShape = [units, seqLen];

    if (this.units % this.heads !== 0) {
      throw new Error(`units (${units}) must be divisible by heads (${heads})`);
    }
    this.headUnits = this.units / this.heads;

    this.q = mj.xavier([this.units, this.units]);
    this.k = mj.xavier([this.units, this.units]);
    this.v = mj.xavier([this.units, this.units]);

    this.wo = new Dense({
      units: this.units,
      outputUnits: this.units,
      activation: "linear",
      alpha,
      clipGradient,
    });

    this.optimizerQ = setOptimizer(this.optimizerName, this.q._shape, alpha);
    this.optimizerK = setOptimizer(this.optimizerName, this.k._shape, alpha);
    this.optimizerV = setOptimizer(this.optimizerName, this.v._shape, alpha);

    this.Q = mj.zeros([this.units, seqLen]);
    this.K = mj.zeros([this.units, seqLen]);
    this.V = mj.zeros([this.units, seqLen]);
    this.concatenated = mj.zeros([this.units, seqLen]);

    this.gradInputBuffer = mj.zeros([this.units, seqLen]);
    this.gradContributionBuffer = mj.zeros([this.units, seqLen]);
    this.dQAll = mj.zeros([this.units, seqLen]);
    this.dKAll = mj.zeros([this.units, seqLen]);
    this.dVAll = mj.zeros([this.units, seqLen]);

    this.gradQBuffer = mj.zeros([this.units, this.units]);
    this.gradKBuffer = mj.zeros([this.units, this.units]);
    this.gradVBuffer = mj.zeros([this.units, this.units]);

    this.params = 3 * this.units * this.units + this.wo.params;
    this.errAttentionScratch = new Float32Array(0);
    this.errScoreScratch = new Float32Array(0);
    this.ensureAttentionBuffers(seqLen, seqLen, seqLen, seqLen, 1);
  }

  compile({ alpha, optimizer, error, clipGradient }: { alpha?: number; optimizer?: Optimizer; error?: Cost; clipGradient?: number | boolean }) {
    if (alpha !== undefined) this.alpha = alpha;
    if (clipGradient !== undefined) this.clipGradient = clipGradient;
    if (optimizer !== undefined) {
      this.optimizerName = optimizer;
      this.optimizerQ = setOptimizer(optimizer, this.q._shape, this.alpha);
      this.optimizerK = setOptimizer(optimizer, this.k._shape, this.alpha);
      this.optimizerV = setOptimizer(optimizer, this.v._shape, this.alpha);
    }
    this.wo.compile({ alpha, optimizer, error, clipGradient });
  }

  getParams(): Matrix[] {
    return [this.q, this.k, this.v, ...this.wo.getParams()];
  }

  update(alpha?: number): void {
    const a = alpha || this.alpha;
    this.optimizerQ.apply(this.q, a);
    this.optimizerK.apply(this.k, a);
    this.optimizerV.apply(this.v, a);
    this.wo.update(a);
  }

  setPadMask(padMask: boolean[]): void {
    this.padMask = padMask;
    this.hasExternalPadMask = true;
    this.padMaskSourceRef = null;
  }

  setAttentionInputs(inputs: MultiHeadAttentionExternalInputs): this {
    this.attentionInputs = inputs;
    return this;
  }

  clearAttentionInputs(): this {
    this.attentionInputs = null;
    return this;
  }

  getLastInputGradients(): MultiHeadAttentionInputGradients {
    return {
      query: this.lastInputGradients.query ? this.lastInputGradients.query.clone() : null,
      key: this.lastInputGradients.key ? this.lastInputGradients.key.clone() : null,
      value: this.lastInputGradients.value ? this.lastInputGradients.value.clone() : null,
    };
  }

  setEffectiveSeqLen(seqLen: number): void {
    this._effectiveSeqLen = seqLen;
  }

  resetEffectiveSeqLen(): void {
    this._effectiveSeqLen = null;
  }

  forward(x?: Matrix): Matrix {
    const ctx = this.resolveAttentionContext(x);
    this.lastContext = ctx;
    this.lastInputGradients = { query: null, key: null, value: null };
    this.input = ctx.querySource;
    this.ensureAttentionBuffers(ctx.totalQueryCols, ctx.totalKeyCols, ctx.querySeqLen, ctx.keySeqLen, ctx.batchSize);

    this.fillProjected(this.Q, this.q, ctx.querySource, ctx.queryProjected);
    this.fillProjected(this.K, this.k, ctx.keySource, ctx.keyProjected);
    this.fillProjected(this.V, this.v, ctx.valueSource, ctx.valueProjected);

    const scale = 1 / Math.sqrt(this.headUnits);
    const canUseNative =
      isNativeAvailable() &&
      ctx.causal &&
      ctx.querySeqLen === ctx.keySeqLen &&
      ctx.totalQueryCols === ctx.totalKeyCols &&
      this.areMasksIdentical(ctx.queryPadMask, ctx.keyPadMask);

    if (canUseNative) {
      multiHeadAttentionForwardNative(
        this.Q._data,
        this.K._data,
        this.V._data,
        ctx.keyPadMask,
        this.heads,
        this.headUnits,
        ctx.querySeqLen,
        ctx.batchSize,
        scale,
        this.concatenated._data,
        this.attentionData
      );
      MultiHeadAttention.zeroMaskedColumnsInPlace(this.concatenated, ctx.queryPadMask);
    } else {
      MultiHeadAttention.forwardFallback(
        this.Q._data,
        this.K._data,
        this.V._data,
        ctx.queryPadMask,
        ctx.keyPadMask,
        this.heads,
        this.headUnits,
        ctx.querySeqLen,
        ctx.keySeqLen,
        ctx.batchSize,
        scale,
        ctx.causal,
        this.concatenated._data,
        this.attentionData
      );
    }

    const tape = engine.tape;
    if (tape) {
      const currentContext = { ...ctx, queryPadMask: [...ctx.queryPadMask], keyPadMask: [...ctx.keyPadMask] };
      const currentScale = scale;
      tape.record([this.Q, this.K, this.V], [this.concatenated], (grad: Matrix) => {
        this.calculateAttentionGradients(grad, currentContext, currentScale);
      });
    }

    this.outputShape = [this.concatenated._shape[0], this.concatenated._shape[1]];
    this.attentionInputs = null;
    return this.wo.forward(this.concatenated);
  }

  backward(y: Matrix, err: Matrix, gradOnly = false): Matrix {
    if (!this.lastContext) {
      throw new Error("MultiHeadAttention.backward: forward must be called before backward");
    }
    const dCat = this.wo.backward(y, err, gradOnly);
    const scale = 1 / Math.sqrt(this.headUnits);
    const gradInput = this.calculateGradients(dCat, this.lastContext, scale);
    if (!gradOnly) this.update(this.alpha);
    return gradInput;
  }

  private calculateAttentionGradients(grad: Matrix, ctx: ResolvedAttentionContext, scale: number): void {
    this.ensureAttentionBuffers(ctx.totalQueryCols, ctx.totalKeyCols, ctx.querySeqLen, ctx.keySeqLen, ctx.batchSize);

    const canUseNative =
      isNativeAvailable() &&
      ctx.causal &&
      ctx.querySeqLen === ctx.keySeqLen &&
      ctx.totalQueryCols === ctx.totalKeyCols &&
      this.areMasksIdentical(ctx.queryPadMask, ctx.keyPadMask);

    if (canUseNative) {
      multiHeadAttentionBackwardNative(
        this.Q._data,
        this.K._data,
        this.V._data,
        this.attentionData,
        grad._data,
        ctx.keyPadMask,
        this.heads,
        this.headUnits,
        ctx.querySeqLen,
        ctx.batchSize,
        scale,
        this.dQAll._data,
        this.dKAll._data,
        this.dVAll._data
      );
      MultiHeadAttention.zeroMaskedColumnsInPlace(this.dQAll, ctx.queryPadMask);
    } else {
      MultiHeadAttention.backwardFallback(
        this.Q._data,
        this.K._data,
        this.V._data,
        this.attentionData,
        grad._data,
        ctx.queryPadMask,
        ctx.keyPadMask,
        this.heads,
        this.headUnits,
        ctx.querySeqLen,
        ctx.keySeqLen,
        ctx.batchSize,
        scale,
        ctx.causal,
        this.dQAll._data,
        this.dKAll._data,
        this.dVAll._data,
        this.errAttentionScratch,
        this.errScoreScratch
      );
    }

    if (this.Q.grad) this.Q.grad.addInPlace(this.dQAll); else this.Q.grad = this.dQAll.clone();
    if (this.K.grad) this.K.grad.addInPlace(this.dKAll); else this.K.grad = this.dKAll.clone();
    if (this.V.grad) this.V.grad.addInPlace(this.dVAll); else this.V.grad = this.dVAll.clone();
  }

  private calculateGradients(dCat: Matrix, ctx: ResolvedAttentionContext, scale: number): Matrix {
    this.calculateAttentionGradients(dCat, ctx, scale);

    const gradQ = ctx.queryProjected ? null : mj.dotProduct(this.dQAll, ctx.querySource, this.gradQBuffer, false, true);
    const gradK = ctx.keyProjected ? null : mj.dotProduct(this.dKAll, ctx.keySource, this.gradKBuffer, false, true);
    const gradV = ctx.valueProjected ? null : mj.dotProduct(this.dVAll, ctx.valueSource, this.gradVBuffer, false, true);

    if (this.clipGradient !== false) {
      const limit = typeof this.clipGradient === "number" ? this.clipGradient : 5.0;
      if (gradQ) this.clipGradients(gradQ, limit);
      if (gradK) this.clipGradients(gradK, limit);
      if (gradV) this.clipGradients(gradV, limit);
    }

    const querySourceGrad = ctx.queryProjected
      ? this.dQAll.clone()
      : mj.dotProduct(this.q, this.dQAll, this.gradInputBuffer, true, false).clone();
    const keySourceGrad = ctx.keyProjected
      ? this.dKAll.clone()
      : mj.dotProduct(this.k, this.dKAll, this.gradContributionBuffer, true, false).clone();
    const valueSourceGrad = ctx.valueProjected
      ? this.dVAll.clone()
      : mj.dotProduct(this.v, this.dVAll, this.gradContributionBuffer, true, false).clone();

    if (gradQ) {
      if (this.q.grad) this.q.grad.addInPlace(gradQ);
      else this.q.grad = gradQ.clone();
    }
    if (gradK) {
      if (this.k.grad) this.k.grad.addInPlace(gradK);
      else this.k.grad = gradK.clone();
    }
    if (gradV) {
      if (this.v.grad) this.v.grad.addInPlace(gradV);
      else this.v.grad = gradV.clone();
    }

    this.lastInputGradients = {
      query: querySourceGrad,
      key: keySourceGrad,
      value: valueSourceGrad,
    };

    if (ctx.recordSourceGradients) {
      this.accumulateMatrixGrad(ctx.querySource, querySourceGrad);
      this.accumulateMatrixGrad(ctx.keySource, keySourceGrad);
      this.accumulateMatrixGrad(ctx.valueSource, valueSourceGrad);
    }

    const gradInput = querySourceGrad.clone();
    if (ctx.keySource === ctx.querySource) gradInput.addInPlace(keySourceGrad);
    if (ctx.valueSource === ctx.querySource) gradInput.addInPlace(valueSourceGrad);

    return gradInput;
  }

  save() {
    return {
      name: this.name,
      status: this.status,
      units: this.units,
      heads: this.heads,
      seqLen: this.seqLen,
      alpha: this.alpha,
      clipGradient: this.clipGradient,
      q: this.q._value,
      k: this.k._value,
      v: this.v._value,
      wo: this.wo.save(),
    };
  }

  toKerasConfig() {
    return {
      class_name: "MultiHeadAttention",
      config: {
        units: this.units,
        heads: this.heads,
        seqLen: this.seqLen,
        name: `multi_head_attention_${Math.floor(Math.random() * 1000)}`,
        trainable: true,
      }
    };
  }

  getWeightsManifest(): { name: string; shape: [number, number]; data: Float32Array }[] {
    const manifest = [
      { name: "q", shape: this.q._shape, data: this.q._data },
      { name: "k", shape: this.k._shape, data: this.k._data },
      { name: "v", shape: this.v._shape, data: this.v._data },
    ];
    const woManifest = this.wo.getWeightsManifest();
    for (const item of woManifest) {
      manifest.push({ name: `wo_${item.name}`, shape: item.shape, data: item.data });
    }
    return manifest;
  }

  setWeightsFromBinary(weights: Record<string, Float32Array>): void {
    if (weights.q) this.q._data.set(weights.q);
    if (weights.k) this.k._data.set(weights.k);
    if (weights.v) this.v._data.set(weights.v);

    const woWeights: Record<string, Float32Array> = {};
    for (const key of Object.keys(weights)) {
      if (key.startsWith("wo_")) {
        woWeights[key.substring(3)] = weights[key];
      }
    }
    if (Object.keys(woWeights).length > 0) {
      this.wo.setWeightsFromBinary(woWeights);
    }
  }

  load(data: any) {
    if (data.q && data.k && data.v) {
      this.q._value = data.q;
      this.k._value = data.k;
      this.v._value = data.v;
    } else if (data.attentionHeads) {
      this.loadLegacyHeads(data.attentionHeads);
    }

    if (data.wo) {
      this.wo.load(data.wo.weight ?? data.wo.kernel, data.wo.bias, data.wo.clipGradient);
    }
    if (data.clipGradient !== undefined) this.clipGradient = data.clipGradient;
    this.optimizerQ = setOptimizer(this.optimizerName, this.q._shape, this.alpha);
    this.optimizerK = setOptimizer(this.optimizerName, this.k._shape, this.alpha);
    this.optimizerV = setOptimizer(this.optimizerName, this.v._shape, this.alpha);
  }

  private resolveAttentionContext(x?: Matrix): ResolvedAttentionContext {
    const cfg = this.attentionInputs;
    const querySource = cfg?.query ?? x;
    const keySource = cfg?.key ?? querySource;
    const valueSource = cfg?.value ?? keySource;

    if (!querySource || !keySource || !valueSource) {
      throw new Error("MultiHeadAttention.forward: query/key/value sources are required");
    }
    this.assertUnits(querySource, "query");
    this.assertUnits(keySource, "key");
    this.assertUnits(valueSource, "value");

    const queryProjected = cfg?.queryProjected ?? false;
    const keyProjected = cfg?.keyProjected ?? false;
    const valueProjected = cfg?.valueProjected ?? false;
    const querySeqLen = this.resolveSeqLen(querySource, cfg?.querySeqLen);
    const keySeqLen = this.resolveSeqLen(keySource, cfg?.keySeqLen);

    if (querySource._shape[1] % querySeqLen !== 0) {
      throw new Error(`MultiHeadAttention.forward: query cols (${querySource._shape[1]}) is not divisible by querySeqLen (${querySeqLen})`);
    }
    if (keySource._shape[1] % keySeqLen !== 0) {
      throw new Error(`MultiHeadAttention.forward: key cols (${keySource._shape[1]}) is not divisible by keySeqLen (${keySeqLen})`);
    }
    if (valueSource._shape[1] % keySeqLen !== 0) {
      throw new Error(`MultiHeadAttention.forward: value cols (${valueSource._shape[1]}) is not divisible by keySeqLen (${keySeqLen})`);
    }

    const batchSize = querySource._shape[1] / querySeqLen;
    const keyBatchSize = keySource._shape[1] / keySeqLen;
    const valueBatchSize = valueSource._shape[1] / keySeqLen;
    if (keyBatchSize !== batchSize || valueBatchSize !== batchSize) {
      throw new Error(
        `MultiHeadAttention.forward: batch mismatch query=${batchSize}, key=${keyBatchSize}, value=${valueBatchSize}`
      );
    }

    const queryPadMask = this.resolveQueryPadMask(querySource, cfg?.queryPadMask);
    const keyPadMask = this.resolveKeyPadMask(querySource, keySource, cfg?.keyPadMask);
    const causal = cfg?.causal ?? !cfg?.key;

    this.padMask = keyPadMask;
    return {
      querySource,
      keySource,
      valueSource,
      queryProjected,
      keyProjected,
      valueProjected,
      queryPadMask,
      keyPadMask,
      querySeqLen,
      keySeqLen,
      batchSize,
      totalQueryCols: querySource._shape[1],
      totalKeyCols: keySource._shape[1],
      causal,
      recordSourceGradients: !!engine.tape,
    };
  }

  private resolveSeqLen(source: Matrix, explicitSeqLen?: number): number {
    if (explicitSeqLen !== undefined) return explicitSeqLen;
    const preferred = this._effectiveSeqLen ?? this.seqLen;
    if (source._shape[1] % preferred === 0) return preferred;
    return source._shape[1];
  }

  private resolveQueryPadMask(querySource: Matrix, explicitPadMask?: boolean[]): boolean[] {
    const totalCols = querySource._shape[1];
    if (explicitPadMask) {
      if (explicitPadMask.length !== totalCols) {
        throw new Error(`MultiHeadAttention.forward: queryPadMask length must be ${totalCols}, got ${explicitPadMask.length}`);
      }
      return explicitPadMask;
    }
    return MultiHeadAttention.detectPadColumns(querySource);
  }

  private resolveKeyPadMask(querySource: Matrix, keySource: Matrix, explicitPadMask?: boolean[]): boolean[] {
    const totalCols = keySource._shape[1];
    if (explicitPadMask) {
      if (explicitPadMask.length !== totalCols) {
        throw new Error(`MultiHeadAttention.forward: keyPadMask length must be ${totalCols}, got ${explicitPadMask.length}`);
      }
      return explicitPadMask;
    }
    if (this.hasExternalPadMask && this.padMask.length === totalCols) {
      return this.padMask;
    }
    if (querySource === keySource && (this.padMask.length !== totalCols || this.padMaskSourceRef !== keySource._data)) {
      this.padMask = MultiHeadAttention.detectPadColumns(keySource, this.padMask);
      this.padMaskSourceRef = keySource._data;
      this.hasExternalPadMask = false;
      return this.padMask;
    }
    return MultiHeadAttention.detectPadColumns(keySource);
  }

  private assertUnits(source: Matrix, label: string): void {
    if (source._shape[0] !== this.units) {
      throw new Error(`MultiHeadAttention.forward: ${label} rows must be ${this.units}, got ${source._shape[0]}`);
    }
  }

  private fillProjected(target: Matrix, weight: Matrix, source: Matrix, projected: boolean): void {
    if (projected) {
      target._data.set(source._data);
      return;
    }
    mj.dotProduct(weight, source, target);
  }

  private accumulateMatrixGrad(target: Matrix, grad: Matrix): void {
    if (target.grad) target.grad.addInPlace(grad);
    else target.grad = grad.clone();
  }

  private areMasksIdentical(a: boolean[], b: boolean[]): boolean {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
      if (a[i] !== b[i]) return false;
    }
    return true;
  }

  private ensureAttentionBuffers(
    totalQueryCols: number,
    totalKeyCols: number,
    querySeqLen: number,
    keySeqLen: number,
    batchSize: number
  ): void {
    const expectedAttentionLen = this.heads * batchSize * keySeqLen * querySeqLen;
    const expectedScratchLen = keySeqLen * querySeqLen;

    this.inputShape = [this.units, totalQueryCols];
    this.outputShape = [this.units, totalQueryCols];
    this.Q = this.bindMatrix("qBuffer", this.units, totalQueryCols);
    this.K = this.bindMatrix("kBuffer", this.units, totalKeyCols);
    this.V = this.bindMatrix("vBuffer", this.units, totalKeyCols);
    this.concatenated = this.bindMatrix("concatenatedBuffer", this.units, totalQueryCols);

    this.gradInputBuffer = this.bindMatrix("gradInputDataBuffer", this.units, totalQueryCols);
    this.gradContributionBuffer = this.bindMatrix("gradContributionDataBuffer", this.units, totalKeyCols);
    this.dQAll = this.bindMatrix("dQAllBuffer", this.units, totalQueryCols);
    this.dKAll = this.bindMatrix("dKAllBuffer", this.units, totalKeyCols);
    this.dVAll = this.bindMatrix("dVAllBuffer", this.units, totalKeyCols);

    if (this.attentionBuffer.length < expectedAttentionLen) {
      const nextCapacity = Math.max(expectedAttentionLen, Math.max(1, this.attentionBuffer.length * 2));
      this.attentionBuffer = new Float32Array(nextCapacity);
    }
    this.attentionData = this.attentionBuffer.subarray(0, expectedAttentionLen);

    if (this.errAttentionBuffer.length < expectedScratchLen) {
      const nextCapacity = Math.max(expectedScratchLen, Math.max(1, this.errAttentionBuffer.length * 2));
      this.errAttentionBuffer = new Float32Array(nextCapacity);
    }
    this.errAttentionScratch = this.errAttentionBuffer.subarray(0, expectedScratchLen);

    if (this.errScoreBuffer.length < expectedScratchLen) {
      const nextCapacity = Math.max(expectedScratchLen, Math.max(1, this.errScoreBuffer.length * 2));
      this.errScoreBuffer = new Float32Array(nextCapacity);
    }
    this.errScoreScratch = this.errScoreBuffer.subarray(0, expectedScratchLen);
  }

  private bindMatrix(
    bufferKey:
      | "qBuffer"
      | "kBuffer"
      | "vBuffer"
      | "concatenatedBuffer"
      | "gradInputDataBuffer"
      | "gradContributionDataBuffer"
      | "dQAllBuffer"
      | "dKAllBuffer"
      | "dVAllBuffer",
    rows: number,
    cols: number
  ): Matrix {
    const requiredLength = rows * cols;
    let buffer = this[bufferKey];
    if (buffer.length < requiredLength) {
      const nextCapacity = Math.max(requiredLength, Math.max(1, buffer.length * 2));
      buffer = new Float32Array(nextCapacity);
      this[bufferKey] = buffer;
    }
    return Matrix.fromFlat(buffer.subarray(0, requiredLength), [rows, cols]);
  }

  private loadLegacyHeads(headsData: Array<{ q: number[][]; k: number[][]; v: number[][] }>) {
    const fusedQ = new Float32Array(this.units * this.units);
    const fusedK = new Float32Array(this.units * this.units);
    const fusedV = new Float32Array(this.units * this.units);

    for (let head = 0; head < this.heads; head++) {
      const legacyHead = headsData[head];
      if (!legacyHead) continue;

      for (let row = 0; row < this.headUnits; row++) {
        const targetRow = head * this.headUnits + row;
        const qRow = legacyHead.q[row] ?? [];
        const kRow = legacyHead.k[row] ?? [];
        const vRow = legacyHead.v[row] ?? [];
        const offset = targetRow * this.units;

        for (let col = 0; col < this.units; col++) {
          fusedQ[offset + col] = qRow[col] ?? 0;
          fusedK[offset + col] = kRow[col] ?? 0;
          fusedV[offset + col] = vRow[col] ?? 0;
        }
      }
    }

    this.q = Matrix.fromFlat(fusedQ, [this.units, this.units]);
    this.k = Matrix.fromFlat(fusedK, [this.units, this.units]);
    this.v = Matrix.fromFlat(fusedV, [this.units, this.units]);
  }

  private clipGradients(m: Matrix, limit: number) {
    mj.clipGradients(m, limit);
  }

  private static detectPadColumns(matrix: Matrix, reuse?: boolean[]): boolean[] {
    const [rows, cols] = matrix._shape;
    const mask = reuse && reuse.length === cols ? reuse : new Array<boolean>(cols);
    mask.fill(true);
    for (let j = 0; j < cols; j++) {
      for (let i = 0; i < rows; i++) {
        if (matrix._data[i * cols + j] !== 0) {
          mask[j] = false;
          break;
        }
      }
    }
    return mask;
  }

  private static zeroMaskedColumnsInPlace(matrix: Matrix, padMask: boolean[]): void {
    const [rows, cols] = matrix._shape;
    for (let col = 0; col < cols; col++) {
      if (!padMask[col]) continue;
      for (let row = 0; row < rows; row++) {
        matrix._data[row * cols + col] = 0;
      }
    }
  }

  private static forwardFallback(
    qData: Float32Array,
    kData: Float32Array,
    vData: Float32Array,
    queryPadMask: boolean[],
    keyPadMask: boolean[],
    heads: number,
    headUnits: number,
    querySeqLen: number,
    keySeqLen: number,
    batchSize: number,
    scale: number,
    causal: boolean,
    outData: Float32Array,
    attentionData: Float32Array
  ): void {
    const totalQueryCols = querySeqLen * batchSize;
    const totalKeyCols = keySeqLen * batchSize;
    outData.fill(0);

    for (let head = 0; head < heads; head++) {
      const rowStart = head * headUnits;
      for (let batch = 0; batch < batchSize; batch++) {
        const queryOffset = batch * querySeqLen;
        const keyOffset = batch * keySeqLen;
        const attnOffset = (head * batchSize + batch) * keySeqLen * querySeqLen;

        for (let qPos = 0; qPos < querySeqLen; qPos++) {
          const qCol = queryOffset + qPos;
          if (queryPadMask[qCol]) {
            for (let kPos = 0; kPos < keySeqLen; kPos++) {
              attentionData[attnOffset + kPos * querySeqLen + qPos] = 0;
            }
            continue;
          }

          let maxScore = -Infinity;
          for (let kPos = 0; kPos < keySeqLen; kPos++) {
            const kCol = keyOffset + kPos;
            const scoreIdx = attnOffset + kPos * querySeqLen + qPos;
            if (keyPadMask[kCol] || (causal && kPos > qPos)) {
              attentionData[scoreIdx] = Number.NEGATIVE_INFINITY;
              continue;
            }
            let score = 0;
            for (let i = 0; i < headUnits; i++) {
              const row = rowStart + i;
              score += kData[row * totalKeyCols + kCol] * qData[row * totalQueryCols + qCol];
            }
            score *= scale;
            attentionData[scoreIdx] = score;
            if (score > maxScore) maxScore = score;
          }

          if (!Number.isFinite(maxScore)) {
            for (let kPos = 0; kPos < keySeqLen; kPos++) {
              attentionData[attnOffset + kPos * querySeqLen + qPos] = 0;
            }
            continue;
          }

          let sumExp = 0;
          for (let kPos = 0; kPos < keySeqLen; kPos++) {
            const scoreIdx = attnOffset + kPos * querySeqLen + qPos;
            const score = attentionData[scoreIdx];
            if (!Number.isFinite(score)) {
              attentionData[scoreIdx] = 0;
              continue;
            }
            const expValue = Math.exp(score - maxScore);
            attentionData[scoreIdx] = expValue;
            sumExp += expValue;
          }

          if (!Number.isFinite(sumExp) || sumExp <= 0) {
            for (let kPos = 0; kPos < keySeqLen; kPos++) {
              attentionData[attnOffset + kPos * querySeqLen + qPos] = 0;
            }
            continue;
          }

          for (let kPos = 0; kPos < keySeqLen; kPos++) {
            attentionData[attnOffset + kPos * querySeqLen + qPos] /= sumExp;
          }

          for (let i = 0; i < headUnits; i++) {
            const row = rowStart + i;
            let sum = 0;
            for (let kPos = 0; kPos < keySeqLen; kPos++) {
              const kCol = keyOffset + kPos;
              sum += vData[row * totalKeyCols + kCol] * attentionData[attnOffset + kPos * querySeqLen + qPos];
            }
            outData[row * totalQueryCols + qCol] = sum;
          }
        }
      }
    }
  }

  private static backwardFallback(
    qData: Float32Array,
    kData: Float32Array,
    vData: Float32Array,
    attentionData: Float32Array,
    dOutData: Float32Array,
    queryPadMask: boolean[],
    keyPadMask: boolean[],
    heads: number,
    headUnits: number,
    querySeqLen: number,
    keySeqLen: number,
    batchSize: number,
    scale: number,
    causal: boolean,
    dQOut: Float32Array,
    dKOut: Float32Array,
    dVOut: Float32Array,
    errAttention: Float32Array,
    errScore: Float32Array
  ): void {
    const totalQueryCols = querySeqLen * batchSize;
    const totalKeyCols = keySeqLen * batchSize;
    dQOut.fill(0);
    dKOut.fill(0);
    dVOut.fill(0);

    for (let head = 0; head < heads; head++) {
      const rowStart = head * headUnits;
      for (let batch = 0; batch < batchSize; batch++) {
        const queryOffset = batch * querySeqLen;
        const keyOffset = batch * keySeqLen;
        const attnOffset = (head * batchSize + batch) * keySeqLen * querySeqLen;
        errAttention.fill(0);
        errScore.fill(0);

        for (let qPos = 0; qPos < querySeqLen; qPos++) {
          const qCol = queryOffset + qPos;
          if (queryPadMask[qCol]) continue;

          for (let i = 0; i < headUnits; i++) {
            const row = rowStart + i;
            const dOutVal = dOutData[row * totalQueryCols + qCol];
            for (let kPos = 0; kPos < keySeqLen; kPos++) {
              const attnIdx = attnOffset + kPos * querySeqLen + qPos;
              const kCol = keyOffset + kPos;
              dVOut[row * totalKeyCols + kCol] += dOutVal * attentionData[attnIdx];
              errAttention[kPos * querySeqLen + qPos] += vData[row * totalKeyCols + kCol] * dOutVal;
            }
          }

          let dot = 0;
          for (let kPos = 0; kPos < keySeqLen; kPos++) {
            const localIdx = kPos * querySeqLen + qPos;
            dot += attentionData[attnOffset + localIdx] * errAttention[localIdx];
          }

          for (let kPos = 0; kPos < keySeqLen; kPos++) {
            const localIdx = kPos * querySeqLen + qPos;
            const kCol = keyOffset + kPos;
            if (keyPadMask[kCol] || (causal && kPos > qPos)) {
              errScore[localIdx] = 0;
              continue;
            }
            errScore[localIdx] = attentionData[attnOffset + localIdx] * (errAttention[localIdx] - dot) * scale;
          }

          for (let i = 0; i < headUnits; i++) {
            const row = rowStart + i;
            let dqSum = 0;
            for (let kPos = 0; kPos < keySeqLen; kPos++) {
              const kCol = keyOffset + kPos;
              const scoreGrad = errScore[kPos * querySeqLen + qPos];
              dqSum += kData[row * totalKeyCols + kCol] * scoreGrad;
              dKOut[row * totalKeyCols + kCol] += qData[row * totalQueryCols + qCol] * scoreGrad;
            }
            dQOut[row * totalQueryCols + qCol] = dqSum;
          }
        }
      }
    }
  }
}
