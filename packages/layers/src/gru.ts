import { Cost, Optimizer, OptimizerType, StatusLayer } from "@oxide-js/core";
import { mj } from "@oxide-js/core";
import { isNativeAvailable, gruForwardNative, gruBackwardNative } from "@oxide-js/core";
import { Matrix } from "@oxide-js/core";
import { setLoss } from "@oxide-js/core";
import { setOptimizer } from "@oxide-js/core";

export interface GRULayerConfig {
  units: number;
  hiddenUnits: number;
  returnSequences?: boolean;
  returnState?: boolean;
  stateful?: boolean;
  bidirectional?: boolean;
  alpha?: number;
  optimizer?: Optimizer;
  status?: StatusLayer;
  clipGradient?: number | boolean;
  loss?: Cost;
}

type DirectionParams = {
  Wxr: Matrix;
  Whr: Matrix;
  br: Matrix;
  Wxz: Matrix;
  Whz: Matrix;
  bz: Matrix;
  Wxh: Matrix;
  Whh: Matrix;
  bh: Matrix;
  optimizerWxr: OptimizerType;
  optimizerWhr: OptimizerType;
  optimizerBr: OptimizerType;
  optimizerWxz: OptimizerType;
  optimizerWhz: OptimizerType;
  optimizerBz: OptimizerType;
  optimizerWxh: OptimizerType;
  optimizerWhh: OptimizerType;
  optimizerBh: OptimizerType;
  hStateful: Matrix;
  xSeq: Float32Array[];
  hSeq: Float32Array[];
  rSeq: Float32Array[];
  zSeq: Float32Array[];
  nSeq: Float32Array[];
  xSeqBuffer: Float32Array;
  hSeqBuffer: Float32Array;
  rSeqBuffer: Float32Array;
  zSeqBuffer: Float32Array;
  nSeqBuffer: Float32Array;
};

export default class GRU {
  name = "gru layer";
  units: number;
  hiddenUnits: number;
  returnSequences: boolean;
  returnState: boolean;
  stateful: boolean;
  bidirectional: boolean;
  inputShape: [number, number];
  outputShape: [number, number];
  params: number;
  loss = 0;
  status: StatusLayer;
  alpha: number;
  clipGradient: number | boolean;

  private optimizerName: Optimizer;
  private lossName: Cost;
  private lossFunc: Function;
  private sumLoss = 0;
  private lossCount = 0;
  private resultBuffer: Matrix = mj.matrix([]);
  private forwardDirection: DirectionParams;
  private backwardDirection?: DirectionParams;
  private batchInputSliceBuffer: Matrix = mj.matrix([]);
  private batchGateXRBuffer: Matrix = mj.matrix([]);
  private batchGateXZBuffer: Matrix = mj.matrix([]);
  private batchGateXNBuffer: Matrix = mj.matrix([]);
  private batchGateSliceRBuffer: Matrix = mj.matrix([]);
  private batchGateSliceZBuffer: Matrix = mj.matrix([]);
  private batchGateSliceNBuffer: Matrix = mj.matrix([]);
  private batchRecRBuffer: Matrix = mj.matrix([]);
  private batchRecZBuffer: Matrix = mj.matrix([]);
  private batchRecNBuffer: Matrix = mj.matrix([]);
  private batchDxStepBuffer: Matrix = mj.matrix([]);
  private batchDhStepBuffer: Matrix = mj.matrix([]);
  private batchOuterInputBuffer: Matrix = mj.matrix([]);
  private batchOuterHiddenBuffer: Matrix = mj.matrix([]);
  private batchBiasGradBuffer: Matrix = mj.matrix([]);
  private batchTransposeProductBuffer: Matrix = mj.matrix([]);
  private errorStepBuffer: Float32Array = new Float32Array(0);
  private batchErrorStepBuffer: Float32Array = new Float32Array(0);
  private splitForwardErrorBuffer: Float32Array = new Float32Array(0);
  private splitBackwardErrorBuffer: Float32Array = new Float32Array(0);
  private batchSplitForwardErrorBuffer: Float32Array = new Float32Array(0);
  private batchSplitBackwardErrorBuffer: Float32Array = new Float32Array(0);

  constructor({
    units,
    hiddenUnits,
    returnSequences = false,
    returnState = false,
    stateful = false,
    bidirectional = false,
    alpha = 0.01,
    optimizer = "adam",
    status = "input",
    clipGradient = 5.0,
    loss = "mse",
  }: GRULayerConfig) {
    this.units = units;
    this.hiddenUnits = hiddenUnits;
    this.returnSequences = returnSequences;
    this.returnState = returnState;
    this.stateful = stateful;
    this.bidirectional = bidirectional;
    this.alpha = alpha;
    this.status = status;
    this.clipGradient = clipGradient;
    this.optimizerName = optimizer;
    this.lossName = loss;
    this.lossFunc = setLoss(loss);

    this.forwardDirection = this.createDirectionParams();
    this.backwardDirection = this.bidirectional ? this.createDirectionParams() : undefined;

    this.inputShape = [units, 0];
    this.outputShape = [this.bidirectional ? hiddenUnits * 2 : hiddenUnits, returnSequences ? 0 : 1];
    const perDirection = 3 * (hiddenUnits * units + hiddenUnits * hiddenUnits + hiddenUnits);
    this.params = this.bidirectional ? perDirection * 2 : perDirection;
  }

  save() {
    return {
      name: this.name,
      units: this.units,
      hiddenUnits: this.hiddenUnits,
      returnSequences: this.returnSequences,
      returnState: this.returnState,
      stateful: this.stateful,
      bidirectional: this.bidirectional,
      alpha: this.alpha,
      optimizer: this.optimizerName,
      status: this.status,
      clipGradient: this.clipGradient,
      loss: this.lossName,
      forward: this.serializeDirection(this.forwardDirection),
      backward: this.backwardDirection ? this.serializeDirection(this.backwardDirection) : null,
    };
  }

  load(data: Record<string, any>) {
    if (!data || typeof data !== "object") {
      throw new Error("GRU.load: expected serialized GRU object.");
    }
    if (!data.forward || typeof data.forward !== "object") {
      throw new Error("GRU.load: expected serialized 'forward' direction.");
    }
    this.assertSerializedDirection(data.forward, "forward");
    this.deserializeDirection(this.forwardDirection, data.forward);
    if (this.backwardDirection) {
      if (!data.backward || typeof data.backward !== "object") {
        throw new Error("GRU.load: expected serialized 'backward' direction for bidirectional GRU.");
      }
      this.assertSerializedDirection(data.backward, "backward");
      this.deserializeDirection(this.backwardDirection, data.backward);
    }
    if (typeof data.clipGradient === "number" || typeof data.clipGradient === "boolean") {
      this.clipGradient = data.clipGradient;
    }
    this.resetOptimizers(this.forwardDirection);
    if (this.backwardDirection) this.resetOptimizers(this.backwardDirection);
  }

  compile({
    alpha,
    optimizer,
    error,
    clipGradient,
  }: {
    alpha?: number;
    optimizer?: Optimizer;
    error?: Cost;
    clipGradient?: number | boolean;
  }) {
    if (alpha !== undefined) this.alpha = alpha;
    if (optimizer !== undefined) {
      this.optimizerName = optimizer;
      this.resetOptimizers(this.forwardDirection);
      if (this.backwardDirection) this.resetOptimizers(this.backwardDirection);
    }
    if (error !== undefined) {
      this.lossName = error;
      this.lossFunc = setLoss(error);
    }
    if (clipGradient !== undefined) this.clipGradient = clipGradient;
  }

  resetState() {
    this.forwardDirection.hStateful._data.fill(0);
    if (this.backwardDirection) this.backwardDirection.hStateful._data.fill(0);
  }

  getState() {
    return {
      forward: this.forwardDirection.hStateful.clone(),
      ...(this.backwardDirection ? { backward: this.backwardDirection.hStateful.clone() } : {}),
    };
  }

  forward(x: Matrix): Matrix {
    if (this.returnState) {
      throw new Error("GRU.forward: returnState=true is not supported yet. Disable returnState for GRU.");
    }
    if (x._shape[0] !== this.units) {
      throw new Error(`GRU.forward: expected input rows ${this.units}, got ${x._shape[0]}`);
    }
    const seqLen = x._shape[1];
    if (seqLen < 1) {
      throw new Error("GRU.forward: expected a non-empty sequence input.");
    }
    const outRows = this.bidirectional ? this.hiddenUnits * 2 : this.hiddenUnits;
    const outCols = this.returnSequences ? seqLen : 1;
    if (this.resultBuffer._shape[0] !== outRows || this.resultBuffer._shape[1] !== outCols) {
      this.resultBuffer = mj.zeros([outRows, outCols]);
    } else {
      this.resultBuffer._data.fill(0);
    }

    this.inputShape = [this.units, seqLen];
    this.outputShape = [outRows, outCols];
    const forwardH = this.runDirectionForward(this.forwardDirection, x, false);
    if (this.returnSequences) {
      for (let t = 0; t < seqLen; t++) {
        this.setColumnData(this.resultBuffer._data, outCols, t, forwardH[t]);
      }
    } else {
      this.resultBuffer._data.set(forwardH[seqLen - 1], 0);
    }

    if (this.backwardDirection) {
      const backwardH = this.runDirectionForward(this.backwardDirection, x, true);
      if (this.returnSequences) {
        for (let t = 0; t < seqLen; t++) {
          const col = this.resultBuffer._data;
          const base = t;
          for (let i = 0; i < this.hiddenUnits; i++) {
            col[(i + this.hiddenUnits) * outCols + base] = backwardH[t][i];
          }
        }
      } else {
        this.resultBuffer._data.set(backwardH[0], this.hiddenUnits);
      }
    }

    return this.resultBuffer;
  }

  forwardBatch(x: Matrix, batchSize: number): Matrix {
    this.assertBatchInputSupported(x, batchSize);
    const totalCols = x._shape[1];
    const seqLen = totalCols / batchSize;
    const outRows = this.bidirectional ? this.hiddenUnits * 2 : this.hiddenUnits;
    const outCols = this.returnSequences ? totalCols : batchSize;
    if (this.resultBuffer._shape[0] !== outRows || this.resultBuffer._shape[1] !== outCols) {
      this.resultBuffer = mj.zeros([outRows, outCols]);
    } else {
      this.resultBuffer._data.fill(0);
    }

    this.inputShape = [this.units, totalCols];
    this.outputShape = [outRows, outCols];
    const forwardH = this.runDirectionForwardBatch(this.forwardDirection, x, batchSize, false);
    if (this.returnSequences) {
      for (let t = 0; t < seqLen; t++) {
        this.writeColumnBlock(this.resultBuffer, t * batchSize, batchSize, forwardH[t]);
      }
    } else {
      this.resultBuffer._data.set(forwardH[seqLen - 1]);
    }

    if (this.backwardDirection) {
      const backwardH = this.runDirectionForwardBatch(this.backwardDirection, x, batchSize, true);
      if (this.returnSequences) {
        for (let t = 0; t < seqLen; t++) {
          this.writeColumnBlockOffset(this.resultBuffer, t * batchSize, batchSize, this.hiddenUnits, backwardH[t]);
        }
      } else {
        this.resultBuffer._data.set(backwardH[0], this.hiddenUnits * batchSize);
      }
    }

    return this.resultBuffer;
  }

  backward(y: Matrix, err: Matrix): Matrix {
    const seqLen = this.inputShape[1];
    if (seqLen <= 0 || this.forwardDirection.hSeq.length !== seqLen + 1) {
      throw new Error("GRU.backward: forward must be called before backward.");
    }
    const external = this.resolveError(y, err, seqLen);
    const extForward = this.buildErrorViews(this.ensureSplitErrorBuffer("forward", seqLen * this.hiddenUnits), seqLen, this.hiddenUnits);
    this.splitForwardErrorBuffer.fill(0, 0, seqLen * this.hiddenUnits);
    const extBackward = this.backwardDirection
      ? this.buildErrorViews(this.ensureSplitErrorBuffer("backward", seqLen * this.hiddenUnits), seqLen, this.hiddenUnits)
      : undefined;
    if (extBackward) this.splitBackwardErrorBuffer.fill(0, 0, seqLen * this.hiddenUnits);

    if (this.returnSequences) {
      for (let t = 0; t < seqLen; t++) {
        for (let i = 0; i < this.hiddenUnits; i++) extForward[t][i] = external[t][i];
        if (extBackward) {
          for (let i = 0; i < this.hiddenUnits; i++) extBackward[t][i] = external[t][i + this.hiddenUnits];
        }
      }
    } else {
      for (let i = 0; i < this.hiddenUnits; i++) extForward[seqLen - 1][i] = external[seqLen - 1][i];
      if (extBackward) {
        for (let i = 0; i < this.hiddenUnits; i++) extBackward[0][i] = external[0][i + this.hiddenUnits];
      }
    }

    const dxForward = this.runDirectionBackward(this.forwardDirection, extForward, false);
    const dx = dxForward.slice();
    if (this.backwardDirection && extBackward) {
      const dxBackward = this.runDirectionBackward(this.backwardDirection, extBackward, true);
      for (let i = 0; i < dx.length; i++) dx[i] += dxBackward[i];
    }
    return Matrix.fromFlat(dx, [this.units, seqLen]);
  }

  backwardBatch(y: Matrix, err: Matrix, batchSize: number): Matrix {
    const totalCols = this.inputShape[1];
    this.assertBatchInputSupportedShape(batchSize, totalCols);
    const seqLen = totalCols / batchSize;
    if (this.forwardDirection.hSeq.length !== seqLen + 1) {
      throw new Error("GRU.backwardBatch: forwardBatch must be called before backwardBatch.");
    }

    const external = this.resolveBatchError(y, err, seqLen, batchSize);
    const stepWidth = this.hiddenUnits * batchSize;
    const extForward = this.buildErrorViews(
      this.ensureBatchSplitErrorBuffer("forward", seqLen * stepWidth),
      seqLen,
      stepWidth
    );
    this.batchSplitForwardErrorBuffer.fill(0, 0, seqLen * stepWidth);
    const extBackward = this.backwardDirection
      ? this.buildErrorViews(this.ensureBatchSplitErrorBuffer("backward", seqLen * stepWidth), seqLen, stepWidth)
      : undefined;
    if (extBackward) this.batchSplitBackwardErrorBuffer.fill(0, 0, seqLen * stepWidth);

    if (this.returnSequences) {
      for (let t = 0; t < seqLen; t++) {
        for (let i = 0; i < this.hiddenUnits * batchSize; i++) extForward[t][i] = external[t][i];
        if (extBackward) {
          for (let i = 0; i < this.hiddenUnits * batchSize; i++) {
            extBackward[t][i] = external[t][i + this.hiddenUnits * batchSize];
          }
        }
      }
    } else {
      extForward[seqLen - 1].set(external[seqLen - 1].subarray(0, this.hiddenUnits * batchSize));
      if (extBackward) {
        extBackward[0].set(external[0].subarray(this.hiddenUnits * batchSize));
      }
    }

    const dxForward = this.runDirectionBackwardBatch(this.forwardDirection, extForward, batchSize, false);
    const dx = dxForward.slice();
    if (this.backwardDirection && extBackward) {
      const dxBackward = this.runDirectionBackwardBatch(this.backwardDirection, extBackward, batchSize, true);
      for (let i = 0; i < dx.length; i++) dx[i] += dxBackward[i];
    }
    return Matrix.fromFlat(dx, [this.units, totalCols]);
  }

  resetLoss() {
    this.sumLoss = 0;
    this.lossCount = 0;
    this.loss = 0;
  }

  private resolveError(y: Matrix, err: Matrix, seqLen: number): Float32Array[] {
    let effectiveErr = err;
    if (this.status === "output") {
      const [lossValue, outputErr] = this.lossFunc(y, this.resultBuffer);
      this.lossCount++;
      this.sumLoss += lossValue;
      this.loss = this.sumLoss / this.lossCount;
      effectiveErr = outputErr;
    }
    const expectedRows = this.bidirectional ? this.hiddenUnits * 2 : this.hiddenUnits;
    const expectedCols = this.returnSequences ? seqLen : 1;
    if (effectiveErr._shape[0] !== expectedRows || effectiveErr._shape[1] !== expectedCols) {
      throw new Error(
        `GRU.backward: error shape mismatch, expected [${expectedRows},${expectedCols}], got [${effectiveErr._shape[0]},${effectiveErr._shape[1]}]`
      );
    }
    const out = this.buildErrorViews(this.ensureErrorBuffer(seqLen * expectedRows), seqLen, expectedRows);
    this.errorStepBuffer.fill(0, 0, seqLen * expectedRows);
    if (this.returnSequences) {
      for (let t = 0; t < seqLen; t++) {
        for (let i = 0; i < expectedRows; i++) out[t][i] = effectiveErr._data[i * seqLen + t];
      }
    } else {
      if (this.bidirectional) {
        for (let i = 0; i < this.hiddenUnits; i++) out[seqLen - 1][i] = effectiveErr._data[i];
        for (let i = 0; i < this.hiddenUnits; i++) out[0][i + this.hiddenUnits] = effectiveErr._data[i + this.hiddenUnits];
      } else {
        for (let i = 0; i < expectedRows; i++) out[seqLen - 1][i] = effectiveErr._data[i];
      }
    }
    return out;
  }

  dispose() {
    this.batchInputSliceBuffer = undefined as any;
    this.batchGateXRBuffer = undefined as any;
    this.batchGateXZBuffer = undefined as any;
    this.batchGateXNBuffer = undefined as any;
    this.batchGateSliceRBuffer = undefined as any;
    this.batchGateSliceZBuffer = undefined as any;
    this.batchGateSliceNBuffer = undefined as any;
    this.batchRecRBuffer = undefined as any;
    this.batchRecZBuffer = undefined as any;
    this.batchRecNBuffer = undefined as any;
    this.batchDxStepBuffer = undefined as any;
    this.batchDhStepBuffer = undefined as any;
    this.batchOuterInputBuffer = undefined as any;
    this.batchOuterHiddenBuffer = undefined as any;
    this.batchBiasGradBuffer = undefined as any;
    this.batchTransposeProductBuffer = undefined as any;

    this.errorStepBuffer = new Float32Array(0);
    this.batchErrorStepBuffer = new Float32Array(0);
    this.splitForwardErrorBuffer = new Float32Array(0);
    this.splitBackwardErrorBuffer = new Float32Array(0);
    this.batchSplitForwardErrorBuffer = new Float32Array(0);
    this.batchSplitBackwardErrorBuffer = new Float32Array(0);

    const disposeDir = (dir: DirectionParams) => {
      dir.xSeqBuffer = new Float32Array(0);
      dir.hSeqBuffer = new Float32Array(0);
      dir.rSeqBuffer = new Float32Array(0);
      dir.zSeqBuffer = new Float32Array(0);
      dir.nSeqBuffer = new Float32Array(0);
      dir.xSeq = [];
      dir.hSeq = [];
      dir.rSeq = [];
      dir.zSeq = [];
      dir.nSeq = [];
    };

    if (this.forwardDirection) disposeDir(this.forwardDirection);
    if (this.backwardDirection) disposeDir(this.backwardDirection);
  }

  private resolveBatchError(y: Matrix, err: Matrix, seqLen: number, batchSize: number): Float32Array[] {
    let effectiveErr = err;
    if (this.status === "output") {
      const [lossValue, outputErr] = this.lossFunc(y, this.resultBuffer);
      this.lossCount++;
      this.sumLoss += lossValue;
      this.loss = this.sumLoss / this.lossCount;
      effectiveErr = outputErr;
    }
    const expectedRows = this.bidirectional ? this.hiddenUnits * 2 : this.hiddenUnits;
    const expectedCols = this.returnSequences ? seqLen * batchSize : batchSize;
    if (effectiveErr._shape[0] !== expectedRows || effectiveErr._shape[1] !== expectedCols) {
      throw new Error(
        `GRU.backwardBatch: error shape mismatch, expected [${expectedRows},${expectedCols}], got [${effectiveErr._shape[0]},${effectiveErr._shape[1]}]`
      );
    }
    const perStep = this.buildErrorViews(
      this.ensureBatchErrorBuffer(seqLen * expectedRows * batchSize),
      seqLen,
      expectedRows * batchSize
    );
    this.batchErrorStepBuffer.fill(0, 0, seqLen * expectedRows * batchSize);
    if (this.returnSequences) {
      for (let t = 0; t < seqLen; t++) {
        this.copyColumnBlockToArray(effectiveErr, t * batchSize, batchSize, perStep[t]);
      }
    } else {
      perStep[seqLen - 1].set(effectiveErr._data.subarray(0, this.hiddenUnits * batchSize));
      if (this.bidirectional) {
        perStep[0].set(effectiveErr._data.subarray(this.hiddenUnits * batchSize), this.hiddenUnits * batchSize);
      }
    }
    return perStep;
  }

  private runDirectionForward(direction: DirectionParams, x: Matrix, reverse: boolean): Float32Array[] {
    const seqLen = x._shape[1];
    this.ensureDirectionSequenceBuffers(direction, seqLen, 1);
    const h0 = direction.hSeq[0];
    h0.fill(0);
    if (this.stateful) h0.set(direction.hStateful._data);

    const outputs: Float32Array[] = new Array(seqLen);
    for (let step = 0; step < seqLen; step++) {
      const t = reverse ? seqLen - 1 - step : step;
      const x_t = direction.xSeq[step];
      this.copyColumnToArray(x, t, x_t);
      const hPrev = direction.hSeq[step];
      const r = direction.rSeq[step];
      const z = direction.zSeq[step];
      const n = direction.nSeq[step];
      const h = direction.hSeq[step + 1];

      for (let i = 0; i < this.hiddenUnits; i++) {
        const rPre = this.gatePre(direction.Wxr, direction.Whr, direction.br, i, x_t, hPrev);
        const zPre = this.gatePre(direction.Wxz, direction.Whz, direction.bz, i, x_t, hPrev);
        r[i] = this.sigmoid(rPre);
        z[i] = this.sigmoid(zPre);
      }

      for (let i = 0; i < this.hiddenUnits; i++) {
        let hMix = 0;
        const whhOffset = i * this.hiddenUnits;
        for (let j = 0; j < this.hiddenUnits; j++) {
          hMix += direction.Whh._data[whhOffset + j] * (r[j] * hPrev[j]);
        }
        let xTerm = direction.bh._data[i];
        const wxhOffset = i * this.units;
        for (let j = 0; j < this.units; j++) xTerm += direction.Wxh._data[wxhOffset + j] * x_t[j];
        n[i] = Math.tanh(xTerm + hMix);
        h[i] = (1 - z[i]) * n[i] + z[i] * hPrev[i];
      }
      outputs[t] = h;
    }

    if (this.stateful) {
      const state = reverse ? direction.hSeq[seqLen] : outputs[seqLen - 1];
      direction.hStateful._data.set(state);
    }
    return outputs;
  }

  private runDirectionForwardBatch(
    direction: DirectionParams,
    x: Matrix,
    batchSize: number,
    reverse: boolean
  ): Float32Array[] {
    const totalCols = x._shape[1];
    const seqLen = totalCols / batchSize;
    this.ensureBatchForwardBuffers(batchSize, totalCols);
    this.ensureDirectionSequenceBuffers(direction, seqLen, batchSize);
    const xGateR = this.batchGateXRBuffer;
    const xGateZ = this.batchGateXZBuffer;
    const xGateN = this.batchGateXNBuffer;
    xGateR._data.fill(0);
    xGateZ._data.fill(0);
    xGateN._data.fill(0);
    if (!isNativeAvailable()) {
      mj.dotProduct(direction.Wxr, x, xGateR);
      mj.dotProduct(direction.Wxz, x, xGateZ);
      mj.dotProduct(direction.Wxh, x, xGateN);

      mj.addBias(xGateR, direction.br);
      mj.addBias(xGateZ, direction.bz);
      mj.addBias(xGateN, direction.bh);
    }

    const h0View = direction.hSeq[0];
    h0View.fill(0);
    if (this.stateful && batchSize === 1) h0View.set(direction.hStateful._data);

    if (
      isNativeAvailable() &&
      gruForwardNative(
        direction.Wxr._data,
        direction.Whr._data,
        direction.br._data,
        direction.Wxz._data,
        direction.Whz._data,
        direction.bz._data,
        direction.Wxh._data,
        direction.Whh._data,
        direction.bh._data,
        x._data,
        h0View,
        this.hiddenUnits,
        this.units,
        seqLen,
        batchSize,
        direction.hSeqBuffer,
        direction.rSeqBuffer,
        direction.zSeqBuffer,
        direction.nSeqBuffer
      )
    ) {
      const outputs: Float32Array[] = new Array(seqLen);
      for (let step = 0; step < seqLen; step++) {
        const t = reverse ? seqLen - 1 - step : step;
        outputs[t] = direction.hSeq[step + 1];
        direction.xSeq[step].set(x._data.subarray(t * batchSize * this.units, (t + 1) * batchSize * this.units));
      }
      if (this.stateful && batchSize === 1) {
        const state = reverse ? direction.hSeq[seqLen] : outputs[seqLen - 1];
        direction.hStateful._data.set(state);
      }
      return outputs;
    }

    const outputs: Float32Array[] = new Array(seqLen);
    for (let step = 0; step < seqLen; step++) {
      const t = reverse ? seqLen - 1 - step : step;
      const colOffset = t * batchSize;
      this.copyColumnBlock(x, colOffset, batchSize, this.batchInputSliceBuffer);
      this.copyColumnBlock(xGateR, colOffset, batchSize, this.batchGateSliceRBuffer);
      this.copyColumnBlock(xGateZ, colOffset, batchSize, this.batchGateSliceZBuffer);
      this.copyColumnBlock(xGateN, colOffset, batchSize, this.batchGateSliceNBuffer);

      const hPrev = Matrix.fromFlat(direction.hSeq[step], [this.hiddenUnits, batchSize]);
      mj.dotProduct(direction.Whr, hPrev, this.batchRecRBuffer);
      mj.dotProduct(direction.Whz, hPrev, this.batchRecZBuffer);

      const r = direction.rSeq[step];
      const z = direction.zSeq[step];
      const n = direction.nSeq[step];
      const h = direction.hSeq[step + 1];

      for (let idx = 0; idx < h.length; idx++) {
        r[idx] = this.sigmoid(this.batchGateSliceRBuffer._data[idx] + this.batchRecRBuffer._data[idx]);
        z[idx] = this.sigmoid(this.batchGateSliceZBuffer._data[idx] + this.batchRecZBuffer._data[idx]);
      }

      const rMulHPrev = this.batchDhStepBuffer._data.subarray(0, this.hiddenUnits * batchSize);
      for (let idx = 0; idx < rMulHPrev.length; idx++) rMulHPrev[idx] = r[idx] * direction.hSeq[step][idx];
      const rMulHPrevMatrix = Matrix.fromFlat(rMulHPrev, [this.hiddenUnits, batchSize]);
      mj.dotProduct(direction.Whh, rMulHPrevMatrix, this.batchRecNBuffer);

      for (let idx = 0; idx < h.length; idx++) {
        n[idx] = Math.tanh(this.batchGateSliceNBuffer._data[idx] + this.batchRecNBuffer._data[idx]);
        h[idx] = (1 - z[idx]) * n[idx] + z[idx] * direction.hSeq[step][idx];
      }
      direction.xSeq[step].set(this.batchInputSliceBuffer._data);
      outputs[t] = h;
    }

    if (this.stateful && batchSize === 1) {
      const state = reverse ? direction.hSeq[seqLen] : outputs[seqLen - 1];
      direction.hStateful._data.set(state);
    }
    return outputs;
  }

  private runDirectionBackward(direction: DirectionParams, externalError: Float32Array[], reverse: boolean): Float32Array {
    const seqLen = this.inputShape[1];
    const dx = new Float32Array(this.units * seqLen);
    const dWxr = Matrix.fromFlat(new Float32Array(this.hiddenUnits * this.units), [this.hiddenUnits, this.units]);
    const dWhr = Matrix.fromFlat(new Float32Array(this.hiddenUnits * this.hiddenUnits), [this.hiddenUnits, this.hiddenUnits]);
    const dBr = Matrix.fromFlat(new Float32Array(this.hiddenUnits), [this.hiddenUnits, 1]);
    const dWxz = Matrix.fromFlat(new Float32Array(this.hiddenUnits * this.units), [this.hiddenUnits, this.units]);
    const dWhz = Matrix.fromFlat(new Float32Array(this.hiddenUnits * this.hiddenUnits), [this.hiddenUnits, this.hiddenUnits]);
    const dBz = Matrix.fromFlat(new Float32Array(this.hiddenUnits), [this.hiddenUnits, 1]);
    const dWxh = Matrix.fromFlat(new Float32Array(this.hiddenUnits * this.units), [this.hiddenUnits, this.units]);
    const dWhh = Matrix.fromFlat(new Float32Array(this.hiddenUnits * this.hiddenUnits), [this.hiddenUnits, this.hiddenUnits]);
    const dBh = Matrix.fromFlat(new Float32Array(this.hiddenUnits), [this.hiddenUnits, 1]);

    let dhNext = new Float32Array(this.hiddenUnits);
    const dhBuffer = new Float32Array(this.hiddenUnits);
    const dn = new Float32Array(this.hiddenUnits);
    const dz = new Float32Array(this.hiddenUnits);
    const daN = new Float32Array(this.hiddenUnits);
    const rMulHPrev = new Float32Array(this.hiddenUnits);
    const dRhPrev = new Float32Array(this.hiddenUnits);
    const drFromN = new Float32Array(this.hiddenUnits);
    const dhPrevFromN = new Float32Array(this.hiddenUnits);
    const daR = new Float32Array(this.hiddenUnits);
    const daZ = new Float32Array(this.hiddenUnits);
    let dhPrev = new Float32Array(this.hiddenUnits);
    for (let step = seqLen - 1; step >= 0; step--) {
      const t = reverse ? seqLen - 1 - step : step;
      const hPrev = direction.hSeq[step];
      const x_t = direction.xSeq[step];
      const r = direction.rSeq[step];
      const z = direction.zSeq[step];
      const n = direction.nSeq[step];
      const dh = dhBuffer;
      dh.set(externalError[t]);
      for (let i = 0; i < this.hiddenUnits; i++) dh[i] += dhNext[i];

      for (let i = 0; i < this.hiddenUnits; i++) {
        dn[i] = dh[i] * (1 - z[i]);
        dz[i] = dh[i] * (hPrev[i] - n[i]);
        daN[i] = dn[i] * (1 - n[i] * n[i]);
      }

      for (let i = 0; i < this.hiddenUnits; i++) rMulHPrev[i] = r[i] * hPrev[i];

      this.outerAccumulate(dWxh._data, this.hiddenUnits, this.units, daN, x_t);
      this.outerAccumulate(dWhh._data, this.hiddenUnits, this.hiddenUnits, daN, rMulHPrev);
      for (let i = 0; i < this.hiddenUnits; i++) dBh._data[i] += daN[i];

      for (let j = 0; j < this.hiddenUnits; j++) {
        let val = 0;
        for (let i = 0; i < this.hiddenUnits; i++) {
          val += direction.Whh._data[i * this.hiddenUnits + j] * daN[i];
        }
        dRhPrev[j] = val;
      }
      for (let i = 0; i < this.hiddenUnits; i++) {
        drFromN[i] = dRhPrev[i] * hPrev[i];
        dhPrevFromN[i] = dRhPrev[i] * r[i];
      }

      for (let i = 0; i < this.hiddenUnits; i++) {
        daR[i] = drFromN[i] * r[i] * (1 - r[i]);
        daZ[i] = dz[i] * z[i] * (1 - z[i]);
      }

      this.outerAccumulate(dWxr._data, this.hiddenUnits, this.units, daR, x_t);
      this.outerAccumulate(dWhr._data, this.hiddenUnits, this.hiddenUnits, daR, hPrev);
      this.outerAccumulate(dWxz._data, this.hiddenUnits, this.units, daZ, x_t);
      this.outerAccumulate(dWhz._data, this.hiddenUnits, this.hiddenUnits, daZ, hPrev);
      for (let i = 0; i < this.hiddenUnits; i++) {
        dBr._data[i] += daR[i];
        dBz._data[i] += daZ[i];
      }

      for (let j = 0; j < this.units; j++) {
        let val = 0;
        for (let i = 0; i < this.hiddenUnits; i++) {
          val += direction.Wxh._data[i * this.units + j] * daN[i];
          val += direction.Wxr._data[i * this.units + j] * daR[i];
          val += direction.Wxz._data[i * this.units + j] * daZ[i];
        }
        dx[j * seqLen + t] = val;
      }

      for (let j = 0; j < this.hiddenUnits; j++) {
        let val = 0;
        for (let i = 0; i < this.hiddenUnits; i++) {
          val += direction.Whr._data[i * this.hiddenUnits + j] * daR[i];
          val += direction.Whz._data[i * this.hiddenUnits + j] * daZ[i];
        }
        dhPrev[j] = val + dhPrevFromN[j] + dh[j] * z[j];
      }
      const prevDhNext = dhNext;
      dhNext = dhPrev;
      dhPrev = prevDhNext;
    }

    this.clipGradientsIfNeeded(dWxr, dWhr, dBr, dWxz, dWhz, dBz, dWxh, dWhh, dBh);
    direction.Wxr.subInPlace(direction.optimizerWxr.calculate(dWxr, this.alpha));
    direction.Whr.subInPlace(direction.optimizerWhr.calculate(dWhr, this.alpha));
    direction.br.subInPlace(direction.optimizerBr.calculate(dBr, this.alpha));
    direction.Wxz.subInPlace(direction.optimizerWxz.calculate(dWxz, this.alpha));
    direction.Whz.subInPlace(direction.optimizerWhz.calculate(dWhz, this.alpha));
    direction.bz.subInPlace(direction.optimizerBz.calculate(dBz, this.alpha));
    direction.Wxh.subInPlace(direction.optimizerWxh.calculate(dWxh, this.alpha));
    direction.Whh.subInPlace(direction.optimizerWhh.calculate(dWhh, this.alpha));
    direction.bh.subInPlace(direction.optimizerBh.calculate(dBh, this.alpha));
    return dx;
  }

  private runDirectionBackwardBatch(
    direction: DirectionParams,
    externalError: Float32Array[],
    batchSize: number,
    reverse: boolean
  ): Float32Array {
    const totalCols = this.inputShape[1];
    const seqLen = totalCols / batchSize;
    const dx = new Float32Array(this.units * totalCols);
    
    const dWxr = Matrix.fromFlat(new Float32Array(this.hiddenUnits * this.units), [this.hiddenUnits, this.units]);
    const dWhr = Matrix.fromFlat(new Float32Array(this.hiddenUnits * this.hiddenUnits), [this.hiddenUnits, this.hiddenUnits]);
    const dBr = Matrix.fromFlat(new Float32Array(this.hiddenUnits), [this.hiddenUnits, 1]);
    const dWxz = Matrix.fromFlat(new Float32Array(this.hiddenUnits * this.units), [this.hiddenUnits, this.units]);
    const dWhz = Matrix.fromFlat(new Float32Array(this.hiddenUnits * this.hiddenUnits), [this.hiddenUnits, this.hiddenUnits]);
    const dBz = Matrix.fromFlat(new Float32Array(this.hiddenUnits), [this.hiddenUnits, 1]);
    const dWxh = Matrix.fromFlat(new Float32Array(this.hiddenUnits * this.units), [this.hiddenUnits, this.units]);
    const dWhh = Matrix.fromFlat(new Float32Array(this.hiddenUnits * this.hiddenUnits), [this.hiddenUnits, this.hiddenUnits]);
    const dBh = Matrix.fromFlat(new Float32Array(this.hiddenUnits), [this.hiddenUnits, 1]);

    this.ensureBatchBackwardBuffers(batchSize);

    if (
      isNativeAvailable() &&
      gruBackwardNative(
        direction.Wxr._data, direction.Whr._data,
        direction.Wxz._data, direction.Whz._data,
        direction.Wxh._data, direction.Whh._data,
        direction.xSeqBuffer, direction.hSeqBuffer,
        direction.rSeqBuffer, direction.zSeqBuffer, direction.nSeqBuffer,
        this.batchErrorStepBuffer,
        this.hiddenUnits, this.units, seqLen, batchSize,
        dWxr._data, dWhr._data, dBr._data,
        dWxz._data, dWhz._data, dBz._data,
        dWxh._data, dWhh._data, dBh._data,
        dx
      )
    ) {
        this.clipGradientsIfNeeded(dWxr, dWhr, dBr, dWxz, dWhz, dBz, dWxh, dWhh, dBh);
        direction.Wxr.subInPlace(direction.optimizerWxr.calculate(dWxr, this.alpha));
        direction.Whr.subInPlace(direction.optimizerWhr.calculate(dWhr, this.alpha));
        direction.br.subInPlace(direction.optimizerBr.calculate(dBr, this.alpha));
        direction.Wxz.subInPlace(direction.optimizerWxz.calculate(dWxz, this.alpha));
        direction.Whz.subInPlace(direction.optimizerWhz.calculate(dWhz, this.alpha));
        direction.bz.subInPlace(direction.optimizerBz.calculate(dBz, this.alpha));
        direction.Wxh.subInPlace(direction.optimizerWxh.calculate(dWxh, this.alpha));
        direction.Whh.subInPlace(direction.optimizerWhh.calculate(dWhh, this.alpha));
        direction.bh.subInPlace(direction.optimizerBh.calculate(dBh, this.alpha));
        return dx;
    }

    let dhNext = new Float32Array(this.hiddenUnits * batchSize);
    const dxMatrix = Matrix.fromFlat(dx, [this.units, totalCols]);
    const dhBuffer = new Float32Array(this.hiddenUnits * batchSize);
    const dn = new Float32Array(this.hiddenUnits * batchSize);
    const dz = new Float32Array(this.hiddenUnits * batchSize);
    const daN = new Float32Array(this.hiddenUnits * batchSize);
    const rMulHPrev = new Float32Array(this.hiddenUnits * batchSize);
    const drFromN = new Float32Array(this.hiddenUnits * batchSize);
    const dhPrevFromN = new Float32Array(this.hiddenUnits * batchSize);
    const daR = new Float32Array(this.hiddenUnits * batchSize);
    const daZ = new Float32Array(this.hiddenUnits * batchSize);
    let dhPrev = new Float32Array(this.hiddenUnits * batchSize);

    for (let step = seqLen - 1; step >= 0; step--) {
      const t = reverse ? seqLen - 1 - step : step;
      const hPrev = direction.hSeq[step];
      const x_t = direction.xSeq[step];
      const r = direction.rSeq[step];
      const z = direction.zSeq[step];
      const n = direction.nSeq[step];
      const dh = dhBuffer;
      dh.set(externalError[t]);
      for (let i = 0; i < dh.length; i++) dh[i] += dhNext[i];

      for (let i = 0; i < daN.length; i++) {
        dn[i] = dh[i] * (1 - z[i]);
        dz[i] = dh[i] * (hPrev[i] - n[i]);
        daN[i] = dn[i] * (1 - n[i] * n[i]);
      }

      for (let i = 0; i < rMulHPrev.length; i++) rMulHPrev[i] = r[i] * hPrev[i];

      const daNMatrix = Matrix.fromFlat(daN, [this.hiddenUnits, batchSize]);
      const xMatrix = Matrix.fromFlat(x_t, [this.units, batchSize]);
      const hPrevMatrix = Matrix.fromFlat(hPrev, [this.hiddenUnits, batchSize]);
      const rMulHPrevMatrix = Matrix.fromFlat(rMulHPrev, [this.hiddenUnits, batchSize]);

      this.accumulateBatchWeightGradients(dWxh, dWhh, dBh, daNMatrix, xMatrix, rMulHPrevMatrix);

      mj.dotProduct(direction.Whh, daNMatrix, this.batchDhStepBuffer, true, false);
      const dRhPrev = this.batchDhStepBuffer._data;
      for (let i = 0; i < drFromN.length; i++) {
        drFromN[i] = dRhPrev[i] * hPrev[i];
        dhPrevFromN[i] = dRhPrev[i] * r[i];
      }

      for (let i = 0; i < daR.length; i++) {
        daR[i] = drFromN[i] * r[i] * (1 - r[i]);
        daZ[i] = dz[i] * z[i] * (1 - z[i]);
      }

      const daRMatrix = Matrix.fromFlat(daR, [this.hiddenUnits, batchSize]);
      const daZMatrix = Matrix.fromFlat(daZ, [this.hiddenUnits, batchSize]);
      this.accumulateBatchWeightGradients(dWxr, dWhr, dBr, daRMatrix, xMatrix, hPrevMatrix);
      this.accumulateBatchWeightGradients(dWxz, dWhz, dBz, daZMatrix, xMatrix, hPrevMatrix);

      this.batchDxStepBuffer._data.fill(0);
      this.accumulateTransposeProduct(direction.Wxh, daNMatrix, this.batchDxStepBuffer);
      this.accumulateTransposeProduct(direction.Wxr, daRMatrix, this.batchDxStepBuffer);
      this.accumulateTransposeProduct(direction.Wxz, daZMatrix, this.batchDxStepBuffer);
      this.writeColumnBlock(dxMatrix, t * batchSize, batchSize, this.batchDxStepBuffer._data);

      this.batchDhStepBuffer._data.fill(0);
      this.accumulateTransposeProduct(direction.Whr, daRMatrix, this.batchDhStepBuffer);
      this.accumulateTransposeProduct(direction.Whz, daZMatrix, this.batchDhStepBuffer);
      for (let i = 0; i < this.batchDhStepBuffer._data.length; i++) {
        this.batchDhStepBuffer._data[i] += dhPrevFromN[i] + dh[i] * z[i];
      }
      dhPrev.set(this.batchDhStepBuffer._data);
      const prevDhNext = dhNext;
      dhNext = dhPrev;
      dhPrev = prevDhNext;
    }

    this.clipGradientsIfNeeded(dWxr, dWhr, dBr, dWxz, dWhz, dBz, dWxh, dWhh, dBh);
    direction.Wxr.subInPlace(direction.optimizerWxr.calculate(dWxr, this.alpha));
    direction.Whr.subInPlace(direction.optimizerWhr.calculate(dWhr, this.alpha));
    direction.br.subInPlace(direction.optimizerBr.calculate(dBr, this.alpha));
    direction.Wxz.subInPlace(direction.optimizerWxz.calculate(dWxz, this.alpha));
    direction.Whz.subInPlace(direction.optimizerWhz.calculate(dWhz, this.alpha));
    direction.bz.subInPlace(direction.optimizerBz.calculate(dBz, this.alpha));
    direction.Wxh.subInPlace(direction.optimizerWxh.calculate(dWxh, this.alpha));
    direction.Whh.subInPlace(direction.optimizerWhh.calculate(dWhh, this.alpha));
    direction.bh.subInPlace(direction.optimizerBh.calculate(dBh, this.alpha));
    return dx;
  }

  private createDirectionParams(): DirectionParams {
    const Wxr = mj.xavier([this.hiddenUnits, this.units]);
    const Whr = mj.xavier([this.hiddenUnits, this.hiddenUnits]);
    const br = mj.zeros([this.hiddenUnits, 1]);
    const Wxz = mj.xavier([this.hiddenUnits, this.units]);
    const Whz = mj.xavier([this.hiddenUnits, this.hiddenUnits]);
    const bz = mj.zeros([this.hiddenUnits, 1]);
    const Wxh = mj.xavier([this.hiddenUnits, this.units]);
    const Whh = mj.xavier([this.hiddenUnits, this.hiddenUnits]);
    const bh = mj.zeros([this.hiddenUnits, 1]);

    return {
      Wxr,
      Whr,
      br,
      Wxz,
      Whz,
      bz,
      Wxh,
      Whh,
      bh,
      optimizerWxr: setOptimizer(this.optimizerName, Wxr._shape, 1e-5),
      optimizerWhr: setOptimizer(this.optimizerName, Whr._shape, 1e-5),
      optimizerBr: setOptimizer(this.optimizerName, br._shape, 1e-5),
      optimizerWxz: setOptimizer(this.optimizerName, Wxz._shape, 1e-5),
      optimizerWhz: setOptimizer(this.optimizerName, Whz._shape, 1e-5),
      optimizerBz: setOptimizer(this.optimizerName, bz._shape, 1e-5),
      optimizerWxh: setOptimizer(this.optimizerName, Wxh._shape, 1e-5),
      optimizerWhh: setOptimizer(this.optimizerName, Whh._shape, 1e-5),
      optimizerBh: setOptimizer(this.optimizerName, bh._shape, 1e-5),
      hStateful: mj.zeros([this.hiddenUnits, 1]),
      xSeq: [],
      hSeq: [],
      rSeq: [],
      zSeq: [],
      nSeq: [],
      xSeqBuffer: new Float32Array(0),
      hSeqBuffer: new Float32Array(0),
      rSeqBuffer: new Float32Array(0),
      zSeqBuffer: new Float32Array(0),
      nSeqBuffer: new Float32Array(0),
    };
  }

  private serializeDirection(direction: DirectionParams) {
    return {
      Wxr: direction.Wxr._value,
      Whr: direction.Whr._value,
      br: direction.br._value,
      Wxz: direction.Wxz._value,
      Whz: direction.Whz._value,
      bz: direction.bz._value,
      Wxh: direction.Wxh._value,
      Whh: direction.Whh._value,
      bh: direction.bh._value,
      hStateful: direction.hStateful._value,
    };
  }

  private deserializeDirection(direction: DirectionParams, value: any) {
    this.loadMatrix(direction.Wxr, value.Wxr);
    this.loadMatrix(direction.Whr, value.Whr);
    this.loadMatrix(direction.br, value.br);
    this.loadMatrix(direction.Wxz, value.Wxz);
    this.loadMatrix(direction.Whz, value.Whz);
    this.loadMatrix(direction.bz, value.bz);
    this.loadMatrix(direction.Wxh, value.Wxh);
    this.loadMatrix(direction.Whh, value.Whh);
    this.loadMatrix(direction.bh, value.bh);
    if (value.hStateful) this.loadMatrix(direction.hStateful, value.hStateful);
  }

  private assertSerializedDirection(value: unknown, directionName: string): asserts value is Record<string, number[][]> {
    if (!value || typeof value !== "object") {
      throw new Error(`GRU.load: expected serialized '${directionName}' direction.`);
    }
    const direction = value as Record<string, unknown>;
    const requiredFields = ["Wxr", "Whr", "br", "Wxz", "Whz", "bz", "Wxh", "Whh", "bh"];
    for (const field of requiredFields) {
      if (!Array.isArray(direction[field])) {
        throw new Error(`GRU.load: expected serialized matrix '${directionName}.${field}'.`);
      }
    }
    if (direction.hStateful !== undefined && !Array.isArray(direction.hStateful)) {
      throw new Error(`GRU.load: expected serialized matrix '${directionName}.hStateful'.`);
    }
  }

  private resetOptimizers(direction: DirectionParams) {
    direction.optimizerWxr = setOptimizer(this.optimizerName, direction.Wxr._shape, 1e-5);
    direction.optimizerWhr = setOptimizer(this.optimizerName, direction.Whr._shape, 1e-5);
    direction.optimizerBr = setOptimizer(this.optimizerName, direction.br._shape, 1e-5);
    direction.optimizerWxz = setOptimizer(this.optimizerName, direction.Wxz._shape, 1e-5);
    direction.optimizerWhz = setOptimizer(this.optimizerName, direction.Whz._shape, 1e-5);
    direction.optimizerBz = setOptimizer(this.optimizerName, direction.bz._shape, 1e-5);
    direction.optimizerWxh = setOptimizer(this.optimizerName, direction.Wxh._shape, 1e-5);
    direction.optimizerWhh = setOptimizer(this.optimizerName, direction.Whh._shape, 1e-5);
    direction.optimizerBh = setOptimizer(this.optimizerName, direction.bh._shape, 1e-5);
  }

  private gatePre(
    Wx: Matrix,
    Wh: Matrix,
    b: Matrix,
    row: number,
    x: Float32Array,
    hPrev: Float32Array
  ): number {
    let sum = b._data[row];
    const wxOffset = row * this.units;
    for (let j = 0; j < this.units; j++) sum += Wx._data[wxOffset + j] * x[j];
    const whOffset = row * this.hiddenUnits;
    for (let j = 0; j < this.hiddenUnits; j++) sum += Wh._data[whOffset + j] * hPrev[j];
    return sum;
  }

  private sigmoid(v: number): number {
    return 1 / (1 + Math.exp(-v));
  }

  private loadMatrix(target: Matrix, value: number[][]) {
    target._value = value;
    target._shape = [value.length, value[0]?.length ?? 0];
  }

  private setColumnData(target: Float32Array, targetCols: number, col: number, data: Float32Array) {
    for (let i = 0; i < data.length; i++) target[i * targetCols + col] = data[i];
  }

  private clipGradientsIfNeeded(...grads: Matrix[]) {
    if (this.clipGradient === false) return;
    const limit = typeof this.clipGradient === "number" ? this.clipGradient : 5.0;
    for (const grad of grads) mj.clipGradients(grad, limit);
  }

  private outerAccumulate(
    target: Float32Array,
    outRows: number,
    outCols: number,
    a: Float32Array,
    b: Float32Array
  ) {
    for (let i = 0; i < outRows; i++) {
      const ai = a[i];
      const offset = i * outCols;
      for (let j = 0; j < outCols; j++) target[offset + j] += ai * b[j];
    }
  }

  private ensureDirectionSequenceBuffers(direction: DirectionParams, seqLen: number, batchSize: number) {
    const inputWidth = this.units * batchSize;
    const hiddenWidth = this.hiddenUnits * batchSize;
    direction.xSeqBuffer = this.ensureCapacity(direction.xSeqBuffer, seqLen * inputWidth);
    direction.hSeqBuffer = this.ensureCapacity(direction.hSeqBuffer, (seqLen + 1) * hiddenWidth);
    direction.rSeqBuffer = this.ensureCapacity(direction.rSeqBuffer, seqLen * hiddenWidth);
    direction.zSeqBuffer = this.ensureCapacity(direction.zSeqBuffer, seqLen * hiddenWidth);
    direction.nSeqBuffer = this.ensureCapacity(direction.nSeqBuffer, seqLen * hiddenWidth);
    direction.xSeq = this.buildErrorViews(direction.xSeqBuffer, seqLen, inputWidth);
    direction.hSeq = this.buildErrorViews(direction.hSeqBuffer, seqLen + 1, hiddenWidth);
    direction.rSeq = this.buildErrorViews(direction.rSeqBuffer, seqLen, hiddenWidth);
    direction.zSeq = this.buildErrorViews(direction.zSeqBuffer, seqLen, hiddenWidth);
    direction.nSeq = this.buildErrorViews(direction.nSeqBuffer, seqLen, hiddenWidth);
  }

  private ensureErrorBuffer(expectedLen: number): Float32Array {
    this.errorStepBuffer = this.ensureCapacity(this.errorStepBuffer, expectedLen);
    return this.errorStepBuffer;
  }

  private ensureBatchErrorBuffer(expectedLen: number): Float32Array {
    this.batchErrorStepBuffer = this.ensureCapacity(this.batchErrorStepBuffer, expectedLen);
    return this.batchErrorStepBuffer;
  }

  private ensureSplitErrorBuffer(kind: "forward" | "backward", expectedLen: number): Float32Array {
    if (kind === "forward") {
      this.splitForwardErrorBuffer = this.ensureCapacity(this.splitForwardErrorBuffer, expectedLen);
      return this.splitForwardErrorBuffer;
    }
    this.splitBackwardErrorBuffer = this.ensureCapacity(this.splitBackwardErrorBuffer, expectedLen);
    return this.splitBackwardErrorBuffer;
  }

  private ensureBatchSplitErrorBuffer(kind: "forward" | "backward", expectedLen: number): Float32Array {
    if (kind === "forward") {
      this.batchSplitForwardErrorBuffer = this.ensureCapacity(this.batchSplitForwardErrorBuffer, expectedLen);
      return this.batchSplitForwardErrorBuffer;
    }
    this.batchSplitBackwardErrorBuffer = this.ensureCapacity(this.batchSplitBackwardErrorBuffer, expectedLen);
    return this.batchSplitBackwardErrorBuffer;
  }

  private ensureCapacity(buffer: Float32Array, expectedLen: number): Float32Array {
    if (buffer.length < expectedLen) {
      return new Float32Array(Math.max(expectedLen, Math.max(1, buffer.length * 2)));
    }
    return buffer;
  }

  private buildErrorViews(buffer: Float32Array, steps: number, width: number): Float32Array[] {
    const views = new Array<Float32Array>(steps);
    for (let step = 0; step < steps; step++) {
      const start = step * width;
      views[step] = buffer.subarray(start, start + width);
    }
    return views;
  }

  private assertBatchInputSupported(x: Matrix, batchSize: number) {
    if (!Number.isInteger(batchSize) || batchSize < 1) {
      throw new Error("GRU.forwardBatch: batchSize must be an integer >= 1.");
    }
    if (x._shape[0] !== this.units) {
      throw new Error(`GRU.forwardBatch: expected input rows ${this.units}, got ${x._shape[0]}`);
    }
    this.assertBatchInputSupportedShape(batchSize, x._shape[1]);
    if (this.stateful && batchSize !== 1) {
      throw new Error("GRU.forwardBatch: stateful=true only supports batchSize=1 in the current batched recurrent path.");
    }
  }

  private assertBatchInputSupportedShape(batchSize: number, totalCols: number) {
    if (totalCols < 1 || totalCols % batchSize !== 0) {
      throw new Error(
        `GRU batched path expects time-major columns divisible by batchSize. Got cols=${totalCols}, batchSize=${batchSize}.`
      );
    }
  }

  private ensureBatchForwardBuffers(batchSize: number, totalCols: number) {
    this.batchInputSliceBuffer = this.ensureBuffer(this.batchInputSliceBuffer, this.units, batchSize);
    this.batchGateXRBuffer = this.ensureBuffer(this.batchGateXRBuffer, this.hiddenUnits, totalCols);
    this.batchGateXZBuffer = this.ensureBuffer(this.batchGateXZBuffer, this.hiddenUnits, totalCols);
    this.batchGateXNBuffer = this.ensureBuffer(this.batchGateXNBuffer, this.hiddenUnits, totalCols);
    this.batchGateSliceRBuffer = this.ensureBuffer(this.batchGateSliceRBuffer, this.hiddenUnits, batchSize);
    this.batchGateSliceZBuffer = this.ensureBuffer(this.batchGateSliceZBuffer, this.hiddenUnits, batchSize);
    this.batchGateSliceNBuffer = this.ensureBuffer(this.batchGateSliceNBuffer, this.hiddenUnits, batchSize);
    this.batchRecRBuffer = this.ensureBuffer(this.batchRecRBuffer, this.hiddenUnits, batchSize);
    this.batchRecZBuffer = this.ensureBuffer(this.batchRecZBuffer, this.hiddenUnits, batchSize);
    this.batchRecNBuffer = this.ensureBuffer(this.batchRecNBuffer, this.hiddenUnits, batchSize);
  }

  private ensureBatchBackwardBuffers(batchSize: number) {
    this.batchDxStepBuffer = this.ensureBuffer(this.batchDxStepBuffer, this.units, batchSize);
    this.batchDhStepBuffer = this.ensureBuffer(this.batchDhStepBuffer, this.hiddenUnits, batchSize);
    this.batchOuterInputBuffer = this.ensureBuffer(this.batchOuterInputBuffer, this.hiddenUnits, this.units);
    this.batchOuterHiddenBuffer = this.ensureBuffer(this.batchOuterHiddenBuffer, this.hiddenUnits, this.hiddenUnits);
    this.batchBiasGradBuffer = this.ensureBuffer(this.batchBiasGradBuffer, this.hiddenUnits, 1);
    this.batchTransposeProductBuffer = this.ensureBuffer(
      this.batchTransposeProductBuffer,
      Math.max(this.units, this.hiddenUnits),
      batchSize
    );
  }

  private ensureBuffer(buffer: Matrix, rows: number, cols: number): Matrix {
    if (buffer._shape[0] !== rows || buffer._shape[1] !== cols) {
      return mj.zeros([rows, cols]);
    }
    return buffer;
  }

  private accumulateBatchWeightGradients(
    dWx: Matrix,
    dWh: Matrix,
    dB: Matrix,
    dz: Matrix,
    x: Matrix,
    hPrev: Matrix
  ) {
    mj.dotProduct(dz, x, this.batchOuterInputBuffer, false, true);
    dWx.addInPlace(this.batchOuterInputBuffer);
    mj.dotProduct(dz, hPrev, this.batchOuterHiddenBuffer, false, true);
    dWh.addInPlace(this.batchOuterHiddenBuffer);
    mj.sumAxis(dz, 1, this.batchBiasGradBuffer);
    dB.addInPlace(this.batchBiasGradBuffer);
  }

  private accumulateTransposeProduct(weight: Matrix, grad: Matrix, target: Matrix) {
    const temp = this.ensureBuffer(this.batchTransposeProductBuffer, target._shape[0], target._shape[1]);
    this.batchTransposeProductBuffer = temp;
    mj.dotProduct(weight, grad, temp, true, false);
    target.addInPlace(temp);
  }

  private copyColumnBlock(source: Matrix, startCol: number, blockCols: number, target: Matrix) {
    const [rows, cols] = source._shape;
    for (let row = 0; row < rows; row++) {
      const srcOffset = row * cols + startCol;
      target._data.set(source._data.subarray(srcOffset, srcOffset + blockCols), row * blockCols);
    }
  }

  private copyColumnBlockToArray(source: Matrix, startCol: number, blockCols: number, target: Float32Array) {
    const [rows, cols] = source._shape;
    for (let row = 0; row < rows; row++) {
      const srcOffset = row * cols + startCol;
      target.set(source._data.subarray(srcOffset, srcOffset + blockCols), row * blockCols);
    }
  }

  private copyColumnToArray(source: Matrix, col: number, target: Float32Array) {
    const [rows, cols] = source._shape;
    for (let row = 0; row < rows; row++) {
      target[row] = source._data[row * cols + col];
    }
  }

  private writeColumnBlock(target: Matrix, startCol: number, blockCols: number, data: Float32Array) {
    const [rows, cols] = target._shape;
    for (let row = 0; row < rows; row++) {
      const srcOffset = row * blockCols;
      target._data.set(data.subarray(srcOffset, srcOffset + blockCols), row * cols + startCol);
    }
  }

  private writeColumnBlockOffset(
    target: Matrix,
    startCol: number,
    blockCols: number,
    rowOffset: number,
    data: Float32Array
  ) {
    const cols = target._shape[1];
    for (let row = 0; row < this.hiddenUnits; row++) {
      const srcOffset = row * blockCols;
      const dstOffset = (row + rowOffset) * cols + startCol;
      target._data.set(data.subarray(srcOffset, srcOffset + blockCols), dstOffset);
    }
  }
}
