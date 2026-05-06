import { Cost, Optimizer, OptimizerType, StatusLayer } from "@oxide-js/core";
import { mj } from "@oxide-js/core";
import { isNativeAvailable, rnnForwardNative, rnnBackwardNative } from "@oxide-js/core";
import { Matrix } from "@oxide-js/core";
import { setLoss } from "@oxide-js/core";
import { setOptimizer } from "@oxide-js/core";

export interface RNNLayerConfig {
  units: number;
  hiddenUnits: number;
  activation?: "tanh" | "relu";
  returnSequences?: boolean;
  returnState?: boolean;
  alpha?: number;
  optimizer?: Optimizer;
  status?: StatusLayer;
  clipGradient?: number | boolean;
  stateful?: boolean;
  loss?: Cost;
}

export default class RNN {
  name = "rnn layer";
  units: number;
  hiddenUnits: number;
  activation: "tanh" | "relu";
  returnSequences: boolean;
  returnState: boolean;
  stateful: boolean;
  inputShape: [number, number];
  outputShape: [number, number];
  params: number;
  loss = 0;
  status: StatusLayer;
  alpha: number;
  clipGradient: number | boolean;

  Wxh: Matrix;
  Whh: Matrix;
  bh: Matrix;

  private optimizerWxh: OptimizerType;
  private optimizerWhh: OptimizerType;
  private optimizerBh: OptimizerType;
  private optimizerName: Optimizer;
  private lossName: Cost;
  private lossFunc: Function;
  private sumLoss = 0;
  private lossCount = 0;

  private h_stateful: Matrix;
  private inputSequence: Float32Array[] = [];
  private hiddenSequence: Float32Array[] = [];
  private activationGradients: Float32Array[] = [];
  private resultBuffer: Matrix = mj.matrix([]);
  private batchInputSequence: Float32Array[] = [];
  private batchHiddenSequence: Float32Array[] = [];
  private batchActivationGradients: Float32Array[] = [];
  private batchInputProjectionBuffer: Matrix = mj.matrix([]);
  private batchInputSliceBuffer: Matrix = mj.matrix([]);
  private batchProjectionSliceBuffer: Matrix = mj.matrix([]);
  private batchRecurrentBuffer: Matrix = mj.matrix([]);
  private batchDxStepBuffer: Matrix = mj.matrix([]);
  private batchDhStepBuffer: Matrix = mj.matrix([]);
  private batchOuterInputBuffer: Matrix = mj.matrix([]);
  private batchOuterHiddenBuffer: Matrix = mj.matrix([]);
  private batchBiasGradBuffer: Matrix = mj.matrix([]);
  private inputSequenceBuffer: Float32Array = new Float32Array(0);
  private hiddenSequenceBuffer: Float32Array = new Float32Array(0);
  private activationGradientBuffer: Float32Array = new Float32Array(0);
  private batchInputSequenceBuffer: Float32Array = new Float32Array(0);
  private batchHiddenSequenceBuffer: Float32Array = new Float32Array(0);
  private batchActivationGradientBuffer: Float32Array = new Float32Array(0);
  private errorStepBuffer: Float32Array = new Float32Array(0);
  private batchErrorStepBuffer: Float32Array = new Float32Array(0);

  constructor({
    units,
    hiddenUnits,
    activation = "tanh",
    returnSequences = false,
    returnState = false,
    alpha = 0.01,
    optimizer = "adam",
    status = "input",
    clipGradient = 5.0,
    stateful = false,
    loss = "mse",
  }: RNNLayerConfig) {
    this.units = units;
    this.hiddenUnits = hiddenUnits;
    this.activation = activation;
    this.returnSequences = returnSequences;
    this.returnState = returnState;
    this.stateful = stateful;
    this.alpha = alpha;
    this.status = status;
    this.clipGradient = clipGradient;
    this.optimizerName = optimizer;
    this.lossName = loss;
    this.lossFunc = setLoss(loss);

    this.Wxh = mj.xavier([hiddenUnits, units]);
    this.Whh = mj.xavier([hiddenUnits, hiddenUnits]);
    this.bh = mj.zeros([hiddenUnits, 1]);

    this.optimizerWxh = setOptimizer(optimizer, this.Wxh._shape, 1e-5);
    this.optimizerWhh = setOptimizer(optimizer, this.Whh._shape, 1e-5);
    this.optimizerBh = setOptimizer(optimizer, this.bh._shape, 1e-5);

    this.inputShape = [units, 0];
    this.outputShape = [hiddenUnits, returnSequences ? 0 : 1];
    this.params = hiddenUnits * units + hiddenUnits * hiddenUnits + hiddenUnits;
    this.h_stateful = mj.zeros([hiddenUnits, 1]);
  }

  save() {
    return {
      name: this.name,
      units: this.units,
      hiddenUnits: this.hiddenUnits,
      activation: this.activation,
      returnSequences: this.returnSequences,
      returnState: this.returnState,
      stateful: this.stateful,
      alpha: this.alpha,
      optimizer: this.optimizerName,
      status: this.status,
      clipGradient: this.clipGradient,
      loss: this.lossName,
      Wxh: this.Wxh._value,
      Whh: this.Whh._value,
      bh: this.bh._value,
      by: this.bh._value,
      hStateful: this.h_stateful._value,
    };
  }

  load(data: {
    Wxh: number[][];
    Whh: number[][];
    bh?: number[][];
    by?: number[][];
    hStateful?: number[][];
    clipGradient?: number | boolean;
  }) {
    this.Wxh._value = data.Wxh;
    this.Wxh._shape = [data.Wxh.length, data.Wxh[0]?.length ?? 0];
    this.Whh._value = data.Whh;
    this.Whh._shape = [data.Whh.length, data.Whh[0]?.length ?? 0];
    const bias = data.bh ?? data.by;
    if (!bias) {
      throw new Error("RNN.load: expected 'bh' (or legacy 'by') in serialized data.");
    }
    this.bh._value = bias;
    this.bh._shape = [bias.length, bias[0]?.length ?? 0];
    if (data.hStateful) {
      this.h_stateful._value = data.hStateful;
      this.h_stateful._shape = [data.hStateful.length, data.hStateful[0]?.length ?? 0];
    } else {
      this.h_stateful = mj.zeros([this.hiddenUnits, 1]);
    }
    if (data.clipGradient !== undefined) this.clipGradient = data.clipGradient;

    this.optimizerWxh = setOptimizer(this.optimizerName, this.Wxh._shape, 1e-5);
    this.optimizerWhh = setOptimizer(this.optimizerName, this.Whh._shape, 1e-5);
    this.optimizerBh = setOptimizer(this.optimizerName, this.bh._shape, 1e-5);
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
      this.optimizerWxh = setOptimizer(optimizer, this.Wxh._shape, 1e-5);
      this.optimizerWhh = setOptimizer(optimizer, this.Whh._shape, 1e-5);
      this.optimizerBh = setOptimizer(optimizer, this.bh._shape, 1e-5);
    }
    if (error !== undefined) {
      this.lossName = error;
      this.lossFunc = setLoss(error);
    }
    if (clipGradient !== undefined) this.clipGradient = clipGradient;
  }

  resetState() {
    this.h_stateful._data.fill(0);
  }

  getState(): Matrix {
    return this.h_stateful.clone();
  }

  forward(x: Matrix): Matrix {
    if (this.returnState) {
      throw new Error("RNN.forward: returnState=true is not supported yet. Disable returnState for RNN.");
    }
    if (x._shape[0] !== this.units) {
      throw new Error(`RNN.forward: expected input rows ${this.units}, got ${x._shape[0]}`);
    }
    const seqLen = x._shape[1];
    if (seqLen < 1) {
      throw new Error("RNN.forward: expected a non-empty sequence input.");
    }
    const outCols = this.returnSequences ? seqLen : 1;
    if (this.resultBuffer._shape[0] !== this.hiddenUnits || this.resultBuffer._shape[1] !== outCols) {
      this.resultBuffer = mj.zeros([this.hiddenUnits, outCols]);
    } else {
      this.resultBuffer._data.fill(0);
    }

    this.inputShape = [this.units, seqLen];
    this.outputShape = [this.hiddenUnits, outCols];
    this.ensureSequenceStateBuffers(seqLen);

    const prev = this.hiddenSequence[0];
    prev.fill(0);
    if (this.stateful) {
      prev.set(this.h_stateful._data);
    }

    for (let t = 0; t < seqLen; t++) {
      const x_t = this.inputSequence[t];
      this.copyColumnToArray(x, t, x_t);
      const h_t = this.hiddenSequence[t + 1];
      const dAct = this.activationGradients[t];
      const hPrev = this.hiddenSequence[t];

      for (let i = 0; i < this.hiddenUnits; i++) {
        let sum = this.bh._data[i];
        const wxhOffset = i * this.units;
        for (let j = 0; j < this.units; j++) sum += this.Wxh._data[wxhOffset + j] * x_t[j];
        const whhOffset = i * this.hiddenUnits;
        for (let j = 0; j < this.hiddenUnits; j++) sum += this.Whh._data[whhOffset + j] * hPrev[j];

        if (this.activation === "relu") {
          if (sum > 0) {
            h_t[i] = sum;
            dAct[i] = 1;
          } else {
            h_t[i] = 0;
            dAct[i] = 0;
          }
        } else {
          const tv = Math.tanh(sum);
          h_t[i] = tv;
          dAct[i] = 1 - tv * tv;
        }
      }

      if (this.returnSequences) {
        this.setColumnData(this.resultBuffer._data, outCols, t, h_t);
      } else if (t === seqLen - 1) {
        this.resultBuffer._data.set(h_t);
      }
    }

    const lastHidden = this.hiddenSequence[seqLen];
    if (this.stateful) this.h_stateful._data.set(lastHidden);
    return this.resultBuffer;
  }

  forwardBatch(x: Matrix, batchSize: number): Matrix {
    this.assertBatchInputSupported(x, batchSize);
    const totalCols = x._shape[1];
    const seqLen = totalCols / batchSize;
    const outCols = this.returnSequences ? totalCols : batchSize;

    this.ensureBatchForwardBuffers(batchSize, totalCols, outCols);
    this.resultBuffer._data.fill(0);
    this.batchInputProjectionBuffer._data.fill(0);
    this.inputShape = [this.units, totalCols];
    this.outputShape = [this.hiddenUnits, outCols];
    this.ensureBatchSequenceStateBuffers(seqLen, batchSize);

    if (!isNativeAvailable()) {
      this.batchInputProjectionBuffer._data.fill(0);
      mj.dotProduct(this.Wxh, x, this.batchInputProjectionBuffer);
      mj.addBias(this.batchInputProjectionBuffer, this.bh);
    }

    const h0View = this.batchHiddenSequence[0];
    h0View.fill(0);
    if (this.stateful && batchSize === 1) h0View.set(this.h_stateful._data);

    if (
      isNativeAvailable() &&
      rnnForwardNative(
        this.Wxh._data,
        this.Whh._data,
        this.bh._data,
        x._data,
        h0View,
        this.hiddenUnits,
        this.units,
        seqLen,
        batchSize,
        this.batchHiddenSequenceBuffer,
        this.batchActivationGradientBuffer
      )
    ) {
      if (this.returnSequences) {
        for (let t = 0; t < seqLen; t++) {
           this.writeColumnBlock(this.resultBuffer, t * batchSize, batchSize, this.batchHiddenSequence[t + 1]);
        }
      } else {
        this.resultBuffer._data.set(this.batchHiddenSequence[seqLen]);
      }

      if (this.stateful && batchSize === 1) {
        this.h_stateful._data.set(this.batchHiddenSequence[seqLen]);
      }
      return this.resultBuffer;
    }

    for (let t = 0; t < seqLen; t++) {
      const colOffset = t * batchSize;
      this.copyColumnBlock(x, colOffset, batchSize, this.batchInputSliceBuffer);
      this.copyColumnBlock(this.batchInputProjectionBuffer, colOffset, batchSize, this.batchProjectionSliceBuffer);

      const hPrev = Matrix.fromFlat(this.batchHiddenSequence[t], [this.hiddenUnits, batchSize]);
      mj.dotProduct(this.Whh, hPrev, this.batchRecurrentBuffer);

      const h_t = this.batchHiddenSequence[t + 1];
      const dAct = this.batchActivationGradients[t];
      const projected = this.batchProjectionSliceBuffer._data;
      const recurrent = this.batchRecurrentBuffer._data;
      for (let i = 0; i < h_t.length; i++) {
        const sum = projected[i] + recurrent[i];
        if (this.activation === "relu") {
          if (sum > 0) {
            h_t[i] = sum;
            dAct[i] = 1;
          } else {
            h_t[i] = 0;
            dAct[i] = 0;
          }
        } else {
          const tv = Math.tanh(sum);
          h_t[i] = tv;
          dAct[i] = 1 - tv * tv;
        }
      }

      this.batchInputSequence[t].set(this.batchInputSliceBuffer._data);

      if (this.returnSequences) {
        this.writeColumnBlock(this.resultBuffer, colOffset, batchSize, h_t);
      } else if (t === seqLen - 1) {
        this.resultBuffer._data.set(h_t);
      }
    }

    if (this.stateful && batchSize === 1) {
      this.h_stateful._data.set(this.batchHiddenSequence[seqLen]);
    }
    return this.resultBuffer;
  }

  backward(y: Matrix, err: Matrix): Matrix {
    const seqLen = this.inputShape[1];
    if (seqLen <= 0 || this.hiddenSequence.length !== seqLen + 1) {
      throw new Error("RNN.backward: forward must be called before backward.");
    }

    const externalError = this.resolveError(y, err, seqLen);
    const dWxh = Matrix.fromFlat(new Float32Array(this.hiddenUnits * this.units), [this.hiddenUnits, this.units]);
    const dWhh = Matrix.fromFlat(new Float32Array(this.hiddenUnits * this.hiddenUnits), [this.hiddenUnits, this.hiddenUnits]);
    const dBh = Matrix.fromFlat(new Float32Array(this.hiddenUnits), [this.hiddenUnits, 1]);
    const dxData = new Float32Array(this.units * seqLen);
    let dhNext = new Float32Array(this.hiddenUnits);
    const dhBuffer = new Float32Array(this.hiddenUnits);
    const dzBuffer = new Float32Array(this.hiddenUnits);
    let dhPrevBuffer = new Float32Array(this.hiddenUnits);

    for (let t = seqLen - 1; t >= 0; t--) {
      const dh = dhBuffer;
      dh.set(externalError[t]);
      for (let i = 0; i < this.hiddenUnits; i++) dh[i] += dhNext[i];

      const dz = dzBuffer;
      for (let i = 0; i < this.hiddenUnits; i++) dz[i] = dh[i] * this.activationGradients[t][i];

      this.outerAccumulate(dWxh._data, this.hiddenUnits, this.units, dz, this.inputSequence[t]);
      this.outerAccumulate(dWhh._data, this.hiddenUnits, this.hiddenUnits, dz, this.hiddenSequence[t]);
      for (let i = 0; i < this.hiddenUnits; i++) dBh._data[i] += dz[i];

      for (let j = 0; j < this.units; j++) {
        let sum = 0;
        for (let i = 0; i < this.hiddenUnits; i++) sum += this.Wxh._data[i * this.units + j] * dz[i];
        dxData[j * seqLen + t] = sum;
      }

      const dhPrev = dhPrevBuffer;
      for (let j = 0; j < this.hiddenUnits; j++) {
        let sum = 0;
        for (let i = 0; i < this.hiddenUnits; i++) sum += this.Whh._data[i * this.hiddenUnits + j] * dz[i];
        dhPrev[j] = sum;
      }
      const prevDhNext = dhNext;
      dhNext = dhPrev;
      dhPrevBuffer = prevDhNext;
    }

    this.clipGradientsIfNeeded(dWxh, dWhh, dBh);
    this.Wxh.subInPlace(this.optimizerWxh.calculate(dWxh, this.alpha));
    this.Whh.subInPlace(this.optimizerWhh.calculate(dWhh, this.alpha));
    this.bh.subInPlace(this.optimizerBh.calculate(dBh, this.alpha));

    return Matrix.fromFlat(dxData, [this.units, seqLen]);
  }

  backwardBatch(y: Matrix, err: Matrix, batchSize: number): Matrix {
    const totalCols = this.inputShape[1];
    this.assertBatchInputSupportedShape(batchSize, totalCols);
    const seqLen = totalCols / batchSize;
    if (this.batchHiddenSequence.length !== seqLen + 1) {
      throw new Error("RNN.backwardBatch: forwardBatch must be called before backwardBatch.");
    }

    const externalError = this.resolveBatchError(y, err, seqLen, batchSize);
    const dWxh = Matrix.fromFlat(new Float32Array(this.hiddenUnits * this.units), [this.hiddenUnits, this.units]);
    const dWhh = Matrix.fromFlat(new Float32Array(this.hiddenUnits * this.hiddenUnits), [this.hiddenUnits, this.hiddenUnits]);
    const dBh = Matrix.fromFlat(new Float32Array(this.hiddenUnits), [this.hiddenUnits, 1]);
    const dx = Matrix.fromFlat(new Float32Array(this.units * totalCols), [this.units, totalCols]);
    let dhNext = new Float32Array(this.hiddenUnits * batchSize);

    this.ensureBatchBackwardBuffers(batchSize);

    if (
      isNativeAvailable() &&
      rnnBackwardNative(
        this.Wxh._data, this.Whh._data,
        this.batchInputSequenceBuffer, this.batchHiddenSequenceBuffer, this.batchActivationGradientBuffer,
        this.batchErrorStepBuffer,
        this.hiddenUnits, this.units, seqLen, batchSize,
        dWxh._data, dWhh._data, dBh._data,
        dx._data
      )
    ) {
        this.clipGradientsIfNeeded(dWxh, dWhh, dBh);
        this.Wxh.subInPlace(this.optimizerWxh.calculate(dWxh, this.alpha));
        this.Whh.subInPlace(this.optimizerWhh.calculate(dWhh, this.alpha));
        this.bh.subInPlace(this.optimizerBh.calculate(dBh, this.alpha));
        return dx;
    }

    const dhBuffer = new Float32Array(this.hiddenUnits * batchSize);
    const dzBuffer = new Float32Array(this.hiddenUnits * batchSize);
    let dhPrevBuffer = new Float32Array(this.hiddenUnits * batchSize);

    for (let t = seqLen - 1; t >= 0; t--) {
      const dh = dhBuffer;
      dh.set(externalError[t]);
      for (let i = 0; i < dh.length; i++) dh[i] += dhNext[i];

      const dz = dzBuffer;
      for (let i = 0; i < dz.length; i++) dz[i] = dh[i] * this.batchActivationGradients[t][i];

      const dzMatrix = Matrix.fromFlat(dz, [this.hiddenUnits, batchSize]);
      const xMatrix = Matrix.fromFlat(this.batchInputSequence[t], [this.units, batchSize]);
      const hPrevMatrix = Matrix.fromFlat(this.batchHiddenSequence[t], [this.hiddenUnits, batchSize]);

      mj.dotProduct(dzMatrix, xMatrix, this.batchOuterInputBuffer, false, true);
      dWxh.addInPlace(this.batchOuterInputBuffer);
      mj.dotProduct(dzMatrix, hPrevMatrix, this.batchOuterHiddenBuffer, false, true);
      dWhh.addInPlace(this.batchOuterHiddenBuffer);
      mj.sumAxis(dzMatrix, 1, this.batchBiasGradBuffer);
      dBh.addInPlace(this.batchBiasGradBuffer);

      mj.dotProduct(this.Wxh, dzMatrix, this.batchDxStepBuffer, true, false);
      this.writeColumnBlock(dx, t * batchSize, batchSize, this.batchDxStepBuffer._data);
      mj.dotProduct(this.Whh, dzMatrix, this.batchDhStepBuffer, true, false);
      dhPrevBuffer.set(this.batchDhStepBuffer._data);
      const prevDhNext = dhNext;
      dhNext = dhPrevBuffer;
      dhPrevBuffer = prevDhNext;
    }

    this.clipGradientsIfNeeded(dWxh, dWhh, dBh);
    this.Wxh.subInPlace(this.optimizerWxh.calculate(dWxh, this.alpha));
    this.Whh.subInPlace(this.optimizerWhh.calculate(dWhh, this.alpha));
    this.bh.subInPlace(this.optimizerBh.calculate(dBh, this.alpha));
    return dx;
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

    const outCols = this.returnSequences ? seqLen : 1;
    if (effectiveErr._shape[0] !== this.hiddenUnits || effectiveErr._shape[1] !== outCols) {
      throw new Error(
        `RNN.backward: error shape mismatch, expected [${this.hiddenUnits},${outCols}], got [${effectiveErr._shape[0]},${effectiveErr._shape[1]}]`
      );
    }

    this.ensureErrorStepBuffers(seqLen);
    const perStep = this.buildStepViews(this.errorStepBuffer, seqLen, this.hiddenUnits);
    this.errorStepBuffer.fill(0, 0, seqLen * this.hiddenUnits);
    if (this.returnSequences) {
      for (let t = 0; t < seqLen; t++) {
        for (let i = 0; i < this.hiddenUnits; i++) {
          perStep[t][i] = effectiveErr._data[i * seqLen + t];
        }
      }
    } else {
      for (let i = 0; i < this.hiddenUnits; i++) {
        perStep[seqLen - 1][i] = effectiveErr._data[i];
      }
    }
    return perStep;
  }

  private clipGradientsIfNeeded(dWxh: Matrix, dWhh: Matrix, dBh: Matrix) {
    if (this.clipGradient === false) return;
    const limit = typeof this.clipGradient === "number" ? this.clipGradient : 5.0;
    mj.clipGradients(dWxh, limit);
    mj.clipGradients(dWhh, limit);
    mj.clipGradients(dBh, limit);
  }

  private setColumnData(target: Float32Array, targetCols: number, col: number, data: Float32Array) {
    for (let i = 0; i < data.length; i++) target[i * targetCols + col] = data[i];
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

  private resolveBatchError(y: Matrix, err: Matrix, seqLen: number, batchSize: number): Float32Array[] {
    let effectiveErr = err;
    if (this.status === "output") {
      const [lossValue, outputErr] = this.lossFunc(y, this.resultBuffer);
      this.lossCount++;
      this.sumLoss += lossValue;
      this.loss = this.sumLoss / this.lossCount;
      effectiveErr = outputErr;
    }

    const expectedCols = this.returnSequences ? seqLen * batchSize : batchSize;
    if (effectiveErr._shape[0] !== this.hiddenUnits || effectiveErr._shape[1] !== expectedCols) {
      throw new Error(
        `RNN.backwardBatch: error shape mismatch, expected [${this.hiddenUnits},${expectedCols}], got [${effectiveErr._shape[0]},${effectiveErr._shape[1]}]`
      );
    }

    const stepWidth = this.hiddenUnits * batchSize;
    this.ensureBatchErrorStepBuffers(seqLen, batchSize);
    const perStep = this.buildStepViews(this.batchErrorStepBuffer, seqLen, stepWidth);
    this.batchErrorStepBuffer.fill(0, 0, seqLen * stepWidth);
    if (this.returnSequences) {
      for (let t = 0; t < seqLen; t++) {
        this.copyColumnBlockToArray(effectiveErr, t * batchSize, batchSize, perStep[t]);
      }
    } else {
      perStep[seqLen - 1].set(effectiveErr._data);
    }
    return perStep;
  }

  private assertBatchInputSupported(x: Matrix, batchSize: number) {
    if (!Number.isInteger(batchSize) || batchSize < 1) {
      throw new Error("RNN.forwardBatch: batchSize must be an integer >= 1.");
    }
    if (x._shape[0] !== this.units) {
      throw new Error(`RNN.forwardBatch: expected input rows ${this.units}, got ${x._shape[0]}`);
    }
    this.assertBatchInputSupportedShape(batchSize, x._shape[1]);
    if (this.stateful && batchSize !== 1) {
      throw new Error("RNN.forwardBatch: stateful=true only supports batchSize=1 in the current batched recurrent path.");
    }
  }

  private assertBatchInputSupportedShape(batchSize: number, totalCols: number) {
    if (totalCols < 1 || totalCols % batchSize !== 0) {
      throw new Error(
        `RNN batched path expects time-major columns divisible by batchSize. Got cols=${totalCols}, batchSize=${batchSize}.`
      );
    }
  }

  private ensureSequenceStateBuffers(seqLen: number) {
    const inputWidth = this.units;
    const hiddenWidth = this.hiddenUnits;
    const inputLen = seqLen * inputWidth;
    const hiddenLen = (seqLen + 1) * hiddenWidth;
    const activationLen = seqLen * hiddenWidth;

    if (this.inputSequenceBuffer.length < inputLen) {
      this.inputSequenceBuffer = new Float32Array(Math.max(inputLen, Math.max(1, this.inputSequenceBuffer.length * 2)));
    }
    if (this.hiddenSequenceBuffer.length < hiddenLen) {
      this.hiddenSequenceBuffer = new Float32Array(Math.max(hiddenLen, Math.max(1, this.hiddenSequenceBuffer.length * 2)));
    }
    if (this.activationGradientBuffer.length < activationLen) {
      this.activationGradientBuffer = new Float32Array(
        Math.max(activationLen, Math.max(1, this.activationGradientBuffer.length * 2))
      );
    }

    this.inputSequence = this.buildStepViews(this.inputSequenceBuffer, seqLen, inputWidth);
    this.hiddenSequence = this.buildStepViews(this.hiddenSequenceBuffer, seqLen + 1, hiddenWidth);
    this.activationGradients = this.buildStepViews(this.activationGradientBuffer, seqLen, hiddenWidth);
  }

  private ensureBatchSequenceStateBuffers(seqLen: number, batchSize: number) {
    const inputWidth = this.units * batchSize;
    const hiddenWidth = this.hiddenUnits * batchSize;
    const inputLen = seqLen * inputWidth;
    const hiddenLen = (seqLen + 1) * hiddenWidth;
    const activationLen = seqLen * hiddenWidth;

    if (this.batchInputSequenceBuffer.length < inputLen) {
      this.batchInputSequenceBuffer = new Float32Array(
        Math.max(inputLen, Math.max(1, this.batchInputSequenceBuffer.length * 2))
      );
    }
    if (this.batchHiddenSequenceBuffer.length < hiddenLen) {
      this.batchHiddenSequenceBuffer = new Float32Array(
        Math.max(hiddenLen, Math.max(1, this.batchHiddenSequenceBuffer.length * 2))
      );
    }
    if (this.batchActivationGradientBuffer.length < activationLen) {
      this.batchActivationGradientBuffer = new Float32Array(
        Math.max(activationLen, Math.max(1, this.batchActivationGradientBuffer.length * 2))
      );
    }

    this.batchInputSequence = this.buildStepViews(this.batchInputSequenceBuffer, seqLen, inputWidth);
    this.batchHiddenSequence = this.buildStepViews(this.batchHiddenSequenceBuffer, seqLen + 1, hiddenWidth);
    this.batchActivationGradients = this.buildStepViews(this.batchActivationGradientBuffer, seqLen, hiddenWidth);
  }

  private ensureErrorStepBuffers(seqLen: number) {
    const expectedLen = seqLen * this.hiddenUnits;
    if (this.errorStepBuffer.length < expectedLen) {
      this.errorStepBuffer = new Float32Array(Math.max(expectedLen, Math.max(1, this.errorStepBuffer.length * 2)));
    }
  }

  private ensureBatchErrorStepBuffers(seqLen: number, batchSize: number) {
    const expectedLen = seqLen * this.hiddenUnits * batchSize;
    if (this.batchErrorStepBuffer.length < expectedLen) {
      this.batchErrorStepBuffer = new Float32Array(
        Math.max(expectedLen, Math.max(1, this.batchErrorStepBuffer.length * 2))
      );
    }
  }

  private buildStepViews(buffer: Float32Array, steps: number, width: number): Float32Array[] {
    const views = new Array<Float32Array>(steps);
    for (let step = 0; step < steps; step++) {
      const start = step * width;
      views[step] = buffer.subarray(start, start + width);
    }
    return views;
  }

  private ensureBatchForwardBuffers(batchSize: number, totalCols: number, outCols: number) {
    if (this.resultBuffer._shape[0] !== this.hiddenUnits || this.resultBuffer._shape[1] !== outCols) {
      this.resultBuffer = mj.zeros([this.hiddenUnits, outCols]);
    }
    if (
      this.batchInputProjectionBuffer._shape[0] !== this.hiddenUnits ||
      this.batchInputProjectionBuffer._shape[1] !== totalCols
    ) {
      this.batchInputProjectionBuffer = mj.zeros([this.hiddenUnits, totalCols]);
    }
    if (this.batchInputSliceBuffer._shape[0] !== this.units || this.batchInputSliceBuffer._shape[1] !== batchSize) {
      this.batchInputSliceBuffer = mj.zeros([this.units, batchSize]);
    }
    if (
      this.batchProjectionSliceBuffer._shape[0] !== this.hiddenUnits ||
      this.batchProjectionSliceBuffer._shape[1] !== batchSize
    ) {
      this.batchProjectionSliceBuffer = mj.zeros([this.hiddenUnits, batchSize]);
      this.batchRecurrentBuffer = mj.zeros([this.hiddenUnits, batchSize]);
    }
  }

  private ensureBatchBackwardBuffers(batchSize: number) {
    if (this.batchDxStepBuffer._shape[0] !== this.units || this.batchDxStepBuffer._shape[1] !== batchSize) {
      this.batchDxStepBuffer = mj.zeros([this.units, batchSize]);
    }
    if (this.batchDhStepBuffer._shape[0] !== this.hiddenUnits || this.batchDhStepBuffer._shape[1] !== batchSize) {
      this.batchDhStepBuffer = mj.zeros([this.hiddenUnits, batchSize]);
    }
    if (
      this.batchOuterInputBuffer._shape[0] !== this.hiddenUnits ||
      this.batchOuterInputBuffer._shape[1] !== this.units
    ) {
      this.batchOuterInputBuffer = mj.zeros([this.hiddenUnits, this.units]);
      this.batchOuterHiddenBuffer = mj.zeros([this.hiddenUnits, this.hiddenUnits]);
      this.batchBiasGradBuffer = mj.zeros([this.hiddenUnits, 1]);
    }
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

  dispose() {
    this.batchInputProjectionBuffer = undefined as any;
    this.batchInputSliceBuffer = undefined as any;
    this.batchProjectionSliceBuffer = undefined as any;
    this.batchRecurrentBuffer = undefined as any;
    this.batchDxStepBuffer = undefined as any;
    this.batchDhStepBuffer = undefined as any;
    this.batchOuterInputBuffer = undefined as any;
    this.batchOuterHiddenBuffer = undefined as any;
    this.batchBiasGradBuffer = undefined as any;

    this.inputSequenceBuffer = new Float32Array(0);
    this.hiddenSequenceBuffer = new Float32Array(0);
    this.activationGradientBuffer = new Float32Array(0);
    this.batchInputSequenceBuffer = new Float32Array(0);
    this.batchHiddenSequenceBuffer = new Float32Array(0);
    this.batchActivationGradientBuffer = new Float32Array(0);
    this.errorStepBuffer = new Float32Array(0);
    this.batchErrorStepBuffer = new Float32Array(0);

    this.inputSequence = [];
    this.hiddenSequence = [];
    this.activationGradients = [];
    this.batchInputSequence = [];
    this.batchHiddenSequence = [];
    this.batchActivationGradients = [];
  }
}

