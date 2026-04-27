import { Cost, Optimzier, OptimzierType, StatusLayer } from "../@types/type";
import concat from "../math/concat";
import { isNativeAvailable, rnnForwardNative, rnnBackwardNative } from "../math/rust_backend";
import mj from "../math";
import Matrix from "../matrix";
import setLoss from "../utils/setLoss";
import setOptimizer from "../utils/setOptimizer";
import { MemoryManager, WorkspaceConfig } from "../utils/memory";

export interface RNNLayerConfig {
  units: number;
  hiddenUnits: number;
  activation?: "tanh" | "relu";
  returnSequences?: boolean;
  returnState?: boolean;
  alpha?: number;
  optimizer?: Optimzier;
  status?: StatusLayer;
  clipGradient?: number | boolean;
  stateful?: boolean;
  loss?: Cost;
  memoryConfig?: WorkspaceConfig;
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
  memoryConfig: WorkspaceConfig;

  Wxh: Matrix;
  Whh: Matrix;
  bh: Matrix;

  private optimizerWxh: OptimzierType;
  private optimizerWhh: OptimzierType;
  private optimizerBh: OptimzierType;
  private optimizerName: Optimzier;
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
  private inputSequenceBuffer: any = new Float32Array(0);
  private hiddenSequenceBuffer: any = new Float32Array(0);
  private activationGradientsBuffer: any = new Float32Array(0);
  private batchInputSequenceBuffer: any = new Float32Array(0);
  private batchHiddenSequenceBuffer: any = new Float32Array(0);
  private batchActivationGradientBuffer: any = new Float32Array(0);
  private errorStepBuffer: any = new Float32Array(0);
  private batchErrorStepBuffer: any = new Float32Array(0);
  private evalBuffers: Record<string, any> = {};

  private bTensors?: {
    dWxh: Matrix;
    dWhh: Matrix;
    dBh: Matrix;
    dxBuffer: Float32Array;
    dhNext: Float32Array;
    dhBuffer: Float32Array;
    dzBuffer: Float32Array;
    dhPrevBuffer: Float32Array;
  };
  private bBatchTensors?: {
    dWxh: Matrix;
    dWhh: Matrix;
    dBh: Matrix;
    dxBuffer: Float32Array;
    dhNext: Float32Array;
    dhBuffer: Float32Array;
    dzBuffer: Float32Array;
    dhPrevBuffer: Float32Array;
  };

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
    memoryConfig = {},
  }: RNNLayerConfig) {
    this.memoryConfig = memoryConfig;
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
    optimizer?: Optimzier;
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

  forward(x: Matrix, options?: { workspace?: "train" | "eval" }): Matrix {
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
    this.ensureSequenceStateBuffers(seqLen, options?.workspace);

    const prev = this.hiddenSequence[0];
    prev.fill(0);
    if (this.stateful) {
      prev.set(this.h_stateful._data);
    }

    if (isNativeAvailable()) {
      this.inputSequenceBuffer.set(x._data);
      rnnForwardNative(
        x._data,
        this.Wxh._data,
        this.Whh._data,
        this.bh._data,
        seqLen,
        1,
        this.units,
        this.hiddenUnits,
        this.activation === "relu",
        this.hiddenSequenceBuffer,
        this.activationGradientsBuffer
      );
      if (this.returnSequences) {
        for (let t = 0; t < seqLen; t++) {
          this.setColumnData(this.resultBuffer._data, outCols, t, this.hiddenSequence[t + 1]);
        }
      } else {
        this.resultBuffer._data.set(this.hiddenSequence[seqLen]);
      }
    } else {
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
    }

    const lastHidden = this.hiddenSequence[seqLen];
    if (this.stateful) this.h_stateful._data.set(lastHidden);
    return this.resultBuffer;
  }

  forwardBatch(x: Matrix, batchSize: number, options?: { workspace?: "train" | "eval" }): Matrix {
    this.assertBatchInputSupported(x, batchSize);
    const totalCols = x._shape[1];
    const seqLen = totalCols / batchSize;
    const outCols = this.returnSequences ? totalCols : batchSize;

    this.ensureBatchForwardBuffers(batchSize, totalCols, outCols);
    this.resultBuffer._data.fill(0);
    this.batchInputProjectionBuffer._data.fill(0);
    mj.dotProduct(this.Wxh, x, this.batchInputProjectionBuffer);
    mj.addBias(this.batchInputProjectionBuffer, this.bh);

    this.inputShape = [this.units, totalCols];
    this.outputShape = [this.hiddenUnits, outCols];
    this.ensureBatchSequenceStateBuffers(seqLen, batchSize);

    const prev = this.batchHiddenSequence[0];
    prev.fill(0);
    if (this.stateful && batchSize === 1) {
      prev.set(this.h_stateful._data);
    }

    if (isNativeAvailable()) {
      rnnForwardNative(
        x._data,
        this.Wxh._data,
        this.Whh._data,
        this.bh._data,
        seqLen,
        batchSize,
        this.units,
        this.hiddenUnits,
        this.activation === "relu",
        this.batchHiddenSequenceBuffer,
        this.batchActivationGradientBuffer
      );
      if (this.returnSequences) {
        for (let t = 0; t < seqLen; t++) {
          this.writeColumnBlock(this.resultBuffer, t * batchSize, batchSize, this.batchHiddenSequence[t + 1]);
        }
      } else {
        this.resultBuffer._data.set(this.batchHiddenSequence[seqLen]);
      }
    } else {
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
    if (!this.bTensors || this.bTensors.dxBuffer.length < this.units * seqLen) {
      this.bTensors = {
        dWxh: Matrix.fromFlat(new Float32Array(this.hiddenUnits * this.units), [this.hiddenUnits, this.units]),
        dWhh: Matrix.fromFlat(new Float32Array(this.hiddenUnits * this.hiddenUnits), [this.hiddenUnits, this.hiddenUnits]),
        dBh: Matrix.fromFlat(new Float32Array(this.hiddenUnits), [this.hiddenUnits, 1]),
        dxBuffer: new Float32Array(Math.max(this.units * seqLen, this.bTensors ? this.bTensors.dxBuffer.length * 2 : 1024)),
        dhNext: new Float32Array(this.hiddenUnits),
        dhBuffer: new Float32Array(this.hiddenUnits),
        dzBuffer: new Float32Array(this.hiddenUnits),
        dhPrevBuffer: new Float32Array(this.hiddenUnits),
      };
    }
    const dWxh = this.bTensors.dWxh; dWxh._data.fill(0);
    const dWhh = this.bTensors.dWhh; dWhh._data.fill(0);
    const dBh = this.bTensors.dBh; dBh._data.fill(0);
    const dxData = this.bTensors.dxBuffer; dxData.fill(0, 0, this.units * seqLen);
    let dhNext = this.bTensors.dhNext; dhNext.fill(0);
    const dhBuffer = this.bTensors.dhBuffer;
    const dzBuffer = this.bTensors.dzBuffer;
    let dhPrevBuffer = this.bTensors.dhPrevBuffer;

    if (isNativeAvailable()) {
      const flatExtError = new Float32Array(this.hiddenUnits * seqLen);
      for (let t = 0; t < seqLen; t++) {
        flatExtError.set(externalError[t], t * this.hiddenUnits);
      }

      rnnBackwardNative(
        this.inputSequenceBuffer,
        this.Wxh._data,
        this.Whh._data,
        this.hiddenSequenceBuffer,
        this.activationGradientsBuffer,
        flatExtError,
        seqLen,
        1,
        this.units,
        this.hiddenUnits,
        dWxh._data,
        dWhh._data,
        dBh._data,
        dxData
      );
    } else {
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
    }

    this.clipGradientsIfNeeded(dWxh, dWhh, dBh);
    this.Wxh.subInPlace(this.optimizerWxh.calculate(dWxh, this.alpha));
    this.Whh.subInPlace(this.optimizerWhh.calculate(dWhh, this.alpha));
    this.Whh.subInPlace(this.optimizerWhh.calculate(dWhh, this.alpha));
    this.bh.subInPlace(this.optimizerBh.calculate(dBh, this.alpha));

    return Matrix.fromFlat(dxData.subarray(0, this.units * seqLen), [this.units, seqLen]);
  }

  backwardBatch(y: Matrix, err: Matrix, batchSize: number): Matrix {
    const totalCols = this.inputShape[1];
    this.assertBatchInputSupportedShape(batchSize, totalCols);
    const seqLen = totalCols / batchSize;
    if (this.batchHiddenSequence.length !== seqLen + 1) {
      throw new Error("RNN.backwardBatch: forwardBatch must be called before backwardBatch.");
    }

    const externalError = this.resolveBatchError(y, err, seqLen, batchSize);
    if (!this.bBatchTensors || this.bBatchTensors.dxBuffer.length < this.units * totalCols || this.bBatchTensors.dhNext.length < this.hiddenUnits * batchSize) {
      this.bBatchTensors = {
        dWxh: Matrix.fromFlat(new Float32Array(this.hiddenUnits * this.units), [this.hiddenUnits, this.units]),
        dWhh: Matrix.fromFlat(new Float32Array(this.hiddenUnits * this.hiddenUnits), [this.hiddenUnits, this.hiddenUnits]),
        dBh: Matrix.fromFlat(new Float32Array(this.hiddenUnits), [this.hiddenUnits, 1]),
        dxBuffer: new Float32Array(Math.max(this.units * totalCols, this.bBatchTensors ? this.bBatchTensors.dxBuffer.length * 2 : 1024)),
        dhNext: new Float32Array(this.hiddenUnits * batchSize),
        dhBuffer: new Float32Array(this.hiddenUnits * batchSize),
        dzBuffer: new Float32Array(this.hiddenUnits * batchSize),
        dhPrevBuffer: new Float32Array(this.hiddenUnits * batchSize),
      };
    }
    const dWxh = this.bBatchTensors.dWxh; dWxh._data.fill(0);
    const dWhh = this.bBatchTensors.dWhh; dWhh._data.fill(0);
    const dBh = this.bBatchTensors.dBh; dBh._data.fill(0);
    const dxData = this.bBatchTensors.dxBuffer; dxData.fill(0, 0, this.units * totalCols);
    const dx = Matrix.fromFlat(dxData.subarray(0, this.units * totalCols), [this.units, totalCols]);
    let dhNext = this.bBatchTensors.dhNext; dhNext.fill(0);
    this.ensureBatchBackwardBuffers(batchSize);
    const dhBuffer = this.bBatchTensors.dhBuffer;
    const dzBuffer = this.bBatchTensors.dzBuffer;
    let dhPrevBuffer = this.bBatchTensors.dhPrevBuffer;

    if (isNativeAvailable()) {
      const flatExtError = new Float32Array(this.hiddenUnits * batchSize * seqLen);
      for (let t = 0; t < seqLen; t++) {
        flatExtError.set(externalError[t], t * this.hiddenUnits * batchSize);
      }

      rnnBackwardNative(
        this.batchInputSequenceBuffer,
        this.Wxh._data,
        this.Whh._data,
        this.batchHiddenSequenceBuffer,
        this.batchActivationGradientBuffer,
        flatExtError,
        seqLen,
        batchSize,
        this.units,
        this.hiddenUnits,
        dWxh._data,
        dWhh._data,
        dBh._data,
        dx._data
      );
    } else {
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

  private ensureSequenceStateBuffers(seqLen: number, workspace: "train" | "eval" = "train") {
    const inputWidth = this.units;
    const hiddenWidth = this.hiddenUnits;
    const inputLen = seqLen * inputWidth;
    const hiddenLen = (seqLen + 1) * hiddenWidth;
    const activationLen = seqLen * hiddenWidth;

    if (workspace === "eval") {
      this.evalBuffers.inputSequenceBuffer = MemoryManager.ensureCapacity(this.evalBuffers.inputSequenceBuffer || new Float32Array(0), inputLen, this.memoryConfig) as any;
      this.evalBuffers.hiddenSequenceBuffer = MemoryManager.ensureCapacity(this.evalBuffers.hiddenSequenceBuffer || new Float32Array(0), hiddenLen, this.memoryConfig) as any;
      this.inputSequenceBuffer = this.evalBuffers.inputSequenceBuffer;
      this.hiddenSequenceBuffer = this.evalBuffers.hiddenSequenceBuffer;
    } else {
      this.inputSequenceBuffer = MemoryManager.ensureCapacity(this.inputSequenceBuffer, inputLen, this.memoryConfig) as any;
      this.hiddenSequenceBuffer = MemoryManager.ensureCapacity(this.hiddenSequenceBuffer, hiddenLen, this.memoryConfig) as any;
    }
    this.activationGradientsBuffer = MemoryManager.ensureCapacity(this.activationGradientsBuffer, activationLen, this.memoryConfig) as any;

    this.inputSequence = this.buildStepViews(this.inputSequenceBuffer, seqLen, inputWidth);
    this.hiddenSequence = this.buildStepViews(this.hiddenSequenceBuffer, seqLen + 1, hiddenWidth);
    this.activationGradients = this.buildStepViews(this.activationGradientsBuffer, seqLen, hiddenWidth);
  }

  private ensureBatchSequenceStateBuffers(seqLen: number, batchSize: number, workspace: "train" | "eval" = "train") {
    const inputWidth = this.units * batchSize;
    const hiddenWidth = this.hiddenUnits * batchSize;
    const inputLen = seqLen * inputWidth;
    const hiddenLen = (seqLen + 1) * hiddenWidth;
    const activationLen = seqLen * hiddenWidth;

    if (workspace === "eval") {
      this.evalBuffers.batchInputSequenceBuffer = MemoryManager.ensureCapacity(this.evalBuffers.batchInputSequenceBuffer || new Float32Array(0), inputLen, this.memoryConfig) as any;
      this.evalBuffers.batchHiddenSequenceBuffer = MemoryManager.ensureCapacity(this.evalBuffers.batchHiddenSequenceBuffer || new Float32Array(0), hiddenLen, this.memoryConfig) as any;
      this.batchInputSequenceBuffer = this.evalBuffers.batchInputSequenceBuffer;
      this.batchHiddenSequenceBuffer = this.evalBuffers.batchHiddenSequenceBuffer;
    } else {
      this.batchInputSequenceBuffer = MemoryManager.ensureCapacity(this.batchInputSequenceBuffer, inputLen, this.memoryConfig) as any;
      this.batchHiddenSequenceBuffer = MemoryManager.ensureCapacity(this.batchHiddenSequenceBuffer, hiddenLen, this.memoryConfig) as any;
    }
    this.batchActivationGradientBuffer = MemoryManager.ensureCapacity(this.batchActivationGradientBuffer, activationLen, this.memoryConfig) as any;

    this.batchInputSequence = this.buildStepViews(this.batchInputSequenceBuffer, seqLen, inputWidth);
    this.batchHiddenSequence = this.buildStepViews(this.batchHiddenSequenceBuffer, seqLen + 1, hiddenWidth);
    this.batchActivationGradients = this.buildStepViews(this.batchActivationGradientBuffer, seqLen, hiddenWidth);
  }

  private ensureErrorStepBuffers(seqLen: number) {
    const expectedLen = seqLen * this.hiddenUnits;
    this.errorStepBuffer = MemoryManager.ensureCapacity(this.errorStepBuffer, expectedLen, this.memoryConfig) as any;
  }

  private ensureBatchErrorStepBuffers(seqLen: number, batchSize: number) {
    const expectedLen = seqLen * this.hiddenUnits * batchSize;
    this.batchErrorStepBuffer = MemoryManager.ensureCapacity(this.batchErrorStepBuffer, expectedLen, this.memoryConfig) as any;
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

  releaseWorkspace(): void {
    this.evalBuffers = {};
    this.inputSequence = [];
    this.hiddenSequence = [];
    this.activationGradients = [];

    this.inputSequenceBuffer = new Float32Array(0);
    this.hiddenSequenceBuffer = new Float32Array(0);
    this.activationGradientsBuffer = new Float32Array(0);
    this.errorStepBuffer = new Float32Array(0);

    this.batchInputSequence = [];
    this.batchHiddenSequence = [];
    this.batchActivationGradients = [];
    this.batchInputSequenceBuffer = new Float32Array(0);
    this.batchHiddenSequenceBuffer = new Float32Array(0);
    this.batchActivationGradientBuffer = new Float32Array(0);
    this.batchErrorStepBuffer = new Float32Array(0);

    this.bTensors = undefined;
    this.bBatchTensors = undefined;

    this.resultBuffer = mj.matrix([]);
    this.batchInputProjectionBuffer = mj.matrix([]);
    this.batchInputSliceBuffer = mj.matrix([]);
    this.batchProjectionSliceBuffer = mj.matrix([]);
    this.batchRecurrentBuffer = mj.matrix([]);
    this.batchDxStepBuffer = mj.matrix([]);
    this.batchDhStepBuffer = mj.matrix([]);
    this.batchOuterInputBuffer = mj.matrix([]);
    this.batchOuterHiddenBuffer = mj.matrix([]);
    this.batchBiasGradBuffer = mj.matrix([]);
  }

  dispose(): void {
    this.releaseWorkspace();
    this.optimizerWxh?.dispose?.();
    this.optimizerWhh?.dispose?.();
    this.optimizerBh?.dispose?.();
    (this as any).Wxh = null;
    (this as any).Whh = null;
    (this as any).bh = null;
    (this as any).optimizerWxh = null;
    (this as any).optimizerWhh = null;
    (this as any).optimizerBh = null;
  }
}
