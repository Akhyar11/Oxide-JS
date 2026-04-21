import { Cost, Optimzier, OptimzierType, StatusLayer } from "../@types/type";
import mj from "../math";
import Matrix from "../matrix";
import setLoss from "../utils/setLoss";
import setOptimizer from "../utils/setOptimizer";

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
  by: Matrix;

  private optimizerWxh: OptimzierType;
  private optimizerWhh: OptimzierType;
  private optimizerBy: OptimzierType;
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
    this.by = mj.zeros([hiddenUnits, 1]);

    this.optimizerWxh = setOptimizer(optimizer, this.Wxh._shape, 1e-5);
    this.optimizerWhh = setOptimizer(optimizer, this.Whh._shape, 1e-5);
    this.optimizerBy = setOptimizer(optimizer, this.by._shape, 1e-5);

    this.inputShape = [units, 1];
    this.outputShape = [hiddenUnits, returnSequences ? 1 : 1];
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
      by: this.by._value,
      hStateful: this.h_stateful._value,
    };
  }

  load(data: {
    Wxh: number[][];
    Whh: number[][];
    by: number[][];
    hStateful?: number[][];
    clipGradient?: number | boolean;
  }) {
    this.Wxh._value = data.Wxh;
    this.Wxh._shape = [data.Wxh.length, data.Wxh[0]?.length ?? 0];
    this.Whh._value = data.Whh;
    this.Whh._shape = [data.Whh.length, data.Whh[0]?.length ?? 0];
    this.by._value = data.by;
    this.by._shape = [data.by.length, data.by[0]?.length ?? 0];
    if (data.hStateful) {
      this.h_stateful._value = data.hStateful;
      this.h_stateful._shape = [data.hStateful.length, data.hStateful[0]?.length ?? 0];
    } else {
      this.h_stateful = mj.zeros([this.hiddenUnits, 1]);
    }
    if (data.clipGradient !== undefined) this.clipGradient = data.clipGradient;

    this.optimizerWxh = setOptimizer(this.optimizerName, this.Wxh._shape, 1e-5);
    this.optimizerWhh = setOptimizer(this.optimizerName, this.Whh._shape, 1e-5);
    this.optimizerBy = setOptimizer(this.optimizerName, this.by._shape, 1e-5);
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
      this.optimizerBy = setOptimizer(optimizer, this.by._shape, 1e-5);
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
    if (x._shape[0] !== this.units) {
      throw new Error(`RNN.forward: expected input rows ${this.units}, got ${x._shape[0]}`);
    }
    const seqLen = x._shape[1];
    const outCols = this.returnSequences ? seqLen : 1;
    if (this.resultBuffer._shape[0] !== this.hiddenUnits || this.resultBuffer._shape[1] !== outCols) {
      this.resultBuffer = mj.zeros([this.hiddenUnits, outCols]);
    } else {
      this.resultBuffer._data.fill(0);
    }

    this.inputShape = [this.units, seqLen];
    this.outputShape = [this.hiddenUnits, outCols];
    this.inputSequence = new Array(seqLen);
    this.hiddenSequence = new Array(seqLen + 1);
    this.activationGradients = new Array(seqLen);

    const prev = new Float32Array(this.hiddenUnits);
    if (this.stateful) {
      prev.set(this.h_stateful._data);
    }
    this.hiddenSequence[0] = prev.slice();

    for (let t = 0; t < seqLen; t++) {
      const x_t = this.getColumn(x, t);
      const z = new Float32Array(this.hiddenUnits);
      const h_t = new Float32Array(this.hiddenUnits);
      const dAct = new Float32Array(this.hiddenUnits);

      for (let i = 0; i < this.hiddenUnits; i++) {
        let sum = this.by._data[i];
        const wxhOffset = i * this.units;
        for (let j = 0; j < this.units; j++) sum += this.Wxh._data[wxhOffset + j] * x_t[j];
        const whhOffset = i * this.hiddenUnits;
        const hPrev = this.hiddenSequence[t];
        for (let j = 0; j < this.hiddenUnits; j++) sum += this.Whh._data[whhOffset + j] * hPrev[j];
        z[i] = sum;

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

      this.inputSequence[t] = x_t;
      this.hiddenSequence[t + 1] = h_t;
      this.activationGradients[t] = dAct;
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

  backward(y: Matrix, err: Matrix): Matrix {
    const seqLen = this.inputShape[1];
    if (seqLen <= 0 || this.hiddenSequence.length !== seqLen + 1) {
      throw new Error("RNN.backward: forward must be called before backward.");
    }

    const externalError = this.resolveError(y, err, seqLen);
    const dWxh = Matrix.fromFlat(new Float32Array(this.hiddenUnits * this.units), [this.hiddenUnits, this.units]);
    const dWhh = Matrix.fromFlat(new Float32Array(this.hiddenUnits * this.hiddenUnits), [this.hiddenUnits, this.hiddenUnits]);
    const dBy = Matrix.fromFlat(new Float32Array(this.hiddenUnits), [this.hiddenUnits, 1]);
    const dxData = new Float32Array(this.units * seqLen);
    let dhNext = new Float32Array(this.hiddenUnits);

    for (let t = seqLen - 1; t >= 0; t--) {
      const dh = externalError[t].slice();
      for (let i = 0; i < this.hiddenUnits; i++) dh[i] += dhNext[i];

      const dz = new Float32Array(this.hiddenUnits);
      for (let i = 0; i < this.hiddenUnits; i++) dz[i] = dh[i] * this.activationGradients[t][i];

      this.outerAccumulate(dWxh._data, this.hiddenUnits, this.units, dz, this.inputSequence[t]);
      this.outerAccumulate(dWhh._data, this.hiddenUnits, this.hiddenUnits, dz, this.hiddenSequence[t]);
      for (let i = 0; i < this.hiddenUnits; i++) dBy._data[i] += dz[i];

      const dx_t = new Float32Array(this.units);
      for (let j = 0; j < this.units; j++) {
        let sum = 0;
        for (let i = 0; i < this.hiddenUnits; i++) sum += this.Wxh._data[i * this.units + j] * dz[i];
        dx_t[j] = sum;
        dxData[j * seqLen + t] = sum;
      }

      const dhPrev = new Float32Array(this.hiddenUnits);
      for (let j = 0; j < this.hiddenUnits; j++) {
        let sum = 0;
        for (let i = 0; i < this.hiddenUnits; i++) sum += this.Whh._data[i * this.hiddenUnits + j] * dz[i];
        dhPrev[j] = sum;
      }
      dhNext = dhPrev;
    }

    this.clipGradientsIfNeeded(dWxh, dWhh, dBy);
    this.Wxh.subInPlace(this.optimizerWxh.calculate(dWxh, this.alpha));
    this.Whh.subInPlace(this.optimizerWhh.calculate(dWhh, this.alpha));
    this.by.subInPlace(this.optimizerBy.calculate(dBy, this.alpha));

    return Matrix.fromFlat(dxData, [this.units, seqLen]);
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

    const perStep: Float32Array[] = Array.from({ length: seqLen }, () => new Float32Array(this.hiddenUnits));
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

  private clipGradientsIfNeeded(dWxh: Matrix, dWhh: Matrix, dBy: Matrix) {
    if (this.clipGradient === false) return;
    const limit = typeof this.clipGradient === "number" ? this.clipGradient : 5.0;
    mj.clipGradients(dWxh, limit);
    mj.clipGradients(dWhh, limit);
    mj.clipGradients(dBy, limit);
  }

  private getColumn(m: Matrix, col: number): Float32Array {
    const [rows, cols] = m._shape;
    const out = new Float32Array(rows);
    for (let i = 0; i < rows; i++) out[i] = m._data[i * cols + col];
    return out;
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
}
