import { Cost, Optimzier, OptimzierType, StatusLayer } from "../@types/type";
import mj from "../math";
import Matrix from "../matrix";
import setLoss from "../utils/setLoss";
import setOptimizer from "../utils/setOptimizer";

export interface GRULayerConfig {
  units: number;
  hiddenUnits: number;
  returnSequences?: boolean;
  returnState?: boolean;
  stateful?: boolean;
  bidirectional?: boolean;
  alpha?: number;
  optimizer?: Optimzier;
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
  optimizerWxr: OptimzierType;
  optimizerWhr: OptimzierType;
  optimizerBr: OptimzierType;
  optimizerWxz: OptimzierType;
  optimizerWhz: OptimzierType;
  optimizerBz: OptimzierType;
  optimizerWxh: OptimzierType;
  optimizerWhh: OptimzierType;
  optimizerBh: OptimzierType;
  hStateful: Matrix;
  xSeq: Float32Array[];
  hSeq: Float32Array[];
  rSeq: Float32Array[];
  zSeq: Float32Array[];
  nSeq: Float32Array[];
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

  private optimizerName: Optimzier;
  private lossName: Cost;
  private lossFunc: Function;
  private sumLoss = 0;
  private lossCount = 0;
  private resultBuffer: Matrix = mj.matrix([]);
  private forwardDirection: DirectionParams;
  private backwardDirection?: DirectionParams;

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

    this.inputShape = [units, 1];
    this.outputShape = [this.bidirectional ? hiddenUnits * 2 : hiddenUnits, 1];
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
    this.deserializeDirection(this.forwardDirection, data.forward);
    if (data.backward && this.backwardDirection) {
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
    optimizer?: Optimzier;
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

  forward(x: Matrix): Matrix {
    if (x._shape[0] !== this.units) {
      throw new Error(`GRU.forward: expected input rows ${this.units}, got ${x._shape[0]}`);
    }
    const seqLen = x._shape[1];
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

  backward(y: Matrix, err: Matrix): Matrix {
    const seqLen = this.inputShape[1];
    if (seqLen <= 0 || this.forwardDirection.hSeq.length !== seqLen + 1) {
      throw new Error("GRU.backward: forward must be called before backward.");
    }
    const external = this.resolveError(y, err, seqLen);
    const extForward = Array.from({ length: seqLen }, () => new Float32Array(this.hiddenUnits));
    const extBackward = this.backwardDirection
      ? Array.from({ length: seqLen }, () => new Float32Array(this.hiddenUnits))
      : undefined;

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
    const out: Float32Array[] = Array.from({ length: seqLen }, () => new Float32Array(expectedRows));
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

  private runDirectionForward(direction: DirectionParams, x: Matrix, reverse: boolean): Float32Array[] {
    const seqLen = x._shape[1];
    direction.xSeq = new Array(seqLen);
    direction.hSeq = new Array(seqLen + 1);
    direction.rSeq = new Array(seqLen);
    direction.zSeq = new Array(seqLen);
    direction.nSeq = new Array(seqLen);
    const h0 = new Float32Array(this.hiddenUnits);
    if (this.stateful) h0.set(direction.hStateful._data);
    direction.hSeq[0] = h0.slice();

    const outputs: Float32Array[] = new Array(seqLen);
    for (let step = 0; step < seqLen; step++) {
      const t = reverse ? seqLen - 1 - step : step;
      const x_t = this.getColumn(x, t);
      const hPrev = direction.hSeq[step];
      const r = new Float32Array(this.hiddenUnits);
      const z = new Float32Array(this.hiddenUnits);
      const n = new Float32Array(this.hiddenUnits);
      const h = new Float32Array(this.hiddenUnits);

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

      direction.xSeq[step] = x_t;
      direction.hSeq[step + 1] = h;
      direction.rSeq[step] = r;
      direction.zSeq[step] = z;
      direction.nSeq[step] = n;
      outputs[t] = h;
    }

    if (this.stateful) {
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
    for (let step = seqLen - 1; step >= 0; step--) {
      const t = reverse ? seqLen - 1 - step : step;
      const hPrev = direction.hSeq[step];
      const x_t = direction.xSeq[step];
      const r = direction.rSeq[step];
      const z = direction.zSeq[step];
      const n = direction.nSeq[step];
      const dh = externalError[t].slice();
      for (let i = 0; i < this.hiddenUnits; i++) dh[i] += dhNext[i];

      const dn = new Float32Array(this.hiddenUnits);
      const dz = new Float32Array(this.hiddenUnits);
      const daN = new Float32Array(this.hiddenUnits);
      for (let i = 0; i < this.hiddenUnits; i++) {
        dn[i] = dh[i] * (1 - z[i]);
        dz[i] = dh[i] * (hPrev[i] - n[i]);
        daN[i] = dn[i] * (1 - n[i] * n[i]);
      }

      const rMulHPrev = new Float32Array(this.hiddenUnits);
      for (let i = 0; i < this.hiddenUnits; i++) rMulHPrev[i] = r[i] * hPrev[i];

      this.outerAccumulate(dWxh._data, this.hiddenUnits, this.units, daN, x_t);
      this.outerAccumulate(dWhh._data, this.hiddenUnits, this.hiddenUnits, daN, rMulHPrev);
      for (let i = 0; i < this.hiddenUnits; i++) dBh._data[i] += daN[i];

      const dRhPrev = new Float32Array(this.hiddenUnits);
      for (let j = 0; j < this.hiddenUnits; j++) {
        let val = 0;
        for (let i = 0; i < this.hiddenUnits; i++) {
          val += direction.Whh._data[i * this.hiddenUnits + j] * daN[i];
        }
        dRhPrev[j] = val;
      }
      const drFromN = new Float32Array(this.hiddenUnits);
      const dhPrevFromN = new Float32Array(this.hiddenUnits);
      for (let i = 0; i < this.hiddenUnits; i++) {
        drFromN[i] = dRhPrev[i] * hPrev[i];
        dhPrevFromN[i] = dRhPrev[i] * r[i];
      }

      const daR = new Float32Array(this.hiddenUnits);
      const daZ = new Float32Array(this.hiddenUnits);
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

      const dhPrev = new Float32Array(this.hiddenUnits);
      for (let j = 0; j < this.hiddenUnits; j++) {
        let val = 0;
        for (let i = 0; i < this.hiddenUnits; i++) {
          val += direction.Whr._data[i * this.hiddenUnits + j] * daR[i];
          val += direction.Whz._data[i * this.hiddenUnits + j] * daZ[i];
        }
        dhPrev[j] = val + dhPrevFromN[j] + dh[j] * z[j];
      }
      dhNext = dhPrev;
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

  private getColumn(m: Matrix, col: number): Float32Array {
    const [rows, cols] = m._shape;
    const out = new Float32Array(rows);
    for (let i = 0; i < rows; i++) out[i] = m._data[i * cols + col];
    return out;
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
}
