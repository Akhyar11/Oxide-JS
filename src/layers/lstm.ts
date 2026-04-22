import { Cost, Optimzier, OptimzierType, StatusLayer } from "../@types/type";
import mj from "../math";
import Matrix from "../matrix";
import setLoss from "../utils/setLoss";
import setOptimizer from "../utils/setOptimizer";

export interface LSTMLayerConfig {
  units: number;
  hiddenUnits: number;
  returnSequences?: boolean;
  returnState?: boolean;
  stateful?: boolean;
  alpha?: number;
  optimizer?: Optimzier;
  status?: StatusLayer;
  clipGradient?: number | boolean;
  loss?: Cost;
}

export default class LSTM {
  name = "lstm layer";
  units: number;
  hiddenUnits: number;
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

  Wxi: Matrix;
  Whi: Matrix;
  bi: Matrix;
  Wxf: Matrix;
  Whf: Matrix;
  bf: Matrix;
  Wxo: Matrix;
  Who: Matrix;
  bo: Matrix;
  Wxg: Matrix;
  Whg: Matrix;
  bg: Matrix;

  private optimizerName: Optimzier;
  private lossName: Cost;
  private lossFunc: Function;
  private sumLoss = 0;
  private lossCount = 0;

  private optimizerWxi: OptimzierType;
  private optimizerWhi: OptimzierType;
  private optimizerBi: OptimzierType;
  private optimizerWxf: OptimzierType;
  private optimizerWhf: OptimzierType;
  private optimizerBf: OptimzierType;
  private optimizerWxo: OptimzierType;
  private optimizerWho: OptimzierType;
  private optimizerBo: OptimzierType;
  private optimizerWxg: OptimzierType;
  private optimizerWhg: OptimzierType;
  private optimizerBg: OptimzierType;

  private h_stateful: Matrix;
  private c_stateful: Matrix;

  private xSeq: Float32Array[] = [];
  private hSeq: Float32Array[] = [];
  private cSeq: Float32Array[] = [];
  private iSeq: Float32Array[] = [];
  private fSeq: Float32Array[] = [];
  private oSeq: Float32Array[] = [];
  private gSeq: Float32Array[] = [];
  private resultBuffer: Matrix = mj.matrix([]);

  constructor({
    units,
    hiddenUnits,
    returnSequences = false,
    returnState = false,
    stateful = false,
    alpha = 0.01,
    optimizer = "adam",
    status = "input",
    clipGradient = 5.0,
    loss = "mse",
  }: LSTMLayerConfig) {
    this.units = units;
    this.hiddenUnits = hiddenUnits;
    this.returnSequences = returnSequences;
    this.returnState = returnState;
    this.stateful = stateful;
    this.alpha = alpha;
    this.status = status;
    this.clipGradient = clipGradient;
    this.optimizerName = optimizer;
    this.lossName = loss;
    this.lossFunc = setLoss(loss);

    this.Wxi = mj.xavier([hiddenUnits, units]);
    this.Whi = mj.xavier([hiddenUnits, hiddenUnits]);
    this.bi = mj.zeros([hiddenUnits, 1]);
    this.Wxf = mj.xavier([hiddenUnits, units]);
    this.Whf = mj.xavier([hiddenUnits, hiddenUnits]);
    this.bf = mj.zeros([hiddenUnits, 1]);
    this.Wxo = mj.xavier([hiddenUnits, units]);
    this.Who = mj.xavier([hiddenUnits, hiddenUnits]);
    this.bo = mj.zeros([hiddenUnits, 1]);
    this.Wxg = mj.xavier([hiddenUnits, units]);
    this.Whg = mj.xavier([hiddenUnits, hiddenUnits]);
    this.bg = mj.zeros([hiddenUnits, 1]);

    this.optimizerWxi = setOptimizer(optimizer, this.Wxi._shape, 1e-5);
    this.optimizerWhi = setOptimizer(optimizer, this.Whi._shape, 1e-5);
    this.optimizerBi = setOptimizer(optimizer, this.bi._shape, 1e-5);
    this.optimizerWxf = setOptimizer(optimizer, this.Wxf._shape, 1e-5);
    this.optimizerWhf = setOptimizer(optimizer, this.Whf._shape, 1e-5);
    this.optimizerBf = setOptimizer(optimizer, this.bf._shape, 1e-5);
    this.optimizerWxo = setOptimizer(optimizer, this.Wxo._shape, 1e-5);
    this.optimizerWho = setOptimizer(optimizer, this.Who._shape, 1e-5);
    this.optimizerBo = setOptimizer(optimizer, this.bo._shape, 1e-5);
    this.optimizerWxg = setOptimizer(optimizer, this.Wxg._shape, 1e-5);
    this.optimizerWhg = setOptimizer(optimizer, this.Whg._shape, 1e-5);
    this.optimizerBg = setOptimizer(optimizer, this.bg._shape, 1e-5);

    this.inputShape = [units, 0];
    this.outputShape = [hiddenUnits, returnSequences ? 0 : 1];
    this.params = 4 * (hiddenUnits * units + hiddenUnits * hiddenUnits + hiddenUnits);
    this.h_stateful = mj.zeros([hiddenUnits, 1]);
    this.c_stateful = mj.zeros([hiddenUnits, 1]);
  }

  save() {
    return {
      name: this.name,
      units: this.units,
      hiddenUnits: this.hiddenUnits,
      returnSequences: this.returnSequences,
      returnState: this.returnState,
      stateful: this.stateful,
      alpha: this.alpha,
      optimizer: this.optimizerName,
      status: this.status,
      clipGradient: this.clipGradient,
      loss: this.lossName,
      Wxi: this.Wxi._value,
      Whi: this.Whi._value,
      bi: this.bi._value,
      Wxf: this.Wxf._value,
      Whf: this.Whf._value,
      bf: this.bf._value,
      Wxo: this.Wxo._value,
      Who: this.Who._value,
      bo: this.bo._value,
      Wxg: this.Wxg._value,
      Whg: this.Whg._value,
      bg: this.bg._value,
      hStateful: this.h_stateful._value,
      cStateful: this.c_stateful._value,
    };
  }

  load(data: Record<string, number[][] | number | boolean | undefined>) {
    this.assertSerializedMatrix(data.Wxi, "Wxi");
    this.assertSerializedMatrix(data.Whi, "Whi");
    this.assertSerializedMatrix(data.bi, "bi");
    this.assertSerializedMatrix(data.Wxf, "Wxf");
    this.assertSerializedMatrix(data.Whf, "Whf");
    this.assertSerializedMatrix(data.bf, "bf");
    this.assertSerializedMatrix(data.Wxo, "Wxo");
    this.assertSerializedMatrix(data.Who, "Who");
    this.assertSerializedMatrix(data.bo, "bo");
    this.assertSerializedMatrix(data.Wxg, "Wxg");
    this.assertSerializedMatrix(data.Whg, "Whg");
    this.assertSerializedMatrix(data.bg, "bg");
    this.loadMatrix(this.Wxi, data.Wxi as number[][]);
    this.loadMatrix(this.Whi, data.Whi as number[][]);
    this.loadMatrix(this.bi, data.bi as number[][]);
    this.loadMatrix(this.Wxf, data.Wxf as number[][]);
    this.loadMatrix(this.Whf, data.Whf as number[][]);
    this.loadMatrix(this.bf, data.bf as number[][]);
    this.loadMatrix(this.Wxo, data.Wxo as number[][]);
    this.loadMatrix(this.Who, data.Who as number[][]);
    this.loadMatrix(this.bo, data.bo as number[][]);
    this.loadMatrix(this.Wxg, data.Wxg as number[][]);
    this.loadMatrix(this.Whg, data.Whg as number[][]);
    this.loadMatrix(this.bg, data.bg as number[][]);
    if (data.hStateful !== undefined) {
      this.assertSerializedMatrix(data.hStateful, "hStateful");
      this.loadMatrix(this.h_stateful, data.hStateful as number[][]);
    }
    if (data.cStateful !== undefined) {
      this.assertSerializedMatrix(data.cStateful, "cStateful");
      this.loadMatrix(this.c_stateful, data.cStateful as number[][]);
    }
    if (typeof data.clipGradient === "number" || typeof data.clipGradient === "boolean") {
      this.clipGradient = data.clipGradient;
    }
    this.resetOptimizers();
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
      this.resetOptimizers();
    }
    if (error !== undefined) {
      this.lossName = error;
      this.lossFunc = setLoss(error);
    }
    if (clipGradient !== undefined) this.clipGradient = clipGradient;
  }

  resetState() {
    this.h_stateful._data.fill(0);
    this.c_stateful._data.fill(0);
  }

  getState() {
    return { h: this.h_stateful.clone(), c: this.c_stateful.clone() };
  }

  forward(x: Matrix): Matrix {
    if (this.returnState) {
      throw new Error("LSTM.forward: returnState=true is not supported yet. Disable returnState for LSTM.");
    }
    if (x._shape[0] !== this.units) {
      throw new Error(`LSTM.forward: expected input rows ${this.units}, got ${x._shape[0]}`);
    }
    const seqLen = x._shape[1];
    if (seqLen < 1) {
      throw new Error("LSTM.forward: expected a non-empty sequence input.");
    }
    const outCols = this.returnSequences ? seqLen : 1;
    if (this.resultBuffer._shape[0] !== this.hiddenUnits || this.resultBuffer._shape[1] !== outCols) {
      this.resultBuffer = mj.zeros([this.hiddenUnits, outCols]);
    } else {
      this.resultBuffer._data.fill(0);
    }

    this.inputShape = [this.units, seqLen];
    this.outputShape = [this.hiddenUnits, outCols];
    this.xSeq = new Array(seqLen);
    this.hSeq = new Array(seqLen + 1);
    this.cSeq = new Array(seqLen + 1);
    this.iSeq = new Array(seqLen);
    this.fSeq = new Array(seqLen);
    this.oSeq = new Array(seqLen);
    this.gSeq = new Array(seqLen);

    const h0 = new Float32Array(this.hiddenUnits);
    const c0 = new Float32Array(this.hiddenUnits);
    if (this.stateful) {
      h0.set(this.h_stateful._data);
      c0.set(this.c_stateful._data);
    }
    this.hSeq[0] = h0.slice();
    this.cSeq[0] = c0.slice();

    for (let t = 0; t < seqLen; t++) {
      const x_t = this.getColumn(x, t);
      const hPrev = this.hSeq[t];
      const cPrev = this.cSeq[t];
      const i = new Float32Array(this.hiddenUnits);
      const f = new Float32Array(this.hiddenUnits);
      const o = new Float32Array(this.hiddenUnits);
      const g = new Float32Array(this.hiddenUnits);
      const c = new Float32Array(this.hiddenUnits);
      const h = new Float32Array(this.hiddenUnits);

      for (let row = 0; row < this.hiddenUnits; row++) {
        const iPre = this.gatePreActivation(this.Wxi, this.Whi, this.bi, row, x_t, hPrev);
        const fPre = this.gatePreActivation(this.Wxf, this.Whf, this.bf, row, x_t, hPrev);
        const oPre = this.gatePreActivation(this.Wxo, this.Who, this.bo, row, x_t, hPrev);
        const gPre = this.gatePreActivation(this.Wxg, this.Whg, this.bg, row, x_t, hPrev);
        i[row] = this.sigmoid(iPre);
        f[row] = this.sigmoid(fPre);
        o[row] = this.sigmoid(oPre);
        g[row] = Math.tanh(gPre);
        c[row] = f[row] * cPrev[row] + i[row] * g[row];
        h[row] = o[row] * Math.tanh(c[row]);
      }

      this.xSeq[t] = x_t;
      this.iSeq[t] = i;
      this.fSeq[t] = f;
      this.oSeq[t] = o;
      this.gSeq[t] = g;
      this.cSeq[t + 1] = c;
      this.hSeq[t + 1] = h;
      if (this.returnSequences) {
        this.setColumnData(this.resultBuffer._data, outCols, t, h);
      } else if (t === seqLen - 1) {
        this.resultBuffer._data.set(h);
      }
    }

    const hLast = this.hSeq[seqLen];
    const cLast = this.cSeq[seqLen];
    if (this.stateful) {
      this.h_stateful._data.set(hLast);
      this.c_stateful._data.set(cLast);
    }
    return this.resultBuffer;
  }

  backward(y: Matrix, err: Matrix): Matrix {
    const seqLen = this.inputShape[1];
    if (seqLen <= 0 || this.hSeq.length !== seqLen + 1 || this.cSeq.length !== seqLen + 1) {
      throw new Error("LSTM.backward: forward must be called before backward.");
    }
    const externalError = this.resolveError(y, err, seqLen);
    const dx = new Float32Array(this.units * seqLen);

    const dWxi = Matrix.fromFlat(new Float32Array(this.hiddenUnits * this.units), [this.hiddenUnits, this.units]);
    const dWhi = Matrix.fromFlat(new Float32Array(this.hiddenUnits * this.hiddenUnits), [this.hiddenUnits, this.hiddenUnits]);
    const dBi = Matrix.fromFlat(new Float32Array(this.hiddenUnits), [this.hiddenUnits, 1]);
    const dWxf = Matrix.fromFlat(new Float32Array(this.hiddenUnits * this.units), [this.hiddenUnits, this.units]);
    const dWhf = Matrix.fromFlat(new Float32Array(this.hiddenUnits * this.hiddenUnits), [this.hiddenUnits, this.hiddenUnits]);
    const dBf = Matrix.fromFlat(new Float32Array(this.hiddenUnits), [this.hiddenUnits, 1]);
    const dWxo = Matrix.fromFlat(new Float32Array(this.hiddenUnits * this.units), [this.hiddenUnits, this.units]);
    const dWho = Matrix.fromFlat(new Float32Array(this.hiddenUnits * this.hiddenUnits), [this.hiddenUnits, this.hiddenUnits]);
    const dBo = Matrix.fromFlat(new Float32Array(this.hiddenUnits), [this.hiddenUnits, 1]);
    const dWxg = Matrix.fromFlat(new Float32Array(this.hiddenUnits * this.units), [this.hiddenUnits, this.units]);
    const dWhg = Matrix.fromFlat(new Float32Array(this.hiddenUnits * this.hiddenUnits), [this.hiddenUnits, this.hiddenUnits]);
    const dBg = Matrix.fromFlat(new Float32Array(this.hiddenUnits), [this.hiddenUnits, 1]);

    let dhNext = new Float32Array(this.hiddenUnits);
    let dcNext = new Float32Array(this.hiddenUnits);

    for (let t = seqLen - 1; t >= 0; t--) {
      const hPrev = this.hSeq[t];
      const cPrev = this.cSeq[t];
      const c = this.cSeq[t + 1];
      const i = this.iSeq[t];
      const f = this.fSeq[t];
      const o = this.oSeq[t];
      const g = this.gSeq[t];
      const x_t = this.xSeq[t];

      const dh = externalError[t].slice();
      for (let k = 0; k < this.hiddenUnits; k++) dh[k] += dhNext[k];

      const di = new Float32Array(this.hiddenUnits);
      const df = new Float32Array(this.hiddenUnits);
      const doGate = new Float32Array(this.hiddenUnits);
      const dg = new Float32Array(this.hiddenUnits);
      const dc = new Float32Array(this.hiddenUnits);
      const dzI = new Float32Array(this.hiddenUnits);
      const dzF = new Float32Array(this.hiddenUnits);
      const dzO = new Float32Array(this.hiddenUnits);
      const dzG = new Float32Array(this.hiddenUnits);

      for (let k = 0; k < this.hiddenUnits; k++) {
        const tanhC = Math.tanh(c[k]);
        doGate[k] = dh[k] * tanhC;
        dc[k] = dh[k] * o[k] * (1 - tanhC * tanhC) + dcNext[k];
        df[k] = dc[k] * cPrev[k];
        di[k] = dc[k] * g[k];
        dg[k] = dc[k] * i[k];
        dzI[k] = di[k] * i[k] * (1 - i[k]);
        dzF[k] = df[k] * f[k] * (1 - f[k]);
        dzO[k] = doGate[k] * o[k] * (1 - o[k]);
        dzG[k] = dg[k] * (1 - g[k] * g[k]);
      }

      this.outerAccumulate(dWxi._data, this.hiddenUnits, this.units, dzI, x_t);
      this.outerAccumulate(dWhi._data, this.hiddenUnits, this.hiddenUnits, dzI, hPrev);
      this.outerAccumulate(dWxf._data, this.hiddenUnits, this.units, dzF, x_t);
      this.outerAccumulate(dWhf._data, this.hiddenUnits, this.hiddenUnits, dzF, hPrev);
      this.outerAccumulate(dWxo._data, this.hiddenUnits, this.units, dzO, x_t);
      this.outerAccumulate(dWho._data, this.hiddenUnits, this.hiddenUnits, dzO, hPrev);
      this.outerAccumulate(dWxg._data, this.hiddenUnits, this.units, dzG, x_t);
      this.outerAccumulate(dWhg._data, this.hiddenUnits, this.hiddenUnits, dzG, hPrev);
      for (let k = 0; k < this.hiddenUnits; k++) {
        dBi._data[k] += dzI[k];
        dBf._data[k] += dzF[k];
        dBo._data[k] += dzO[k];
        dBg._data[k] += dzG[k];
      }

      const dx_t = new Float32Array(this.units);
      const dhPrev = new Float32Array(this.hiddenUnits);
      for (let j = 0; j < this.units; j++) {
        let val = 0;
        for (let k = 0; k < this.hiddenUnits; k++) {
          val += this.Wxi._data[k * this.units + j] * dzI[k];
          val += this.Wxf._data[k * this.units + j] * dzF[k];
          val += this.Wxo._data[k * this.units + j] * dzO[k];
          val += this.Wxg._data[k * this.units + j] * dzG[k];
        }
        dx_t[j] = val;
        dx[j * seqLen + t] = val;
      }

      for (let j = 0; j < this.hiddenUnits; j++) {
        let val = 0;
        for (let k = 0; k < this.hiddenUnits; k++) {
          val += this.Whi._data[k * this.hiddenUnits + j] * dzI[k];
          val += this.Whf._data[k * this.hiddenUnits + j] * dzF[k];
          val += this.Who._data[k * this.hiddenUnits + j] * dzO[k];
          val += this.Whg._data[k * this.hiddenUnits + j] * dzG[k];
        }
        dhPrev[j] = val;
      }

      const dcPrev = new Float32Array(this.hiddenUnits);
      for (let k = 0; k < this.hiddenUnits; k++) dcPrev[k] = dc[k] * f[k];
      dhNext = dhPrev;
      dcNext = dcPrev;
    }

    this.clipGradientsIfNeeded(dWxi, dWhi, dBi, dWxf, dWhf, dBf, dWxo, dWho, dBo, dWxg, dWhg, dBg);
    this.Wxi.subInPlace(this.optimizerWxi.calculate(dWxi, this.alpha));
    this.Whi.subInPlace(this.optimizerWhi.calculate(dWhi, this.alpha));
    this.bi.subInPlace(this.optimizerBi.calculate(dBi, this.alpha));
    this.Wxf.subInPlace(this.optimizerWxf.calculate(dWxf, this.alpha));
    this.Whf.subInPlace(this.optimizerWhf.calculate(dWhf, this.alpha));
    this.bf.subInPlace(this.optimizerBf.calculate(dBf, this.alpha));
    this.Wxo.subInPlace(this.optimizerWxo.calculate(dWxo, this.alpha));
    this.Who.subInPlace(this.optimizerWho.calculate(dWho, this.alpha));
    this.bo.subInPlace(this.optimizerBo.calculate(dBo, this.alpha));
    this.Wxg.subInPlace(this.optimizerWxg.calculate(dWxg, this.alpha));
    this.Whg.subInPlace(this.optimizerWhg.calculate(dWhg, this.alpha));
    this.bg.subInPlace(this.optimizerBg.calculate(dBg, this.alpha));

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
    const outCols = this.returnSequences ? seqLen : 1;
    if (effectiveErr._shape[0] !== this.hiddenUnits || effectiveErr._shape[1] !== outCols) {
      throw new Error(
        `LSTM.backward: error shape mismatch, expected [${this.hiddenUnits},${outCols}], got [${effectiveErr._shape[0]},${effectiveErr._shape[1]}]`
      );
    }
    const perStep: Float32Array[] = Array.from({ length: seqLen }, () => new Float32Array(this.hiddenUnits));
    if (this.returnSequences) {
      for (let t = 0; t < seqLen; t++) {
        for (let i = 0; i < this.hiddenUnits; i++) perStep[t][i] = effectiveErr._data[i * seqLen + t];
      }
    } else {
      for (let i = 0; i < this.hiddenUnits; i++) perStep[seqLen - 1][i] = effectiveErr._data[i];
    }
    return perStep;
  }

  private gatePreActivation(
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

  private getColumn(m: Matrix, col: number): Float32Array {
    const [rows, cols] = m._shape;
    const out = new Float32Array(rows);
    for (let i = 0; i < rows; i++) out[i] = m._data[i * cols + col];
    return out;
  }

  private setColumnData(target: Float32Array, targetCols: number, col: number, data: Float32Array) {
    for (let i = 0; i < data.length; i++) target[i * targetCols + col] = data[i];
  }

  private loadMatrix(target: Matrix, value: number[][]) {
    target._value = value;
    target._shape = [value.length, value[0]?.length ?? 0];
  }

  private assertSerializedMatrix(value: unknown, fieldName: string): asserts value is number[][] {
    if (!Array.isArray(value)) {
      throw new Error(`LSTM.load: expected serialized matrix '${fieldName}'.`);
    }
  }

  private resetOptimizers() {
    this.optimizerWxi = setOptimizer(this.optimizerName, this.Wxi._shape, 1e-5);
    this.optimizerWhi = setOptimizer(this.optimizerName, this.Whi._shape, 1e-5);
    this.optimizerBi = setOptimizer(this.optimizerName, this.bi._shape, 1e-5);
    this.optimizerWxf = setOptimizer(this.optimizerName, this.Wxf._shape, 1e-5);
    this.optimizerWhf = setOptimizer(this.optimizerName, this.Whf._shape, 1e-5);
    this.optimizerBf = setOptimizer(this.optimizerName, this.bf._shape, 1e-5);
    this.optimizerWxo = setOptimizer(this.optimizerName, this.Wxo._shape, 1e-5);
    this.optimizerWho = setOptimizer(this.optimizerName, this.Who._shape, 1e-5);
    this.optimizerBo = setOptimizer(this.optimizerName, this.bo._shape, 1e-5);
    this.optimizerWxg = setOptimizer(this.optimizerName, this.Wxg._shape, 1e-5);
    this.optimizerWhg = setOptimizer(this.optimizerName, this.Whg._shape, 1e-5);
    this.optimizerBg = setOptimizer(this.optimizerName, this.bg._shape, 1e-5);
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
