import { Cost, Optimizer, OptimizerType, StatusLayer } from "@oxide-js/core";
import { mj, engine } from "@oxide-js/core";
import { isNativeAvailable, lstmForwardNative, lstmBackwardNative } from "@oxide-js/core";
import { Matrix } from "@oxide-js/core";
import { setLoss } from "@oxide-js/core";
import { setOptimizer } from "@oxide-js/core";

export interface LSTMLayerConfig {
  units: number;
  hiddenUnits: number;
  forgetBias?: number;
  returnSequences?: boolean;
  returnState?: boolean;
  stateful?: boolean;
  alpha?: number;
  optimizer?: Optimizer;
  status?: StatusLayer;
  clipGradient?: number | boolean;
  loss?: Cost;
}

export default class LSTM {
  name = "lstm layer";
  units: number;
  hiddenUnits: number;
  forgetBias: number;
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

  private optimizerName: Optimizer;
  private lossName: Cost;
  private lossFunc: Function;
  private sumLoss = 0;
  private lossCount = 0;

  private optimizerWxi: OptimizerType;
  private optimizerWhi: OptimizerType;
  private optimizerBi: OptimizerType;
  private optimizerWxf: OptimizerType;
  private optimizerWhf: OptimizerType;
  private optimizerBf: OptimizerType;
  private optimizerWxo: OptimizerType;
  private optimizerWho: OptimizerType;
  private optimizerBo: OptimizerType;
  private optimizerWxg: OptimizerType;
  private optimizerWhg: OptimizerType;
  private optimizerBg: OptimizerType;

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
  private batchXSeq: Float32Array[] = [];
  private batchHSeq: Float32Array[] = [];
  private batchCSeq: Float32Array[] = [];
  private batchISeq: Float32Array[] = [];
  private batchFSeq: Float32Array[] = [];
  private batchOSeq: Float32Array[] = [];
  private batchGSeq: Float32Array[] = [];
  private batchInputSliceBuffer: Matrix = mj.matrix([]);
  private batchGateXIBuffer: Matrix = mj.matrix([]);
  private batchGateXFBuffer: Matrix = mj.matrix([]);
  private batchGateXOBuffer: Matrix = mj.matrix([]);
  private batchGateXGBuffer: Matrix = mj.matrix([]);
  private batchGateSliceIBuffer: Matrix = mj.matrix([]);
  private batchGateSliceFBuffer: Matrix = mj.matrix([]);
  private batchGateSliceOBuffer: Matrix = mj.matrix([]);
  private batchGateSliceGBuffer: Matrix = mj.matrix([]);
  private batchRecIBuffer: Matrix = mj.matrix([]);
  private batchRecFBuffer: Matrix = mj.matrix([]);
  private batchRecOBuffer: Matrix = mj.matrix([]);
  private batchRecGBuffer: Matrix = mj.matrix([]);
  private batchDxStepBuffer: Matrix = mj.matrix([]);
  private batchDhStepBuffer: Matrix = mj.matrix([]);
  private batchOuterInputBuffer: Matrix = mj.matrix([]);
  private batchOuterHiddenBuffer: Matrix = mj.matrix([]);
  private batchBiasGradBuffer: Matrix = mj.matrix([]);
  private batchTransposeProductBuffer: Matrix = mj.matrix([]);
  private xSeqBuffer: Float32Array = new Float32Array(0);
  private hSeqBuffer: Float32Array = new Float32Array(0);
  private cSeqBuffer: Float32Array = new Float32Array(0);
  private iSeqBuffer: Float32Array = new Float32Array(0);
  private fSeqBuffer: Float32Array = new Float32Array(0);
  private oSeqBuffer: Float32Array = new Float32Array(0);
  private gSeqBuffer: Float32Array = new Float32Array(0);
  private batchXSeqBuffer: Float32Array = new Float32Array(0);
  private batchHSeqBuffer: Float32Array = new Float32Array(0);
  private batchCSeqBuffer: Float32Array = new Float32Array(0);
  private batchISeqBuffer: Float32Array = new Float32Array(0);
  private batchFSeqBuffer: Float32Array = new Float32Array(0);
  private batchOSeqBuffer: Float32Array = new Float32Array(0);
  private batchGSeqBuffer: Float32Array = new Float32Array(0);
  private errorStepBuffer: Float32Array = new Float32Array(0);
  private batchErrorStepBuffer: Float32Array = new Float32Array(0);

  constructor({
    units,
    hiddenUnits,
    forgetBias = 1,
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
    this.forgetBias = forgetBias;
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
    this.bf._data.fill(forgetBias);
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

    // Ensure all weight matrices have names for better debugging in Tape
    this.Wxi.name = "Wxi"; this.Wxf.name = "Wxf"; this.Wxo.name = "Wxo"; this.Wxg.name = "Wxg";
    this.Whi.name = "Whi"; this.Whf.name = "Whf"; this.Who.name = "Who"; this.Whg.name = "Whg";
    this.bi.name = "bi"; this.bf.name = "bf"; this.bo.name = "bo"; this.bg.name = "bg";
  }

  getParams(): Matrix[] {
    return [
      this.Wxi, this.Wxf, this.Wxo, this.Wxg,
      this.Whi, this.Whf, this.Who, this.Whg,
      this.bi, this.bf, this.bo, this.bg
    ];
  }

  update(alpha?: number): void {
    const a = alpha || this.alpha;
    this.optimizerWxi.apply(this.Wxi, a);
    this.optimizerWxf.apply(this.Wxf, a);
    this.optimizerWxo.apply(this.Wxo, a);
    this.optimizerWxg.apply(this.Wxg, a);
    this.optimizerWhi.apply(this.Whi, a);
    this.optimizerWhf.apply(this.Whf, a);
    this.optimizerWho.apply(this.Who, a);
    this.optimizerWhg.apply(this.Whg, a);
    this.optimizerBi.apply(this.bi, a);
    this.optimizerBf.apply(this.bf, a);
    this.optimizerBo.apply(this.bo, a);
    this.optimizerBg.apply(this.bg, a);
  }

  save() {
    return {
      name: this.name,
      units: this.units,
      hiddenUnits: this.hiddenUnits,
      forgetBias: this.forgetBias,
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

  toKerasConfig() {
    return {
      class_name: "LSTM",
      config: {
        units: this.hiddenUnits,
        activation: "tanh",
        recurrent_activation: "sigmoid",
        use_bias: true,
        unit_forget_bias: true,
        return_sequences: this.returnSequences,
        return_state: this.returnState,
        stateful: this.stateful,
        name: `lstm_${Math.floor(Math.random() * 1000)}`,
        trainable: true,
      }
    };
  }

  getWeightsManifest(): { name: string; shape: [number, number]; data: Float32Array }[] {
    return [
      { name: "Wxi", shape: this.Wxi._shape, data: this.Wxi._data },
      { name: "Whi", shape: this.Whi._shape, data: this.Whi._data },
      { name: "bi", shape: this.bi._shape, data: this.bi._data },
      { name: "Wxf", shape: this.Wxf._shape, data: this.Wxf._data },
      { name: "Whf", shape: this.Whf._shape, data: this.Whf._data },
      { name: "bf", shape: this.bf._shape, data: this.bf._data },
      { name: "Wxo", shape: this.Wxo._shape, data: this.Wxo._data },
      { name: "Who", shape: this.Who._shape, data: this.Who._data },
      { name: "bo", shape: this.bo._shape, data: this.bo._data },
      { name: "Wxg", shape: this.Wxg._shape, data: this.Wxg._data },
      { name: "Whg", shape: this.Whg._shape, data: this.Whg._data },
      { name: "bg", shape: this.bg._shape, data: this.bg._data },
    ];
  }

  setWeightsFromBinary(weights: Record<string, Float32Array>): void {
    const names = ["Wxi", "Whi", "bi", "Wxf", "Whf", "bf", "Wxo", "Who", "bo", "Wxg", "Whg", "bg"];
    
    // Check if input units needs to be determined
    if (this.units === 0 && weights.Wxi) {
      this.units = weights.Wxi.length / this.hiddenUnits;
      
      // Re-initialize all weight shapes
      this.Wxi._shape = [this.hiddenUnits, this.units]; this.Wxi._data = new Float32Array(this.hiddenUnits * this.units);
      this.Wxf._shape = [this.hiddenUnits, this.units]; this.Wxf._data = new Float32Array(this.hiddenUnits * this.units);
      this.Wxo._shape = [this.hiddenUnits, this.units]; this.Wxo._data = new Float32Array(this.hiddenUnits * this.units);
      this.Wxg._shape = [this.hiddenUnits, this.units]; this.Wxg._data = new Float32Array(this.hiddenUnits * this.units);
    }
    
    for (const name of names) {
      if (weights[name]) {
        (this as any)[name]._data.set(weights[name]);
      }
    }
  }

  load(data: Record<string, number[][] | number | boolean | undefined>) {
    if (typeof data.forgetBias === "number") {
      this.forgetBias = data.forgetBias;
    }
    if (data.Wxi) {
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
    }
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
    optimizer?: Optimizer;
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
    this.ensureSequenceStateBuffers(seqLen);

    const h0 = this.hSeq[0];
    const c0 = this.cSeq[0];
    h0.fill(0);
    c0.fill(0);
    if (this.stateful) {
      h0.set(this.h_stateful._data);
      c0.set(this.c_stateful._data);
    }

    for (let t = 0; t < seqLen; t++) {
      const x_t = this.xSeq[t];
      this.copyColumnToArray(x, t, x_t);
      const hPrev = this.hSeq[t];
      const cPrev = this.cSeq[t];
      const i = this.iSeq[t];
      const f = this.fSeq[t];
      const o = this.oSeq[t];
      const g = this.gSeq[t];
      const c = this.cSeq[t + 1];
      const h = this.hSeq[t + 1];

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

    // RECORD FOR AUTO-DIFF
    const tape = engine.tape;
    if (tape) {
      tape.record(this.getParams().concat([x]), [this.resultBuffer], (grad: Matrix) => {
        const dx = this.backward(mj.zeros([1, 1]), grad, true); // true = gradOnly
        if (x.grad) x.grad.addInPlace(dx);
        else x.grad = dx;
      });
    }

    return this.resultBuffer;
  }

  forwardBatch(x: Matrix, batchSize: number): Matrix {
    this.assertBatchInputSupported(x, batchSize);
    const totalCols = x._shape[1];
    const seqLen = totalCols / batchSize;
    const outCols = this.returnSequences ? totalCols : batchSize;

    this.ensureBatchForwardBuffers(batchSize, totalCols, outCols);
    this.resultBuffer._data.fill(0);
    this.batchGateXIBuffer._data.fill(0);
    this.batchGateXFBuffer._data.fill(0);
    this.batchGateXOBuffer._data.fill(0);
    this.batchGateXGBuffer._data.fill(0);
    if (!isNativeAvailable()) {
      mj.dotProduct(this.Wxi, x, this.batchGateXIBuffer);
      mj.dotProduct(this.Wxf, x, this.batchGateXFBuffer);
      mj.dotProduct(this.Wxo, x, this.batchGateXOBuffer);
      mj.dotProduct(this.Wxg, x, this.batchGateXGBuffer);

      mj.addBias(this.batchGateXIBuffer, this.bi);
      mj.addBias(this.batchGateXFBuffer, this.bf);
      mj.addBias(this.batchGateXOBuffer, this.bo);
      mj.addBias(this.batchGateXGBuffer, this.bg);
    }

    this.inputShape = [this.units, totalCols];
    this.outputShape = [this.hiddenUnits, outCols];
    this.ensureBatchSequenceStateBuffers(seqLen, batchSize);

    const h0View = this.batchHSeq[0];
    const c0View = this.batchCSeq[0];
    h0View.fill(0);
    c0View.fill(0);
    if (this.stateful && batchSize === 1) {
      h0View.set(this.h_stateful._data);
      c0View.set(this.c_stateful._data);
    }

    if (
      isNativeAvailable() &&
      lstmForwardNative(
        this.Wxi._data,
        this.Wxf._data,
        this.Wxo._data,
        this.Wxg._data,
        this.Whi._data,
        this.Whf._data,
        this.Who._data,
        this.Whg._data,
        this.bi._data,
        this.bf._data,
        this.bo._data,
        this.bg._data,
        x._data,
        h0View,
        c0View,
        this.hiddenUnits,
        this.units,
        seqLen,
        batchSize,
        this.batchHSeqBuffer,
        this.batchCSeqBuffer,
        this.batchISeqBuffer,
        this.batchFSeqBuffer,
        this.batchOSeqBuffer,
        this.batchGSeqBuffer
      )
    ) {
      // Sync resultBuffer with the final h values
      if (this.returnSequences) {
        // Result buffer was already zeros, we need to copy batchHSeqBuffer[bs*hu .. ] into it
        // Or better: the native kernel could write to the resultBuffer directly if layout matched.
        // Actually, LSTM resultBuffer layout is [hiddenUnits, totalCols]
        // Native h_seq_out layout is [sl+1, bs, hu] -> flattened as (sl+1)*bs*hu
        // Wait, I need to check the layout mapping.
        for (let t = 0; t < seqLen; t++) {
          const h_t = this.batchHSeq[t + 1];
          this.writeColumnBlock(this.resultBuffer, t * batchSize, batchSize, h_t);
        }
      } else {
        this.resultBuffer._data.set(this.batchHSeq[seqLen]);
      }

      if (this.stateful && batchSize === 1) {
        this.h_stateful._data.set(this.batchHSeq[seqLen]);
        this.c_stateful._data.set(this.batchCSeq[seqLen]);
      }
      return this.resultBuffer;
    }

    for (let t = 0; t < seqLen; t++) {
      const colOffset = t * batchSize;
      this.copyColumnBlock(x, colOffset, batchSize, this.batchInputSliceBuffer);
      this.copyColumnBlock(this.batchGateXIBuffer, colOffset, batchSize, this.batchGateSliceIBuffer);
      this.copyColumnBlock(this.batchGateXFBuffer, colOffset, batchSize, this.batchGateSliceFBuffer);
      this.copyColumnBlock(this.batchGateXOBuffer, colOffset, batchSize, this.batchGateSliceOBuffer);
      this.copyColumnBlock(this.batchGateXGBuffer, colOffset, batchSize, this.batchGateSliceGBuffer);

      const hPrevMatrix = Matrix.fromFlat(this.batchHSeq[t], [this.hiddenUnits, batchSize]);
      mj.dotProduct(this.Whi, hPrevMatrix, this.batchRecIBuffer);
      mj.dotProduct(this.Whf, hPrevMatrix, this.batchRecFBuffer);
      mj.dotProduct(this.Who, hPrevMatrix, this.batchRecOBuffer);
      mj.dotProduct(this.Whg, hPrevMatrix, this.batchRecGBuffer);

      const i = this.batchISeq[t];
      const f = this.batchFSeq[t];
      const o = this.batchOSeq[t];
      const g = this.batchGSeq[t];
      const c = this.batchCSeq[t + 1];
      const h = this.batchHSeq[t + 1];
      const cPrev = this.batchCSeq[t];
      for (let idx = 0; idx < h.length; idx++) {
        const iVal = this.sigmoid(this.batchGateSliceIBuffer._data[idx] + this.batchRecIBuffer._data[idx]);
        const fVal = this.sigmoid(this.batchGateSliceFBuffer._data[idx] + this.batchRecFBuffer._data[idx]);
        const oVal = this.sigmoid(this.batchGateSliceOBuffer._data[idx] + this.batchRecOBuffer._data[idx]);
        const gVal = Math.tanh(this.batchGateSliceGBuffer._data[idx] + this.batchRecGBuffer._data[idx]);
        const cVal = fVal * cPrev[idx] + iVal * gVal;
        i[idx] = iVal;
        f[idx] = fVal;
        o[idx] = oVal;
        g[idx] = gVal;
        c[idx] = cVal;
        h[idx] = oVal * Math.tanh(cVal);
      }
      this.batchXSeq[t].set(this.batchInputSliceBuffer._data);
      if (this.returnSequences) {
        this.writeColumnBlock(this.resultBuffer, colOffset, batchSize, h);
      } else if (t === seqLen - 1) {
        this.resultBuffer._data.set(h);
      }
    }

    if (this.stateful && batchSize === 1) {
      this.h_stateful._data.set(this.batchHSeq[seqLen]);
      this.c_stateful._data.set(this.batchCSeq[seqLen]);
    }

    // RECORD FOR AUTO-DIFF
    const tape = engine.tape;
    if (tape) {
      tape.record(this.getParams().concat([x]), [this.resultBuffer], (grad: Matrix) => {
        const dx = this.backwardBatch(mj.zeros([1, 1]), grad, batchSize, true); // true = gradOnly
        if (x.grad) x.grad.addInPlace(dx);
        else x.grad = dx;
      });
    }

    return this.resultBuffer;
  }

  backward(y: Matrix, err: Matrix, gradOnly = false): Matrix {
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
    const dh = new Float32Array(this.hiddenUnits);
    const di = new Float32Array(this.hiddenUnits);
    const df = new Float32Array(this.hiddenUnits);
    const doGate = new Float32Array(this.hiddenUnits);
    const dg = new Float32Array(this.hiddenUnits);
    const dc = new Float32Array(this.hiddenUnits);
    const dzI = new Float32Array(this.hiddenUnits);
    const dzF = new Float32Array(this.hiddenUnits);
    const dzO = new Float32Array(this.hiddenUnits);
    const dzG = new Float32Array(this.hiddenUnits);
    let dhPrev = new Float32Array(this.hiddenUnits);
    let dcPrev = new Float32Array(this.hiddenUnits);

    for (let t = seqLen - 1; t >= 0; t--) {
      const hPrev = this.hSeq[t];
      const cPrev = this.cSeq[t];
      const c = this.cSeq[t + 1];
      const i = this.iSeq[t];
      const f = this.fSeq[t];
      const o = this.oSeq[t];
      const g = this.gSeq[t];
      const x_t = this.xSeq[t];

      dh.set(externalError[t]);
      for (let k = 0; k < this.hiddenUnits; k++) dh[k] += dhNext[k];

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

      for (let j = 0; j < this.units; j++) {
        let val = 0;
        for (let k = 0; k < this.hiddenUnits; k++) {
          val += this.Wxi._data[k * this.units + j] * dzI[k];
          val += this.Wxf._data[k * this.units + j] * dzF[k];
          val += this.Wxo._data[k * this.units + j] * dzO[k];
          val += this.Wxg._data[k * this.units + j] * dzG[k];
        }
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

      for (let k = 0; k < this.hiddenUnits; k++) dcPrev[k] = dc[k] * f[k];
      const prevDhNext = dhNext;
      dhNext = dhPrev;
      dhPrev = prevDhNext;
      const prevDcNext = dcNext;
      dcNext = dcPrev;
      dcPrev = prevDcNext;
    }

    this.clipGradientsIfNeeded(dWxi, dWhi, dBi, dWxf, dWhf, dBf, dWxo, dWho, dBo, dWxg, dWhg, dBg);
    
    // Populate .grad for Tape support
    if (this.Wxi.grad) this.Wxi.grad.addInPlace(dWxi); else this.Wxi.grad = dWxi;
    if (this.Whi.grad) this.Whi.grad.addInPlace(dWhi); else this.Whi.grad = dWhi;
    if (this.bi.grad) this.bi.grad.addInPlace(dBi); else this.bi.grad = dBi;
    if (this.Wxf.grad) this.Wxf.grad.addInPlace(dWxf); else this.Wxf.grad = dWxf;
    if (this.Whf.grad) this.Whf.grad.addInPlace(dWhf); else this.Whf.grad = dWhf;
    if (this.bf.grad) this.bf.grad.addInPlace(dBf); else this.bf.grad = dBf;
    if (this.Wxo.grad) this.Wxo.grad.addInPlace(dWxo); else this.Wxo.grad = dWxo;
    if (this.Who.grad) this.Who.grad.addInPlace(dWho); else this.Who.grad = dWho;
    if (this.bo.grad) this.bo.grad.addInPlace(dBo); else this.bo.grad = dBo;
    if (this.Wxg.grad) this.Wxg.grad.addInPlace(dWxg); else this.Wxg.grad = dWxg;
    if (this.Whg.grad) this.Whg.grad.addInPlace(dWhg); else this.Whg.grad = dWhg;
    if (this.bg.grad) this.bg.grad.addInPlace(dBg); else this.bg.grad = dBg;

    if (!gradOnly) {
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
    }

    return Matrix.fromFlat(dx, [this.units, seqLen]);
  }

  backwardBatch(y: Matrix, err: Matrix, batchSize: number, gradOnly = false): Matrix {
    const totalCols = this.inputShape[1];
    this.assertBatchInputSupportedShape(batchSize, totalCols);
    const seqLen = totalCols / batchSize;
    if (this.batchHSeq.length !== seqLen + 1 || this.batchCSeq.length !== seqLen + 1) {
      throw new Error("LSTM.backwardBatch: forwardBatch must be called before backwardBatch.");
    }

    const externalError = this.resolveBatchError(y, err, seqLen, batchSize);
    const dx = Matrix.fromFlat(new Float32Array(this.units * totalCols), [this.units, totalCols]);

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

    if (
      isNativeAvailable() &&
      lstmBackwardNative(
        this.Wxi._data, this.Wxf._data, this.Wxo._data, this.Wxg._data,
        this.Whi._data, this.Whf._data, this.Who._data, this.Whg._data,
        this.batchXSeqBuffer,
        this.batchHSeqBuffer,
        this.batchCSeqBuffer,
        this.batchISeqBuffer,
        this.batchFSeqBuffer,
        this.batchOSeqBuffer,
        this.batchGSeqBuffer,
        this.batchErrorStepBuffer,
        this.hiddenUnits, this.units, seqLen, batchSize,
        dWxi._data, dWhi._data, dBi._data,
        dWxf._data, dWhf._data, dBf._data,
        dWxo._data, dWho._data, dBo._data,
        dWxg._data, dWhg._data, dBg._data,
        dx._data
      )
    ) {
      this.clipGradientsIfNeeded(dWxi, dWhi, dBi, dWxf, dWhf, dBf, dWxo, dWho, dBo, dWxg, dWhg, dBg);

      // Populate .grad for Tape support
      if (this.Wxi.grad) this.Wxi.grad.addInPlace(dWxi); else this.Wxi.grad = dWxi;
      if (this.Whi.grad) this.Whi.grad.addInPlace(dWhi); else this.Whi.grad = dWhi;
      if (this.bi.grad) this.bi.grad.addInPlace(dBi); else this.bi.grad = dBi;
      if (this.Wxf.grad) this.Wxf.grad.addInPlace(dWxf); else this.Wxf.grad = dWxf;
      if (this.Whf.grad) this.Whf.grad.addInPlace(dWhf); else this.Whf.grad = dWhf;
      if (this.bf.grad) this.bf.grad.addInPlace(dBf); else this.bf.grad = dBf;
      if (this.Wxo.grad) this.Wxo.grad.addInPlace(dWxo); else this.Wxo.grad = dWxo;
      if (this.Who.grad) this.Who.grad.addInPlace(dWho); else this.Who.grad = dWho;
      if (this.bo.grad) this.bo.grad.addInPlace(dBo); else this.bo.grad = dBo;
      if (this.Wxg.grad) this.Wxg.grad.addInPlace(dWxg); else this.Wxg.grad = dWxg;
      if (this.Whg.grad) this.Whg.grad.addInPlace(dWhg); else this.Whg.grad = dWhg;
      if (this.bg.grad) this.bg.grad.addInPlace(dBg); else this.bg.grad = dBg;

      if (!gradOnly) {
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
      }
      return dx;
    }

    this.ensureBatchBackwardBuffers(batchSize);

    let dhNext = new Float32Array(this.hiddenUnits * batchSize);
    let dcNext = new Float32Array(this.hiddenUnits * batchSize);
    const dh = new Float32Array(this.hiddenUnits * batchSize);
    const dzI = new Float32Array(this.hiddenUnits * batchSize);
    const dzF = new Float32Array(this.hiddenUnits * batchSize);
    const dzO = new Float32Array(this.hiddenUnits * batchSize);
    const dzG = new Float32Array(this.hiddenUnits * batchSize);
    let dcPrev = new Float32Array(this.hiddenUnits * batchSize);
    let dhPrev = new Float32Array(this.hiddenUnits * batchSize);

    for (let t = seqLen - 1; t >= 0; t--) {
      const hPrev = this.batchHSeq[t];
      const cPrev = this.batchCSeq[t];
      const c = this.batchCSeq[t + 1];
      const i = this.batchISeq[t];
      const f = this.batchFSeq[t];
      const o = this.batchOSeq[t];
      const g = this.batchGSeq[t];
      const x_t = this.batchXSeq[t];

      dh.set(externalError[t]);
      for (let idx = 0; idx < dh.length; idx++) dh[idx] += dhNext[idx];

      for (let idx = 0; idx < dh.length; idx++) {
        const tanhC = Math.tanh(c[idx]);
        const doGate = dh[idx] * tanhC;
        const dc = dh[idx] * o[idx] * (1 - tanhC * tanhC) + dcNext[idx];
        const df = dc * cPrev[idx];
        const di = dc * g[idx];
        const dg = dc * i[idx];
        dzI[idx] = di * i[idx] * (1 - i[idx]);
        dzF[idx] = df * f[idx] * (1 - f[idx]);
        dzO[idx] = doGate * o[idx] * (1 - o[idx]);
        dzG[idx] = dg * (1 - g[idx] * g[idx]);
        dcPrev[idx] = dc * f[idx];
      }

      const dzIMatrix = Matrix.fromFlat(dzI, [this.hiddenUnits, batchSize]);
      const dzFMatrix = Matrix.fromFlat(dzF, [this.hiddenUnits, batchSize]);
      const dzOMatrix = Matrix.fromFlat(dzO, [this.hiddenUnits, batchSize]);
      const dzGMatrix = Matrix.fromFlat(dzG, [this.hiddenUnits, batchSize]);
      const xMatrix = Matrix.fromFlat(x_t, [this.units, batchSize]);
      const hPrevMatrix = Matrix.fromFlat(hPrev, [this.hiddenUnits, batchSize]);

      this.accumulateBatchWeightGradients(dWxi, dWhi, dBi, dzIMatrix, xMatrix, hPrevMatrix);
      this.accumulateBatchWeightGradients(dWxf, dWhf, dBf, dzFMatrix, xMatrix, hPrevMatrix);
      this.accumulateBatchWeightGradients(dWxo, dWho, dBo, dzOMatrix, xMatrix, hPrevMatrix);
      this.accumulateBatchWeightGradients(dWxg, dWhg, dBg, dzGMatrix, xMatrix, hPrevMatrix);

      this.batchDxStepBuffer._data.fill(0);
      this.accumulateTransposeProduct(this.Wxi, dzIMatrix, this.batchDxStepBuffer);
      this.accumulateTransposeProduct(this.Wxf, dzFMatrix, this.batchDxStepBuffer);
      this.accumulateTransposeProduct(this.Wxo, dzOMatrix, this.batchDxStepBuffer);
      this.accumulateTransposeProduct(this.Wxg, dzGMatrix, this.batchDxStepBuffer);
      this.writeColumnBlock(dx, t * batchSize, batchSize, this.batchDxStepBuffer._data);

      this.batchDhStepBuffer._data.fill(0);
      this.accumulateTransposeProduct(this.Whi, dzIMatrix, this.batchDhStepBuffer);
      this.accumulateTransposeProduct(this.Whf, dzFMatrix, this.batchDhStepBuffer);
      this.accumulateTransposeProduct(this.Who, dzOMatrix, this.batchDhStepBuffer);
      this.accumulateTransposeProduct(this.Whg, dzGMatrix, this.batchDhStepBuffer);
      dhPrev.set(this.batchDhStepBuffer._data);
      const prevDhNext = dhNext;
      dhNext = dhPrev;
      dhPrev = prevDhNext;
      const prevDcNext = dcNext;
      dcNext = dcPrev;
      dcPrev = prevDcNext;
    }

    this.clipGradientsIfNeeded(dWxi, dWhi, dBi, dWxf, dWhf, dBf, dWxo, dWho, dBo, dWxg, dWhg, dBg);
    
    // Populate .grad for Tape support
    if (this.Wxi.grad) this.Wxi.grad.addInPlace(dWxi); else this.Wxi.grad = dWxi;
    if (this.Whi.grad) this.Whi.grad.addInPlace(dWhi); else this.Whi.grad = dWhi;
    if (this.bi.grad) this.bi.grad.addInPlace(dBi); else this.bi.grad = dBi;
    if (this.Wxf.grad) this.Wxf.grad.addInPlace(dWxf); else this.Wxf.grad = dWxf;
    if (this.Whf.grad) this.Whf.grad.addInPlace(dWhf); else this.Whf.grad = dWhf;
    if (this.bf.grad) this.bf.grad.addInPlace(dBf); else this.bf.grad = dBf;
    if (this.Wxo.grad) this.Wxo.grad.addInPlace(dWxo); else this.Wxo.grad = dWxo;
    if (this.Who.grad) this.Who.grad.addInPlace(dWho); else this.Who.grad = dWho;
    if (this.bo.grad) this.bo.grad.addInPlace(dBo); else this.bo.grad = dBo;
    if (this.Wxg.grad) this.Wxg.grad.addInPlace(dWxg); else this.Wxg.grad = dWxg;
    if (this.Whg.grad) this.Whg.grad.addInPlace(dWhg); else this.Whg.grad = dWhg;
    if (this.bg.grad) this.bg.grad.addInPlace(dBg); else this.bg.grad = dBg;

    if (!gradOnly) {
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
    }
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
        `LSTM.backward: error shape mismatch, expected [${this.hiddenUnits},${outCols}], got [${effectiveErr._shape[0]},${effectiveErr._shape[1]}]`
      );
    }
    const perStep = this.buildStepViews(this.ensureErrorBuffer(seqLen * this.hiddenUnits), seqLen, this.hiddenUnits);
    this.errorStepBuffer.fill(0, 0, seqLen * this.hiddenUnits);
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
        `LSTM.backwardBatch: error shape mismatch, expected [${this.hiddenUnits},${expectedCols}], got [${effectiveErr._shape[0]},${effectiveErr._shape[1]}]`
      );
    }

    const perStep = this.buildStepViews(
      this.ensureBatchErrorBuffer(seqLen * this.hiddenUnits * batchSize),
      seqLen,
      this.hiddenUnits * batchSize
    );
    this.batchErrorStepBuffer.fill(0, 0, seqLen * this.hiddenUnits * batchSize);
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
      throw new Error("LSTM.forwardBatch: batchSize must be an integer >= 1.");
    }
    if (x._shape[0] !== this.units) {
      throw new Error(`LSTM.forwardBatch: expected input rows ${this.units}, got ${x._shape[0]}`);
    }
    this.assertBatchInputSupportedShape(batchSize, x._shape[1]);
    if (this.stateful && batchSize !== 1) {
      throw new Error("LSTM.forwardBatch: stateful=true only supports batchSize=1 in the current batched recurrent path.");
    }
  }

  private ensureSequenceStateBuffers(seqLen: number) {
    this.xSeqBuffer = this.ensureCapacity(this.xSeqBuffer, seqLen * this.units);
    this.hSeqBuffer = this.ensureCapacity(this.hSeqBuffer, (seqLen + 1) * this.hiddenUnits);
    this.cSeqBuffer = this.ensureCapacity(this.cSeqBuffer, (seqLen + 1) * this.hiddenUnits);
    this.iSeqBuffer = this.ensureCapacity(this.iSeqBuffer, seqLen * this.hiddenUnits);
    this.fSeqBuffer = this.ensureCapacity(this.fSeqBuffer, seqLen * this.hiddenUnits);
    this.oSeqBuffer = this.ensureCapacity(this.oSeqBuffer, seqLen * this.hiddenUnits);
    this.gSeqBuffer = this.ensureCapacity(this.gSeqBuffer, seqLen * this.hiddenUnits);
    this.xSeq = this.buildStepViews(this.xSeqBuffer, seqLen, this.units);
    this.hSeq = this.buildStepViews(this.hSeqBuffer, seqLen + 1, this.hiddenUnits);
    this.cSeq = this.buildStepViews(this.cSeqBuffer, seqLen + 1, this.hiddenUnits);
    this.iSeq = this.buildStepViews(this.iSeqBuffer, seqLen, this.hiddenUnits);
    this.fSeq = this.buildStepViews(this.fSeqBuffer, seqLen, this.hiddenUnits);
    this.oSeq = this.buildStepViews(this.oSeqBuffer, seqLen, this.hiddenUnits);
    this.gSeq = this.buildStepViews(this.gSeqBuffer, seqLen, this.hiddenUnits);
  }

  private ensureBatchSequenceStateBuffers(seqLen: number, batchSize: number) {
    const inputWidth = this.units * batchSize;
    const hiddenWidth = this.hiddenUnits * batchSize;
    this.batchXSeqBuffer = this.ensureCapacity(this.batchXSeqBuffer, seqLen * inputWidth);
    this.batchHSeqBuffer = this.ensureCapacity(this.batchHSeqBuffer, (seqLen + 1) * hiddenWidth);
    this.batchCSeqBuffer = this.ensureCapacity(this.batchCSeqBuffer, (seqLen + 1) * hiddenWidth);
    this.batchISeqBuffer = this.ensureCapacity(this.batchISeqBuffer, seqLen * hiddenWidth);
    this.batchFSeqBuffer = this.ensureCapacity(this.batchFSeqBuffer, seqLen * hiddenWidth);
    this.batchOSeqBuffer = this.ensureCapacity(this.batchOSeqBuffer, seqLen * hiddenWidth);
    this.batchGSeqBuffer = this.ensureCapacity(this.batchGSeqBuffer, seqLen * hiddenWidth);
    this.batchXSeq = this.buildStepViews(this.batchXSeqBuffer, seqLen, inputWidth);
    this.batchHSeq = this.buildStepViews(this.batchHSeqBuffer, seqLen + 1, hiddenWidth);
    this.batchCSeq = this.buildStepViews(this.batchCSeqBuffer, seqLen + 1, hiddenWidth);
    this.batchISeq = this.buildStepViews(this.batchISeqBuffer, seqLen, hiddenWidth);
    this.batchFSeq = this.buildStepViews(this.batchFSeqBuffer, seqLen, hiddenWidth);
    this.batchOSeq = this.buildStepViews(this.batchOSeqBuffer, seqLen, hiddenWidth);
    this.batchGSeq = this.buildStepViews(this.batchGSeqBuffer, seqLen, hiddenWidth);
  }

  private ensureErrorBuffer(expectedLen: number): Float32Array {
    this.errorStepBuffer = this.ensureCapacity(this.errorStepBuffer, expectedLen);
    return this.errorStepBuffer;
  }

  private ensureBatchErrorBuffer(expectedLen: number): Float32Array {
    this.batchErrorStepBuffer = this.ensureCapacity(this.batchErrorStepBuffer, expectedLen);
    return this.batchErrorStepBuffer;
  }

  private ensureCapacity(buffer: Float32Array, expectedLen: number): Float32Array {
    if (buffer.length < expectedLen) {
      return new Float32Array(Math.max(expectedLen, Math.max(1, buffer.length * 2)));
    }
    return buffer;
  }

  private buildStepViews(buffer: Float32Array, steps: number, width: number): Float32Array[] {
    const views = new Array<Float32Array>(steps);
    for (let step = 0; step < steps; step++) {
      const start = step * width;
      views[step] = buffer.subarray(start, start + width);
    }
    return views;
  }

  private assertBatchInputSupportedShape(batchSize: number, totalCols: number) {
    if (totalCols < 1 || totalCols % batchSize !== 0) {
      throw new Error(
        `LSTM batched path expects time-major columns divisible by batchSize. Got cols=${totalCols}, batchSize=${batchSize}.`
      );
    }
  }

  private ensureBatchForwardBuffers(batchSize: number, totalCols: number, outCols: number) {
    if (this.resultBuffer._shape[0] !== this.hiddenUnits || this.resultBuffer._shape[1] !== outCols) {
      this.resultBuffer = mj.zeros([this.hiddenUnits, outCols]);
    }
    if (this.batchInputSliceBuffer._shape[0] !== this.units || this.batchInputSliceBuffer._shape[1] !== batchSize) {
      this.batchInputSliceBuffer = mj.zeros([this.units, batchSize]);
    }
    this.batchGateXIBuffer = this.ensureBuffer(this.batchGateXIBuffer, this.hiddenUnits, totalCols);
    this.batchGateXFBuffer = this.ensureBuffer(this.batchGateXFBuffer, this.hiddenUnits, totalCols);
    this.batchGateXOBuffer = this.ensureBuffer(this.batchGateXOBuffer, this.hiddenUnits, totalCols);
    this.batchGateXGBuffer = this.ensureBuffer(this.batchGateXGBuffer, this.hiddenUnits, totalCols);
    this.batchGateSliceIBuffer = this.ensureBuffer(this.batchGateSliceIBuffer, this.hiddenUnits, batchSize);
    this.batchGateSliceFBuffer = this.ensureBuffer(this.batchGateSliceFBuffer, this.hiddenUnits, batchSize);
    this.batchGateSliceOBuffer = this.ensureBuffer(this.batchGateSliceOBuffer, this.hiddenUnits, batchSize);
    this.batchGateSliceGBuffer = this.ensureBuffer(this.batchGateSliceGBuffer, this.hiddenUnits, batchSize);
    this.batchRecIBuffer = this.ensureBuffer(this.batchRecIBuffer, this.hiddenUnits, batchSize);
    this.batchRecFBuffer = this.ensureBuffer(this.batchRecFBuffer, this.hiddenUnits, batchSize);
    this.batchRecOBuffer = this.ensureBuffer(this.batchRecOBuffer, this.hiddenUnits, batchSize);
    this.batchRecGBuffer = this.ensureBuffer(this.batchRecGBuffer, this.hiddenUnits, batchSize);
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

  dispose() {
    this.batchInputSliceBuffer = undefined as any;
    this.batchGateXIBuffer = undefined as any;
    this.batchGateXFBuffer = undefined as any;
    this.batchGateXOBuffer = undefined as any;
    this.batchGateXGBuffer = undefined as any;
    this.batchGateSliceIBuffer = undefined as any;
    this.batchGateSliceFBuffer = undefined as any;
    this.batchGateSliceOBuffer = undefined as any;
    this.batchGateSliceGBuffer = undefined as any;
    this.batchRecIBuffer = undefined as any;
    this.batchRecFBuffer = undefined as any;
    this.batchRecOBuffer = undefined as any;
    this.batchRecGBuffer = undefined as any;
    this.batchDxStepBuffer = undefined as any;
    this.batchDhStepBuffer = undefined as any;
    this.batchOuterInputBuffer = undefined as any;
    this.batchOuterHiddenBuffer = undefined as any;
    this.batchBiasGradBuffer = undefined as any;
    this.batchTransposeProductBuffer = undefined as any;

    this.xSeqBuffer = new Float32Array(0);
    this.hSeqBuffer = new Float32Array(0);
    this.cSeqBuffer = new Float32Array(0);
    this.iSeqBuffer = new Float32Array(0);
    this.fSeqBuffer = new Float32Array(0);
    this.oSeqBuffer = new Float32Array(0);
    this.gSeqBuffer = new Float32Array(0);

    this.batchXSeqBuffer = new Float32Array(0);
    this.batchHSeqBuffer = new Float32Array(0);
    this.batchCSeqBuffer = new Float32Array(0);
    this.batchISeqBuffer = new Float32Array(0);
    this.batchFSeqBuffer = new Float32Array(0);
    this.batchOSeqBuffer = new Float32Array(0);
    this.batchGSeqBuffer = new Float32Array(0);

    this.errorStepBuffer = new Float32Array(0);
    this.batchErrorStepBuffer = new Float32Array(0);

    this.xSeq = [];
    this.hSeq = [];
    this.cSeq = [];
    this.iSeq = [];
    this.fSeq = [];
    this.oSeq = [];
    this.gSeq = [];

    this.batchXSeq = [];
    this.batchHSeq = [];
    this.batchCSeq = [];
    this.batchISeq = [];
    this.batchFSeq = [];
    this.batchOSeq = [];
    this.batchGSeq = [];
  }
}
