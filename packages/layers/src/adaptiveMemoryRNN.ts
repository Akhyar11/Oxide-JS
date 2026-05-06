import { Cost, Optimizer, OptimizerType, StatusLayer } from "@oxide-js/core";
import { mj } from "@oxide-js/core";
import { adaptiveMemoryRnnBackwardNative, isNativeAvailable } from "@oxide-js/core";
import { Matrix } from "@oxide-js/core";
import { setLoss } from "@oxide-js/core";
import { setOptimizer } from "@oxide-js/core";

export interface AdaptiveMemoryRNNConfig {
  units: number;
  hiddenUnits: number;
  activation?: "tanh" | "relu";
  memorySlots?: number;
  memoryDim?: number;
  returnSequences?: boolean;
  returnState?: boolean;
  stateful?: boolean;
  alpha?: number;
  optimizer?: Optimizer;
  status?: StatusLayer;
  clipGradient?: number | boolean;
  loss?: Cost;
}

type AdaptiveMemoryStepCache = {
  x: Float32Array;
  hPrev: Float32Array;
  h: Float32Array;
  dAct: Float32Array;
  combined: Float32Array;
  read: Float32Array;
  queryInput: Float32Array;
  query: Float32Array;
  attention: Float32Array;
  gateInput: Float32Array;
  gate: Float32Array;
  candidate: Float32Array;
  memoryKeysBefore: Float32Array;
  memoryValuesBefore: Float32Array;
  writeSlot: number;
};

export default class AdaptiveMemoryRNN {
  name = "adaptive memory rnn layer";
  units: number;
  hiddenUnits: number;
  activation: "tanh" | "relu";
  memorySlots: number;
  memoryDim: number;
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
  Wq: Matrix;
  Wm: Matrix;
  memoryKeys: Matrix;
  memoryValues: Matrix;
  memoryUsage: Float32Array;
  Wg: Matrix;
  bg: Matrix;
  hStateful: Matrix;

  private optimizerWxh: OptimizerType;
  private optimizerWhh: OptimizerType;
  private optimizerBh: OptimizerType;
  private optimizerWq: OptimizerType;
  private optimizerWm: OptimizerType;
  private optimizerWg: OptimizerType;
  private optimizerBg: OptimizerType;
  private optimizerName: Optimizer;
  private lossName: Cost;
  private lossFunc: Function;
  private sumLoss = 0;
  private lossCount = 0;

  private combinedInputSequence: Float32Array<ArrayBufferLike>[] = [];
  private rawInputSequence: Float32Array<ArrayBufferLike>[] = [];
  private memoryReadSequence: Float32Array<ArrayBufferLike>[] = [];
  private hiddenSequence: Float32Array<ArrayBufferLike>[] = [];
  private activationGradients: Float32Array<ArrayBufferLike>[] = [];
  private stepCaches: AdaptiveMemoryStepCache[] = [];
  private batchStepCaches: AdaptiveMemoryStepCache[][] = [];
  private batchCombinedInputSequence: Float32Array<ArrayBufferLike>[] = [];
  private batchRawInputSequence: Float32Array<ArrayBufferLike>[] = [];
  private batchHiddenSequence: Float32Array<ArrayBufferLike>[] = [];
  private batchActivationGradients: Float32Array<ArrayBufferLike>[] = [];
  private combinedInputBuffer: Float32Array<ArrayBufferLike> = new Float32Array(0);
  private rawInputBuffer: Float32Array<ArrayBufferLike> = new Float32Array(0);
  private memoryReadBuffer: Float32Array<ArrayBufferLike> = new Float32Array(0);
  private hiddenSequenceBuffer: Float32Array<ArrayBufferLike> = new Float32Array(0);
  private activationGradientBuffer: Float32Array<ArrayBufferLike> = new Float32Array(0);
  private batchCombinedInputBuffer: Float32Array<ArrayBufferLike> = new Float32Array(0);
  private batchRawInputBuffer: Float32Array<ArrayBufferLike> = new Float32Array(0);
  private batchHiddenSequenceBuffer: Float32Array<ArrayBufferLike> = new Float32Array(0);
  private batchActivationGradientBuffer: Float32Array<ArrayBufferLike> = new Float32Array(0);
  private batchErrorStepBuffer: Float32Array<ArrayBufferLike> = new Float32Array(0);
  private batchMemoryKeysBuffer: Float32Array<ArrayBufferLike> = new Float32Array(0);
  private batchMemoryValuesBuffer: Float32Array<ArrayBufferLike> = new Float32Array(0);
  private batchMemoryUsageBuffer: Float32Array<ArrayBufferLike> = new Float32Array(0);
  private batchQueryBlockBuffer: Float32Array<ArrayBufferLike> = new Float32Array(0);
  private batchReadBlockBuffer: Float32Array<ArrayBufferLike> = new Float32Array(0);
  private batchGateBlockBuffer: Float32Array<ArrayBufferLike> = new Float32Array(0);
  private batchCandidateBlockBuffer: Float32Array<ArrayBufferLike> = new Float32Array(0);
  private batchBestSlots = new Int32Array(0);
  private batchDxBuffer: Matrix = mj.matrix([]);
  private batchDhNextBuffer: Float32Array<ArrayBufferLike> = new Float32Array(0);
  private batchDhBuffer: Float32Array<ArrayBufferLike> = new Float32Array(0);
  private batchDzBuffer: Float32Array<ArrayBufferLike> = new Float32Array(0);
  private batchDhPrevBuffer: Float32Array<ArrayBufferLike> = new Float32Array(0);
  private errorStepBuffer: Float32Array<ArrayBufferLike> = new Float32Array(0);
  // Flat combined buffer used by native forward/backward: [seqLen*(units+memoryDim)*batchSize]
  private batchCombinedFlatBuffer: Float32Array<ArrayBufferLike> = new Float32Array(0);
  private resultBuffer: Matrix = mj.matrix([]);
  private queryInputScratch: Float32Array<ArrayBufferLike> = new Float32Array(0);
  private queryScratch: Float32Array<ArrayBufferLike> = new Float32Array(0);
  private scoresScratch: Float32Array<ArrayBufferLike> = new Float32Array(0);
  private attentionScratch: Float32Array<ArrayBufferLike> = new Float32Array(0);
  private readScratch: Float32Array<ArrayBufferLike> = new Float32Array(0);
  private gateInputScratch: Float32Array<ArrayBufferLike> = new Float32Array(0);
  private gateScratch: Float32Array<ArrayBufferLike> = new Float32Array(0);
  private candidateScratch: Float32Array<ArrayBufferLike> = new Float32Array(0);
  private batchXSampleScratch: Float32Array<ArrayBufferLike> = new Float32Array(0);
  private batchHPrevSampleScratch: Float32Array<ArrayBufferLike> = new Float32Array(0);
  private batchCombinedSampleScratch: Float32Array<ArrayBufferLike> = new Float32Array(0);
  private batchHSampleScratch: Float32Array<ArrayBufferLike> = new Float32Array(0);
  private batchDActSampleScratch: Float32Array<ArrayBufferLike> = new Float32Array(0);
  private dWxhBuffer: Matrix = mj.matrix([]);
  private dWhhBuffer: Matrix = mj.matrix([]);
  private dBhBuffer: Matrix = mj.matrix([]);
  private dxBuffer: Matrix = mj.matrix([]);
  private dhNextBuffer: Float32Array<ArrayBufferLike> = new Float32Array(0);
  private dhBuffer: Float32Array<ArrayBufferLike> = new Float32Array(0);
  private dzBuffer: Float32Array<ArrayBufferLike> = new Float32Array(0);
  private dhPrevBuffer: Float32Array<ArrayBufferLike> = new Float32Array(0);

  constructor({
    units,
    hiddenUnits,
    activation = "tanh",
    memorySlots = 32,
    memoryDim = hiddenUnits,
    returnSequences = false,
    returnState = false,
    stateful = false,
    alpha = 0.001,
    optimizer = "adam",
    status = "input",
    clipGradient = 5.0,
    loss = "mse",
  }: AdaptiveMemoryRNNConfig) {
    this.assertPositiveInteger(units, "units");
    this.assertPositiveInteger(hiddenUnits, "hiddenUnits");
    this.assertPositiveInteger(memorySlots, "memorySlots");
    this.assertPositiveInteger(memoryDim, "memoryDim");

    this.units = units;
    this.hiddenUnits = hiddenUnits;
    this.activation = activation;
    this.memorySlots = memorySlots;
    this.memoryDim = memoryDim;
    this.returnSequences = returnSequences;
    this.returnState = returnState;
    this.stateful = stateful;
    this.alpha = alpha;
    this.status = status;
    this.clipGradient = clipGradient;
    this.optimizerName = optimizer;
    this.lossName = loss;
    this.lossFunc = setLoss(loss);

    this.Wxh = mj.xavier([hiddenUnits, units + memoryDim]);
    this.Whh = mj.xavier([hiddenUnits, hiddenUnits]);
    this.bh = mj.zeros([hiddenUnits, 1]);
    this.Wq = mj.xavier([memoryDim, units + hiddenUnits]);
    this.Wm = mj.xavier([memoryDim, hiddenUnits]);
    this.Wg = mj.xavier([memoryDim, units + hiddenUnits + memoryDim]);
    this.bg = mj.zeros([memoryDim, 1]);
    this.memoryKeys = mj.zeros([memoryDim, memorySlots]);
    this.memoryValues = mj.zeros([memoryDim, memorySlots]);
    this.memoryUsage = new Float32Array(memorySlots);
    this.hStateful = mj.zeros([hiddenUnits, 1]);

    this.optimizerWxh = setOptimizer(optimizer, this.Wxh._shape, 1e-5);
    this.optimizerWhh = setOptimizer(optimizer, this.Whh._shape, 1e-5);
    this.optimizerBh = setOptimizer(optimizer, this.bh._shape, 1e-5);
    this.optimizerWq = setOptimizer(optimizer, this.Wq._shape, 1e-5);
    this.optimizerWm = setOptimizer(optimizer, this.Wm._shape, 1e-5);
    this.optimizerWg = setOptimizer(optimizer, this.Wg._shape, 1e-5);
    this.optimizerBg = setOptimizer(optimizer, this.bg._shape, 1e-5);

    this.inputShape = [units, 0];
    this.outputShape = [hiddenUnits, returnSequences ? 0 : 1];
    this.params = this.computeParams();
  }

  save() {
    return {
      name: this.name,
      units: this.units,
      hiddenUnits: this.hiddenUnits,
      activation: this.activation,
      memorySlots: this.memorySlots,
      memoryDim: this.memoryDim,
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
      Wq: this.Wq._value,
      Wm: this.Wm._value,
      Wg: this.Wg._value,
      bg: this.bg._value,
      memoryKeys: this.memoryKeys._value,
      memoryValues: this.memoryValues._value,
      memoryUsage: Array.from(this.memoryUsage),
      hStateful: this.hStateful._value,
    };
  }

  load(data: any): void {
    if (data.units !== undefined) this.units = data.units;
    if (data.hiddenUnits !== undefined) this.hiddenUnits = data.hiddenUnits;
    if (data.activation !== undefined) this.activation = data.activation;
    if (data.memorySlots !== undefined) this.memorySlots = data.memorySlots;
    if (data.memoryDim !== undefined) this.memoryDim = data.memoryDim;
    if (data.returnSequences !== undefined) this.returnSequences = data.returnSequences;
    if (data.returnState !== undefined) this.returnState = data.returnState;
    if (data.stateful !== undefined) this.stateful = data.stateful;
    if (data.alpha !== undefined) this.alpha = data.alpha;
    if (data.optimizer !== undefined) this.optimizerName = data.optimizer;
    if (data.status !== undefined) this.status = data.status;
    if (data.loss !== undefined) {
      this.lossName = data.loss;
      this.lossFunc = setLoss(data.loss);
    }

    this.loadMatrix("Wxh", data.Wxh);
    this.loadMatrix("Whh", data.Whh);
    this.loadMatrix("bh", data.bh);
    this.loadMatrix("Wq", data.Wq);
    this.loadMatrix("Wm", data.Wm);
    this.loadMatrix("Wg", data.Wg);
    this.loadMatrix("bg", data.bg);
    this.loadMatrix("memoryKeys", data.memoryKeys);
    this.loadMatrix("memoryValues", data.memoryValues);

    this.memoryUsage = new Float32Array(data.memoryUsage ?? this.memorySlots);
    if (this.memoryUsage.length !== this.memorySlots) {
      const resized = new Float32Array(this.memorySlots);
      resized.set(this.memoryUsage.subarray(0, this.memorySlots));
      this.memoryUsage = resized;
    }

    if (data.hStateful) {
      this.hStateful._value = data.hStateful;
      this.hStateful._shape = [data.hStateful.length, data.hStateful[0]?.length ?? 0];
    } else {
      this.hStateful = mj.zeros([this.hiddenUnits, 1]);
    }
    if (data.clipGradient !== undefined) this.clipGradient = data.clipGradient;

    this.inputShape = [this.units, 0];
    this.outputShape = [this.hiddenUnits, this.returnSequences ? 0 : 1];
    this.params = this.computeParams();
    this.resetOptimizers(this.optimizerName);
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
  }): void {
    if (alpha !== undefined) this.alpha = alpha;
    if (optimizer !== undefined) {
      this.optimizerName = optimizer;
      this.resetOptimizers(optimizer);
    }
    if (error !== undefined) {
      this.lossName = error;
      this.lossFunc = setLoss(error);
    }
    if (clipGradient !== undefined) this.clipGradient = clipGradient;
  }

  resetState(): void {
    this.hStateful._data.fill(0);
    this.memoryKeys._data.fill(0);
    this.memoryValues._data.fill(0);
    this.memoryUsage.fill(0);
  }

  getState() {
    return {
      h: this.hStateful.clone(),
      memoryKeys: this.memoryKeys.clone(),
      memoryValues: this.memoryValues.clone(),
      memoryUsage: Array.from(this.memoryUsage),
    };
  }

  forward(x: Matrix): Matrix {
    if (this.returnState) {
      throw new Error("AdaptiveMemoryRNN.forward: returnState=true is not supported yet.");
    }
    if (x._shape[0] !== this.units) {
      throw new Error(`AdaptiveMemoryRNN.forward: expected input rows ${this.units}, got ${x._shape[0]}`);
    }
    const seqLen = x._shape[1];
    if (seqLen < 1) {
      throw new Error("AdaptiveMemoryRNN.forward: expected a non-empty sequence input.");
    }
    if (!this.stateful) {
      this.memoryKeys._data.fill(0);
      this.memoryValues._data.fill(0);
      this.memoryUsage.fill(0);
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
    this.stepCaches = new Array<AdaptiveMemoryStepCache>(seqLen);

    const h0 = this.hiddenSequence[0];
    h0.fill(0);
    if (this.stateful) h0.set(this.hStateful._data);

    for (let t = 0; t < seqLen; t++) {
      const x_t = this.rawInputSequence[t];
      this.copyColumnToArray(x, t, x_t);
      const hPrev = this.hiddenSequence[t];
      const memoryKeysBefore = this.memoryKeys._data.slice();
      const memoryValuesBefore = this.memoryValues._data.slice();
      const query = this.computeQueryInto(x_t, hPrev);
      const bestSlot = this.retrieveMemoryInto(query);
      const h_t = this.hiddenSequence[t + 1];
      const dAct = this.activationGradients[t];
      const combined = this.combinedInputSequence[t];

      this.memoryReadSequence[t].set(this.readScratch);
      this.concatArrays(combined, x_t, this.readScratch);
      this.rnnCellForward(combined, hPrev, h_t, dAct);

      const gate = this.computeWriteGateInto(x_t, h_t, this.readScratch);
      const candidate = this.projectCandidateMemoryInto(h_t);
      const writeSlot = this.selectWriteSlot(bestSlot);
      this.stepCaches[t] = {
        x: x_t.slice(),
        hPrev: hPrev.slice(),
        h: h_t.slice(),
        dAct: dAct.slice(),
        combined: combined.slice(),
        read: this.readScratch.slice(),
        queryInput: this.queryInputScratch.slice(),
        query: query.slice(),
        attention: this.attentionScratch.slice(),
        gateInput: this.gateInputScratch.slice(),
        gate: gate.slice(),
        candidate: candidate.slice(),
        memoryKeysBefore,
        memoryValuesBefore,
        writeSlot,
      };
      this.updateMemory(writeSlot, query, candidate, gate);

      if (this.returnSequences) {
        this.setColumnData(this.resultBuffer._data, outCols, t, h_t);
      } else if (t === seqLen - 1) {
        this.resultBuffer._data.set(h_t);
      }
    }

    if (this.stateful) this.hStateful._data.set(this.hiddenSequence[seqLen]);
    return this.resultBuffer;
  }

  forwardBatch(x: Matrix, batchSize: number): Matrix {
    this.assertBatchInputSupported(x, batchSize);
    const totalCols = x._shape[1];
    const seqLen = totalCols / batchSize;
    const outCols = this.returnSequences ? totalCols : batchSize;

    if (this.resultBuffer._shape[0] !== this.hiddenUnits || this.resultBuffer._shape[1] !== outCols) {
      this.resultBuffer = mj.zeros([this.hiddenUnits, outCols]);
    } else {
      this.resultBuffer._data.fill(0);
    }

    this.inputShape = [this.units, totalCols];
    this.outputShape = [this.hiddenUnits, outCols];
    this.ensureBatchSequenceStateBuffers(seqLen, batchSize);
    this.ensureBatchMemoryBuffers(batchSize);
    this.ensureScratchBuffers();
    this.batchStepCaches = Array.from({ length: batchSize }, () => new Array<AdaptiveMemoryStepCache>(seqLen));

    if (!this.stateful || batchSize > 1) {
      this.batchMemoryKeysBuffer.fill(0, 0, batchSize * this.memoryDim * this.memorySlots);
      this.batchMemoryValuesBuffer.fill(0, 0, batchSize * this.memoryDim * this.memorySlots);
      this.batchMemoryUsageBuffer.fill(0, 0, batchSize * this.memorySlots);
    } else {
      this.batchMemoryKeysBuffer.set(this.memoryKeys._data, 0);
      this.batchMemoryValuesBuffer.set(this.memoryValues._data, 0);
      this.batchMemoryUsageBuffer.set(this.memoryUsage, 0);
    }

    const h0 = this.batchHiddenSequence[0];
    h0.fill(0);
    if (this.stateful && batchSize === 1) h0.set(this.hStateful._data);
    for (let t = 0; t < seqLen; t++) {
      const rawBlock = this.batchRawInputSequence[t];
      this.copyColumnBlockToArray(x, t * batchSize, batchSize, rawBlock);
      const combinedBlock = this.batchCombinedInputSequence[t];
      const hPrevBlock = this.batchHiddenSequence[t];
      const hBlock = this.batchHiddenSequence[t + 1];
      const dActBlock = this.batchActivationGradients[t];

      for (let b = 0; b < batchSize; b++) {
        const x_t = this.copyBatchSampleToScratch(rawBlock, this.units, batchSize, b, this.batchXSampleScratch);
        const hPrev = this.copyBatchSampleToScratch(hPrevBlock, this.hiddenUnits, batchSize, b, this.batchHPrevSampleScratch);
        const query = this.computeQueryInto(x_t, hPrev);
        const memoryOffset = b * this.memoryDim * this.memorySlots;
        const usageOffset = b * this.memorySlots;
        const memoryKeysBefore = this.batchMemoryKeysBuffer.slice(memoryOffset, memoryOffset + this.memoryDim * this.memorySlots);
        const memoryValuesBefore = this.batchMemoryValuesBuffer.slice(memoryOffset, memoryOffset + this.memoryDim * this.memorySlots);
        const bestSlot = this.retrieveMemoryFromBuffersInto(
          query,
          this.batchMemoryKeysBuffer,
          this.batchMemoryValuesBuffer,
          memoryOffset
        );
        this.writeBatchSampleFromScratch(
          combinedBlock,
          this.units + this.memoryDim,
          batchSize,
          b,
          x_t,
          this.readScratch,
          this.batchCombinedSampleScratch
        );

        const h_t = this.batchHSampleScratch;
        const dAct = this.batchDActSampleScratch;
        this.rnnCellForward(this.batchCombinedSampleScratch, hPrev, h_t, dAct);
        this.copyScratchToBatchSample(hBlock, this.hiddenUnits, batchSize, b, h_t);
        this.copyScratchToBatchSample(dActBlock, this.hiddenUnits, batchSize, b, dAct);

        const gate = this.computeWriteGateInto(x_t, h_t, this.readScratch);
        const candidate = this.projectCandidateMemoryInto(h_t);
        const writeSlot = this.selectWriteSlotFromUsage(bestSlot, this.batchMemoryUsageBuffer, usageOffset);
        this.batchStepCaches[b][t] = {
          x: x_t.slice(),
          hPrev: hPrev.slice(),
          h: h_t.slice(),
          dAct: dAct.slice(),
          combined: this.batchCombinedSampleScratch.slice(),
          read: this.readScratch.slice(),
          queryInput: this.queryInputScratch.slice(),
          query: query.slice(),
          attention: this.attentionScratch.slice(),
          gateInput: this.gateInputScratch.slice(),
          gate: gate.slice(),
          candidate: candidate.slice(),
          memoryKeysBefore,
          memoryValuesBefore,
          writeSlot,
        };
        this.updateMemoryBuffers(
          writeSlot, query, candidate, gate,
          this.batchMemoryKeysBuffer, this.batchMemoryValuesBuffer, this.batchMemoryUsageBuffer,
          memoryOffset, usageOffset
        );
      }

      if (this.returnSequences) {
        this.writeColumnBlock(this.resultBuffer, t * batchSize, batchSize, hBlock);
      } else if (t === seqLen - 1) {
        this.resultBuffer._data.set(hBlock);
      }
    }

    if (this.stateful && batchSize === 1) {
      this.hStateful._data.set(this.batchHiddenSequence[seqLen]);
      this.memoryKeys._data.set(this.batchMemoryKeysBuffer.subarray(0, this.memoryDim * this.memorySlots));
      this.memoryValues._data.set(this.batchMemoryValuesBuffer.subarray(0, this.memoryDim * this.memorySlots));
      this.memoryUsage.set(this.batchMemoryUsageBuffer.subarray(0, this.memorySlots));
    }
    return this.resultBuffer;
  }

  backward(y: Matrix, err: Matrix): Matrix {
    const seqLen = this.inputShape[1];
    if (seqLen <= 0 || this.stepCaches.length !== seqLen) {
      throw new Error("AdaptiveMemoryRNN.backward: forward must be called before backward.");
    }

    const externalError = this.resolveError(y, err, seqLen);
    this.ensureBackwardBuffers(seqLen);
    const dWxh = this.dWxhBuffer;
    const dWhh = this.dWhhBuffer;
    const dBh = this.dBhBuffer;
    const dWq = Matrix.fromFlat(new Float32Array(this.memoryDim * (this.units + this.hiddenUnits)), [this.memoryDim, this.units + this.hiddenUnits]);
    const dWm = Matrix.fromFlat(new Float32Array(this.memoryDim * this.hiddenUnits), [this.memoryDim, this.hiddenUnits]);
    const dWg = Matrix.fromFlat(new Float32Array(this.memoryDim * (this.units + this.hiddenUnits + this.memoryDim)), [this.memoryDim, this.units + this.hiddenUnits + this.memoryDim]);
    const dBg = Matrix.fromFlat(new Float32Array(this.memoryDim), [this.memoryDim, 1]);
    dWxh._data.fill(0);
    dWhh._data.fill(0);
    dBh._data.fill(0);
    this.dxBuffer._data.fill(0);

    const nativeOk = this.runNativeBackward(
      [this.stepCaches],
      [externalError],
      dWxh,
      dWhh,
      dBh,
      dWq,
      dWm,
      dWg,
      dBg,
      this.dxBuffer._data,
      seqLen,
      1
    );
    if (!nativeOk) {
      this.backwardThroughStepCaches(
        this.stepCaches,
        externalError,
        dWxh,
        dWhh,
        dBh,
        dWq,
        dWm,
        dWg,
        dBg,
        this.dxBuffer._data,
        seqLen,
        0,
        1
      );
    }

    this.clipGradientsIfNeeded(dWxh, dWhh, dBh, dWq, dWm, dWg, dBg);
    this.Wxh.subInPlace(this.optimizerWxh.calculate(dWxh, this.alpha));
    this.Whh.subInPlace(this.optimizerWhh.calculate(dWhh, this.alpha));
    this.bh.subInPlace(this.optimizerBh.calculate(dBh, this.alpha));
    this.Wq.subInPlace(this.optimizerWq.calculate(dWq, this.alpha));
    this.Wm.subInPlace(this.optimizerWm.calculate(dWm, this.alpha));
    this.Wg.subInPlace(this.optimizerWg.calculate(dWg, this.alpha));
    this.bg.subInPlace(this.optimizerBg.calculate(dBg, this.alpha));

    return this.dxBuffer;
  }

  backwardBatch(y: Matrix, err: Matrix, batchSize: number): Matrix {
    const totalCols = this.inputShape[1];
    this.assertBatchInputSupportedShape(batchSize, totalCols);
    const seqLen = totalCols / batchSize;
    if (this.batchStepCaches.length !== batchSize || this.batchStepCaches[0]?.length !== seqLen) {
      throw new Error("AdaptiveMemoryRNN.backwardBatch: forwardBatch must be called before backwardBatch.");
    }

    const externalError = this.resolveBatchError(y, err, seqLen, batchSize);
    this.ensureBatchBackwardBuffers(seqLen, batchSize);
    const dWxh = this.dWxhBuffer;
    const dWhh = this.dWhhBuffer;
    const dBh = this.dBhBuffer;
    const dx = this.batchDxBuffer;
    const dWq = Matrix.fromFlat(new Float32Array(this.memoryDim * (this.units + this.hiddenUnits)), [this.memoryDim, this.units + this.hiddenUnits]);
    const dWm = Matrix.fromFlat(new Float32Array(this.memoryDim * this.hiddenUnits), [this.memoryDim, this.hiddenUnits]);
    const dWg = Matrix.fromFlat(new Float32Array(this.memoryDim * (this.units + this.hiddenUnits + this.memoryDim)), [this.memoryDim, this.units + this.hiddenUnits + this.memoryDim]);
    const dBg = Matrix.fromFlat(new Float32Array(this.memoryDim), [this.memoryDim, 1]);
    dWxh._data.fill(0);
    dWhh._data.fill(0);
    dBh._data.fill(0);
    dx._data.fill(0);

    const nativeOk = this.runNativeBackward(
      this.batchStepCaches,
      this.buildPerSampleExternalErrors(externalError, seqLen, batchSize),
      dWxh,
      dWhh,
      dBh,
      dWq,
      dWm,
      dWg,
      dBg,
      dx._data,
      totalCols,
      batchSize
    );

    if (!nativeOk) {
      for (let sample = 0; sample < batchSize; sample++) {
        const sampleError = new Array<Float32Array>(seqLen);
        for (let t = 0; t < seqLen; t++) {
          const stepError = new Float32Array(this.hiddenUnits);
          const source = externalError[t];
          for (let i = 0; i < this.hiddenUnits; i++) {
            stepError[i] = source[i * batchSize + sample];
          }
          sampleError[t] = stepError;
        }
        this.backwardThroughStepCaches(
          this.batchStepCaches[sample],
          sampleError,
          dWxh,
          dWhh,
          dBh,
          dWq,
          dWm,
          dWg,
          dBg,
          dx._data,
          totalCols,
          sample,
          batchSize
        );
      }
    }

    this.clipGradientsIfNeeded(dWxh, dWhh, dBh, dWq, dWm, dWg, dBg);
    this.Wxh.subInPlace(this.optimizerWxh.calculate(dWxh, this.alpha));
    this.Whh.subInPlace(this.optimizerWhh.calculate(dWhh, this.alpha));
    this.bh.subInPlace(this.optimizerBh.calculate(dBh, this.alpha));
    this.Wq.subInPlace(this.optimizerWq.calculate(dWq, this.alpha));
    this.Wm.subInPlace(this.optimizerWm.calculate(dWm, this.alpha));
    this.Wg.subInPlace(this.optimizerWg.calculate(dWg, this.alpha));
    this.bg.subInPlace(this.optimizerBg.calculate(dBg, this.alpha));
    return dx;
  }

  resetLoss(): void {
    this.sumLoss = 0;
    this.lossCount = 0;
    this.loss = 0;
  }

  dispose(): void {
    this.combinedInputSequence = [];
    this.rawInputSequence = [];
    this.memoryReadSequence = [];
    this.hiddenSequence = [];
    this.activationGradients = [];
    this.stepCaches = [];
    this.batchStepCaches = [];
    this.batchCombinedInputSequence = [];
    this.batchRawInputSequence = [];
    this.batchHiddenSequence = [];
    this.batchActivationGradients = [];
    this.combinedInputBuffer = new Float32Array(0);
    this.rawInputBuffer = new Float32Array(0);
    this.memoryReadBuffer = new Float32Array(0);
    this.hiddenSequenceBuffer = new Float32Array(0);
    this.activationGradientBuffer = new Float32Array(0);
    this.batchCombinedInputBuffer = new Float32Array(0);
    this.batchRawInputBuffer = new Float32Array(0);
    this.batchHiddenSequenceBuffer = new Float32Array(0);
    this.batchActivationGradientBuffer = new Float32Array(0);
    this.batchErrorStepBuffer = new Float32Array(0);
    this.batchMemoryKeysBuffer = new Float32Array(0);
    this.batchMemoryValuesBuffer = new Float32Array(0);
    this.batchMemoryUsageBuffer = new Float32Array(0);
    this.batchQueryBlockBuffer = new Float32Array(0);
    this.batchReadBlockBuffer = new Float32Array(0);
    this.batchGateBlockBuffer = new Float32Array(0);
    this.batchCandidateBlockBuffer = new Float32Array(0);
    this.batchBestSlots = new Int32Array(0);
    this.errorStepBuffer = new Float32Array(0);
    this.queryInputScratch = new Float32Array(0);
    this.queryScratch = new Float32Array(0);
    this.scoresScratch = new Float32Array(0);
    this.attentionScratch = new Float32Array(0);
    this.readScratch = new Float32Array(0);
    this.gateInputScratch = new Float32Array(0);
    this.gateScratch = new Float32Array(0);
    this.candidateScratch = new Float32Array(0);
    this.batchXSampleScratch = new Float32Array(0);
    this.batchHPrevSampleScratch = new Float32Array(0);
    this.batchCombinedSampleScratch = new Float32Array(0);
    this.batchHSampleScratch = new Float32Array(0);
    this.batchDActSampleScratch = new Float32Array(0);
    this.dWxhBuffer = undefined as any;
    this.dWhhBuffer = undefined as any;
    this.dBhBuffer = undefined as any;
    this.dxBuffer = undefined as any;
    this.batchDxBuffer = undefined as any;
    this.dhNextBuffer = new Float32Array(0);
    this.dhBuffer = new Float32Array(0);
    this.dzBuffer = new Float32Array(0);
    this.dhPrevBuffer = new Float32Array(0);
    this.batchDhNextBuffer = new Float32Array(0);
    this.batchDhBuffer = new Float32Array(0);
    this.batchDzBuffer = new Float32Array(0);
    this.batchDhPrevBuffer = new Float32Array(0);
    this.resultBuffer = undefined as any;
  }

  private computeQueryInto(x_t: Float32Array, hPrev: Float32Array): Float32Array {
    const input = this.queryInputScratch;
    this.concatArrays(input, x_t, hPrev);
    const query = this.queryScratch;
    const inputLength = input.length;
    const wqData = this.Wq._data;
    for (let i = 0; i < this.memoryDim; i++) {
      let sum = 0;
      const offset = i * inputLength;
      for (let j = 0; j < inputLength; j++) sum += wqData[offset + j] * input[j];
      query[i] = sum;
    }
    return query;
  }







  private retrieveMemoryInto(query: Float32Array): number {
    const scores = this.scoresScratch;
    const keys = this.memoryKeys._data;
    const values = this.memoryValues._data;
    const memoryDim = this.memoryDim;
    const memorySlots = this.memorySlots;
    let bestSlot = 0;
    let bestScore = -Infinity;
    const scoreScale = 1 / Math.sqrt(memoryDim);
    for (let slot = 0; slot < memorySlots; slot++) {
      let score = 0;
      for (let i = 0; i < memoryDim; i++) {
        score += query[i] * keys[i * memorySlots + slot];
      }
      score *= scoreScale;
      scores[slot] = score;
      if (score > bestScore) {
        bestScore = score;
        bestSlot = slot;
      }
    }

    const attention = this.stableSoftmaxInto(scores);
    const read = this.readScratch;
    for (let i = 0; i < memoryDim; i++) {
      let sum = 0;
      const offset = i * memorySlots;
      for (let slot = 0; slot < memorySlots; slot++) {
        sum += values[offset + slot] * attention[slot];
      }
      read[i] = sum;
    }

    return bestSlot;
  }

  private retrieveMemoryFromBuffersInto(
    query: Float32Array,
    keys: Float32Array,
    values: Float32Array,
    memoryOffset: number
  ): number {
    const scores = this.scoresScratch;
    const memoryDim = this.memoryDim;
    const memorySlots = this.memorySlots;
    let bestSlot = 0;
    let bestScore = -Infinity;
    const scoreScale = 1 / Math.sqrt(memoryDim);
    for (let slot = 0; slot < memorySlots; slot++) {
      let score = 0;
      for (let i = 0; i < memoryDim; i++) {
        score += query[i] * keys[memoryOffset + i * memorySlots + slot];
      }
      score *= scoreScale;
      scores[slot] = score;
      if (score > bestScore) {
        bestScore = score;
        bestSlot = slot;
      }
    }

    const attention = this.stableSoftmaxInto(scores);
    const read = this.readScratch;
    for (let i = 0; i < memoryDim; i++) {
      let sum = 0;
      const offset = memoryOffset + i * memorySlots;
      for (let slot = 0; slot < memorySlots; slot++) sum += values[offset + slot] * attention[slot];
      read[i] = sum;
    }
    return bestSlot;
  }

  private selectWriteSlot(retrievedSlot: number): number {
    for (let slot = 0; slot < this.memorySlots; slot++) {
      if (this.memoryUsage[slot] === 0) return slot;
    }

    let bestSlot = retrievedSlot;
    let lowestUsage = this.memoryUsage[retrievedSlot] ?? Infinity;
    for (let slot = 0; slot < this.memorySlots; slot++) {
      const usage = this.memoryUsage[slot];
      if (usage < lowestUsage) {
        lowestUsage = usage;
        bestSlot = slot;
      }
    }
    return bestSlot;
  }

  private selectWriteSlotFromUsage(retrievedSlot: number, usage: Float32Array, usageOffset: number): number {
    for (let slot = 0; slot < this.memorySlots; slot++) {
      if (usage[usageOffset + slot] === 0) return slot;
    }

    let bestSlot = retrievedSlot;
    let lowestUsage = usage[usageOffset + retrievedSlot] ?? Infinity;
    for (let slot = 0; slot < this.memorySlots; slot++) {
      const currentUsage = usage[usageOffset + slot];
      if (currentUsage < lowestUsage) {
        lowestUsage = currentUsage;
        bestSlot = slot;
      }
    }
    return bestSlot;
  }

  private stableSoftmaxInto(scores: Float32Array): Float32Array {
    const attention = this.attentionScratch;
    let maxScore = -Infinity;
    for (let i = 0; i < scores.length; i++) {
      if (scores[i] > maxScore) maxScore = scores[i];
    }
    let denom = 0;
    for (let i = 0; i < scores.length; i++) {
      const value = Math.exp(scores[i] - maxScore);
      attention[i] = value;
      denom += value;
    }
    if (denom === 0 || !Number.isFinite(denom)) {
      attention.fill(1 / scores.length, 0, scores.length);
      return attention;
    }
    for (let i = 0; i < attention.length; i++) attention[i] /= denom;
    return attention;
  }

  private rnnCellForward(combined: Float32Array, hPrev: Float32Array, h_t: Float32Array, dAct: Float32Array): void {
    const combinedUnits = this.units + this.memoryDim;
    const wxhData = this.Wxh._data;
    const whhData = this.Whh._data;
    for (let i = 0; i < this.hiddenUnits; i++) {
      let sum = this.bh._data[i];
      const wxhOffset = i * combinedUnits;
      for (let j = 0; j < combinedUnits; j++) sum += wxhData[wxhOffset + j] * combined[j];
      const whhOffset = i * this.hiddenUnits;
      for (let j = 0; j < this.hiddenUnits; j++) sum += whhData[whhOffset + j] * hPrev[j];
      this.applyActivation(sum, i, h_t, dAct);
    }
  }

  private computeWriteGateInto(x_t: Float32Array, h_t: Float32Array, read: Float32Array): Float32Array {
    const input = this.gateInputScratch;
    let offset = 0;
    input.set(x_t, offset);
    offset += x_t.length;
    input.set(h_t, offset);
    offset += h_t.length;
    input.set(read, offset);

    const gate = this.gateScratch;
    const inputLength = input.length;
    const wgData = this.Wg._data;
    const bgData = this.bg._data;
    for (let i = 0; i < this.memoryDim; i++) {
      let sum = bgData[i];
      const rowOffset = i * inputLength;
      for (let j = 0; j < inputLength; j++) sum += wgData[rowOffset + j] * input[j];
      gate[i] = this.sigmoid(sum);
    }
    return gate;
  }

  private projectCandidateMemoryInto(h_t: Float32Array): Float32Array {
    const candidate = this.candidateScratch;
    const wmData = this.Wm._data;
    const hiddenUnits = this.hiddenUnits;
    for (let i = 0; i < this.memoryDim; i++) {
      let sum = 0;
      const offset = i * hiddenUnits;
      for (let j = 0; j < hiddenUnits; j++) sum += wmData[offset + j] * h_t[j];
      candidate[i] = sum;
    }
    return candidate;
  }

  private updateMemory(slot: number, query: Float32Array, candidate: Float32Array, gate: Float32Array): void {
    for (let i = 0; i < this.memoryDim; i++) {
      const idx = i * this.memorySlots + slot;
      const g = gate[i];
      this.memoryKeys._data[idx] = (1 - g) * this.memoryKeys._data[idx] + g * query[i];
      this.memoryValues._data[idx] = (1 - g) * this.memoryValues._data[idx] + g * candidate[i];
    }
    this.memoryUsage[slot] += 1;
  }

  private updateMemoryBuffers(
    slot: number,
    query: Float32Array,
    candidate: Float32Array,
    gate: Float32Array,
    keys: Float32Array,
    values: Float32Array,
    usage: Float32Array,
    memoryOffset: number,
    usageOffset: number,
    batchIndex: number = 0,
    batchSize: number = 1
  ): void {
    for (let i = 0; i < this.memoryDim; i++) {
      const idx = memoryOffset + i * this.memorySlots + slot;
      const sourceIdx = i * batchSize + batchIndex;
      const g = gate[sourceIdx];
      keys[idx] = (1 - g) * keys[idx] + g * query[sourceIdx];
      values[idx] = (1 - g) * values[idx] + g * candidate[sourceIdx];
    }
    usage[usageOffset + slot] += 1;
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
        `AdaptiveMemoryRNN.backward: error shape mismatch, expected [${this.hiddenUnits},${outCols}], got [${effectiveErr._shape[0]},${effectiveErr._shape[1]}]`
      );
    }

    this.ensureErrorStepBuffers(seqLen);
    const perStep = this.buildStepViews(this.errorStepBuffer, seqLen, this.hiddenUnits);
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
        `AdaptiveMemoryRNN.backwardBatch: error shape mismatch, expected [${this.hiddenUnits},${expectedCols}], got [${effectiveErr._shape[0]},${effectiveErr._shape[1]}]`
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

  private applyActivation(sum: number, i: number, h_t: Float32Array, dAct: Float32Array): void {
    if (this.activation === "relu") {
      if (sum > 0) {
        h_t[i] = sum;
        dAct[i] = 1;
      } else {
        h_t[i] = 0;
        dAct[i] = 0;
      }
      return;
    }
    const tv = Math.tanh(sum);
    h_t[i] = tv;
    dAct[i] = 1 - tv * tv;
  }

  private sigmoid(value: number): number {
    if (value >= 0) {
      const z = Math.exp(-value);
      return 1 / (1 + z);
    }
    const z = Math.exp(value);
    return z / (1 + z);
  }

  private backwardThroughStepCaches(
    stepCaches: AdaptiveMemoryStepCache[],
    externalError: Float32Array[],
    dWxh: Matrix,
    dWhh: Matrix,
    dBh: Matrix,
    dWq: Matrix,
    dWm: Matrix,
    dWg: Matrix,
    dBg: Matrix,
    dxData: Float32Array,
    dxCols: number,
    batchIndex: number,
    batchSize: number
  ): void {
    const combinedUnits = this.units + this.memoryDim;
    const memoryStateSize = this.memoryDim * this.memorySlots;
    const scoreScale = 1 / Math.sqrt(this.memoryDim);
    let dhNext = new Float32Array(this.hiddenUnits);
    let dMemoryKeysNext = new Float32Array(memoryStateSize);
    let dMemoryValuesNext = new Float32Array(memoryStateSize);

    for (let t = stepCaches.length - 1; t >= 0; t--) {
      const step = stepCaches[t];
      const dQuery = new Float32Array(this.memoryDim);
      const dCandidate = new Float32Array(this.memoryDim);
      const dGate = new Float32Array(this.memoryDim);
      const dRead = new Float32Array(this.memoryDim);
      const dMemoryKeysBefore = dMemoryKeysNext.slice();
      const dMemoryValuesBefore = dMemoryValuesNext.slice();
      const writeSlot = step.writeSlot;

      for (let i = 0; i < this.memoryDim; i++) {
        const idx = i * this.memorySlots + writeSlot;
        const gate = step.gate[i];
        const oldKey = step.memoryKeysBefore[idx];
        const oldValue = step.memoryValuesBefore[idx];
        const dKeyAfter = dMemoryKeysNext[idx];
        const dValueAfter = dMemoryValuesNext[idx];
        dMemoryKeysBefore[idx] = dKeyAfter * (1 - gate);
        dMemoryValuesBefore[idx] = dValueAfter * (1 - gate);
        dQuery[i] += dKeyAfter * gate;
        dCandidate[i] += dValueAfter * gate;
        dGate[i] += dKeyAfter * (step.query[i] - oldKey) + dValueAfter * (step.candidate[i] - oldValue);
      }

      const dh = externalError[t].slice();
      for (let i = 0; i < this.hiddenUnits; i++) dh[i] += dhNext[i];

      this.outerAccumulate(dWm._data, this.memoryDim, this.hiddenUnits, dCandidate, step.h);
      for (let j = 0; j < this.hiddenUnits; j++) {
        let sum = 0;
        for (let i = 0; i < this.memoryDim; i++) sum += this.Wm._data[i * this.hiddenUnits + j] * dCandidate[i];
        dh[j] += sum;
      }

      const dGatePre = new Float32Array(this.memoryDim);
      for (let i = 0; i < this.memoryDim; i++) {
        dGatePre[i] = dGate[i] * step.gate[i] * (1 - step.gate[i]);
        dBg._data[i] += dGatePre[i];
      }
      this.outerAccumulate(dWg._data, this.memoryDim, this.units + this.hiddenUnits + this.memoryDim, dGatePre, step.gateInput);
      for (let j = 0; j < this.units; j++) {
        let sum = 0;
        for (let i = 0; i < this.memoryDim; i++) sum += this.Wg._data[i * (this.units + this.hiddenUnits + this.memoryDim) + j] * dGatePre[i];
        const col = t * batchSize + batchIndex;
        dxData[j * dxCols + col] += sum;
      }
      for (let j = 0; j < this.hiddenUnits; j++) {
        let sum = 0;
        for (let i = 0; i < this.memoryDim; i++) sum += this.Wg._data[i * (this.units + this.hiddenUnits + this.memoryDim) + this.units + j] * dGatePre[i];
        dh[j] += sum;
      }
      for (let j = 0; j < this.memoryDim; j++) {
        let sum = 0;
        for (let i = 0; i < this.memoryDim; i++) sum += this.Wg._data[i * (this.units + this.hiddenUnits + this.memoryDim) + this.units + this.hiddenUnits + j] * dGatePre[i];
        dRead[j] += sum;
      }

      const dz = new Float32Array(this.hiddenUnits);
      for (let i = 0; i < this.hiddenUnits; i++) {
        dz[i] = dh[i] * step.dAct[i];
        dBh._data[i] += dz[i];
      }
      this.outerAccumulate(dWxh._data, this.hiddenUnits, combinedUnits, dz, step.combined);
      this.outerAccumulate(dWhh._data, this.hiddenUnits, this.hiddenUnits, dz, step.hPrev);

      for (let j = 0; j < this.units; j++) {
        let sum = 0;
        for (let i = 0; i < this.hiddenUnits; i++) sum += this.Wxh._data[i * combinedUnits + j] * dz[i];
        const col = t * batchSize + batchIndex;
        dxData[j * dxCols + col] += sum;
      }
      for (let j = 0; j < this.memoryDim; j++) {
        let sum = 0;
        for (let i = 0; i < this.hiddenUnits; i++) sum += this.Wxh._data[i * combinedUnits + this.units + j] * dz[i];
        dRead[j] += sum;
      }

      const dhPrev = new Float32Array(this.hiddenUnits);
      for (let j = 0; j < this.hiddenUnits; j++) {
        let sum = 0;
        for (let i = 0; i < this.hiddenUnits; i++) sum += this.Whh._data[i * this.hiddenUnits + j] * dz[i];
        dhPrev[j] = sum;
      }

      const dAttention = new Float32Array(this.memorySlots);
      for (let slot = 0; slot < this.memorySlots; slot++) {
        let attnGrad = 0;
        for (let i = 0; i < this.memoryDim; i++) {
          const idx = i * this.memorySlots + slot;
          dMemoryValuesBefore[idx] += dRead[i] * step.attention[slot];
          attnGrad += step.memoryValuesBefore[idx] * dRead[i];
        }
        dAttention[slot] = attnGrad;
      }

      let softmaxInner = 0;
      for (let slot = 0; slot < this.memorySlots; slot++) {
        softmaxInner += dAttention[slot] * step.attention[slot];
      }
      const dScores = new Float32Array(this.memorySlots);
      for (let slot = 0; slot < this.memorySlots; slot++) {
        dScores[slot] = step.attention[slot] * (dAttention[slot] - softmaxInner);
      }
      for (let slot = 0; slot < this.memorySlots; slot++) {
        const scoreGrad = dScores[slot] * scoreScale;
        for (let i = 0; i < this.memoryDim; i++) {
          const idx = i * this.memorySlots + slot;
          dMemoryKeysBefore[idx] += step.query[i] * scoreGrad;
          dQuery[i] += step.memoryKeysBefore[idx] * scoreGrad;
        }
      }

      this.outerAccumulate(dWq._data, this.memoryDim, this.units + this.hiddenUnits, dQuery, step.queryInput);
      for (let j = 0; j < this.units; j++) {
        let sum = 0;
        for (let i = 0; i < this.memoryDim; i++) sum += this.Wq._data[i * (this.units + this.hiddenUnits) + j] * dQuery[i];
        const col = t * batchSize + batchIndex;
        dxData[j * dxCols + col] += sum;
      }
      for (let j = 0; j < this.hiddenUnits; j++) {
        let sum = 0;
        for (let i = 0; i < this.memoryDim; i++) sum += this.Wq._data[i * (this.units + this.hiddenUnits) + this.units + j] * dQuery[i];
        dhPrev[j] += sum;
      }

      dhNext = dhPrev;
      dMemoryKeysNext = dMemoryKeysBefore;
      dMemoryValuesNext = dMemoryValuesBefore;
    }
  }

  private runNativeBackward(
    sampleCaches: AdaptiveMemoryStepCache[][],
    sampleErrors: Float32Array[][],
    dWxh: Matrix,
    dWhh: Matrix,
    dBh: Matrix,
    dWq: Matrix,
    dWm: Matrix,
    dWg: Matrix,
    dBg: Matrix,
    dxData: Float32Array,
    dxCols: number,
    batchSize: number
  ): boolean {
    if (!isNativeAvailable()) return false;
    const seqLen = sampleCaches[0]?.length ?? 0;
    if (seqLen === 0) return false;
    const flattened = this.flattenStepCachesForNative(sampleCaches, sampleErrors, seqLen, batchSize);
    return adaptiveMemoryRnnBackwardNative(
      this.Wxh._data,
      this.Whh._data,
      this.Wq._data,
      this.Wm._data,
      this.Wg._data,
      flattened.hPrev,
      flattened.h,
      flattened.dAct,
      flattened.combined,
      flattened.read,
      flattened.queryInput,
      flattened.query,
      flattened.attention,
      flattened.gateInput,
      flattened.gate,
      flattened.candidate,
      flattened.memoryKeysBefore,
      flattened.memoryValuesBefore,
      flattened.writeSlots,
      flattened.errH,
      this.hiddenUnits,
      this.units,
      this.memoryDim,
      this.memorySlots,
      seqLen,
      batchSize,
      dWxh._data,
      dWhh._data,
      dBh._data,
      dWq._data,
      dWm._data,
      dWg._data,
      dBg._data,
      dxData
    );
  }

  private buildPerSampleExternalErrors(externalError: Float32Array[], seqLen: number, batchSize: number): Float32Array[][] {
    const perSample = new Array<Float32Array[]>(batchSize);
    for (let sample = 0; sample < batchSize; sample++) {
      const sampleSteps = new Array<Float32Array>(seqLen);
      for (let t = 0; t < seqLen; t++) {
        const stepError = new Float32Array(this.hiddenUnits);
        const source = externalError[t];
        for (let i = 0; i < this.hiddenUnits; i++) {
          stepError[i] = source[i * batchSize + sample];
        }
        sampleSteps[t] = stepError;
      }
      perSample[sample] = sampleSteps;
    }
    return perSample;
  }

  private flattenStepCachesForNative(
    sampleCaches: AdaptiveMemoryStepCache[][],
    sampleErrors: Float32Array[][],
    seqLen: number,
    batchSize: number
  ): {
    hPrev: Float32Array;
    h: Float32Array;
    dAct: Float32Array;
    combined: Float32Array;
    read: Float32Array;
    queryInput: Float32Array;
    query: Float32Array;
    attention: Float32Array;
    gateInput: Float32Array;
    gate: Float32Array;
    candidate: Float32Array;
    memoryKeysBefore: Float32Array;
    memoryValuesBefore: Float32Array;
    writeSlots: Int32Array;
    errH: Float32Array;
  } {
    const totalSteps = seqLen * batchSize;
    const hPrev = new Float32Array(totalSteps * this.hiddenUnits);
    const h = new Float32Array(totalSteps * this.hiddenUnits);
    const dAct = new Float32Array(totalSteps * this.hiddenUnits);
    const combined = new Float32Array(totalSteps * (this.units + this.memoryDim));
    const read = new Float32Array(totalSteps * this.memoryDim);
    const queryInput = new Float32Array(totalSteps * (this.units + this.hiddenUnits));
    const query = new Float32Array(totalSteps * this.memoryDim);
    const attention = new Float32Array(totalSteps * this.memorySlots);
    const gateInput = new Float32Array(totalSteps * (this.units + this.hiddenUnits + this.memoryDim));
    const gate = new Float32Array(totalSteps * this.memoryDim);
    const candidate = new Float32Array(totalSteps * this.memoryDim);
    const memoryKeysBefore = new Float32Array(totalSteps * this.memoryDim * this.memorySlots);
    const memoryValuesBefore = new Float32Array(totalSteps * this.memoryDim * this.memorySlots);
    const writeSlots = new Int32Array(totalSteps);
    const errH = new Float32Array(totalSteps * this.hiddenUnits);

    for (let sample = 0; sample < batchSize; sample++) {
      for (let t = 0; t < seqLen; t++) {
        const stepIndex = sample * seqLen + t;
        const cache = sampleCaches[sample][t];
        hPrev.set(cache.hPrev, stepIndex * this.hiddenUnits);
        h.set(cache.h, stepIndex * this.hiddenUnits);
        dAct.set(cache.dAct, stepIndex * this.hiddenUnits);
        combined.set(cache.combined, stepIndex * (this.units + this.memoryDim));
        read.set(cache.read, stepIndex * this.memoryDim);
        queryInput.set(cache.queryInput, stepIndex * (this.units + this.hiddenUnits));
        query.set(cache.query, stepIndex * this.memoryDim);
        attention.set(cache.attention, stepIndex * this.memorySlots);
        gateInput.set(cache.gateInput, stepIndex * (this.units + this.hiddenUnits + this.memoryDim));
        gate.set(cache.gate, stepIndex * this.memoryDim);
        candidate.set(cache.candidate, stepIndex * this.memoryDim);
        memoryKeysBefore.set(cache.memoryKeysBefore, stepIndex * this.memoryDim * this.memorySlots);
        memoryValuesBefore.set(cache.memoryValuesBefore, stepIndex * this.memoryDim * this.memorySlots);
        writeSlots[stepIndex] = cache.writeSlot;
        errH.set(sampleErrors[sample][t], stepIndex * this.hiddenUnits);
      }
    }

    return {
      hPrev,
      h,
      dAct,
      combined,
      read,
      queryInput,
      query,
      attention,
      gateInput,
      gate,
      candidate,
      memoryKeysBefore,
      memoryValuesBefore,
      writeSlots,
      errH,
    };
  }

  private clipGradientsIfNeeded(dWxh: Matrix, dWhh: Matrix, dBh: Matrix, dWq: Matrix, dWm: Matrix, dWg: Matrix, dBg: Matrix): void {
    if (this.clipGradient === false) return;
    const limit = typeof this.clipGradient === "number" ? this.clipGradient : 5.0;
    mj.clipGradients(dWxh, limit);
    mj.clipGradients(dWhh, limit);
    mj.clipGradients(dBh, limit);
    mj.clipGradients(dWq, limit);
    mj.clipGradients(dWm, limit);
    mj.clipGradients(dWg, limit);
    mj.clipGradients(dBg, limit);
  }

  private assertBatchInputSupported(x: Matrix, batchSize: number): void {
    if (!Number.isInteger(batchSize) || batchSize < 1) {
      throw new Error("AdaptiveMemoryRNN.forwardBatch: batchSize must be an integer >= 1.");
    }
    if (this.returnState) {
      throw new Error("AdaptiveMemoryRNN.forwardBatch: returnState=true is not supported yet.");
    }
    if (x._shape[0] !== this.units) {
      throw new Error(`AdaptiveMemoryRNN.forwardBatch: expected input rows ${this.units}, got ${x._shape[0]}`);
    }
    this.assertBatchInputSupportedShape(batchSize, x._shape[1]);
    if (this.stateful && batchSize !== 1) {
      throw new Error("AdaptiveMemoryRNN.forwardBatch: stateful=true only supports batchSize=1.");
    }
  }

  private assertBatchInputSupportedShape(batchSize: number, totalCols: number): void {
    if (totalCols < 1 || totalCols % batchSize !== 0) {
      throw new Error(
        `AdaptiveMemoryRNN batched path expects time-major columns divisible by batchSize. Got cols=${totalCols}, batchSize=${batchSize}.`
      );
    }
  }

  private ensureSequenceStateBuffers(seqLen: number): void {
    const combinedWidth = this.units + this.memoryDim;
    this.combinedInputBuffer = this.ensureBuffer(this.combinedInputBuffer, seqLen * combinedWidth);
    this.rawInputBuffer = this.ensureBuffer(this.rawInputBuffer, seqLen * this.units);
    this.memoryReadBuffer = this.ensureBuffer(this.memoryReadBuffer, seqLen * this.memoryDim);
    this.hiddenSequenceBuffer = this.ensureBuffer(this.hiddenSequenceBuffer, (seqLen + 1) * this.hiddenUnits);
    this.activationGradientBuffer = this.ensureBuffer(this.activationGradientBuffer, seqLen * this.hiddenUnits);
    this.ensureScratchBuffers();

    this.combinedInputSequence = this.buildStepViews(this.combinedInputBuffer, seqLen, combinedWidth);
    this.rawInputSequence = this.buildStepViews(this.rawInputBuffer, seqLen, this.units);
    this.memoryReadSequence = this.buildStepViews(this.memoryReadBuffer, seqLen, this.memoryDim);
    this.hiddenSequence = this.buildStepViews(this.hiddenSequenceBuffer, seqLen + 1, this.hiddenUnits);
    this.activationGradients = this.buildStepViews(this.activationGradientBuffer, seqLen, this.hiddenUnits);

    this.combinedInputBuffer.fill(0, 0, seqLen * combinedWidth);
    this.rawInputBuffer.fill(0, 0, seqLen * this.units);
    this.memoryReadBuffer.fill(0, 0, seqLen * this.memoryDim);
    this.hiddenSequenceBuffer.fill(0, 0, (seqLen + 1) * this.hiddenUnits);
    this.activationGradientBuffer.fill(0, 0, seqLen * this.hiddenUnits);
  }

  private ensureBatchSequenceStateBuffers(seqLen: number, batchSize: number): void {
    const combinedWidth = (this.units + this.memoryDim) * batchSize;
    const rawWidth = this.units * batchSize;
    const hiddenWidth = this.hiddenUnits * batchSize;

    this.batchCombinedInputBuffer = this.ensureBuffer(this.batchCombinedInputBuffer, seqLen * combinedWidth);
    this.batchRawInputBuffer = this.ensureBuffer(this.batchRawInputBuffer, seqLen * rawWidth);
    this.batchHiddenSequenceBuffer = this.ensureBuffer(this.batchHiddenSequenceBuffer, (seqLen + 1) * hiddenWidth);
    this.batchActivationGradientBuffer = this.ensureBuffer(this.batchActivationGradientBuffer, seqLen * hiddenWidth);

    this.batchCombinedInputSequence = this.buildStepViews(this.batchCombinedInputBuffer, seqLen, combinedWidth);
    this.batchRawInputSequence = this.buildStepViews(this.batchRawInputBuffer, seqLen, rawWidth);
    this.batchHiddenSequence = this.buildStepViews(this.batchHiddenSequenceBuffer, seqLen + 1, hiddenWidth);
    this.batchActivationGradients = this.buildStepViews(this.batchActivationGradientBuffer, seqLen, hiddenWidth);

    this.batchCombinedInputBuffer.fill(0, 0, seqLen * combinedWidth);
    this.batchRawInputBuffer.fill(0, 0, seqLen * rawWidth);
    this.batchHiddenSequenceBuffer.fill(0, 0, (seqLen + 1) * hiddenWidth);
    this.batchActivationGradientBuffer.fill(0, 0, seqLen * hiddenWidth);
  }

  private ensureErrorStepBuffers(seqLen: number): void {
    this.errorStepBuffer = this.ensureBuffer(this.errorStepBuffer, seqLen * this.hiddenUnits);
  }

  private ensureBatchErrorStepBuffers(seqLen: number, batchSize: number): void {
    this.batchErrorStepBuffer = this.ensureBuffer(this.batchErrorStepBuffer, seqLen * this.hiddenUnits * batchSize);
  }

  private ensureBatchMemoryBuffers(batchSize: number): void {
    this.batchMemoryKeysBuffer = this.ensureBuffer(
      this.batchMemoryKeysBuffer,
      batchSize * this.memoryDim * this.memorySlots
    );
    this.batchMemoryValuesBuffer = this.ensureBuffer(
      this.batchMemoryValuesBuffer,
      batchSize * this.memoryDim * this.memorySlots
    );
    this.batchMemoryUsageBuffer = this.ensureBuffer(this.batchMemoryUsageBuffer, batchSize * this.memorySlots);
    this.batchQueryBlockBuffer = this.ensureExactBuffer(this.batchQueryBlockBuffer, this.memoryDim * batchSize);
    this.batchReadBlockBuffer = this.ensureExactBuffer(this.batchReadBlockBuffer, this.memoryDim * batchSize);
    this.batchGateBlockBuffer = this.ensureExactBuffer(this.batchGateBlockBuffer, this.memoryDim * batchSize);
    this.batchCandidateBlockBuffer = this.ensureExactBuffer(this.batchCandidateBlockBuffer, this.memoryDim * batchSize);
    if (this.batchBestSlots.length !== batchSize) this.batchBestSlots = new Int32Array(batchSize);
  }

  private ensureBackwardBuffers(seqLen: number): void {
    const combinedUnits = this.units + this.memoryDim;
    if (this.dWxhBuffer._shape[0] !== this.hiddenUnits || this.dWxhBuffer._shape[1] !== combinedUnits) {
      this.dWxhBuffer = mj.zeros([this.hiddenUnits, combinedUnits]);
    }
    if (this.dWhhBuffer._shape[0] !== this.hiddenUnits || this.dWhhBuffer._shape[1] !== this.hiddenUnits) {
      this.dWhhBuffer = mj.zeros([this.hiddenUnits, this.hiddenUnits]);
    }
    if (this.dBhBuffer._shape[0] !== this.hiddenUnits || this.dBhBuffer._shape[1] !== 1) {
      this.dBhBuffer = mj.zeros([this.hiddenUnits, 1]);
    }
    if (this.dxBuffer._shape[0] !== this.units || this.dxBuffer._shape[1] !== seqLen) {
      this.dxBuffer = mj.zeros([this.units, seqLen]);
    }
    this.dhNextBuffer = this.ensureExactBuffer(this.dhNextBuffer, this.hiddenUnits);
    this.dhBuffer = this.ensureExactBuffer(this.dhBuffer, this.hiddenUnits);
    this.dzBuffer = this.ensureExactBuffer(this.dzBuffer, this.hiddenUnits);
    this.dhPrevBuffer = this.ensureExactBuffer(this.dhPrevBuffer, this.hiddenUnits);
  }

  private ensureScratchBuffers(): void {
    const queryInputSize = this.units + this.hiddenUnits;
    const gateInputSize = this.units + this.hiddenUnits + this.memoryDim;
    this.queryInputScratch = this.ensureExactBuffer(this.queryInputScratch, queryInputSize);
    this.queryScratch = this.ensureExactBuffer(this.queryScratch, this.memoryDim);
    this.scoresScratch = this.ensureExactBuffer(this.scoresScratch, this.memorySlots);
    this.attentionScratch = this.ensureExactBuffer(this.attentionScratch, this.memorySlots);
    this.readScratch = this.ensureExactBuffer(this.readScratch, this.memoryDim);
    this.gateInputScratch = this.ensureExactBuffer(this.gateInputScratch, gateInputSize);
    this.gateScratch = this.ensureExactBuffer(this.gateScratch, this.memoryDim);
    this.candidateScratch = this.ensureExactBuffer(this.candidateScratch, this.memoryDim);
    this.batchXSampleScratch = this.ensureExactBuffer(this.batchXSampleScratch, this.units);
    this.batchHPrevSampleScratch = this.ensureExactBuffer(this.batchHPrevSampleScratch, this.hiddenUnits);
    this.batchCombinedSampleScratch = this.ensureExactBuffer(this.batchCombinedSampleScratch, this.units + this.memoryDim);
    this.batchHSampleScratch = this.ensureExactBuffer(this.batchHSampleScratch, this.hiddenUnits);
    this.batchDActSampleScratch = this.ensureExactBuffer(this.batchDActSampleScratch, this.hiddenUnits);
  }

  private ensureBatchBackwardBuffers(seqLen: number, batchSize: number): void {
    this.ensureBackwardBuffers(seqLen);
    const totalCols = seqLen * batchSize;
    if (this.batchDxBuffer._shape[0] !== this.units || this.batchDxBuffer._shape[1] !== totalCols) {
      this.batchDxBuffer = mj.zeros([this.units, totalCols]);
    }
    const hiddenWidth = this.hiddenUnits * batchSize;
    this.batchDhNextBuffer = this.ensureExactBuffer(this.batchDhNextBuffer, hiddenWidth);
    this.batchDhBuffer = this.ensureExactBuffer(this.batchDhBuffer, hiddenWidth);
    this.batchDzBuffer = this.ensureExactBuffer(this.batchDzBuffer, hiddenWidth);
    this.batchDhPrevBuffer = this.ensureExactBuffer(this.batchDhPrevBuffer, hiddenWidth);
  }

  private ensureBuffer(buffer: Float32Array<ArrayBufferLike>, size: number): Float32Array<ArrayBufferLike> {
    if (buffer.length >= size) return buffer;
    return new Float32Array(Math.max(size, Math.max(1, buffer.length * 2)));
  }

  private ensureExactBuffer(buffer: Float32Array<ArrayBufferLike>, size: number): Float32Array<ArrayBufferLike> {
    if (buffer.length === size) return buffer;
    return new Float32Array(size);
  }

  private buildStepViews(buffer: Float32Array<ArrayBufferLike>, steps: number, width: number): Float32Array<ArrayBufferLike>[] {
    const views = new Array<Float32Array<ArrayBufferLike>>(steps);
    for (let step = 0; step < steps; step++) {
      const start = step * width;
      views[step] = new Float32Array(buffer.buffer, buffer.byteOffset + start * 4, width);
    }
    return views;
  }

  private concatArrays(target: Float32Array, a: Float32Array, b: Float32Array): void {
    target.set(a, 0);
    target.set(b, a.length);
  }

  private copyColumnToArray(source: Matrix, col: number, target: Float32Array): void {
    const [rows, cols] = source._shape;
    for (let row = 0; row < rows; row++) target[row] = source._data[row * cols + col];
  }

  private copyColumnBlockToArray(source: Matrix, startCol: number, blockCols: number, target: Float32Array): void {
    const [rows, cols] = source._shape;
    for (let row = 0; row < rows; row++) {
      const srcOffset = row * cols + startCol;
      target.set(source._data.subarray(srcOffset, srcOffset + blockCols), row * blockCols);
    }
  }

  private writeColumnBlock(target: Matrix, startCol: number, blockCols: number, data: Float32Array): void {
    const [rows, cols] = target._shape;
    for (let row = 0; row < rows; row++) {
      const srcOffset = row * blockCols;
      target._data.set(data.subarray(srcOffset, srcOffset + blockCols), row * cols + startCol);
    }
  }

  private copyBatchSampleToScratch(
    source: Float32Array,
    rows: number,
    batchSize: number,
    batchIndex: number,
    target: Float32Array
  ): Float32Array {
    for (let row = 0; row < rows; row++) target[row] = source[row * batchSize + batchIndex];
    return target;
  }

  private copyScratchToBatchSample(
    target: Float32Array,
    rows: number,
    batchSize: number,
    batchIndex: number,
    source: Float32Array
  ): void {
    for (let row = 0; row < rows; row++) target[row * batchSize + batchIndex] = source[row];
  }

  private writeBatchSampleFromScratch(
    target: Float32Array,
    rows: number,
    batchSize: number,
    batchIndex: number,
    input: Float32Array,
    read: Float32Array,
    combinedScratch: Float32Array
  ): void {
    for (let i = 0; i < input.length; i++) {
      combinedScratch[i] = input[i];
      target[i * batchSize + batchIndex] = input[i];
    }
    for (let i = 0; i < read.length; i++) {
      const row = input.length + i;
      combinedScratch[row] = read[i];
      target[row * batchSize + batchIndex] = read[i];
    }
    if (rows !== combinedScratch.length) {
      throw new Error("AdaptiveMemoryRNN.forwardBatch: combined scratch shape mismatch.");
    }
  }

  private setColumnData(target: Float32Array, targetCols: number, col: number, data: Float32Array): void {
    for (let i = 0; i < data.length; i++) target[i * targetCols + col] = data[i];
  }

  private outerAccumulate(
    target: Float32Array,
    outRows: number,
    outCols: number,
    a: Float32Array,
    b: Float32Array
  ): void {
    for (let i = 0; i < outRows; i++) {
      const ai = a[i];
      const offset = i * outCols;
      for (let j = 0; j < outCols; j++) target[offset + j] += ai * b[j];
    }
  }

  private loadMatrix(name: "Wxh" | "Whh" | "bh" | "Wq" | "Wm" | "Wg" | "bg" | "memoryKeys" | "memoryValues", value: number[][]): void {
    if (!value) throw new Error(`AdaptiveMemoryRNN.load: expected '${name}' in serialized data.`);
    this[name]._value = value;
    this[name]._shape = [value.length, value[0]?.length ?? 0];
  }

  private resetOptimizers(optimizer: Optimizer): void {
    this.optimizerWxh = setOptimizer(optimizer, this.Wxh._shape, 1e-5);
    this.optimizerWhh = setOptimizer(optimizer, this.Whh._shape, 1e-5);
    this.optimizerBh = setOptimizer(optimizer, this.bh._shape, 1e-5);
    this.optimizerWq = setOptimizer(optimizer, this.Wq._shape, 1e-5);
    this.optimizerWm = setOptimizer(optimizer, this.Wm._shape, 1e-5);
    this.optimizerWg = setOptimizer(optimizer, this.Wg._shape, 1e-5);
    this.optimizerBg = setOptimizer(optimizer, this.bg._shape, 1e-5);
  }

  private computeParams(): number {
    return (
      this.hiddenUnits * (this.units + this.memoryDim) +
      this.hiddenUnits * this.hiddenUnits +
      this.hiddenUnits +
      this.memoryDim * (this.units + this.hiddenUnits) +
      this.memoryDim * this.hiddenUnits +
      this.memoryDim * (this.units + this.hiddenUnits + this.memoryDim) +
      this.memoryDim
    );
  }

  private assertPositiveInteger(value: number, name: string): void {
    if (!Number.isInteger(value) || value < 1) {
      throw new Error(`AdaptiveMemoryRNN: ${name} must be an integer >= 1.`);
    }
  }
}
