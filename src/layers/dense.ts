import mj from "../math";
import { softmaxBackward } from "../activation";
import {
  ActivationType,
  Cost,
  Optimzier,
  OptimzierType,
  StatusLayer,
  matrix2d,
} from "../@types/type";
import setActivation from "../utils/setActivation";
import Matrix from "../matrix";
import setOptimizer from "../utils/setOptimizer";
import { MemoryManager, WorkspaceConfig } from "../utils/memory";
import setLoss from "../utils/setLoss";
import {
  denseLinearBackwardNative,
  isNativeAvailable,
  projectLastTokenLogitsNative,
  reluNative,
  shouldUseNativeDenseLinearBackward,
  sigmoidNative,
  tanhNative
} from "../math/rust_backend";

interface DenseLayers {
  units: number;
  outputUnits: number;
  alpha?: number;
  loss?: Cost;
  activation?: ActivationType;
  optimizer?: Optimzier;
  status?: StatusLayer;
  clipGradient?: number | boolean;
  bias?: boolean;
  memoryConfig?: WorkspaceConfig;
}

export interface CompileDenseLayers {
  alpha?: number;
  optimizer?: Optimzier;
  error?: Cost;
  clipGradient?: number | boolean;
}

export default class Dense {
  name = "dense layer";
  units: number;
  outputUnits: number;
  alpha: number;
  loss: number = 0;
  params: number;
  inputShape: [number, number];
  outputShape: [number, number];
  status: StatusLayer;
  clipGradient: number | boolean;
  bias: Matrix;
  weight: Matrix;
  private sumLoss: number = 0;
  private index: number = 0;
  private optimizerWeight: OptimzierType;
  private optimizerBias: OptimzierType;
  memoryConfig: WorkspaceConfig;

  private zData: any = new Float32Array(0);
  private resultData: any = new Float32Array(0);
  private dInputData: any = new Float32Array(0);
  private errorData: any = new Float32Array(0);
  private prevLayerErrData: any = new Float32Array(0);
  private lastTokenProjectBufferData: any = new Float32Array(0);
  private evalBuffers: Record<string, any> = {};

  private z: Matrix;
  private result: Matrix;
  private dInput: Matrix;
  private err: Matrix;
  private activationName: ActivationType;
  private optimizerName: Optimzier;
  private lossName: Cost;
  private input: Matrix = mj.matrix([]);
  private lossFunc: Function;
  private activation: (a: Matrix, out?: { result: Matrix; dResult: Matrix }) => [Matrix, Matrix];

  // Pre-allocated buffers for speed (REUSE)
  private errWeightBuffer: Matrix;
  private errBiasBuffer: Matrix;
  private prevLayerErrBuffer: Matrix;
  private lastTokenProjectBuffer: Matrix;

  constructor({
    units,
    outputUnits,
    activation = "linear",
    optimizer = "sgd",
    status = "input",
    alpha = 0.1,
    loss = "mse",
    clipGradient = 5.0,
    memoryConfig = {},
  }: DenseLayers) {
    this.memoryConfig = memoryConfig;
    this.z = mj.matrix([]);
    this.result = mj.matrix([]);
    this.err = mj.matrix([]);
    this.dInput = mj.matrix([]);
    this.errWeightBuffer = mj.matrix([]);
    this.errBiasBuffer = mj.matrix([]);
    this.prevLayerErrBuffer = mj.matrix([]);
    this.lastTokenProjectBuffer = mj.matrix([]);
    // Guard: combining softmax activation with softmaxCrossEntropy loss applies softmax twice,
    // which produces incorrect gradients. Users should set activation='linear' when using
    // softmaxCrossEntropy loss.
    if (activation === "softmax" && loss === "softmaxCrossEntropy") {
      throw new Error(
        "Dense: activation='softmax' combined with loss='softmaxCrossEntropy' applies softmax twice. " +
        "Use activation='linear' with loss='softmaxCrossEntropy'."
      );
    }
    this.units = units;
    this.outputUnits = outputUnits;
    this.inputShape = [units, 1];
    this.outputShape = [outputUnits, 1];

    // Gunakan Xavier initialization untuk stabilitas lebih baik
    this.weight = mj.xavier([outputUnits, units]);
    this.bias = mj.zeros([outputUnits, 1]);

    this.z = mj.zeros([outputUnits, 1]); // Buffer for dotProduct + bias
    this.result = mj.zeros([outputUnits, 1]); // Buffer hasil aktivasi
    this.dInput = mj.zeros([outputUnits, 1]); // Buffer grad aktivasi
    this.errWeightBuffer = mj.zeros([outputUnits, units]); // Buffer for errWeight
    this.errBiasBuffer = mj.zeros([outputUnits, 1]);
    this.prevLayerErrBuffer = mj.zeros([units, 1]);
    this.lastTokenProjectBuffer = mj.zeros([outputUnits, 1]);
    this.activation = setActivation(activation);
    this.activationName = activation;
    this.optimizerName = optimizer;
    this.lossName = loss;
    this.status = status;
    this.optimizerWeight = setOptimizer(optimizer, this.weight._shape, 1e-5);
    this.optimizerBias = setOptimizer(optimizer, this.bias._shape, 1e-5);
    this.lossFunc = setLoss(loss);
    this.alpha = alpha;
    this.clipGradient = clipGradient;
    this.params = outputUnits * units + outputUnits;
  }

  save() {
    return {
      name: this.name,
      status: this.status,
      units: this.units,
      outputUnits: this.outputUnits,
      activation: this.activationName,
      optimizer: this.optimizerName,
      loss: this.lossName,
      clipGradient: this.clipGradient,
      weight: this.weight._value,
      bias: this.bias._value,
    };
  }

  getLossName(): Cost {
    return this.lossName;
  }

  load(weight: matrix2d, bias: matrix2d, clipGradient?: number | boolean): void {
    this.weight._value = weight;
    this.weight._shape = [weight.length, weight[0]?.length ?? 0];
    this.bias._value = bias;
    this.bias._shape = [bias.length, bias[0]?.length ?? 0];
    this.units = this.weight._shape[1];
    this.outputUnits = this.weight._shape[0];
    this.params = this.outputUnits * this.units + this.outputUnits;
    if (clipGradient !== undefined) {
        this.clipGradient = clipGradient;
    }
    this.z = mj.zeros([this.outputUnits, 1]);
    this.result = mj.zeros([this.outputUnits, 1]);
    this.dInput = mj.zeros([this.outputUnits, 1]);
    this.errWeightBuffer = mj.zeros([this.outputUnits, this.units]);
    this.errBiasBuffer = mj.zeros([this.outputUnits, 1]);
    this.prevLayerErrBuffer = mj.zeros([this.units, 1]);
    this.lastTokenProjectBuffer = mj.zeros([this.outputUnits, 1]);
    this.optimizerWeight = setOptimizer(this.optimizerName, this.weight._shape, 1e-5);
    this.optimizerBias = setOptimizer(this.optimizerName, this.bias._shape, 1e-5);
  }

  compile({
    alpha,
    optimizer,
    error,
    clipGradient,
  }: CompileDenseLayers): void {
    if (alpha !== undefined) this.alpha = alpha;

    if (optimizer !== undefined) {
      this.optimizerWeight = setOptimizer(optimizer, this.weight._shape, 1e-5);
      this.optimizerBias = setOptimizer(optimizer, this.bias._shape, 1e-5);
      this.optimizerName = optimizer;
    }

    if (error !== undefined) {
      this.lossFunc = setLoss(error);
      this.lossName = error;
    }
    if (clipGradient !== undefined) this.clipGradient = clipGradient;
  }

  forward(x: Matrix, options?: { workspace?: "train" | "eval" }): Matrix {
    const [, seqLen] = x._shape;
    this.input = x;

    this.ensureForwardBuffers(seqLen, options?.workspace);

    // 1. MatMul weight * input -> simpan di this.z 
    // [outputUnits, units] * [units, seqLen] -> [outputUnits, seqLen]
    mj.dotProduct(this.weight, this.input, this.z);

    // 2. Tambahkan bias secara broadcast (per kolom) - OPTIMIZED WITH NATIVE
    mj.addBias(this.z, this.bias);

    // 3. Activation
    if (this.activationName === "linear") {
      // Linear activation is an identity. Reuse the pre-activation buffer directly
      // to avoid a full output copy on large projector layers.
      this.result = this.z;
      return this.result;
    }

    if (this.activationName === "relu") {
      if (isNativeAvailable()) {
        reluNative(this.z._data, this.result._data, this.dInput._data);
      } else {
        const zData = this.z._data;
        const outData = this.result._data;
        const gradData = this.dInput._data;
        for (let i = 0; i < zData.length; i++) {
          const v = zData[i];
          if (v > 0) {
            outData[i] = v;
            gradData[i] = 1;
          } else {
            outData[i] = 0;
            gradData[i] = 0;
          }
        }
      }
      return this.result;
    }

    if (this.activationName === "sigmoid") {
      if (isNativeAvailable()) {
        sigmoidNative(this.z._data, this.result._data, this.dInput._data);
      } else {
        const zData = this.z._data;
        const outData = this.result._data;
        const gradData = this.dInput._data;
        for (let i = 0; i < zData.length; i++) {
          const sig = 1 / (1 + Math.exp(-zData[i]));
          outData[i] = sig;
          gradData[i] = sig * (1 - sig);
        }
      }
      return this.result;
    }

    if (this.activationName === "tanh") {
      if (isNativeAvailable()) {
        tanhNative(this.z._data, this.result._data, this.dInput._data);
      } else {
        const zData = this.z._data;
        const outData = this.result._data;
        const gradData = this.dInput._data;
        for (let i = 0; i < zData.length; i++) {
          const tv = Math.tanh(zData[i]);
          outData[i] = tv;
          gradData[i] = 1 - tv * tv;
        }
      }
      return this.result;
    }

    if (this.activationName === "lRelu") {
      const zData = this.z._data;
      const outData = this.result._data;
      const gradData = this.dInput._data;
      for (let i = 0; i < zData.length; i++) {
        const v = zData[i];
        if (v < 0) {
          outData[i] = v * 1e-5;
          gradData[i] = 1e-5;
        } else {
          outData[i] = v;
          gradData[i] = 1;
        }
      }
      return this.result;
    }

    this.activation(this.z, { result: this.result, dResult: this.dInput });
    return this.result;
  }

  backward(y: Matrix, err: Matrix): Matrix {
    const [rows, seqLen] = this.result._shape;
    this.ensureBackwardBuffers(seqLen);

    let e: Matrix = mj.matrix([]);
    let lossValue = 0;
    const hasExternalError = err._data.length > 0;
    if (this.status === "output" && !hasExternalError) {
      // Safety check: Jika target adalah sparse index (1xN) tapi output bukan 1xN, 
      // dan loss function saat ini adalah MSE, maka PASTI akan error shape.
      // Paksa gunakan SoftmaxCrossEntropy untuk kasus klasifikasi sparse.
      const isSparseTarget = y._shape[0] === 1 && this.result._shape[0] > 1;
      const SoftmaxCrossEntropy = require("../cost/softmaxCrossEntropy").default;
      if (isSparseTarget && this.lossName === "mse") {
        [lossValue, e] = SoftmaxCrossEntropy(y, this.result, this.err);
      } else if (this.lossName === "softmaxCrossEntropy") {
        [lossValue, e] = SoftmaxCrossEntropy(y, this.result, this.err);
      } else {
        [lossValue, e] = (this.lossFunc as any)(y, this.result, this.err);
      }
      this.index++;
      this.sumLoss += lossValue;
      this.loss = this.sumLoss / this.index;
    } else {
      e = err;
    }

    let errActivation: Matrix;
    if (this.activationName === "softmax") {
      errActivation = softmaxBackward(this.result, e, false);
    } else if (this.activationName === "linear") {
      errActivation = e;
    } else {
      errActivation = mj.mul(e, this.dInput, this.err);
    }

    if (this.errBiasBuffer._shape[0] !== this.outputUnits) {
      this.errBiasBuffer = mj.zeros([this.outputUnits, 1]);
    }

    const canUseNativeLinearBackward =
      this.activationName === "linear" &&
      isNativeAvailable() &&
      shouldUseNativeDenseLinearBackward(this.outputUnits, this.units, seqLen);

    let gradWeight: Matrix;
    let gradBias: Matrix;
    let prevErr: Matrix;

    if (canUseNativeLinearBackward) {
      denseLinearBackwardNative(
        errActivation._data,
        this.input._data,
        this.weight._data,
        this.outputUnits,
        this.units,
        seqLen,
        this.clipGradient === false ? -1 : (typeof this.clipGradient === "number" ? this.clipGradient : 5.0),
        this.errWeightBuffer._data,
        this.errBiasBuffer._data,
        this.prevLayerErrBuffer._data
      );
      gradWeight = this.errWeightBuffer;
      gradBias = this.errBiasBuffer;
      prevErr = this.prevLayerErrBuffer;
    } else {
      // 1. Hitung gradien weight
      // [outputUnits, seqLen] * [seqLen, units] -> [outputUnits, units]
      gradWeight = mj.dotProduct(errActivation, this.input, this.errWeightBuffer, false, true);

      // 2. Hitung gradien bias (Sum sepanjang sequence/kolom)
      gradBias = mj.sumAxis(errActivation, 1, this.errBiasBuffer);

      if (this.clipGradient !== false) {
        const limit = typeof this.clipGradient === "number" ? this.clipGradient : 5.0;
        mj.clipGradients(gradWeight, limit);
        mj.clipGradients(gradBias, limit);
      }

      // 3. Hitung gradien ke layer sebelumnya dengan bobot sebelum update
      // [units, outputUnits] * [outputUnits, seqLen] -> [units, seqLen]
      prevErr = mj.dotProduct(this.weight, errActivation, this.prevLayerErrBuffer, true, false);
    }

    // 4. Dapatkan update dari optimizer
    const updateWeight = this.optimizerWeight.calculate(gradWeight, this.alpha);
    const updateBias = this.optimizerBias.calculate(gradBias, this.alpha);

    // 5. Update In-Place!
    this.weight.subInPlace(updateWeight);
    this.bias.subInPlace(updateBias);
    return prevErr;
  }

  /** @deprecated Use mj.clipGradients instead */
  private clipGradients(m: Matrix, limit: number) {
    mj.clipGradients(m, limit);
  }

  /**
   * Resize output units (e.g., when vocab size increases)
   * @param newOutputUnits - New number of output units
   */
  resize(newOutputUnits: number): void {
    if (newOutputUnits <= this.outputUnits) return;

    console.log(`[Dense] Resizing output units: ${this.outputUnits} -> ${newOutputUnits}`);

    // 1. Resize weights [newOutputUnits, units]
    const newWeight = mj.random([newOutputUnits, this.units]);
    const oldWeightData = this.weight._data;
    const newWeightData = newWeight._data;

    for (let i = 0; i < this.outputUnits; i++) {
      for (let j = 0; j < this.units; j++) {
        newWeightData[i * this.units + j] = oldWeightData[i * this.units + j];
      }
    }

    // 2. Resize bias [newOutputUnits, 1]
    const newBias = mj.zeros([newOutputUnits, 1]);
    const oldBiasData = this.bias._data;
    const newBiasData = newBias._data;
    for (let i = 0; i < this.outputUnits; i++) {
      newBiasData[i] = oldBiasData[i];
    }

    // 3. Update state
    this.weight = newWeight;
    this.bias = newBias;
    this.outputUnits = newOutputUnits;
    this.outputShape = [newOutputUnits, 1];
    this.params = newOutputUnits * this.units + newOutputUnits;

    // 4. Re-allocate buffers
    this.z = mj.zeros([newOutputUnits, 1]);
    this.result = mj.zeros([newOutputUnits, 1]);
    this.dInput = mj.zeros([newOutputUnits, 1]);
    this.errWeightBuffer = mj.zeros([newOutputUnits, this.units]);

    // 5. Reset optimizer for new shape
    this.optimizerWeight = setOptimizer(this.optimizerName, this.weight._shape, 1e-5);
    this.optimizerBias = setOptimizer(this.optimizerName, this.bias._shape, 1e-5);
  }

  /** Reset akumulasi loss — panggil di awal setiap epoch */
  resetLoss(): void {
    this.sumLoss = 0;
    this.index = 0;
    this.loss = 0;
  }

  getLastOutput(): Matrix {
    return this.result;
  }

  projectLastTokenFromSequence(sequence: Matrix, seqLen: number, batchSize: number, options?: { workspace?: "train" | "eval" }): Matrix {
    if (this.activationName !== "linear") {
      throw new Error("Dense.projectLastTokenFromSequence hanya mendukung activation='linear'.");
    }

    if (options?.workspace === "eval") {
      this.evalBuffers.lastTokenProjectBufferData = MemoryManager.ensureCapacity(this.evalBuffers.lastTokenProjectBufferData || new Float32Array(0), this.outputUnits * batchSize, this.memoryConfig) as any;
      this.lastTokenProjectBuffer = Matrix.fromFlat(this.evalBuffers.lastTokenProjectBufferData.subarray(0, this.outputUnits * batchSize) as any, [this.outputUnits, batchSize]);
    } else {
      this.lastTokenProjectBufferData = MemoryManager.ensureCapacity(this.lastTokenProjectBufferData, this.outputUnits * batchSize, this.memoryConfig) as any;
      this.lastTokenProjectBuffer = Matrix.fromFlat(this.lastTokenProjectBufferData.subarray(0, this.outputUnits * batchSize) as any, [this.outputUnits, batchSize]);
    }

    if (isNativeAvailable()) {
      projectLastTokenLogitsNative(
        sequence._data,
        this.weight._data,
        this.bias._data,
        this.units,
        seqLen,
        batchSize,
        this.outputUnits,
        this.lastTokenProjectBuffer._data
      );
      return this.lastTokenProjectBuffer;
    }

    const sourceData = sequence._data;
    const outData = this.lastTokenProjectBuffer._data;
    const totalCols = sequence._shape[1];
    const weightData = this.weight._data;
    const biasData = this.bias._data;

    for (let outIdx = 0; outIdx < this.outputUnits; outIdx++) {
      const weightOffset = outIdx * this.units;
      for (let b = 0; b < batchSize; b++) {
        const tokenCol = (b + 1) * seqLen - 1;
        let sum = biasData[outIdx];
        for (let unitIdx = 0; unitIdx < this.units; unitIdx++) {
          sum += weightData[weightOffset + unitIdx] * sourceData[unitIdx * totalCols + tokenCol];
        }
        outData[outIdx * batchSize + b] = sum;
      }
    }

    return this.lastTokenProjectBuffer;
  }

  private ensureForwardBuffers(seqLen: number, workspace: "train" | "eval" = "train"): void {
    const required = this.outputUnits * seqLen;
    if (workspace === "eval") {
      this.evalBuffers.zData = MemoryManager.ensureCapacity(this.evalBuffers.zData || new Float32Array(0), required, this.memoryConfig) as any;
      this.evalBuffers.resultData = MemoryManager.ensureCapacity(this.evalBuffers.resultData || new Float32Array(0), required, this.memoryConfig) as any;
      this.evalBuffers.dInputData = MemoryManager.ensureCapacity(this.evalBuffers.dInputData || new Float32Array(0), required, this.memoryConfig) as any;
      this.zData = this.evalBuffers.zData;
      this.resultData = this.evalBuffers.resultData;
      this.dInputData = this.evalBuffers.dInputData;
    } else {
      this.zData = MemoryManager.ensureCapacity(this.zData, required, this.memoryConfig) as any;
      this.resultData = MemoryManager.ensureCapacity(this.resultData, required, this.memoryConfig) as any;
      this.dInputData = MemoryManager.ensureCapacity(this.dInputData, required, this.memoryConfig) as any;
    }

    this.z = Matrix.fromFlat(this.zData.subarray(0, required) as any, [this.outputUnits, seqLen]);
    this.result = Matrix.fromFlat(this.resultData.subarray(0, required) as any, [this.outputUnits, seqLen]);
    this.dInput = Matrix.fromFlat(this.dInputData.subarray(0, required) as any, [this.outputUnits, seqLen]);
  }

  private ensureBackwardBuffers(seqLen: number): void {
    const required = this.outputUnits * seqLen;
    const prevRequired = this.units * seqLen;

    this.errorData = MemoryManager.ensureCapacity(this.errorData, required, this.memoryConfig) as any;
    this.err = Matrix.fromFlat(this.errorData.subarray(0, required) as any, [this.outputUnits, seqLen]);

    this.prevLayerErrData = MemoryManager.ensureCapacity(this.prevLayerErrData, prevRequired, this.memoryConfig) as any;
    this.prevLayerErrBuffer = Matrix.fromFlat(this.prevLayerErrData.subarray(0, prevRequired) as any, [this.units, seqLen]);
  }

  releaseWorkspace(): void {
    this.evalBuffers = {};
    this.zData = new Float32Array(0);
    this.resultData = new Float32Array(0);
    this.errorData = new Float32Array(0);
    this.prevLayerErrData = new Float32Array(0);
    this.z = mj.matrix([]);
    this.result = mj.matrix([]);
    this.err = mj.matrix([]);
  }

  dispose(): void {
    this.releaseWorkspace();
    this.optimizerWeight?.dispose?.();
    this.optimizerBias?.dispose?.();
    (this as any).weight = null;
    (this as any).bias = null;
    (this as any).optimizerWeight = null;
    (this as any).optimizerBias = null;
  }
}
