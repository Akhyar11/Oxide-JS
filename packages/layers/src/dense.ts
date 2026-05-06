import { mj } from "@oxide-js/core";
import { softmaxBackward } from "@oxide-js/core";
import {
  ActivationType,
  Cost,
  Optimizer,
  OptimizerType,
  StatusLayer,
  matrix2d,
} from "@oxide-js/core";
import { setActivation } from "@oxide-js/core";
import { Matrix } from "@oxide-js/core";
import { setOptimizer } from "@oxide-js/core";
import { setLoss } from "@oxide-js/core";
import {
  denseLinearBackwardNative,
  isNativeAvailable,
  projectLastTokenLogitsNative,
  reluNative,
  shouldUseNativeDenseLinearBackward,
  sigmoidNative,
  tanhNative
} from "@oxide-js/core";

interface DenseLayers {
  units: number;
  outputUnits: number;
  alpha?: number;
  loss?: Cost;
  activation?: ActivationType;
  optimizer?: Optimizer;
  status?: StatusLayer;
  clipGradient?: number | boolean;
}

export interface CompileDenseLayers {
  alpha?: number;
  optimizer?: Optimizer;
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
  private optimizerWeight: OptimizerType;
  private optimizerBias: OptimizerType;
  private activationName: ActivationType;
  private optimizerName: Optimizer;
  private lossName: Cost;
  private input: Matrix = mj.matrix([]);
  private dInput: Matrix;
  private result: Matrix;
  private lossFunc: Function;
  private activation: (a: Matrix) => [Matrix, Matrix];

  // Pre-allocated buffers for speed (REUSE)
  private z: Matrix;
  private errWeightBuffer: Matrix;
  private errBiasBuffer: Matrix;
  private errActivationBuffer: Matrix;
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
  }: DenseLayers) {
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
    this.errActivationBuffer = mj.zeros([outputUnits, 1]);
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

  getActivationName(): ActivationType {
    return this.activationName;
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
    this.errActivationBuffer = mj.zeros([this.outputUnits, 1]);
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

  forward(x: Matrix): Matrix {
    const [, seqLen] = x._shape;
    this.input = x;

    this.ensureForwardBuffers(seqLen);

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

    const [result, dResult] = this.activation(this.z);
    this.dInput = dResult;
    this.result = result;
    return this.result;
  }

  backward(y: Matrix, err: Matrix): Matrix {
    const [rows, seqLen] = this.result._shape;
    let e: Matrix = mj.matrix([]);
    let lossValue = 0;
    const hasExternalError = err._data.length > 0;
    if (this.status === "output" && !hasExternalError) {
      // Safety check: Jika target adalah sparse index (1xN) tapi output bukan 1xN, 
      // dan loss function saat ini adalah MSE, maka PASTI akan error shape.
      // Paksa gunakan SoftmaxCrossEntropy untuk kasus klasifikasi sparse.
      const isSparseTarget = y._shape[0] === 1 && this.result._shape[0] > 1;
      if (isSparseTarget && this.lossName === "mse") {
        if (this.activationName === "softmax") {
          throw new Error(
            "Sparse multiclass target requires activation='linear' with loss='softmaxCrossEntropy', or use one-hot target with loss='crossEntropy'. Do not use activation='softmax' with implicit softmaxCrossEntropy."
          );
        }
        const SoftmaxCrossEntropy = require("../cost/softmaxCrossEntropy").default;
        [lossValue, e] = SoftmaxCrossEntropy(y, this.result);
      } else {
        [lossValue, e] = this.lossFunc(y, this.result);
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
      this.errActivationBuffer = this.ensureCapacityMatrix(this.errActivationBuffer, "errActivationData", e._shape[0], seqLen);
      errActivation = mj.mul(e, this.dInput, this.errActivationBuffer);
    }

    this.prevLayerErrBuffer = this.ensureCapacityMatrix(this.prevLayerErrBuffer, "prevLayerErrData", this.units, seqLen);
    
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

  projectLastTokenFromSequence(sequence: Matrix, seqLen: number, batchSize: number): Matrix {
    if (this.activationName !== "linear") {
      throw new Error("Dense.projectLastTokenFromSequence hanya mendukung activation='linear'.");
    }

    if (this.lastTokenProjectBuffer._shape[0] !== this.outputUnits || this.lastTokenProjectBuffer._shape[1] !== batchSize) {
      this.lastTokenProjectBuffer = mj.zeros([this.outputUnits, batchSize]);
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

  private ensureForwardBuffers(seqLen: number): void {
    this.z = this.ensureCapacityMatrix(this.z, "zData", this.outputUnits, seqLen);
    this.result = this.ensureCapacityMatrix(this.result, "resultData", this.outputUnits, seqLen);
    this.dInput = this.ensureCapacityMatrix(this.dInput, "dInputData", this.outputUnits, seqLen);
  }

  private ensureCapacityMatrix(matrix: Matrix, prop: string, rows: number, cols: number): Matrix {
    const requiredLen = rows * cols;
    let data = (this as any)[prop] as Float32Array | undefined;
    if (!data || data.length < requiredLen) {
      data = new Float32Array(Math.max(requiredLen, Math.max(1, (data?.length ?? 0) * 2)));
      (this as any)[prop] = data;
    }
    const newMatrix = Matrix.fromFlat(data.subarray(0, requiredLen), [rows, cols]);
    // preserve old values if we are just slicing unless we explicitly want zeros
    return newMatrix;
  }

  dispose() {
    this.z = undefined as any;
    this.result = undefined as any;
    this.dInput = undefined as any;
    this.errWeightBuffer = undefined as any;
    this.errBiasBuffer = undefined as any;
    this.errActivationBuffer = undefined as any;
    this.prevLayerErrBuffer = undefined as any;
    this.lastTokenProjectBuffer = undefined as any;
    
    (this as any).zData = new Float32Array(0);
    (this as any).resultData = new Float32Array(0);
    (this as any).dInputData = new Float32Array(0);
    (this as any).errActivationData = new Float32Array(0);
    (this as any).prevLayerErrData = new Float32Array(0);
  }
}
