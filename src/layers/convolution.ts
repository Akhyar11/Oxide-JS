import {
  ActivationType,
  Cost,
  Optimzier,
  OptimzierType,
  StatusLayer,
  matrix2d,
} from "../@types/type";
import mj from "../math";
import Matrix from "../matrix";
import setActivation from "../utils/setActivation";
import setLoss from "../utils/setLoss";
import setOptimizer from "../utils/setOptimizer";
import { CompileDenseLayers } from "./dense";
import { isNativeAvailable, convBackwardInputNative } from "../math/rust_backend";
import { MemoryManager, WorkspaceConfig } from "../utils/memory";

interface ConvolutionLayers {
  kernelSize: [number, number];
  inputShape: [number, number];
  alpha?: number;
  status?: StatusLayer;
  activation?: ActivationType;
  optimizer?: Optimzier;
  loss?: Cost;
  clipGradient?: number | boolean;
  memoryConfig?: WorkspaceConfig;
}

export default class Convolution {
  name = "convolution layer";
  kernel: Matrix;
  bias: Matrix;
  activationName: ActivationType;
  status: StatusLayer;
  optimizerName: Optimzier;
  lossName: Cost;
  loss: number = 0;
  alpha = 0.1;
  clipGradient: number | boolean = true;
  params: number;
  inputShape: [number, number];
  outputShape: [number, number];
  memoryConfig: WorkspaceConfig;

  private sumLoss: number = 0;
  private index: number = 0;
  private activation: Function;
  private lossFunc: Function;
  private optimizerKernel: OptimzierType;
  private optimizerBias: OptimzierType;

  private input: Matrix = mj.matrix([]);
  private result: Matrix;
  private dResult: Matrix;
  private convBuffer: Matrix;
  private backwardInputBuffer: Matrix;
  private errActivationBuffer: Matrix;
  private errKernelBuffer: Matrix;

  private outputData: any = new Float32Array(0);
  private dResultData: any = new Float32Array(0);
  private convBufferData: any = new Float32Array(0);
  private evalBuffers: Record<string, any> = {};

  private backwardInputData: any = new Float32Array(0);
  private errActivationData: any = new Float32Array(0);
  private errKernelData: any = new Float32Array(0);

  constructor({
    kernelSize,
    inputShape,
    alpha = 0.1,
    status = "input",
    activation = "linear",
    optimizer = "sgd",
    loss = "mse",
    clipGradient = 5.0,
    memoryConfig = {},
  }: ConvolutionLayers) {
    this.kernel = mj.random(kernelSize);
    this.bias = mj.zeros([
      inputShape[0] - kernelSize[0] + 1,
      inputShape[1] - kernelSize[1] + 1,
    ]);
    this.inputShape = inputShape;
    this.alpha = alpha;
    this.outputShape =
      status === "convOutput"
        ? [this.bias._shape[0] * this.bias._shape[1], 1]
        : this.bias._shape;
    this.activationName = activation;
    this.status = status;
    this.optimizerName = optimizer;
    this.lossName = loss;
    this.clipGradient = clipGradient;
    this.activation = setActivation(this.activationName);
    this.lossFunc = setLoss(this.lossName);
    this.memoryConfig = memoryConfig;
    this.optimizerKernel = setOptimizer(this.optimizerName, this.kernel._shape, 1e-5);
    this.optimizerBias = setOptimizer(
      this.optimizerName,
      this.bias._shape,
      1e-5
    );
    this.params =
      this.kernel._shape[0] * this.kernel._shape[1] +
      this.bias._shape[0] * this.bias._shape[1];

    this.result = mj.matrix([]);
    this.dResult = mj.matrix([]);
    this.convBuffer = mj.matrix([]);
    this.backwardInputBuffer = mj.matrix([]);
    this.errActivationBuffer = mj.matrix([]);
    this.errKernelBuffer = mj.matrix([]);
  }

  save() {
    const data = {
      name: this.name,
      status: this.status,
      kernelSize: this.kernel._shape,
      inputShape: this.inputShape,
      outputShape: this.outputShape,
      activation: this.activationName,
      optimizer: this.optimizerName,
      loss: this.lossName,
      kernel: this.kernel._value,
      bias: this.bias._value,
      clipGradient: this.clipGradient,
    };
    return data;
  }

  load(kernel: matrix2d, bias: matrix2d, clipGradient?: number | boolean): void {
    this.kernel._value = kernel;
    this.kernel._shape = [kernel.length, kernel[0]?.length ?? 0];
    this.bias._value = bias;
    this.bias._shape = [bias.length, bias[0]?.length ?? 0];
    if (clipGradient !== undefined) this.clipGradient = clipGradient;
  }

  compile({
    alpha = 0.1,
    optimizer = "sgd",
    error = "mse",
    clipGradient,
  }: CompileDenseLayers): void {
    this.alpha = alpha;
    this.optimizerKernel = setOptimizer(optimizer, this.kernel._shape, 1e-5);
    this.optimizerBias = setOptimizer(optimizer, this.bias._shape, 1e-5);
    this.lossFunc = setLoss(error);
    this.optimizerName = optimizer;
    this.lossName = error;
    if (clipGradient !== undefined) this.clipGradient = clipGradient;
  }

  private calculateErrInput(err: Matrix, input: Matrix) {
    if (isNativeAvailable()) {
      const res = convBackwardInputNative(
        err._data,
        err._shape[0],
        err._shape[1],
        input._data,
        input._shape[0],
        input._shape[1],
        this.inputShape[0],
        this.inputShape[1]
      );
      return Matrix.fromFlat(res, this.inputShape);
    }

    const matrix = mj.zeros(this.inputShape);
    const matrixData = matrix._data;
    const errData = err._data;
    const inputData = input._data;
    const errCols = err._shape[1];
    const inputCols = input._shape[1];
    const outCols = matrix._shape[1];
    for (let k = 0; k < err._shape[0]; k++) {
      for (let l = 0; l < err._shape[1]; l++) {
        for (let m = 0; m < input._shape[0]; m++) {
          for (let n = 0; n < input._shape[1]; n++) {
            matrixData[(m + k) * outCols + (n + l)] +=
              errData[k * errCols + l] * inputData[m * inputCols + n];
          }
        }
      }
    }
    return matrix;
  }

  forward(x: Matrix, options?: { workspace?: "train" | "eval" }) {
    this.input = x;
    const outShape = this.bias._shape;
    this.ensureForwardBuffers(outShape[0] * outShape[1], options?.workspace);

    mj.convolution(x, this.kernel, this.convBuffer);
    this.convBuffer.addInPlace(this.bias);

    this.activation(this.convBuffer, { result: this.result, dResult: this.dResult });

    let result = this.result;
    if (this.status === "convOutput") {
      // Flatten is handled by view if possible, but sequential model might expect a new Matrix object.
      // For simplicity, we return the reshaped result matrix.
      // Sequential.ts needs to handle this.
      result = mj.reshape(result, [
        this.bias._shape[0] * this.bias._shape[1],
        1,
      ]);
    }
    return result;
  }

  backward(y: Matrix, err: Matrix) {
    let e = err;
    let loss = 0;
    if (this.status === "convOutput") e = mj.reshape(err, this.bias._shape);
    if (this.status === "output") {
      [loss, e] = this.lossFunc(y, this.result);
      this.index++;
      this.sumLoss += loss;
      this.loss = this.sumLoss / this.index;
    }

    this.ensureBackwardBuffers();
    this.errActivationBuffer.copyFrom(e);
    this.errActivationBuffer.mulInPlace(this.dResult);

    // [New] Gradient Clipping
    if (this.clipGradient !== false) {
      const limit = typeof this.clipGradient === "number" ? this.clipGradient : 5.0;
      mj.clipGradients(this.errActivationBuffer, limit);
    }

    mj.convolution(this.input, this.errActivationBuffer, this.errKernelBuffer);
    
    const optimizerKernel = this.optimizerKernel.calculate(
      this.errKernelBuffer,
      this.alpha
    );
    const optimizerBias = this.optimizerBias.calculate(
      this.errActivationBuffer,
      this.alpha
    );
    const errOutput = this.calculateErrInput(this.errActivationBuffer, this.kernel);
    
    this.kernel.subInPlace(optimizerKernel);
    this.bias.subInPlace(optimizerBias);
    return errOutput;
  }

  /** Reset akumulasi loss — panggil di awal setiap epoch */
  resetLoss(): void {
    this.sumLoss = 0;
    this.index = 0;
    this.loss = 0;
  }

  private ensureForwardBuffers(n: number, workspace: "train" | "eval" = "train"): void {
    if (workspace === "eval") {
      this.evalBuffers.outputData = MemoryManager.ensureCapacity(this.evalBuffers.outputData || new Float32Array(0), n, this.memoryConfig) as any;
      this.evalBuffers.dResultData = MemoryManager.ensureCapacity(this.evalBuffers.dResultData || new Float32Array(0), n, this.memoryConfig) as any;
      this.evalBuffers.convBufferData = MemoryManager.ensureCapacity(this.evalBuffers.convBufferData || new Float32Array(0), n, this.memoryConfig) as any;

      this.outputData = this.evalBuffers.outputData;
      this.dResultData = this.evalBuffers.dResultData;
      this.convBufferData = this.evalBuffers.convBufferData;
    } else {
      this.outputData = MemoryManager.ensureCapacity(this.outputData, n, this.memoryConfig) as any;
      this.dResultData = MemoryManager.ensureCapacity(this.dResultData, n, this.memoryConfig) as any;
      this.convBufferData = MemoryManager.ensureCapacity(this.convBufferData, n, this.memoryConfig) as any;
    }

    this.result = Matrix.fromFlat(this.outputData.subarray(0, n) as any, this.bias._shape);
    this.dResult = Matrix.fromFlat(this.dResultData.subarray(0, n) as any, this.bias._shape);
    this.convBuffer = Matrix.fromFlat(this.convBufferData.subarray(0, n) as any, this.bias._shape);
  }

  private ensureBackwardBuffers(): void {
    const n = this.bias._shape[0] * this.bias._shape[1];
    const kn = this.kernel._shape[0] * this.kernel._shape[1];
    this.errActivationData = MemoryManager.ensureCapacity(this.errActivationData, n, this.memoryConfig) as any;
    this.errKernelData = MemoryManager.ensureCapacity(this.errKernelData, kn, this.memoryConfig) as any;

    this.errActivationBuffer = Matrix.fromFlat(this.errActivationData.subarray(0, n) as any, this.bias._shape);
    this.errKernelBuffer = Matrix.fromFlat(this.errKernelData.subarray(0, kn) as any, this.kernel._shape);
  }

  releaseWorkspace(): void {
    this.evalBuffers = {};
    this.outputData = new Float32Array(0);
    this.dResultData = new Float32Array(0);
    this.convBufferData = new Float32Array(0);
    this.backwardInputData = new Float32Array(0);
    this.errActivationData = new Float32Array(0);
    this.errKernelData = new Float32Array(0);

    this.result = mj.matrix([]);
    this.dResult = mj.matrix([]);
    this.convBuffer = mj.matrix([]);
    this.backwardInputBuffer = mj.matrix([]);
    this.errActivationBuffer = mj.matrix([]);
    this.errKernelBuffer = mj.matrix([]);
  }

  dispose(): void {
    this.releaseWorkspace();
    this.optimizerKernel?.dispose?.();
    this.optimizerBias?.dispose?.();
    (this as any).kernel = null;
    (this as any).bias = null;
    (this as any).optimizerKernel = null;
    (this as any).optimizerBias = null;
  }
}
