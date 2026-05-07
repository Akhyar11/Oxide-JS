import {
  ActivationType,
  Cost,
  Optimizer,
  OptimizerType,
  StatusLayer,
  matrix2d,
} from "@oxide-js/core";
import { mj, engine } from "@oxide-js/core";
import { Matrix } from "@oxide-js/core";
import { setActivation } from "@oxide-js/core";
import { setLoss } from "@oxide-js/core";
import { setOptimizer } from "@oxide-js/core";
import { CompileDenseLayers } from "./dense.js";
import { isNativeAvailable, convBackwardInputNative } from "@oxide-js/core";

interface ConvolutionLayers {
  kernelSize: [number, number];
  inputShape: [number, number];
  alpha?: number;
  status?: StatusLayer;
  activation?: ActivationType;
  optimizer?: Optimizer;
  loss?: Cost;
  clipGradient?: number | boolean;
}

export default class Convolution {
  name = "convolution layer";
  kernel: Matrix;
  bias: Matrix;
  activationName: ActivationType;
  status: StatusLayer;
  optimizerName: Optimizer;
  lossName: Cost;
  loss: number = 0;
  alpha = 0.1;
  clipGradient: number | boolean = true;
  params: number;
  inputShape: [number, number];
  outputShape: [number, number];
  private sumLoss: number = 0;
  private index: number = 0;
  private activation: Function;
  private lossFunc: Function;
  private optimizerKernel: OptimizerType;
  private optimizerBias: OptimizerType;
  private input: Matrix = mj.matrix([]);
  private result: Matrix = mj.matrix([]);
  private dResult: Matrix = mj.matrix([]);
  constructor({
    kernelSize,
    inputShape,
    alpha = 0.1,
    status = "input",
    activation = "linear",
    optimizer = "sgd",
    loss = "mse",
    clipGradient = 5.0,
  }: ConvolutionLayers) {
    this.kernel = kernelSize[0] > 0 && kernelSize[1] > 0 ? mj.random(kernelSize) : mj.matrix([]);
    
    const outRows = Math.max(0, inputShape[0] - kernelSize[0] + 1);
    const outCols = Math.max(0, inputShape[1] - kernelSize[1] + 1);
    this.bias = outRows > 0 && outCols > 0 ? mj.zeros([outRows, outCols]) : mj.matrix([]);
    
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
    this.optimizerKernel = setOptimizer(this.optimizerName, kernelSize, 1e-5);
    this.optimizerBias = setOptimizer(
      this.optimizerName,
      [inputShape[0] - kernelSize[0] + 1, inputShape[1] - kernelSize[1] + 1],
      1e-5
    );
    this.params =
      kernelSize[0] * kernelSize[1] +
      this.bias._shape[0] * this.bias._shape[1];
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

  toKerasConfig() {
    return {
      class_name: "Conv2D",
      config: {
        filters: 1, // Oxide-JS implementation currently seems to support 1 filter per layer
        kernel_size: this.kernel._shape,
        strides: [1, 1],
        padding: "valid",
        data_format: "channels_last",
        dilation_rate: [1, 1],
        activation: this.activationName,
        use_bias: true,
        kernel_initializer: { class_name: "VarianceScaling", config: { scale: 1.0, mode: "fan_avg", distribution: "uniform" } },
        bias_initializer: { class_name: "Zeros", config: {} },
        name: `conv2d_${Math.floor(Math.random() * 1000)}`,
        trainable: true,
      }
    };
  }

  getWeightsManifest(): { name: string; shape: [number, number]; data: Float32Array }[] {
    return [
      { name: "kernel", shape: this.kernel._shape, data: this.kernel._data },
      { name: "bias", shape: this.bias._shape, data: this.bias._data },
    ];
  }

  setWeightsFromBinary(weights: Record<string, Float32Array>): void {
    if (weights.kernel) {
      if (this.kernel._data.length === 0 || this.kernel._data.length !== weights.kernel.length) {
        this.kernel._data = new Float32Array(weights.kernel.length);
        // Assuming kernel is square if shape is lost, otherwise it relies on compile/load to fix it
        const dim = Math.sqrt(weights.kernel.length);
        this.kernel._shape = [dim, dim];
      }
      this.kernel._data.set(weights.kernel);
    }
    if (weights.bias) {
      if (this.bias._data.length === 0 || this.bias._data.length !== weights.bias.length) {
        this.bias._data = new Float32Array(weights.bias.length);
        const dim = Math.sqrt(weights.bias.length);
        this.bias._shape = [dim, dim];
      }
      this.bias._data.set(weights.bias);
    }
  }

  load(kernel?: matrix2d, bias?: matrix2d, clipGradient?: number | boolean): void {
    if (kernel) {
      this.kernel._value = kernel;
      this.kernel._shape = [kernel.length, kernel[0]?.length ?? 0];
    }
    if (bias) {
      this.bias._value = bias;
      this.bias._shape = [bias.length, bias[0]?.length ?? 0];
    }
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
    if (error !== undefined) {
      this.lossFunc = setLoss(error);
      this.lossName = error;
    }
    if (clipGradient !== undefined) this.clipGradient = clipGradient;
  }

  getParams(): Matrix[] {
    return [this.kernel, this.bias];
  }

  update(alpha: number): void {
    const a = alpha || this.alpha;
    this.optimizerKernel.apply(this.kernel, a);
    this.optimizerBias.apply(this.bias, a);
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

  forward(x: Matrix) {
    this.input = x;
    const calculateWeightBias = mj.add(
      mj.convolution(x, this.kernel),
      this.bias
    );
    [this.result, this.dResult] = this.activation(calculateWeightBias);
    let result = this.result;
    if (this.status === "convOutput") {
      result = mj.reshape(result, [
        this.bias._shape[0] * this.bias._shape[1],
        1,
      ]);
    }
    return result;
  }

  backward(y: Matrix, err: Matrix, gradOnly = false) {
    const dx = this.calculateGradients(y, err);
    if (!gradOnly) {
      this.update(this.alpha);
    }
    return dx;
  }

  private calculateGradients(y: Matrix, err: Matrix): Matrix {
    let e = err;
    if (this.status === "convOutput") e = mj.reshape(err, this.bias._shape);
    if (this.status === "output") {
      const [lossValue, outputErr] = this.lossFunc(y, this.result);
      this.index++;
      this.sumLoss += lossValue;
      this.loss = this.sumLoss / this.index;
      e = outputErr;
    }

    const errActivation = mj.mul(e, this.dResult);
    
    if (this.clipGradient !== false) {
      const limit = typeof this.clipGradient === "number" ? this.clipGradient : 5.0;
      mj.clipGradients(errActivation, limit);
    }

    const dKernel = mj.convolution(this.input, errActivation);
    const dBias = errActivation;

    // Accumulate gradients
    if (this.kernel.grad) this.kernel.grad.addInPlace(dKernel); else this.kernel.grad = dKernel;
    if (this.bias.grad) this.bias.grad.addInPlace(dBias); else this.bias.grad = dBias;

    return this.calculateErrInput(errActivation, this.kernel);
  }

  /** Reset akumulasi loss — panggil di awal setiap epoch */
  resetLoss(): void {
    this.sumLoss = 0;
    this.index = 0;
    this.loss = 0;
  }
}
