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

interface ConvolutionLayers {
  kernelSize: [number, number];
  inputShape: [number, number];
  alpha?: number;
  status?: StatusLayer;
  activation?: ActivationType;
  optimizer?: Optimzier;
  loss?: Cost;
  clipGradient?: number | boolean;
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
  private sumLoss: number = 0;
  private index: number = 0;
  private activation: Function;
  private lossFunc: Function;
  private optimizerKernel: OptimzierType;
  private optimizerBias: OptimzierType;
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

    const errActivation = mj.mul(e, this.dResult);
    
    // [New] Gradient Clipping
    if (this.clipGradient !== false) {
      const limit = typeof this.clipGradient === "number" ? this.clipGradient : 5.0;
      mj.clipGradients(errActivation, limit);
    }

    const errKernel = mj.convolution(this.input, errActivation);
    const optimizerKernel = this.optimizerKernel.calculate(
      errKernel,
      this.alpha
    );
    const optimizerBias = this.optimizerBias.calculate(
      errActivation,
      this.alpha
    );
    const errOutput = this.calculateErrInput(errActivation, this.kernel);
    this.kernel = mj.sub(this.kernel, optimizerKernel);
    this.bias = mj.sub(this.bias, optimizerBias);
    return errOutput;
  }

  /** Reset akumulasi loss — panggil di awal setiap epoch */
  resetLoss(): void {
    this.sumLoss = 0;
    this.index = 0;
    this.loss = 0;
  }
}
