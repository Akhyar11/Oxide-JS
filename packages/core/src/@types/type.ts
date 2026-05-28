import Matrix from "../matrix/index.js";
import AdaGrad from "../optimizer/adaGrad.js";
import Adam from "../optimizer/adam.js";
import Momentum from "../optimizer/momentum.js";
import NAG from "../optimizer/nag.js";
import SGD from "../optimizer/sgd.js";
export type { FitConfig, FitResult } from "./fitConfig.js";

export type vector = number[];
export type matrix2d = number[][];
export type matrix3d = number[][][];
export type MatrixCollection = Matrix | number;
export type MatrixShape = [number, number];
export type MatrixFlatData = Float32Array | Float64Array;
export { Matrix };
export type ActivationType = "sigmoid" | "tanh" | "relu" | "lRelu" | "linear" | "softmax" | "elu" | "gelu" | "hardsigmoid" | "hardswish" | "mish" | "selu" | "softplus" | "softsign" | "swish" | "threshold";
export type StatusLayer =
  | "input"
  | "output"
  | "norm"
  | "outputReduction"
  | "convOutput"
  | "train"
  | "test";
export type Optimizer = "sgd" | "adaGrad" | "momentum" | "nag" | "adam";
export type OptimizerType = SGD | AdaGrad | NAG | Momentum | Adam;
export type Cost = "mse" | "mae" | "huber" | "logCosh" | "hinge" | "squaredHinge" | "klDivergence" | "poisson" | "crossEntropy" | "binaryCrossEntropy" | "softmaxCrossEntropy";
// Layers type is now in @oxide-js/layers


export type WorkerData = {
  value: number;
  i: number;
  k: number;
};
