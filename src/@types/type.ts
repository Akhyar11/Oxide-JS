import Activation from "../layers/activation";
import Convolution from "../layers/convolution";
import Dense from "../layers/dense";
import SelfAttantion from "../layers/selfAttantion";
import Embedding from "../layers/embedding";
import Flatten from "../layers/flatten";
import PositionalEncoding from "../layers/positionalEncoding";
import Matrix from "../matrix";
import AdaGrad from "../optimizer/adaGrad";
import Adam from "../optimizer/adam";
import Momentum from "../optimizer/momentum";
import NAG from "../optimizer/nag";
import SGD from "../optimizer/sgd";

export type vector = number[];
export type matrix2d = number[][];
export type matrix3d = number[][][];
export type MatrixCollection = Matrix | number;
export type MatrixShape = [number, number];
export { Matrix };
export type ActivationType = "sigmoid" | "tanh" | "relu" | "lRelu" | "linear" | "softmax";
export type StatusLayer =
  | "input"
  | "output"
  | "norm"
  | "outputReduction"
  | "convOutput";
export type Optimzier = "sgd" | "adaGrad" | "momentum" | "nag" | "adam";
export type OptimzierType = SGD | AdaGrad | NAG | Momentum | Adam;
export type Cost = "mse" | "crossEntropy" | "binaryCrossEntropy" | "softmaxCrossEntropy";
export type Layers = Dense | Activation | Convolution | SelfAttantion | Embedding | Flatten | PositionalEncoding;

export type WorkerData = {
  value: number;
  i: number;
  k: number;
};
