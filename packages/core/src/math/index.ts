import absm from "./absm.js";
import Matrix from "../matrix/index.js";
import add, { addInto } from "./add.js";
import argmax from "./argmax.js";
import concat from "./concat.js";
import div from "./div.js";
import dotDiv, { dotDivScalar } from "./dotDiv.js";
import dotMul, { dotMulScalar } from "./dotMul.js";
import dotProduct from "./dotProduct.js";
import dotSub, { dotSubScalar } from "./dotSub.js";
import dotSum, { dotSumScalar } from "./dotSum.js";
import expm from "./expm.js";
import flatten from "./flatten.js";
import logm from "./logm.js";
import map from "./map.js";
import matrix from "./matrix.js";
import mean from "./mean.js";
import mul from "./mul.js";
import ones from "./ones.js";
import random from "./random.js";
import reshape from "./reshape.js";
import sub, { subInto } from "./sub.js";
import transpose from "./transpose.js";
import zeros from "./zeros.js";
import convolution from "./convolution.js";
import norm, { normScalar } from "./norm.js";
import xavier from "./xavier.js";
import he from "./he.js";
import addBias from "./addBias.js";
import addBiasRow from "./addBiasRow.js";
import sumAxis from "./sumAxis.js";
import clipGradients from "./clipGradients.js";
import pow from "./pow.js";

// === Activation Functions ===
import linear, {
  sigmoid,
  tanh,
  relu,
  lRelu,
  softmax,
  elu,
  selu,
  softplus,
  softsign,
  swish,
  gelu,
  mish,
  hardSigmoid,
  hardSwish,
  threshold,
} from "../activation/index.js";

// === Cost / Loss Functions ===
import MeanSquaredError from "../cost/mse.js";
import CategoricalCrossEntropy, { BinaryCrossEntropy } from "../cost/crossEntropy.js";
import SoftmaxCrossEntropy from "../cost/softmaxCrossEntropy.js";

// === Optimizers ===
import SGD from "../optimizer/sgd.js";
import Adam from "../optimizer/adam.js";
import NAG from "../optimizer/nag.js";
import AdaGrad from "../optimizer/adaGrad.js";
import Momentum from "../optimizer/momentum.js";

const mj = {
  // --- Math ---
  absm,
  add,
  addInto,
  argmax,
  concat,
  div,
  dotDiv,
  dotDivScalar,
  dotMul,
  dotMulScalar,
  dotProduct,
  dotSub,
  dotSubScalar,
  dotSum,
  dotSumScalar,
  expm,
  flatten,
  logm,
  map,
  matrix,
  mean,
  mul,
  ones,
  pow,
  random,
  reshape,
  sub,
  subInto,
  transpose,
  zeros,
  convolution,
  norm,
  normScalar,
  xavier,
  he,
  addBias,
  addBiasRow,
  sumAxis,
  clipGradients,

  // --- Activation (Returns Matrix, Supports Auto-Diff & Rust) ---
  sigmoid,
  tanh,
  relu,
  lRelu,
  linear,
  softmax,
  elu,
  selu,
  softplus,
  softsign,
  swish,
  gelu,
  mish,
  hardSigmoid,
  hardSwish,
  threshold,

  // --- Cost / Loss (Functional API) ---
  mse: (pred: Matrix, target: Matrix) => MeanSquaredError(target, pred)[0],
  crossEntropy: (pred: Matrix, target: Matrix) => CategoricalCrossEntropy(target, pred)[0],
  binaryCrossEntropy: (pred: Matrix, target: Matrix) => BinaryCrossEntropy(target, pred)[0],
  softmaxCrossEntropy: (pred: Matrix, target: Matrix) => SoftmaxCrossEntropy(target, pred)[0],

  // --- Optimizers ---
  SGD,
  Adam,
  NAG,
  AdaGrad,
  Momentum,
};

export default mj;
