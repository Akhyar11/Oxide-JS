/**
 * @oxide-js/core: Foundation of Oxide-JS
 */

// === Matrix & Math ===
export { default as Matrix } from "./matrix/index.js";
export { default as mj } from "./math/index.js";

// === Activation Functions ===
export {
  default as linear,
  sigmoid,
  tanh,
  relu,
  lRelu,
  softmax,
  softmaxOnly,
  softmaxInto,
  softmaxBackward,
  softmaxBackwardInto,
  softmaxGradient,
} from "./activation/index.js";

// === Cost Functions ===
export {
  MeanSquaredError,
  CategoricalCrossEntropy,
  BinaryCrossEntropy,
  SoftmaxCrossEntropy,
} from "./cost/index.js";

export {
  SGD,
  Adam,
  NAG,
  AdaGrad,
  Momentum,
} from "./optimizer/index.js";

// === Tokenizer ===
export {
  BPETokenizer,
  charPreTokenizer,
  unicodeGraphemePreTokenizer,
  unicodeWordPreTokenizer,
  whitespacePreTokenizer,
  scriptAwarePreTokenizer,
} from "./tokenizer/index.js";
export type { BPEConfig, BPETrainingEncodeOptions, BPEVocabData, BPETokenizerOptions, BuiltInPreTokenizer, PreTokenizer } from "./tokenizer/index.js";

// === Utils ===
export {
  setActivation,
  setLoss,
  setOptimizer,
  cosineSimilarity,
  shuffleInPlace,
  splitTrainValidation,
  formatLoss,
  formatProgressBar,
  formatTime,
  trimPaddingBatch,
} from "./utils/index.js";

// === Auto-Diff ===
export { default as Tape } from "./autodiff/index.js";
export { engine } from "./autodiff/engine.js";

// === Native Backend ===
export * from "./math/rust_backend.js";

// === Types ===
export type {
  vector,
  matrix2d,
  matrix3d,
  MatrixCollection,
  MatrixShape,
  MatrixFlatData,
  ActivationType,
  StatusLayer,
  Optimizer,
  OptimizerType,
  Cost,
  WorkerData,
  FitConfig,
  FitResult,
} from "./@types/type.js";
