/**
 * ML-V1: TypeScript + Rust Native Machine Learning Library
 *
 * Library machine learning custom berbasis TypeScript dengan
 * backend Rust (N-API) untuk akselerasi operasi numerik kritikal.
 *
 * @packageDocumentation
 */

// === Core ===
export { default as Matrix } from "./matrix";
export { default as mj } from "./math";

// === Layers ===
export {
  Dense,
  Convolution,
  Activation,
  CompileDenseLayers,
  SelfAttention,
  MultiHeadAttention,
  Embedding,
  Flatten,
  PositionalEncoding,
  LayerNormalization,
  Dropout,
  RNN,
  LSTM,
  GRU,
  AdaptiveMemoryRNN,
} from "./layers";

// === Models ===
export { Sequential, Transformers, DimentionalityReduction } from "./models";

// === Tokenizer ===
export {
  BPETokenizer,
  charPreTokenizer,
  unicodeGraphemePreTokenizer,
  unicodeWordPreTokenizer,
  whitespacePreTokenizer,
  scriptAwarePreTokenizer,
} from "./tokenizer";
export type { BPEConfig, BPEVocabData, BPETokenizerOptions, BuiltInPreTokenizer, PreTokenizer } from "./tokenizer";

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
} from "./activation";

// === Cost Functions ===
export {
  MeanSquerError,
  CategoricalCrossEntropy,
  BinaryCrossEntropy,
  SoftmaxCrossEntropy,
} from "./cost";

// === Utils ===
export {
  setActivation,
  setLayers,
  registerLayer,
  setLoss,
  setOptimizer,
  cosineSimilarity,
  shuffleInPlace,
  splitTrainValidation,
  formatLoss,
  formatProgressBar,
  formatTime,
} from "./utils";

// === Native Backend ===
export { isNativeAvailable } from "./math/rust_backend";

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
  Optimzier,
  Cost,
  Layers,
  FitConfig,
  FitResult,
} from "./@types/type";

export type { PaddingSide } from "./@types/fitConfig";
