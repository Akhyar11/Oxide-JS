import { MatrixShape } from "../@types/type";

let native: any = null;
const disableNativeByEnv = process.env.ML_DISABLE_NATIVE === "1";

if (!disableNativeByEnv) {
  try {
    // Gunakan loader NAPI-RS agar memilih artefak platform yang benar.
    // Menghindari pemakaian binary stale ketika file *.node lama masih ada di repo root.
    native = require("../../index.js");
  } catch (e) {
    // console.warn("Rust Backend: Native module failed to load.");
  }
}

let forceDisable = false;

export type MaskedSparseSoftmaxCrossEntropyResult = {
  loss: number;
  validTokens: number;
};

export const setForceDisableNative = (v: boolean) => {
  forceDisable = v;
};

export const isNativeAvailable = () => native !== null && !forceDisable;

// Threshold adaptif untuk menyeimbangkan overhead JS<->native:
// - payload kecil lebih cepat di JS karena overhead FFI/dispatch,
// - payload besar lebih cepat di native karena compute lebih dominan.
// Nilai ini sengaja disentralisasi agar mudah dituning dari satu tempat.
const DOT_NATIVE_WORKLOAD_THRESHOLD = 32 * 32 * 32;
const ELEMENTWISE_NATIVE_LENGTH_THRESHOLD = 4 * 1024;
const ADAM_NATIVE_LENGTH_THRESHOLD = 2 * 1024;

export const shouldUseNativeDotProduct = (aRows: number, aCols: number, bCols: number): boolean => {
  return aRows * aCols * bCols >= DOT_NATIVE_WORKLOAD_THRESHOLD;
};

export const shouldUseNativeElementwise = (length: number): boolean => {
  return length >= ELEMENTWISE_NATIVE_LENGTH_THRESHOLD;
};

export const shouldUseNativeAdam = (length: number): boolean => {
  return length >= ADAM_NATIVE_LENGTH_THRESHOLD;
};


export const dotProductNative = (
  aData: Float32Array,
  aRows: number,
  aCols: number,
  bData: Float32Array,
  bRows: number,
  bCols: number,
  transA: boolean,
  transB: boolean,
  outData: Float32Array
): void => {
  if (!native) throw new Error("Native backend not available");
  if (typeof native.dotProductIntoDims === "function") {
    native.dotProductIntoDims(aData, aRows, aCols, bData, bRows, bCols, outData, transA, transB);
    return;
  }
  // Backward compatibility untuk binary native lama.
  native.dotProductInto(aData, [aRows, aCols] as MatrixShape, bData, [bRows, bCols] as MatrixShape, outData, transA, transB);
};

export const addNative = (a: Float32Array, b: Float32Array, out: Float32Array): void => {
  if (!native) throw new Error("Native backend not available");
  native.addMatricesInto(a, b, out);
};

export const subNative = (a: Float32Array, b: Float32Array, out: Float32Array): void => {
  if (!native) throw new Error("Native backend not available");
  native.subMatricesInto(a, b, out);
};

export const mulNative = (a: Float32Array, b: Float32Array, out: Float32Array): void => {
  if (!native) throw new Error("Native backend not available");
  native.mulMatricesInto(a, b, out);
};

export const divNative = (a: Float32Array, b: Float32Array, out: Float32Array): void => {
  if (!native) throw new Error("Native backend not available");
  native.divMatricesInto(a, b, out);
};

export const addInPlaceNative = (a: Float32Array, b: Float32Array): void => {
  if (!native) throw new Error("Native backend not available");
  native.addInPlace(a, b);
};

export const subInPlaceNative = (a: Float32Array, b: Float32Array): void => {
  if (!native) throw new Error("Native backend not available");
  native.subInPlace(a, b);
};

export const mulInPlaceNative = (a: Float32Array, b: Float32Array): void => {
  if (!native) throw new Error("Native backend not available");
  native.mulInPlace(a, b);
};

export const softmaxNative = (
  data: Float32Array,
  rows: number,
  cols: number,
  isRow: boolean,
  out: Float32Array
): void => {
  if (!native) throw new Error("Native backend not available");
  native.softmaxNativeInto(data, rows, cols, isRow, out);
};

export const softmaxBackwardNative = (
  sData: Float32Array,
  gData: Float32Array,
  rows: number,
  cols: number,
  isRow: boolean,
  out: Float32Array
): void => {
  if (!native) throw new Error("Native backend not available");
  native.softmaxBackwardNativeInto(sData, gData, rows, cols, isRow, out);
};

export const maskedSparseSoftmaxCrossEntropyNative = (
  logits: Float32Array,
  inputTokens: Float32Array,
  targets: Float32Array,
  seqLen: number,
  batchSize: number,
  vocabSize: number,
  padTokenId: number | null,
  outGrad: Float32Array
): MaskedSparseSoftmaxCrossEntropyResult => {
  if (!native) throw new Error("Native backend not available");
  const result = native.maskedSparseSoftmaxCrossEntropyInto(
    logits,
    inputTokens,
    targets,
    seqLen,
    batchSize,
    vocabSize,
    padTokenId,
    outGrad
  );
  return {
    loss: result.loss,
    validTokens: result.validTokens ?? result.valid_tokens,
  };
};

export const projectLastTokenLogitsNative = (
  hidden: Float32Array,
  weight: Float32Array,
  bias: Float32Array,
  units: number,
  seqLen: number,
  batchSize: number,
  vocabSize: number,
  out: Float32Array
): void => {
  if (!native) throw new Error("Native backend not available");
  native.projectLastTokenLogitsNativeInto(hidden, weight, bias, units, seqLen, batchSize, vocabSize, out);
};

export const layerNormNative = (
  xData: Float32Array,
  gamma: Float32Array,
  beta: Float32Array,
  rows: number,
  cols: number,
  eps: number,
  outRes: Float32Array,
  outNorm: Float32Array,
  outMeans: Float32Array,
  outStds: Float32Array
): void => {
  if (!native) throw new Error("Native backend not available");
  native.layerNormNativeInto(xData, gamma, beta, rows, cols, eps, outRes, outNorm, outMeans, outStds);
};

export const layerNormBackwardNative = (
  errData: Float32Array,
  normData: Float32Array,
  gammaData: Float32Array,
  rows: number,
  cols: number,
  stdData: Float32Array,
  dGammaOut: Float32Array,
  dBetaOut: Float32Array,
  dxOut: Float32Array
): void => {
  if (!native) throw new Error("Native backend not available");
  native.layerNormBackwardNativeInto(
    errData,
    normData,
    gammaData,
    rows,
    cols,
    stdData,
    dGammaOut,
    dBetaOut,
    dxOut
  );
};

export const applyAttentionMaskNative = (
  data: Float32Array,
  padMask: boolean[],
  rows: number,
  cols: number,
  scale: number
): void => {
  if (!native) throw new Error("Native backend not available");
  native.applyAttentionMaskNative(data, padMask, rows, cols, scale);
};

export const multiHeadAttentionForwardNative = (
  qData: Float32Array,
  kData: Float32Array,
  vData: Float32Array,
  padMask: boolean[],
  heads: number,
  headUnits: number,
  seqLen: number,
  batchSize: number,
  scale: number,
  outData: Float32Array,
  attentionData: Float32Array
): void => {
  if (!native) throw new Error("Native backend not available");
  native.multiHeadAttentionForwardNativeInto(
    qData,
    kData,
    vData,
    padMask,
    heads,
    headUnits,
    seqLen,
    batchSize,
    scale,
    outData,
    attentionData
  );
};

export const multiHeadAttentionBackwardNative = (
  qData: Float32Array,
  kData: Float32Array,
  vData: Float32Array,
  attentionData: Float32Array,
  dOutData: Float32Array,
  padMask: boolean[],
  heads: number,
  headUnits: number,
  seqLen: number,
  batchSize: number,
  scale: number,
  dQOut: Float32Array,
  dKOut: Float32Array,
  dVOut: Float32Array
): void => {
  if (!native) throw new Error("Native backend not available");
  native.multiHeadAttentionBackwardNativeInto(
    qData,
    kData,
    vData,
    attentionData,
    dOutData,
    padMask,
    heads,
    headUnits,
    seqLen,
    batchSize,
    scale,
    dQOut,
    dKOut,
    dVOut
  );
};

export const adamUpdateNative = (
  grad: Float32Array,
  m: Float32Array,
  v: Float32Array,
  buffer: Float32Array,
  t: number,
  alpha: number,
  beta1: number,
  beta2: number,
  epsilon: number
): void => {
  if (!native) throw new Error("Native backend not available");
  native.adamUpdateNative(grad, m, v, buffer, t, alpha, beta1, beta2, epsilon);
};

export const reluNative = (input: Float32Array, outRes: Float32Array, outGrad: Float32Array): void => {
  if (!native) throw new Error("Native backend not available");
  native.reluNativeInto(input, outRes, outGrad);
};

export const sigmoidNative = (input: Float32Array, outRes: Float32Array, outGrad: Float32Array): void => {
  if (!native) throw new Error("Native backend not available");
  native.sigmoidNativeInto(input, outRes, outGrad);
};

export const tanhNative = (input: Float32Array, outRes: Float32Array, outGrad: Float32Array): void => {
  if (!native) throw new Error("Native backend not available");
  native.tanhNativeInto(input, outRes, outGrad);
};

export const mseNative = (yTrue: Float32Array, yPred: Float32Array): number[] => {
  if (!native) throw new Error("Native backend not available");
  return native.mseNative(yTrue, yPred);
};

export const embeddingForwardNative = (
  indices: number[],
  weightData: Float32Array,
  vocabSize: number,
  embeddingDim: number,
  padTokenId: number | null,
  out: Float32Array
): void => {
  if (!native) throw new Error("Native backend not available");
  native.embeddingForwardNativeInto(indices, weightData, vocabSize, embeddingDim, padTokenId, out);
};

export const embeddingBackwardNative = (
  indices: number[],
  errData: Float32Array,
  gradData: Float32Array,
  vocabSize: number,
  embeddingDim: number,
  padTokenId: number | null
): void => {
  if (!native) throw new Error("Native backend not available");
  native.embeddingBackwardNative(indices, errData, gradData, vocabSize, embeddingDim, padTokenId);
};

export const convolutionNative = (
  aData: Float32Array,
  aRows: number,
  aCols: number,
  kData: Float32Array,
  kRows: number,
  kCols: number
): Float32Array => {
  if (!native) throw new Error("Native backend not available");
  const outRows = aRows - kRows + 1;
  const outCols = aCols - kCols + 1;
  const out = new Float32Array(outRows * outCols);
  native.convolutionNativeInto(aData, aRows, aCols, kData, kRows, kCols, out);
  return out;
};

export const convBackwardInputNative = (
  errData: Float32Array,
  errRows: number,
  errCols: number,
  inputData: Float32Array,
  inputRows: number,
  inputCols: number,
  outRows: number,
  outCols: number
): Float32Array => {
  if (!native) throw new Error("Native backend not available");
  const out = new Float32Array(outRows * outCols);
  native.convBackwardInputNativeInto(errData, errRows, errCols, inputData, inputRows, inputCols, outRows, outCols, out);
  return out;
};
export const addBiasNative = (data: Float32Array, bias: Float32Array, rows: number, cols: number): void => {
  if (!native) throw new Error("Native backend not available");
  native.addBiasNative(data, bias, rows, cols);
};

export const sumAxisNative = (data: Float32Array, rows: number, cols: number, axis: number, out: Float32Array): void => {
  if (!native) throw new Error("Native backend not available");
  native.sumAxisNative(data, rows, cols, axis, out);
};

export const clipGradientsNative = (data: Float32Array, limit: number): void => {
  if (!native) throw new Error("Native backend not available");
  native.clipGradientsNative(data, limit);
};
