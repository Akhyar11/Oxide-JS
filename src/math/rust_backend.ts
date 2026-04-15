import { MatrixShape } from "../@types/type";

let native: any = null;

try {
  // Hanya gunakan satu nama konsisten yang di-generate oleh script build
  native = require("../../ml-native.node");
} catch (e) {
  // console.warn("Rust Backend: Native module failed to load.");
}

export const isNativeAvailable = () => native !== null;

export const dotProductNative = (
  aData: Float64Array,
  aShape: MatrixShape,
  bData: Float64Array,
  bShape: MatrixShape,
  transA: boolean,
  transB: boolean
): Float64Array => {
  if (!native) throw new Error("Native backend not available");
  return native.dotProduct(aData, Array.from(aShape), bData, Array.from(bShape), transA, transB);
};

export const addNative = (a: Float64Array, b: Float64Array): Float64Array => {
  if (!native) throw new Error("Native backend not available");
  return native.addMatrices(a, b);
};

export const subNative = (a: Float64Array, b: Float64Array): Float64Array => {
  if (!native) throw new Error("Native backend not available");
  return native.subMatrices(a, b);
};

export const mulNative = (a: Float64Array, b: Float64Array): Float64Array => {
  if (!native) throw new Error("Native backend not available");
  return native.mulMatrices(a, b);
};

export const divNative = (a: Float64Array, b: Float64Array): Float64Array => {
  if (!native) throw new Error("Native backend not available");
  return native.divMatrices(a, b);
};

export const addInPlaceNative = (a: Float64Array, b: Float64Array): void => {
  if (!native) throw new Error("Native backend not available");
  native.addInPlace(a, b);
};

export const subInPlaceNative = (a: Float64Array, b: Float64Array): void => {
  if (!native) throw new Error("Native backend not available");
  native.subInPlace(a, b);
};

export const mulInPlaceNative = (a: Float64Array, b: Float64Array): void => {
  if (!native) throw new Error("Native backend not available");
  native.mulInPlace(a, b);
};

export const softmaxNative = (
  data: Float64Array,
  rows: number,
  cols: number,
  isRow: boolean
): Float64Array => {
  if (!native) throw new Error("Native backend not available");
  return native.softmaxNative(data, rows, cols, isRow);
};

export const softmaxBackwardNative = (
  sData: Float64Array,
  gData: Float64Array,
  rows: number,
  cols: number,
  isRow: boolean
): Float64Array => {
  if (!native) throw new Error("Native backend not available");
  return native.softmaxBackwardNative(sData, gData, rows, cols, isRow);
};

export const layerNormNative = (
  xData: Float64Array,
  gamma: Float64Array,
  beta: Float64Array,
  rows: number,
  cols: number,
  eps: number
): Float64Array[] => {
  if (!native) throw new Error("Native backend not available");
  return native.layerNormNative(xData, gamma, beta, rows, cols, eps);
};

export const applyAttentionMaskNative = (
  data: Float64Array,
  padMask: boolean[],
  rows: number,
  cols: number,
  scale: number
): void => {
  if (!native) throw new Error("Native backend not available");
  native.applyAttentionMaskNative(data, padMask, rows, cols, scale);
};

export const adamUpdateNative = (
  grad: Float64Array,
  m: Float64Array,
  v: Float64Array,
  buffer: Float64Array,
  t: number,
  alpha: number,
  beta1: number,
  beta2: number,
  epsilon: number
): void => {
  if (!native) throw new Error("Native backend not available");
  native.adamUpdateNative(grad, m, v, buffer, t, alpha, beta1, beta2, epsilon);
};

export const reluNative = (input: Float64Array): Float64Array[] => {
  if (!native) throw new Error("Native backend not available");
  return native.reluNative(input);
};

export const sigmoidNative = (input: Float64Array): Float64Array[] => {
  if (!native) throw new Error("Native backend not available");
  return native.sigmoidNative(input);
};

export const tanhNative = (input: Float64Array): Float64Array[] => {
  if (!native) throw new Error("Native backend not available");
  return native.tanhNative(input);
};

export const mseNative = (yTrue: Float64Array, yPred: Float64Array): number[] => {
  if (!native) throw new Error("Native backend not available");
  return native.mseNative(yTrue, yPred);
};

export const embeddingForwardNative = (
  indices: number[],
  weightData: Float64Array,
  vocabSize: number,
  embeddingDim: number,
  padTokenId: number | null
): Float64Array => {
  if (!native) throw new Error("Native backend not available");
  return native.embeddingForwardNative(indices, weightData, vocabSize, embeddingDim, padTokenId);
};

export const embeddingBackwardNative = (
  indices: number[],
  errData: Float64Array,
  gradData: Float64Array,
  vocabSize: number,
  embeddingDim: number,
  padTokenId: number | null
): void => {
  if (!native) throw new Error("Native backend not available");
  native.embeddingBackwardNative(indices, errData, gradData, vocabSize, embeddingDim, padTokenId);
};

export const convolutionNative = (
  aData: Float64Array,
  aRows: number,
  aCols: number,
  kData: Float64Array,
  kRows: number,
  kCols: number
): Float64Array => {
  if (!native) throw new Error("Native backend not available");
  return native.convolutionNative(aData, aRows, aCols, kData, kRows, kCols);
};

export const convBackwardInputNative = (
  errData: Float64Array,
  errRows: number,
  errCols: number,
  inputData: Float64Array,
  inputRows: number,
  inputCols: number,
  outRows: number,
  outCols: number
): Float64Array => {
  if (!native) throw new Error("Native backend not available");
  return native.convBackwardInputNative(errData, errRows, errCols, inputData, inputRows, inputCols, outRows, outCols);
};
