import { MatrixShape } from "../@types/type";

let native: any = null;

try {
  // Hanya gunakan satu nama konsisten yang di-generate oleh script build
  native = require("../../ml-native.node");
} catch (e) {
  // console.warn("Rust Backend: Native module failed to load.");
}

let forceDisable = false;

export const setForceDisableNative = (v: boolean) => {
  forceDisable = v;
};

export const isNativeAvailable = () => native !== null && !forceDisable;


export const dotProductNative = (
  aData: Float64Array,
  aShape: MatrixShape,
  bData: Float64Array,
  bShape: MatrixShape,
  transA: boolean,
  transB: boolean,
  outData: Float64Array
): void => {
  if (!native) throw new Error("Native backend not available");
  native.dotProductInto(aData, Array.from(aShape), bData, Array.from(bShape), outData, transA, transB);
};

export const addNative = (a: Float64Array, b: Float64Array, out: Float64Array): void => {
  if (!native) throw new Error("Native backend not available");
  native.addMatricesInto(a, b, out);
};

export const subNative = (a: Float64Array, b: Float64Array, out: Float64Array): void => {
  if (!native) throw new Error("Native backend not available");
  native.subMatricesInto(a, b, out);
};

export const mulNative = (a: Float64Array, b: Float64Array, out: Float64Array): void => {
  if (!native) throw new Error("Native backend not available");
  native.mulMatricesInto(a, b, out);
};

export const divNative = (a: Float64Array, b: Float64Array, out: Float64Array): void => {
  if (!native) throw new Error("Native backend not available");
  native.divMatricesInto(a, b, out);
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
  isRow: boolean,
  out: Float64Array
): void => {
  if (!native) throw new Error("Native backend not available");
  native.softmaxNativeInto(data, rows, cols, isRow, out);
};

export const softmaxBackwardNative = (
  sData: Float64Array,
  gData: Float64Array,
  rows: number,
  cols: number,
  isRow: boolean,
  out: Float64Array
): void => {
  if (!native) throw new Error("Native backend not available");
  native.softmaxBackwardNativeInto(sData, gData, rows, cols, isRow, out);
};

export const layerNormNative = (
  xData: Float64Array,
  gamma: Float64Array,
  beta: Float64Array,
  rows: number,
  cols: number,
  eps: number,
  outRes: Float64Array,
  outNorm: Float64Array,
  outMeans: Float64Array,
  outStds: Float64Array
): void => {
  if (!native) throw new Error("Native backend not available");
  native.layerNormNativeInto(xData, gamma, beta, rows, cols, eps, outRes, outNorm, outMeans, outStds);
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

export const reluNative = (input: Float64Array, outRes: Float64Array, outGrad: Float64Array): void => {
  if (!native) throw new Error("Native backend not available");
  native.reluNativeInto(input, outRes, outGrad);
};

export const sigmoidNative = (input: Float64Array, outRes: Float64Array, outGrad: Float64Array): void => {
  if (!native) throw new Error("Native backend not available");
  native.sigmoidNativeInto(input, outRes, outGrad);
};

export const tanhNative = (input: Float64Array, outRes: Float64Array, outGrad: Float64Array): void => {
  if (!native) throw new Error("Native backend not available");
  native.tanhNativeInto(input, outRes, outGrad);
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
  padTokenId: number | null,
  out: Float64Array
): void => {
  if (!native) throw new Error("Native backend not available");
  native.embeddingForwardNativeInto(indices, weightData, vocabSize, embeddingDim, padTokenId, out);
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
  const outRows = aRows - kRows + 1;
  const outCols = aCols - kCols + 1;
  const out = new Float64Array(outRows * outCols);
  native.convolutionNativeInto(aData, aRows, aCols, kData, kRows, kCols, out);
  return out;
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
  const out = new Float64Array(outRows * outCols);
  native.convBackwardInputNativeInto(errData, errRows, errCols, inputData, inputRows, inputCols, outRows, outCols, out);
  return out;
};
export const addBiasNative = (data: Float64Array, bias: Float64Array, rows: number, cols: number): void => {
  if (!native) throw new Error("Native backend not available");
  native.addBiasNative(data, bias, rows, cols);
};

export const sumAxisNative = (data: Float64Array, rows: number, cols: number, axis: number, out: Float64Array): void => {
  if (!native) throw new Error("Native backend not available");
  native.sumAxisNative(data, rows, cols, axis, out);
};

export const clipGradientsNative = (data: Float64Array, limit: number): void => {
  if (!native) throw new Error("Native backend not available");
  native.clipGradientsNative(data, limit);
};
