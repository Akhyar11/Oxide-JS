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

export interface EmbeddingSparseBackwardResult {
  uniqueIndices: Int32Array;
  grad: Float32Array;
}

export const setForceDisableNative = (v: boolean) => {
  forceDisable = v;
};

export const isNativeAvailable = () => native !== null && !forceDisable;

// Threshold adaptif untuk menyeimbangkan overhead JS<->native:
// - payload kecil lebih cepat di JS karena overhead FFI/dispatch,
// - payload besar lebih cepat di native karena compute lebih dominan.
// Nilai ini sengaja disentralisasi agar mudah dituning dari satu tempat.
const DOT_NATIVE_WORKLOAD_THRESHOLD_BASE = 32 * 32 * 32;
const ELEMENTWISE_NATIVE_LENGTH_THRESHOLD = 4 * 1024;
const ADAM_NATIVE_LENGTH_THRESHOLD = 2 * 1024;
const DENSE_LINEAR_BACKWARD_NATIVE_WORKLOAD_THRESHOLD_BASE = 64 * 128 * 256;

function readPositiveEnvOrDefault(name: string, fallback: number): number {
  const raw = process.env[name];
  if (!raw) return fallback;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed <= 0) return fallback;
  return parsed;
}

const DOT_NATIVE_THRESHOLD_SCALE = readPositiveEnvOrDefault("ML_DOT_NATIVE_THRESHOLD_SCALE", 0.25);
const DENSE_LINEAR_BACKWARD_NATIVE_THRESHOLD_SCALE = readPositiveEnvOrDefault(
  "ML_DENSE_LINEAR_BACKWARD_THRESHOLD_SCALE",
  0.5
);

const DOT_NATIVE_WORKLOAD_THRESHOLD = Math.max(
  1,
  Math.floor(DOT_NATIVE_WORKLOAD_THRESHOLD_BASE * DOT_NATIVE_THRESHOLD_SCALE)
);
const DENSE_LINEAR_BACKWARD_NATIVE_WORKLOAD_THRESHOLD = Math.max(
  1,
  Math.floor(
    DENSE_LINEAR_BACKWARD_NATIVE_WORKLOAD_THRESHOLD_BASE * DENSE_LINEAR_BACKWARD_NATIVE_THRESHOLD_SCALE
  )
);

export const shouldUseNativeDotProduct = (aRows: number, aCols: number, bCols: number): boolean => {
  const workload = aRows * aCols * bCols;
  if (workload >= DOT_NATIVE_WORKLOAD_THRESHOLD) return true;

  // Heuristik untuk workload transformer menengah:
  // walau workload belum melewati threshold utama, matmul dengan K besar tetap
  // cenderung lebih cepat di native dibanding loop JS.
  if (aCols >= 64 && Math.max(aRows, bCols) >= 32) {
    return workload >= Math.floor(DOT_NATIVE_WORKLOAD_THRESHOLD / 2);
  }
  return false;
};

export const shouldUseNativeElementwise = (length: number): boolean => {
  return length >= ELEMENTWISE_NATIVE_LENGTH_THRESHOLD;
};

export const shouldUseNativeAdam = (length: number): boolean => {
  return length >= ADAM_NATIVE_LENGTH_THRESHOLD;
};

export const shouldUseNativeDenseLinearBackward = (
  outputUnits: number,
  units: number,
  seqLen: number
): boolean => {
  const workload = outputUnits * units * seqLen;
  if (workload >= DENSE_LINEAR_BACKWARD_NATIVE_WORKLOAD_THRESHOLD) return true;

  // Heuristik medium transformer workload:
  // jalur linear backward native biasanya unggul saat token count dan dimensi
  // cukup besar walau volume total belum menyentuh threshold utama.
  if (seqLen >= 64 && Math.min(outputUnits, units) >= 32) {
    return workload >= Math.floor(DENSE_LINEAR_BACKWARD_NATIVE_WORKLOAD_THRESHOLD / 2);
  }
  return false;
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

export const shouldUseNativeOptimizer = (length: number): boolean => {
  return !forceDisable && length >= 2048;
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

export const adamSparseUpdateNative = (
  indices: Int32Array,
  grad: Float32Array,
  weight: Float32Array,
  m: Float32Array,
  v: Float32Array,
  t: number,
  alpha: number,
  beta1: number,
  beta2: number,
  epsilon: number,
  vocabSize: number,
  embeddingDim: number
): void => {
  if (!native) throw new Error("Native backend not available");
  native.adamSparseUpdateNative(
    indices,
    grad,
    weight,
    m,
    v,
    t,
    alpha,
    beta1,
    beta2,
    epsilon,
    vocabSize,
    embeddingDim
  );
};

export const sgdUpdateNative = (grad: Float32Array, out: Float32Array, alpha: number): void => {
  if (!native) throw new Error("Native backend not available");
  native.sgdUpdateNative(grad, out, alpha);
};

export const sgdSparseUpdateNative = (
  indices: Int32Array,
  grad: Float32Array,
  weight: Float32Array,
  alpha: number,
  vocabSize: number,
  embeddingDim: number
): void => {
  if (!native) throw new Error("Native backend not available");
  native.sgdSparseUpdateNative(indices, grad, weight, alpha, vocabSize, embeddingDim);
};

export const adagradUpdateNative = (
  grad: Float32Array,
  sum: Float32Array,
  out: Float32Array,
  alpha: number,
  epsilon: number
): void => {
  if (!native) throw new Error("Native backend not available");
  native.adagradUpdateNative(grad, sum, out, alpha, epsilon);
};

export const adagradSparseUpdateNative = (
  indices: Int32Array,
  grad: Float32Array,
  weight: Float32Array,
  sum: Float32Array,
  alpha: number,
  epsilon: number,
  vocabSize: number,
  embeddingDim: number
): void => {
  if (!native) throw new Error("Native backend not available");
  native.adagradSparseUpdateNative(indices, grad, weight, sum, alpha, epsilon, vocabSize, embeddingDim);
};

export const momentumUpdateNative = (
  grad: Float32Array,
  v: Float32Array,
  out: Float32Array,
  alpha: number,
  beta: number
): void => {
  if (!native) throw new Error("Native backend not available");
  native.momentumUpdateNative(grad, v, out, alpha, beta);
};

export const momentumSparseUpdateNative = (
  indices: Int32Array,
  grad: Float32Array,
  weight: Float32Array,
  v: Float32Array,
  alpha: number,
  beta: number,
  vocabSize: number,
  embeddingDim: number
): void => {
  if (!native) throw new Error("Native backend not available");
  native.momentumSparseUpdateNative(indices, grad, weight, v, alpha, beta, vocabSize, embeddingDim);
};

export const nagUpdateNative = (
  grad: Float32Array,
  v: Float32Array,
  out: Float32Array,
  alpha: number,
  beta: number
): void => {
  if (!native) throw new Error("Native backend not available");
  native.nagUpdateNative(grad, v, out, alpha, beta);
};

export const nagSparseUpdateNative = (
  indices: Int32Array,
  grad: Float32Array,
  weight: Float32Array,
  v: Float32Array,
  alpha: number,
  beta: number,
  vocabSize: number,
  embeddingDim: number
): void => {
  if (!native) throw new Error("Native backend not available");
  native.nagSparseUpdateNative(indices, grad, weight, v, alpha, beta, vocabSize, embeddingDim);
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
  indices: ArrayLike<number>,
  weightData: Float32Array,
  vocabSize: number,
  embeddingDim: number,
  padTokenId: number | null,
  out: Float32Array
): void => {
  if (!native) throw new Error("Native backend not available");
  if (typeof native.embeddingForwardNativeInt32Into === "function" && indices instanceof Int32Array) {
    native.embeddingForwardNativeInt32Into(indices, weightData, vocabSize, embeddingDim, padTokenId, out);
    return;
  }
  native.embeddingForwardNativeInto(Array.from(indices), weightData, vocabSize, embeddingDim, padTokenId, out);
};

export const embeddingBackwardNative = (
  indices: ArrayLike<number>,
  errData: Float32Array,
  gradData: Float32Array,
  vocabSize: number,
  embeddingDim: number,
  padTokenId: number | null
): void => {
  if (!native) throw new Error("Native backend not available");
  if (typeof native.embeddingBackwardNativeInt32 === "function" && indices instanceof Int32Array) {
    native.embeddingBackwardNativeInt32(indices, errData, gradData, vocabSize, embeddingDim, padTokenId);
    return;
  }
  native.embeddingBackwardNative(Array.from(indices), errData, gradData, vocabSize, embeddingDim, padTokenId);
};

export const embeddingBackwardSparseNative = (
  indices: Int32Array,
  errData: Float32Array,
  embeddingDim: number,
  padTokenId: number | null
): EmbeddingSparseBackwardResult => {
  if (!native) throw new Error("Native backend not available");
  return native.embeddingBackwardSparseNative(indices, errData, embeddingDim, padTokenId);
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

export const denseLinearBackwardNative = (
  errActivation: Float32Array,
  input: Float32Array,
  weight: Float32Array,
  outputUnits: number,
  units: number,
  seqLen: number,
  clipLimit: number,
  gradWeightOut: Float32Array,
  gradBiasOut: Float32Array,
  prevErrOut: Float32Array
): void => {
  if (!native) throw new Error("Native backend not available");
  native.denseLinearBackwardNativeInto(
    errActivation,
    input,
    weight,
    outputUnits,
    units,
    seqLen,
    clipLimit,
    gradWeightOut,
    gradBiasOut,
    prevErrOut
  );
};

export const rnnForwardNative = (
  xData: Float32Array,
  wxhData: Float32Array,
  whhData: Float32Array,
  bhData: Float32Array,
  seqLen: number,
  batchSize: number,
  units: number,
  hiddenUnits: number,
  isRelu: boolean,
  hSeqData: Float32Array,
  dactData: Float32Array
): void => {
  if (!native) throw new Error("Native backend not available");
  native.rnnForwardNativeInto(
    xData,
    wxhData,
    whhData,
    bhData,
    seqLen,
    batchSize,
    units,
    hiddenUnits,
    isRelu,
    hSeqData,
    dactData
  );
};

export const rnnBackwardNative = (
  xData: Float32Array,
  wxhData: Float32Array,
  whhData: Float32Array,
  hSeqData: Float32Array,
  dactData: Float32Array,
  extErrData: Float32Array,
  seqLen: number,
  batchSize: number,
  units: number,
  hiddenUnits: number,
  dwxhData: Float32Array,
  dwhhData: Float32Array,
  dbhData: Float32Array,
  dxData: Float32Array
): void => {
  if (!native) throw new Error("Native backend not available");
  native.rnnBackwardNativeInto(
    xData,
    wxhData,
    whhData,
    hSeqData,
    dactData,
    extErrData,
    seqLen,
    batchSize,
    units,
    hiddenUnits,
    dwxhData,
    dwhhData,
    dbhData,
    dxData
  );
};

export const lstmForwardNative = (
  xData: Float32Array,
  wxiData: Float32Array,
  whiData: Float32Array,
  biData: Float32Array,
  wxfData: Float32Array,
  whfData: Float32Array,
  bfData: Float32Array,
  wxoData: Float32Array,
  whoData: Float32Array,
  boData: Float32Array,
  wxgData: Float32Array,
  whgData: Float32Array,
  bgData: Float32Array,
  seqLen: number,
  batchSize: number,
  units: number,
  hiddenUnits: number,
  hSeqData: Float32Array,
  cSeqData: Float32Array,
  iSeqData: Float32Array,
  fSeqData: Float32Array,
  oSeqData: Float32Array,
  gSeqData: Float32Array
): void => {
  if (!native) throw new Error("Native backend not available");
  native.lstmForwardNativeInto(
    xData,
    wxiData,
    whiData,
    biData,
    wxfData,
    whfData,
    bfData,
    wxoData,
    whoData,
    boData,
    wxgData,
    whgData,
    bgData,
    seqLen,
    batchSize,
    units,
    hiddenUnits,
    hSeqData,
    cSeqData,
    iSeqData,
    fSeqData,
    oSeqData,
    gSeqData
  );
};

export const lstmBackwardNative = (
  xData: Float32Array,
  wxiData: Float32Array,
  whiData: Float32Array,
  wxfData: Float32Array,
  whfData: Float32Array,
  wxoData: Float32Array,
  whoData: Float32Array,
  wxgData: Float32Array,
  whgData: Float32Array,
  hSeqData: Float32Array,
  cSeqData: Float32Array,
  iSeqData: Float32Array,
  fSeqData: Float32Array,
  oSeqData: Float32Array,
  gSeqData: Float32Array,
  extErrData: Float32Array,
  seqLen: number,
  batchSize: number,
  units: number,
  hiddenUnits: number,
  dwxiData: Float32Array,
  dwhiData: Float32Array,
  dbiData: Float32Array,
  dwxfData: Float32Array,
  dwhfData: Float32Array,
  dbfData: Float32Array,
  dwxoData: Float32Array,
  dwhoData: Float32Array,
  dboData: Float32Array,
  dwxgData: Float32Array,
  dwhgData: Float32Array,
  dbgData: Float32Array,
  dxData: Float32Array
): void => {
  if (!native) throw new Error("Native backend not available");
  native.lstmBackwardNativeInto(
    xData,
    wxiData,
    whiData,
    wxfData,
    whfData,
    wxoData,
    whoData,
    wxgData,
    whgData,
    hSeqData,
    cSeqData,
    iSeqData,
    fSeqData,
    oSeqData,
    gSeqData,
    extErrData,
    seqLen,
    batchSize,
    units,
    hiddenUnits,
    dwxiData,
    dwhiData,
    dbiData,
    dwxfData,
    dwhfData,
    dbfData,
    dwxoData,
    dwhoData,
    dboData,
    dwxgData,
    dwhgData,
    dbgData,
    dxData
  );
};

export const gruForwardNative = (
  xData: Float32Array,
  wxrData: Float32Array,
  whrData: Float32Array,
  brData: Float32Array,
  wxzData: Float32Array,
  whzData: Float32Array,
  bzData: Float32Array,
  wxhData: Float32Array,
  whhData: Float32Array,
  bhData: Float32Array,
  seqLen: number,
  batchSize: number,
  units: number,
  hiddenUnits: number,
  reverse: boolean,
  hSeqData: Float32Array,
  rSeqData: Float32Array,
  zSeqData: Float32Array,
  nSeqData: Float32Array
): void => {
  if (!native) throw new Error("Native backend not available");
  native.gruForwardNativeInto(
    xData,
    wxrData,
    whrData,
    brData,
    wxzData,
    whzData,
    bzData,
    wxhData,
    whhData,
    bhData,
    seqLen,
    batchSize,
    units,
    hiddenUnits,
    reverse,
    hSeqData,
    rSeqData,
    zSeqData,
    nSeqData
  );
};

export const gruBackwardNative = (
  xData: Float32Array,
  wxrData: Float32Array,
  whrData: Float32Array,
  wxzData: Float32Array,
  whzData: Float32Array,
  wxhData: Float32Array,
  whhData: Float32Array,
  hSeqData: Float32Array,
  rSeqData: Float32Array,
  zSeqData: Float32Array,
  nSeqData: Float32Array,
  extErrData: Float32Array,
  seqLen: number,
  batchSize: number,
  units: number,
  hiddenUnits: number,
  reverse: boolean,
  dwxrData: Float32Array,
  dwhrData: Float32Array,
  dbrData: Float32Array,
  dwxzData: Float32Array,
  dwhzData: Float32Array,
  dbzData: Float32Array,
  dwxhData: Float32Array,
  dwhhData: Float32Array,
  dbhData: Float32Array,
  dxData: Float32Array
): void => {
  if (!native) throw new Error("Native backend not available");
  native.gruBackwardNativeInto(
    xData,
    wxrData,
    whrData,
    wxzData,
    whzData,
    wxhData,
    whhData,
    hSeqData,
    rSeqData,
    zSeqData,
    nSeqData,
    extErrData,
    seqLen,
    batchSize,
    units,
    hiddenUnits,
    reverse,
    dwxrData,
    dwhrData,
    dbrData,
    dwxzData,
    dwhzData,
    dbzData,
    dwxhData,
    dwhhData,
    dbhData,
    dxData
  );
};
