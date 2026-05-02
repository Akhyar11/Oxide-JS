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

/**
 * Fused embedding backward + Adam update in one NAPI call.
 * Eliminates all JS↔native round-trips and intermediate allocations on the hot path.
 *
 * Returns false (and is a no-op) when the native binary does not export this symbol
 * so that callers can fall back to the existing split path transparently.
 */
export const embeddingAdamBackwardUpdateNative = (
  indices: Int32Array,
  errData: Float32Array,
  weight: Float32Array,
  m: Float32Array,
  v: Float32Array,
  t: number,
  alpha: number,
  beta1: number,
  beta2: number,
  epsilon: number,
  vocabSize: number,
  embeddingDim: number,
  padTokenId: number | null
): boolean => {
  if (!native) return false;
  if (typeof native.embeddingAdamBackwardUpdateNative !== "function") return false;
  native.embeddingAdamBackwardUpdateNative(
    indices,
    errData,
    weight,
    m,
    v,
    t,
    alpha,
    beta1,
    beta2,
    epsilon,
    vocabSize,
    embeddingDim,
    padTokenId
  );
  return true;
};

/**
 * Fused embedding backward + SGD update (single NAPI call).
 * Returns false when the native binary does not export this symbol.
 */
export const embeddingSgdBackwardUpdateNative = (
  indices: Int32Array,
  errData: Float32Array,
  weight: Float32Array,
  alpha: number,
  vocabSize: number,
  embeddingDim: number,
  padTokenId: number | null
): boolean => {
  if (!native) return false;
  if (typeof native.embeddingSgdBackwardUpdateNative !== "function") return false;
  native.embeddingSgdBackwardUpdateNative(indices, errData, weight, alpha, vocabSize, embeddingDim, padTokenId);
  return true;
};

/**
 * Fused embedding backward + AdaGrad update (single NAPI call).
 * Returns false when the native binary does not export this symbol.
 */
export const embeddingAdagradBackwardUpdateNative = (
  indices: Int32Array,
  errData: Float32Array,
  weight: Float32Array,
  sumData: Float32Array,
  alpha: number,
  epsilon: number,
  vocabSize: number,
  embeddingDim: number,
  padTokenId: number | null
): boolean => {
  if (!native) return false;
  if (typeof native.embeddingAdagradBackwardUpdateNative !== "function") return false;
  native.embeddingAdagradBackwardUpdateNative(indices, errData, weight, sumData, alpha, epsilon, vocabSize, embeddingDim, padTokenId);
  return true;
};

/**
 * Fused embedding backward + Momentum update (single NAPI call).
 * Returns false when the native binary does not export this symbol.
 */
export const embeddingMomentumBackwardUpdateNative = (
  indices: Int32Array,
  errData: Float32Array,
  weight: Float32Array,
  vData: Float32Array,
  alpha: number,
  beta: number,
  vocabSize: number,
  embeddingDim: number,
  padTokenId: number | null
): boolean => {
  if (!native) return false;
  if (typeof native.embeddingMomentumBackwardUpdateNative !== "function") return false;
  native.embeddingMomentumBackwardUpdateNative(indices, errData, weight, vData, alpha, beta, vocabSize, embeddingDim, padTokenId);
  return true;
};

/**
 * Fused embedding backward + NAG update (single NAPI call).
 * Returns false when the native binary does not export this symbol.
 */
export const embeddingNagBackwardUpdateNative = (
  indices: Int32Array,
  errData: Float32Array,
  weight: Float32Array,
  vData: Float32Array,
  alpha: number,
  beta: number,
  vocabSize: number,
  embeddingDim: number,
  padTokenId: number | null
): boolean => {
  if (!native) return false;
  if (typeof native.embeddingNagBackwardUpdateNative !== "function") return false;
  native.embeddingNagBackwardUpdateNative(indices, errData, weight, vData, alpha, beta, vocabSize, embeddingDim, padTokenId);
  return true;
};

/**
 * Native LSTM Forward (Fuses the sequence loop).
 */
export const lstmForwardNative = (
  wxi: Float32Array, wxf: Float32Array, wxo: Float32Array, wxg: Float32Array,
  whi: Float32Array, whf: Float32Array, who: Float32Array, whg: Float32Array,
  bi: Float32Array, bf: Float32Array, bo: Float32Array, bg: Float32Array,
  x_seq: Float32Array, h0: Float32Array, c0: Float32Array,
  hiddenUnits: number, inputUnits: number, seqLen: number, batchSize: number,
  h_seq_out: Float32Array, c_seq_out: Float32Array,
  i_seq_out: Float32Array, f_seq_out: Float32Array, o_seq_out: Float32Array, g_seq_out: Float32Array
): boolean => {
  if (!native) return false;
  if (typeof native.lstmForwardNativeInto !== "function") return false;
  native.lstmForwardNativeInto(
    wxi, wxf, wxo, wxg,
    whi, whf, who, whg,
    bi, bf, bo, bg,
    x_seq, h0, c0,
    hiddenUnits, inputUnits, seqLen, batchSize,
    h_seq_out, c_seq_out,
    i_seq_out, f_seq_out, o_seq_out, g_seq_out
  );
  return true;
};

/**
 * Native LSTM Backward (Fuses the BPTT sequence loop).
 */
export const lstmBackwardNative = (
  wx_i: Float32Array, wx_f: Float32Array, wx_o: Float32Array, wx_g: Float32Array,
  wh_i: Float32Array, wh_f: Float32Array, wh_o: Float32Array, wh_g: Float32Array,
  x_seq: Float32Array, h_seq: Float32Array, c_seq: Float32Array,
  i_seq: Float32Array, f_seq: Float32Array, o_seq: Float32Array, g_seq: Float32Array,
  err_h: Float32Array,
  hiddenUnits: number, inputUnits: number, seqLen: number, batchSize: number,
  dwx_i: Float32Array, dwh_i: Float32Array, dbi: Float32Array,
  dwx_f: Float32Array, dwh_f: Float32Array, dbf: Float32Array,
  dwx_o: Float32Array, dwho: Float32Array, dbo: Float32Array,
  dwx_g: Float32Array, dwh_g: Float32Array, dbg: Float32Array,
  dx_out: Float32Array
): boolean => {
  if (!native) return false;
  if (typeof native.lstmBackwardNativeInto !== "function") return false;
  native.lstmBackwardNativeInto(
    wx_i, wx_f, wx_o, wx_g,
    wh_i, wh_f, wh_o, wh_g,
    x_seq, h_seq, c_seq,
    i_seq, f_seq, o_seq, g_seq,
    err_h,
    hiddenUnits, inputUnits, seqLen, batchSize,
    dwx_i, dwh_i, dbi,
    dwx_f, dwh_f, dbf,
    dwx_o, dwho, dbo,
    dwx_g, dwh_g, dbg,
    dx_out
  );
  return true;
};

/**
 * Native GRU Forward (Fuses the sequence loop).
 */
export const gruForwardNative = (
  wxr: Float32Array, whr: Float32Array, br: Float32Array,
  wxz: Float32Array, whz: Float32Array, bz: Float32Array,
  wxh: Float32Array, whh: Float32Array, bh: Float32Array,
  x_seq: Float32Array, h0: Float32Array,
  hiddenUnits: number, inputUnits: number,
  seqLen: number,
  batchSize: number,
  h_seq_out: Float32Array,
  r_seq_out: Float32Array,
  z_seq_out: Float32Array,
  n_seq_out: Float32Array
): boolean => {
  if (!native) return false;
  if (typeof native.gruForwardNativeInto !== "function") return false;
  native.gruForwardNativeInto(
    wxr, whr, br,
    wxz, whz, bz,
    wxh, whh, bh,
    x_seq, h0,
    hiddenUnits, inputUnits, seqLen, batchSize,
    h_seq_out, r_seq_out, z_seq_out, n_seq_out
  );
  return true;
};

/**
 * Native GRU Backward (Fuses the BPTT sequence loop).
 */
export const gruBackwardNative = (
  wxr: Float32Array, whr: Float32Array,
  wxz: Float32Array, whz: Float32Array,
  wxh: Float32Array, whh: Float32Array,
  x_seq: Float32Array, h_seq: Float32Array,
  r_seq: Float32Array, z_seq: Float32Array, n_seq: Float32Array,
  err_h: Float32Array,
  hiddenUnits: number, inputUnits: number, seqLen: number, batchSize: number,
  dwxr: Float32Array, dwhr: Float32Array, dbr: Float32Array,
  dwxz: Float32Array, dwhz: Float32Array, dbz: Float32Array,
  dwxh: Float32Array, dwhh: Float32Array, dbh: Float32Array,
  dx_out: Float32Array
): boolean => {
  if (!native) return false;
  if (typeof native.gruBackwardNativeInto !== "function") return false;
  native.gruBackwardNativeInto(
    wxr, whr,
    wxz, whz,
    wxh, whh,
    x_seq, h_seq,
    r_seq, z_seq, n_seq,
    err_h,
    hiddenUnits, inputUnits, seqLen, batchSize,
    dwxr, dwhr, dbr,
    dwxz, dwhz, dbz,
    dwxh, dwhh, dbh,
    dx_out
  );
  return true;
};

/**
 * Native Simple RNN Forward
 */
export const rnnForwardNative = (
  wxh: Float32Array,
  whh: Float32Array,
  bias: Float32Array,
  x_seq: Float32Array,
  h0: Float32Array,
  hiddenUnits: number,
  inputUnits: number,
  seqLen: number,
  batchSize: number,
  h_seq_out: Float32Array,
  d_act_out: Float32Array
): boolean => {
  if (!native) return false;
  if (typeof native.rnnForwardNativeInto !== "function") return false;
  native.rnnForwardNativeInto(
    wxh, whh, bias,
    x_seq, h0,
    hiddenUnits, inputUnits, seqLen, batchSize,
    h_seq_out, d_act_out
  );
  return true;
};

/**
 * Native Simple RNN Backward
 */
export const rnnBackwardNative = (
  wxh: Float32Array, whh: Float32Array,
  x_seq: Float32Array, h_seq: Float32Array, d_act_seq: Float32Array,
  err_h: Float32Array,
  hiddenUnits: number, inputUnits: number, seqLen: number, batchSize: number,
  dwxh: Float32Array, dwhh: Float32Array, dbh: Float32Array,
  dx_out: Float32Array
): boolean => {
  if (!native) return false;
  if (typeof native.rnnBackwardNativeInto !== "function") return false;
  native.rnnBackwardNativeInto(
    wxh, whh,
    x_seq, h_seq, d_act_seq,
    err_h,
    hiddenUnits, inputUnits, seqLen, batchSize,
    dwxh, dwhh, dbh,
    dx_out
  );
  return true;
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

/**
 * Native AdaptiveMemoryRNN Forward (fused sequence loop + memory module).
 *
 * combined_out layout: [seq_len * (input_units+memory_dim) * batch_size]
 * h_seq_out layout   : [(seq_len+1) * batch_size * hidden_units]
 * act_grad layout    : [seq_len * batch_size * hidden_units]
 * mem_keys/values    : [batch_size * memory_dim * memory_slots]  (mutated in-place)
 * mem_usage          : [batch_size * memory_slots]               (mutated in-place)
 *
 * Returns false when the native binary does not export this symbol.
 */
export const adaptiveMemoryRnnForwardNative = (
  wq: Float32Array, wm: Float32Array,
  wxh: Float32Array, whh: Float32Array, bh: Float32Array,
  wg: Float32Array, bg: Float32Array,
  xSeq: Float32Array, h0: Float32Array,
  hiddenUnits: number, inputUnits: number, memoryDim: number, memorySlots: number,
  seqLen: number, batchSize: number,
  useRelu: boolean,
  hSeqOut: Float32Array,
  actGrad: Float32Array,
  memKeys: Float32Array,
  memValues: Float32Array,
  memUsage: Float32Array,
  combinedOut: Float32Array,
): boolean => {
  if (!native) return false;
  if (typeof native.adaptiveMemoryRnnForwardNativeInto !== "function") return false;
  native.adaptiveMemoryRnnForwardNativeInto(
    wq, wm,
    wxh, whh, bh,
    wg, bg,
    xSeq, h0,
    hiddenUnits, inputUnits, memoryDim, memorySlots,
    seqLen, batchSize,
    useRelu,
    hSeqOut,
    actGrad,
    memKeys,
    memValues,
    memUsage,
    combinedOut,
  );
  return true;
};

/**
 * Native AdaptiveMemoryRNN Backward (BPTT through Wxh/Whh/bh; dx propagated
 * through the combined input — only the raw-input portion is written to dx_out).
 *
 * combined layout : [seq_len * (input_units+memory_dim) * batch_size]
 * h_seq layout    : [(seq_len+1) * batch_size * hidden_units]
 * act_grad layout : [seq_len * batch_size * hidden_units]
 * err_h layout    : [seq_len * batch_size * hidden_units]  (pre-built per-step errors)
 * dx_out layout   : [input_units, total_cols]
 *
 * Returns false when the native binary does not export this symbol.
 */
export const adaptiveMemoryRnnBackwardNative = (
  wxh: Float32Array, whh: Float32Array,
  combined: Float32Array, hSeq: Float32Array,
  actGrad: Float32Array, errH: Float32Array,
  hiddenUnits: number, inputUnits: number, memoryDim: number,
  seqLen: number, batchSize: number,
  dwxh: Float32Array, dwhh: Float32Array, dbh: Float32Array,
  dxOut: Float32Array,
): boolean => {
  if (!native) return false;
  if (typeof native.adaptiveMemoryRnnBackwardNativeInto !== "function") return false;
  native.adaptiveMemoryRnnBackwardNativeInto(
    wxh, whh,
    combined, hSeq,
    actGrad, errH,
    hiddenUnits, inputUnits, memoryDim,
    seqLen, batchSize,
    dwxh, dwhh, dbh,
    dxOut,
  );
  return true;
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
