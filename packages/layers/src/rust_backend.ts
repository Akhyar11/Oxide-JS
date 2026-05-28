import { createRequire } from "node:module";
const require = createRequire(import.meta.url);

let native: any = null;
const disableNativeByEnv = process.env.ML_DISABLE_NATIVE === "1";

function isMusl(): boolean {
  if (!process.report || typeof process.report.getReport !== "function") {
    try {
      const lddPath = require("child_process").execSync("which ldd").toString().trim();
      return require("fs").readFileSync(lddPath, "utf8").includes("musl");
    } catch {
      return true;
    }
  }
  const report = process.report.getReport() as {
    header?: {
      glibcVersionRuntime?: string;
    };
  };
  return !report.header?.glibcVersionRuntime;
}

function getLocalNativeCandidates(): string[] {
  const { platform, arch } = process;
  if (platform === "linux" && arch === "x64") {
    return [isMusl() ? "../layers-native.linux-x64-musl.node" : "../layers-native.linux-x64-gnu.node"];
  }
  if (platform === "linux" && arch === "arm64") {
    return [isMusl() ? "../layers-native.linux-arm64-musl.node" : "../layers-native.linux-arm64-gnu.node"];
  }
  if (platform === "darwin") {
    return arch === "arm64"
      ? ["../layers-native.darwin-universal.node", "../layers-native.darwin-arm64.node"]
      : ["../layers-native.darwin-universal.node", "../layers-native.darwin-x64.node"];
  }
  if (platform === "win32" && arch === "x64") {
    return ["../layers-native.win32-x64-msvc.node"];
  }
  if (platform === "win32" && arch === "arm64") {
    return ["../layers-native.win32-arm64-msvc.node"];
  }
  if (platform === "win32" && arch === "ia32") {
    return ["../layers-native.win32-ia32-msvc.node"];
  }
  if (platform === "android" && arch === "arm64") {
    return ["../layers-native.android-arm64.node"];
  }
  if (platform === "android" && arch === "arm") {
    return ["../layers-native.android-arm-eabi.node"];
  }
  if (platform === "freebsd" && arch === "x64") {
    return ["../layers-native.freebsd-x64.node"];
  }
  return [];
}

if (!disableNativeByEnv) {
  try {
    for (const candidate of getLocalNativeCandidates()) {
      try {
        native = require(candidate);
        break;
      } catch {
        native = null;
      }
    }
    if (native == null) {
      native = require("../index.js");
      if (native != null && Object.keys(native).length === 0) {
        native = null;
      }
    }
  } catch (err) {
    // console.warn("Layers native backend failed to load, falling back to JS implementation.", err);
  }
}

export const isNativeAvailable = (): boolean => {
  return native !== null;
};

export const seq2ColNative = (
  inputs: Float32Array,
  batchSize: number,
  sequenceLength: number,
  inputDim: number,
  kernelSize: number,
  strides: number,
  padLeft: number,
  out: Float32Array
): void => {
  if (!native) throw new Error("Layers native backend not available");
  native.seq2ColNative(inputs, batchSize, sequenceLength, inputDim, kernelSize, strides, padLeft, out);
};

export const col2SeqNative = (
  gradOut: Float32Array,
  batchSize: number,
  sequenceLength: number,
  inputDim: number,
  kernelSize: number,
  strides: number,
  padLeft: number,
  gradIn: Float32Array
): void => {
  if (!native) throw new Error("Layers native backend not available");
  native.col2SeqNative(gradOut, batchSize, sequenceLength, inputDim, kernelSize, strides, padLeft, gradIn);
};

export const grid2ColNative = (
  inputs: Float32Array,
  batchSize: number,
  height: number,
  width: number,
  channels: number,
  kernelRows: number,
  kernelCols: number,
  strideRows: number,
  strideCols: number,
  padTop: number,
  padLeft: number,
  hOut: number,
  wOut: number,
  out: Float32Array
): void => {
  if (!native) throw new Error("Layers native backend not available");
  native.grid2ColNative(inputs, batchSize, height, width, channels, kernelRows, kernelCols, strideRows, strideCols, padTop, padLeft, hOut, wOut, out);
};

export const col2GridNative = (
  gradOut: Float32Array,
  batchSize: number,
  height: number,
  width: number,
  channels: number,
  kernelRows: number,
  kernelCols: number,
  strideRows: number,
  strideCols: number,
  padTop: number,
  padLeft: number,
  hOut: number,
  wOut: number,
  gradIn: Float32Array
): void => {
  if (!native) throw new Error("Layers native backend not available");
  native.col2GridNative(gradOut, batchSize, height, width, channels, kernelRows, kernelCols, strideRows, strideCols, padTop, padLeft, hOut, wOut, gradIn);
};

export const maxPooling1DForwardNative = (
  inputs: Float32Array,
  batchSize: number,
  sequenceLength: number,
  inputDim: number,
  poolSize: number,
  strides: number,
  padLeft: number,
  out: Float32Array,
  maxIndices: Int32Array
): void => {
  if (!native) throw new Error("Layers native backend not available");
  native.maxPooling1DForwardNative(inputs, batchSize, sequenceLength, inputDim, poolSize, strides, padLeft, out, maxIndices);
};

export const maxPooling1DBackwardNative = (
  gradOut: Float32Array,
  maxIndices: Int32Array,
  gradIn: Float32Array
): void => {
  if (!native) throw new Error("Layers native backend not available");
  native.maxPooling1DBackwardNative(gradOut, maxIndices, gradIn);
};

export const maxPooling2DForwardNative = (
  inputs: Float32Array,
  batchSize: number,
  height: number,
  width: number,
  channels: number,
  poolRows: number,
  poolCols: number,
  strideRows: number,
  strideCols: number,
  padTop: number,
  padLeft: number,
  hOut: number,
  wOut: number,
  out: Float32Array,
  maxIndices: Int32Array
): void => {
  if (!native) throw new Error("Layers native backend not available");
  native.maxPooling2DForwardNative(inputs, batchSize, height, width, channels, poolRows, poolCols, strideRows, strideCols, padTop, padLeft, hOut, wOut, out, maxIndices);
};

export const maxPooling2DBackwardNative = (
  gradOut: Float32Array,
  maxIndices: Int32Array,
  gradIn: Float32Array
): void => {
  if (!native) throw new Error("Layers native backend not available");
  native.maxPooling2DBackwardNative(gradOut, maxIndices, gradIn);
};

export const averagePooling1DForwardNative = (
  inputs: Float32Array,
  batchSize: number,
  sequenceLength: number,
  inputDim: number,
  poolSize: number,
  strides: number,
  padLeft: number,
  out: Float32Array,
  windowCounts: Int32Array
): void => {
  if (!native) throw new Error("Layers native backend not available");
  native.averagePooling1DForwardNative(inputs, batchSize, sequenceLength, inputDim, poolSize, strides, padLeft, out, windowCounts);
};

export const averagePooling1DBackwardNative = (
  gradOut: Float32Array,
  windowCounts: Int32Array,
  batchSize: number,
  sequenceLength: number,
  inputDim: number,
  poolSize: number,
  strides: number,
  padLeft: number,
  gradIn: Float32Array
): void => {
  if (!native) throw new Error("Layers native backend not available");
  native.averagePooling1DBackwardNative(gradOut, windowCounts, batchSize, sequenceLength, inputDim, poolSize, strides, padLeft, gradIn);
};

export const averagePooling2DForwardNative = (
  inputs: Float32Array,
  batchSize: number,
  height: number,
  width: number,
  channels: number,
  poolRows: number,
  poolCols: number,
  strideRows: number,
  strideCols: number,
  padTop: number,
  padLeft: number,
  hOut: number,
  wOut: number,
  out: Float32Array,
  windowCounts: Int32Array
): void => {
  if (!native) throw new Error("Layers native backend not available");
  native.averagePooling2DForwardNative(inputs, batchSize, height, width, channels, poolRows, poolCols, strideRows, strideCols, padTop, padLeft, hOut, wOut, out, windowCounts);
};

export const averagePooling2DBackwardNative = (
  gradOut: Float32Array,
  windowCounts: Int32Array,
  batchSize: number,
  height: number,
  width: number,
  channels: number,
  poolRows: number,
  poolCols: number,
  strideRows: number,
  strideCols: number,
  padTop: number,
  padLeft: number,
  hOut: number,
  wOut: number,
  gradIn: Float32Array
): void => {
  if (!native) throw new Error("Layers native backend not available");
  native.averagePooling2DBackwardNative(gradOut, windowCounts, batchSize, height, width, channels, poolRows, poolCols, strideRows, strideCols, padTop, padLeft, hOut, wOut, gradIn);
};

export const embeddingForwardNative = (
  inputs: Float32Array,
  embeddings: Float32Array,
  vocabSize: number,
  embeddingDim: number,
  out: Float32Array
): void => {
  if (!native) throw new Error("Layers native backend not available");
  native.embeddingForwardNative(inputs, embeddings, vocabSize, embeddingDim, out);
};

export const embeddingBackwardNative = (
  gradOut: Float32Array,
  inputs: Float32Array,
  vocabSize: number,
  embeddingDim: number,
  gradEmbed: Float32Array
): void => {
  if (!native) throw new Error("Layers native backend not available");
  native.embeddingBackwardNative(gradOut, inputs, vocabSize, embeddingDim, gradEmbed);
};

export const layerNormalizationForwardNative = (
  inputs: Float32Array,
  gamma: Float32Array,
  beta: Float32Array,
  epsilon: number,
  out: Float32Array,
  mean: Float32Array,
  invStd: Float32Array
): void => {
  if (!native) throw new Error("Layers native backend not available");
  native.layerNormalizationForwardNative(inputs, gamma, beta, epsilon, out, mean, invStd);
};

export const layerNormalizationBackwardNative = (
  gradOut: Float32Array,
  inputs: Float32Array,
  mean: Float32Array,
  invStd: Float32Array,
  gamma: Float32Array,
  gradIn: Float32Array,
  gradGamma: Float32Array,
  gradBeta: Float32Array
): void => {
  if (!native) throw new Error("Layers native backend not available");
  native.layerNormalizationBackwardNative(gradOut, inputs, mean, invStd, gamma, gradIn, gradGamma, gradBeta);
};

export const batchNormalizationForwardNative = (
  inputs: Float32Array,
  gamma: Float32Array,
  beta: Float32Array,
  movingMean: Float32Array,
  movingVariance: Float32Array,
  epsilon: number,
  momentum: number,
  training: boolean,
  out: Float32Array,
  mean: Float32Array,
  invStd: Float32Array
): void => {
  if (!native) throw new Error("Layers native backend not available");
  native.batchNormalizationForwardNative(inputs, gamma, beta, movingMean, movingVariance, epsilon, momentum, training, out, mean, invStd);
};

export const batchNormalizationBackwardNative = (
  gradOut: Float32Array,
  inputs: Float32Array,
  mean: Float32Array,
  invStd: Float32Array,
  gamma: Float32Array,
  training: boolean,
  gradIn: Float32Array,
  gradGamma: Float32Array,
  gradBeta: Float32Array
): void => {
  if (!native) throw new Error("Layers native backend not available");
  native.batchNormalizationBackwardNative(gradOut, inputs, mean, invStd, gamma, training, gradIn, gradGamma, gradBeta);
};

export const rnnForwardNative = (
  inputs: Float32Array,
  kernel: Float32Array,
  recurrentKernel: Float32Array,
  bias: Float32Array,
  batchSize: number,
  sequenceLength: number,
  inputDim: number,
  units: number,
  activation: string,
  returnSequences: boolean,
  out: Float32Array,
  hiddenStates: Float32Array
): void => {
  if (!native) throw new Error("Layers native backend not available");
  native.rnnForwardNative(inputs, kernel, recurrentKernel, bias, batchSize, sequenceLength, inputDim, units, activation, returnSequences, out, hiddenStates);
};

export const rnnBackwardNative = (
  gradOut: Float32Array,
  inputs: Float32Array,
  hiddenStates: Float32Array,
  kernel: Float32Array,
  recurrentKernel: Float32Array,
  batchSize: number,
  sequenceLength: number,
  inputDim: number,
  units: number,
  activation: string,
  returnSequences: boolean,
  gradIn: Float32Array,
  gradKernel: Float32Array,
  gradRecurrentKernel: Float32Array,
  gradBias: Float32Array
): void => {
  if (!native) throw new Error("Layers native backend not available");
  native.rnnBackwardNative(gradOut, inputs, hiddenStates, kernel, recurrentKernel, batchSize, sequenceLength, inputDim, units, activation, returnSequences, gradIn, gradKernel, gradRecurrentKernel, gradBias);
};

export const lstmForwardNative = (
  inputs: Float32Array,
  kernel: Float32Array,
  recurrentKernel: Float32Array,
  bias: Float32Array,
  batchSize: number,
  sequenceLength: number,
  inputDim: number,
  units: number,
  returnSequences: boolean,
  out: Float32Array,
  hiddenStates: Float32Array,
  cellStates: Float32Array,
  gateValues: Float32Array
): void => {
  if (!native) throw new Error("Layers native backend not available");
  native.lstmForwardNative(inputs, kernel, recurrentKernel, bias, batchSize, sequenceLength, inputDim, units, returnSequences, out, hiddenStates, cellStates, gateValues);
};

export const lstmBackwardNative = (
  gradOut: Float32Array,
  inputs: Float32Array,
  hiddenStates: Float32Array,
  cellStates: Float32Array,
  gateValues: Float32Array,
  kernel: Float32Array,
  recurrentKernel: Float32Array,
  batchSize: number,
  sequenceLength: number,
  inputDim: number,
  units: number,
  returnSequences: boolean,
  gradIn: Float32Array,
  gradKernel: Float32Array,
  gradRecurrentKernel: Float32Array,
  gradBias: Float32Array
): void => {
  if (!native) throw new Error("Layers native backend not available");
  native.lstmBackwardNative(gradOut, inputs, hiddenStates, cellStates, gateValues, kernel, recurrentKernel, batchSize, sequenceLength, inputDim, units, returnSequences, gradIn, gradKernel, gradRecurrentKernel, gradBias);
};

export const gruForwardNative = (
  inputs: Float32Array,
  kernel: Float32Array,
  recurrentKernel: Float32Array,
  bias: Float32Array,
  batchSize: number,
  sequenceLength: number,
  inputDim: number,
  units: number,
  returnSequences: boolean,
  out: Float32Array,
  hiddenStates: Float32Array,
  gateValues: Float32Array
): void => {
  if (!native) throw new Error("Layers native backend not available");
  native.gruForwardNative(inputs, kernel, recurrentKernel, bias, batchSize, sequenceLength, inputDim, units, returnSequences, out, hiddenStates, gateValues);
};

export const gruBackwardNative = (
  gradOut: Float32Array,
  inputs: Float32Array,
  hiddenStates: Float32Array,
  gateValues: Float32Array,
  kernel: Float32Array,
  recurrentKernel: Float32Array,
  batchSize: number,
  sequenceLength: number,
  inputDim: number,
  units: number,
  returnSequences: boolean,
  gradIn: Float32Array,
  gradKernel: Float32Array,
  gradRecurrentKernel: Float32Array,
  gradBias: Float32Array
): void => {
  if (!native) throw new Error("Layers native backend not available");
  native.gruBackwardNative(gradOut, inputs, hiddenStates, gateValues, kernel, recurrentKernel, batchSize, sequenceLength, inputDim, units, returnSequences, gradIn, gradKernel, gradRecurrentKernel, gradBias);
};

export const attentionForwardNative = (
  inputsQ: Float32Array,
  inputsK: Float32Array,
  wQ: Float32Array,
  wK: Float32Array,
  wV: Float32Array,
  bQ: Float32Array,
  bK: Float32Array,
  bV: Float32Array,
  batchSize: number,
  seqLenQ: number,
  seqLenK: number,
  inputDim: number,
  units: number,
  useBias: boolean,
  out: Float32Array,
  q: Float32Array,
  k: Float32Array,
  v: Float32Array,
  scores: Float32Array,
  probs: Float32Array
): void => {
  if (!native) throw new Error("Layers native backend not available");
  native.attentionForwardNative(inputsQ, inputsK, wQ, wK, wV, bQ, bK, bV, batchSize, seqLenQ, seqLenK, inputDim, units, useBias, out, q, k, v, scores, probs);
};

export const attentionBackwardNative = (
  gradOut: Float32Array,
  inputsQ: Float32Array,
  inputsK: Float32Array,
  q: Float32Array,
  k: Float32Array,
  v: Float32Array,
  probs: Float32Array,
  wQ: Float32Array,
  wK: Float32Array,
  wV: Float32Array,
  batchSize: number,
  seqLenQ: number,
  seqLenK: number,
  inputDim: number,
  units: number,
  useBias: boolean,
  gradInQ: Float32Array,
  gradInK: Float32Array,
  gradWQ: Float32Array,
  gradWK: Float32Array,
  gradWV: Float32Array,
  gradBQ: Float32Array,
  gradBK: Float32Array,
  gradBV: Float32Array
): void => {
  if (!native) throw new Error("Layers native backend not available");
  native.attentionBackwardNative(gradOut, inputsQ, inputsK, q, k, v, probs, wQ, wK, wV, batchSize, seqLenQ, seqLenK, inputDim, units, useBias, gradInQ, gradInK, gradWQ, gradWK, gradWV, gradBQ, gradBK, gradBV);
};

export const multiHeadAttentionForwardNative = (
  inputsQ: Float32Array,
  inputsK: Float32Array,
  inputsV: Float32Array,
  wQ: Float32Array,
  wK: Float32Array,
  wV: Float32Array,
  wO: Float32Array,
  bQ: Float32Array,
  bK: Float32Array,
  bV: Float32Array,
  bO: Float32Array,
  batchSize: number,
  seqLenQ: number,
  seqLenK: number,
  inputDimQ: number,
  inputDimK: number,
  inputDimV: number,
  numHeads: number,
  keyDim: number,
  valueDim: number,
  outputDim: number,
  useBias: boolean,
  out: Float32Array,
  q: Float32Array,
  k: Float32Array,
  v: Float32Array,
  scores: Float32Array,
  probs: Float32Array,
  outConcat: Float32Array
): void => {
  if (!native) throw new Error("Layers native backend not available");
  native.multiHeadAttentionForwardNative(
    inputsQ, inputsK, inputsV, wQ, wK, wV, wO, bQ, bK, bV, bO,
    batchSize, seqLenQ, seqLenK, inputDimQ, inputDimK, inputDimV,
    numHeads, keyDim, valueDim, outputDim, useBias,
    out, q, k, v, scores, probs, outConcat
  );
};

export const multiHeadAttentionBackwardNative = (
  gradOut: Float32Array,
  inputsQ: Float32Array,
  inputsK: Float32Array,
  inputsV: Float32Array,
  q: Float32Array,
  k: Float32Array,
  v: Float32Array,
  probs: Float32Array,
  outConcat: Float32Array,
  wQ: Float32Array,
  wK: Float32Array,
  wV: Float32Array,
  wO: Float32Array,
  batchSize: number,
  seqLenQ: number,
  seqLenK: number,
  inputDimQ: number,
  inputDimK: number,
  inputDimV: number,
  numHeads: number,
  keyDim: number,
  valueDim: number,
  outputDim: number,
  useBias: boolean,
  gradInQ: Float32Array,
  gradInK: Float32Array,
  gradInV: Float32Array,
  gradWQ: Float32Array,
  gradWK: Float32Array,
  gradWV: Float32Array,
  gradWO: Float32Array,
  gradBQ: Float32Array,
  gradBK: Float32Array,
  gradBV: Float32Array,
  gradBO: Float32Array
): void => {
  if (!native) throw new Error("Layers native backend not available");
  native.multiHeadAttentionBackwardNative(
    gradOut, inputsQ, inputsK, inputsV, q, k, v, probs, outConcat,
    wQ, wK, wV, wO, batchSize, seqLenQ, seqLenK, inputDimQ, inputDimK, inputDimV,
    numHeads, keyDim, valueDim, outputDim, useBias,
    gradInQ, gradInK, gradInV, gradWQ, gradWK, gradWV, gradWO, gradBQ, gradBK, gradBV, gradBO
  );
};






