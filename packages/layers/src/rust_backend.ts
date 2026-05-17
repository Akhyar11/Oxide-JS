import { createRequire } from "module";
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
