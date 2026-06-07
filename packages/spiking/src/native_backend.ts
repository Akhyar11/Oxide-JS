import { createRequire } from "node:module";
const require = createRequire(import.meta.url);

let native: any = null;
const disableNativeByEnv = process.env.ML_DISABLE_NATIVE === "1";

if (!disableNativeByEnv) {
  try {
    native = require("../../index.js");
    if (native != null && Object.keys(native).length === 0) {
      native = null;
    }
  } catch {
    // Silently ignore if native binding fails to load
  }
}

export const isNativeAvailable = () => native !== null;

export const dotProductAddOnlyNativeWrapper = (
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
  if (!native) throw new Error("Spiking Native backend not available");
  native.dotProductAddOnlyNative(
    aData,
    aRows,
    aCols,
    bData,
    bRows,
    bCols,
    transA,
    transB,
    outData
  );
};

export const lifStepNativeWrapper = (
  potentials: Float32Array,
  dot: Float32Array,
  spikes: Float32Array,
  lastPotentials: Float32Array,
  beta: Float32Array,
  threshold: Float32Array
): void => {
  if (!native) throw new Error("Spiking Native backend not available");
  native.lifStepNative(potentials, dot, spikes, lastPotentials, beta, threshold);
};

export const maskSurrogateNativeWrapper = (
  errorSignal: Float32Array,
  potentials: Float32Array,
  threshold: Float32Array,
  windowSize: number
): void => {
  if (!native) throw new Error("Spiking Native backend not available");
  native.maskSurrogateNative(errorSignal, potentials, threshold, windowSize);
};

export const applyAddOnlyDeltaNativeWrapper = (
  kernel: Float32Array,
  bias: Float32Array,
  inputs: Float32Array,
  errorSignal: Float32Array,
  learningRate: number,
  batch: number,
  inFeatures: number,
  units: number,
  useBias: boolean
): void => {
  if (!native) throw new Error("Spiking Native backend not available");
  native.applyAddOnlyDeltaNative(
    kernel,
    bias,
    inputs,
    errorSignal,
    learningRate,
    batch,
    inFeatures,
    units,
    useBias
  );
};

export const applyEmbeddingDeltaNativeWrapper = (
  embeddings: Float32Array,
  inputs: Float32Array,
  errorSignal: Float32Array,
  learningRate: number,
  inputDim: number,
  outputDim: number
): void => {
  if (!native) throw new Error("Spiking Native backend not available");
  native.applyEmbeddingDeltaNative(
    embeddings,
    inputs,
    errorSignal,
    learningRate,
    inputDim,
    outputDim
  );
};

export const contrastiveHebbianNativeWrapper = (
  spikes: Float32Array,
  errData: Float32Array,
  numPairs: number,
  sequenceLength: number,
  dModel: number
): number => {
  if (!native) throw new Error("Spiking Native backend not available");
  return native.contrastiveHebbianNative(
    spikes,
    errData,
    numPairs,
    sequenceLength,
    dModel
  );
};
