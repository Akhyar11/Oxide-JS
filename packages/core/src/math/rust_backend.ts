import { createRequire } from "node:module";
const require = createRequire(import.meta.url);

import { MatrixShape } from "../@types/type.js";

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
    return [isMusl() ? "../../oxide-native.linux-x64-musl.node" : "../../oxide-native.linux-x64-gnu.node"];
  }
  if (platform === "linux" && arch === "arm64") {
    return [isMusl() ? "../../oxide-native.linux-arm64-musl.node" : "../../oxide-native.linux-arm64-gnu.node"];
  }
  if (platform === "darwin") {
    return arch === "arm64"
      ? ["../../oxide-native.darwin-universal.node", "../../oxide-native.darwin-arm64.node"]
      : ["../../oxide-native.darwin-universal.node", "../../oxide-native.darwin-x64.node"];
  }
  if (platform === "win32" && arch === "x64") {
    return ["../../oxide-native.win32-x64-msvc.node"];
  }
  if (platform === "win32" && arch === "arm64") {
    return ["../../oxide-native.win32-arm64-msvc.node"];
  }
  if (platform === "win32" && arch === "ia32") {
    return ["../../oxide-native.win32-ia32-msvc.node"];
  }
  if (platform === "android" && arch === "arm64") {
    return ["../../oxide-native.android-arm64.node"];
  }
  if (platform === "android" && arch === "arm") {
    return ["../../oxide-native.android-arm-eabi.node"];
  }
  if (platform === "freebsd" && arch === "x64") {
    return ["../../oxide-native.freebsd-x64.node"];
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
      native = require("../../index.js");
      if (native != null && Object.keys(native).length === 0) {
        native = null;
      }
    }
  } catch {
    // console.warn("Rust Backend: Native module failed to load.");
  }
}

let forceDisable = false;

export const setForceDisableNative = (v: boolean) => {
  forceDisable = v;
};

export const isNativeAvailable = () => native !== null && !forceDisable;

function requireNativeMethod(name: string): any {
  if (!native) {
    throw new Error("Native backend not available");
  }
  const method = native[name];
  if (typeof method !== "function") {
    throw new Error(`Native backend is active but method '${name}' is unavailable in the loaded binary`);
  }
  return method;
}


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

export const sgdUpdateNative = (grad: Float32Array, out: Float32Array, alpha: number): void => {
  if (!native) throw new Error("Native backend not available");
  native.sgdUpdateNative(grad, out, alpha);
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

export const maeNativeInto = (yTrue: Float32Array, yPred: Float32Array, outGrad: Float32Array): number[] => {
  if (!native) throw new Error("Native backend not available");
  return native.maeNativeInto(yTrue, yPred, outGrad);
};

export const huberNativeInto = (yTrue: Float32Array, yPred: Float32Array, outGrad: Float32Array, delta: number): number[] => {
  if (!native) throw new Error("Native backend not available");
  return native.huberNativeInto(yTrue, yPred, outGrad, delta);
};

export const logcoshNativeInto = (yTrue: Float32Array, yPred: Float32Array, outGrad: Float32Array): number[] => {
  if (!native) throw new Error("Native backend not available");
  return native.logcoshNativeInto(yTrue, yPred, outGrad);
};

export const hingeNativeInto = (yTrue: Float32Array, yPred: Float32Array, outGrad: Float32Array): number[] => {
  if (!native) throw new Error("Native backend not available");
  return native.hingeNativeInto(yTrue, yPred, outGrad);
};

export const squaredHingeNativeInto = (yTrue: Float32Array, yPred: Float32Array, outGrad: Float32Array): number[] => {
  if (!native) throw new Error("Native backend not available");
  return native.squaredHingeNativeInto(yTrue, yPred, outGrad);
};

export const kldivergenceNativeInto = (yTrue: Float32Array, yPred: Float32Array, outGrad: Float32Array): number[] => {
  if (!native) throw new Error("Native backend not available");
  return native.kldivergenceNativeInto(yTrue, yPred, outGrad);
};

export const poissonNativeInto = (yTrue: Float32Array, yPred: Float32Array, outGrad: Float32Array): number[] => {
  if (!native) throw new Error("Native backend not available");
  return native.poissonNativeInto(yTrue, yPred, outGrad);
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
export const addBiasNative = (data: Float32Array, bias: Float32Array, rows: number, cols: number): void => {
  if (!native) throw new Error("Native backend not available");
  native.addBiasNative(data, bias, rows, cols);
};

export const addBiasRowNative = (data: Float32Array, bias: Float32Array, rows: number, cols: number): void => {
  if (!native) throw new Error("Native backend not available");
  native.addBiasRowNative(data, bias, rows, cols);
};

export const sumAxisNative = (data: Float32Array, rows: number, cols: number, axis: number, out: Float32Array): void => {
  if (!native) throw new Error("Native backend not available");
  native.sumAxisNative(data, rows, cols, axis, out);
};

export const clipGradientsNative = (data: Float32Array, limit: number): void => {
  if (!native) throw new Error("Native backend not available");
  native.clipGradientsNative(data, limit);
};

export const lReluNative = (input: Float32Array, outRes: Float32Array, outGrad: Float32Array): void => {
  if (!native) throw new Error("Native backend not available");
  native.lReluNativeInto(input, outRes, outGrad);
};

export const powNative = (a: Float32Array, n: number, out: Float32Array): void => {
  if (!native) throw new Error("Native backend not available");
  native.powNative(a, n, out);
};

export const absmNative = (a: Float32Array, out: Float32Array): void => {
  if (!native) throw new Error("Native backend not available");
  native.absmNative(a, out);
};

export const expmNative = (a: Float32Array, out: Float32Array): void => {
  if (!native) throw new Error("Native backend not available");
  native.expmNative(a, out);
};

export const logmNative = (a: Float32Array, out: Float32Array): void => {
  if (!native) throw new Error("Native backend not available");
  native.logmNative(a, out);
};

export const transposeNative = (a: Float32Array, rows: number, cols: number, out: Float32Array): void => {
  if (!native) throw new Error("Native backend not available");
  native.transposeNative(a, rows, cols, out);
};

export const eluNative = (input: Float32Array, alpha: number, outRes: Float32Array, outGrad: Float32Array): void => {
  if (!native) throw new Error("Native backend not available");
  native.eluNativeInto(input, alpha, outRes, outGrad);
};

export const thresholdNative = (input: Float32Array, threshold: number, leak: number, outRes: Float32Array, outGrad: Float32Array): void => {
  if (!native) throw new Error("Native backend not available");
  native.thresholdNativeInto(input, threshold, leak, outRes, outGrad);
};



export const seluNative = (input: Float32Array, outRes: Float32Array, outGrad: Float32Array): void => {
  if (!native) throw new Error("Native backend not available");
  native.seluNativeInto(input, outRes, outGrad);
};

export const softplusNative = (input: Float32Array, outRes: Float32Array, outGrad: Float32Array): void => {
  if (!native) throw new Error("Native backend not available");
  native.softplusNativeInto(input, outRes, outGrad);
};

export const softsignNative = (input: Float32Array, outRes: Float32Array, outGrad: Float32Array): void => {
  if (!native) throw new Error("Native backend not available");
  native.softsignNativeInto(input, outRes, outGrad);
};

export const swishNative = (input: Float32Array, outRes: Float32Array, outGrad: Float32Array): void => {
  if (!native) throw new Error("Native backend not available");
  native.swishNativeInto(input, outRes, outGrad);
};

export const geluNative = (input: Float32Array, outRes: Float32Array, outGrad: Float32Array): void => {
  if (!native) throw new Error("Native backend not available");
  native.geluNativeInto(input, outRes, outGrad);
};

export const mishNative = (input: Float32Array, outRes: Float32Array, outGrad: Float32Array): void => {
  if (!native) throw new Error("Native backend not available");
  native.mishNativeInto(input, outRes, outGrad);
};

export const hardSigmoidNative = (input: Float32Array, outRes: Float32Array, outGrad: Float32Array): void => {
  if (!native) throw new Error("Native backend not available");
  native.hardSigmoidNativeInto(input, outRes, outGrad);
};

export const hardSwishNative = (input: Float32Array, outRes: Float32Array, outGrad: Float32Array): void => {
  if (!native) throw new Error("Native backend not available");
  native.hardSwishNativeInto(input, outRes, outGrad);
};

export const dotSumNative = (a: Float32Array): number => {
  if (!native) throw new Error("Native backend not available");
  return native.dotSumNative(a);
};

export const dotSubNative = (a: Float32Array): number => {
  if (!native) throw new Error("Native backend not available");
  return native.dotSubNative(a);
};

export const dotMulNative = (a: Float32Array): number => {
  if (!native) throw new Error("Native backend not available");
  return native.dotMulNative(a);
};

export const dotDivNative = (a: Float32Array): number => {
  if (!native) throw new Error("Native backend not available");
  return native.dotDivNative(a);
};

export const sumSquaresNative = (a: Float32Array): number => {
  if (!native) throw new Error("Native backend not available");
  return native.sumSquaresNative(a);
};

export const scaleVectorNative = (a: Float32Array, scale: number): void => {
  if (!native) throw new Error("Native backend not available");
  native.scaleVectorNative(a, scale);
};

export const getNativeBpeTrainer = (): any => {
  if (!native) return null;
  return native.NativeBpeTrainer;
};
