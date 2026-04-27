import { Cost, Optimzier, OptimzierType, StatusLayer } from "../@types/type";
import mj from "../math";
import Matrix from "../matrix";
import { isNativeAvailable, layerNormNative, layerNormBackwardNative } from "../math/rust_backend";
import setOptimizer from "../utils/setOptimizer";
import { MemoryManager, WorkspaceConfig } from "../utils/memory";

/**
 * Layer Normalization
 * 
 * Menormalkan output dari sublayer (misal: Attention) agar training lebih stabil.
 * Rumus: y = ((x - mean) / sqrt(var + epsilon)) * gamma + beta
 * 
 * gamma dan beta adalah parameter yang di-train.
 */
export default class LayerNormalization {
  name = "layer normalization";
  units: number;
  gamma: Matrix;
  beta: Matrix;
  status: StatusLayer;
  params: number;
  inputShape: [number, number] = [0, 0];
  outputShape: [number, number] = [0, 0];
  loss: number = 0;
  clipGradient: number | boolean = 5.0;
  memoryConfig: WorkspaceConfig;

  private epsilon = 1e-5;
  private input: Matrix = mj.matrix([]);
  private alpha: number = 0.01;
  private optimizerName: Optimzier = "sgd";
  private optimizerGamma: OptimzierType;
  private optimizerBeta: OptimzierType;

  private resultData: any = new Float32Array(0);
  private normalizedData: any = new Float32Array(0);
  private meanData: any = new Float32Array(0);
  private stdData: any = new Float32Array(0);
  private evalBuffers: Record<string, any> = {};

  private dGammaData: any = new Float32Array(0);
  private dBetaData: any = new Float32Array(0);
  private dxData: any = new Float32Array(0);

  private resultBuffer: Matrix;
  private normalized: Matrix;
  private mean: Matrix;
  private std: Matrix;
  private dGammaBuffer: Matrix;
  private dBetaBuffer: Matrix;
  private dxBuffer: Matrix;

  constructor({
    units,
    status = "norm",
    alpha = 0.01,
    optimizer = "sgd",
    clipGradient,
    memoryConfig = {},
  }: {
    units: number;
    status?: StatusLayer;
    alpha?: number;
    optimizer?: Optimzier;
    clipGradient?: number | boolean;
    memoryConfig?: WorkspaceConfig;
  }) {
    this.units = units;
    this.status = status;
    this.alpha = alpha;
    this.optimizerName = optimizer;
    this.clipGradient = clipGradient ?? 5.0;
    this.gamma = mj.ones([units, 1]);
    this.beta = mj.zeros([units, 1]);
    this.params = units * 2;
    this.optimizerGamma = setOptimizer(this.optimizerName, this.gamma._shape, this.alpha);
    this.optimizerBeta = setOptimizer(this.optimizerName, this.beta._shape, this.alpha);
    this.memoryConfig = memoryConfig;
    this.resultBuffer = mj.matrix([]);
    this.normalized = mj.matrix([]);
    this.mean = mj.matrix([]);
    this.std = mj.matrix([]);
    this.dGammaBuffer = mj.matrix([]);
    this.dBetaBuffer = mj.matrix([]);
    this.dxBuffer = mj.matrix([]);
  }

  save() {
    return {
      name: this.name,
      status: this.status,
      units: this.units,
      clipGradient: this.clipGradient,
      gamma: this.gamma._value,
      beta: this.beta._value,
    };
  }

  load(gamma: number[][], beta: number[][], clipGradient?: number | boolean): void {
    this.gamma._value = gamma;
    this.gamma._shape = [gamma.length, gamma[0]?.length ?? 0];
    this.beta._value = beta;
    this.beta._shape = [beta.length, beta[0]?.length ?? 0];
    this.units = this.gamma._shape[0];
    this.params = this.units * 2;
    if (clipGradient !== undefined) this.clipGradient = clipGradient;
    this.optimizerGamma = setOptimizer(this.optimizerName, this.gamma._shape, this.alpha);
    this.optimizerBeta = setOptimizer(this.optimizerName, this.beta._shape, this.alpha);
  }

  forward(x: Matrix, options?: { workspace?: "train" | "eval" }): Matrix {
    const [rows, cols] = x._shape;
    this.input = x;
    this.inputShape = [rows, cols];
    this.outputShape = [rows, cols];
    this.ensureForwardBuffers(rows, cols, options?.workspace);

    if (isNativeAvailable()) {
      layerNormNative(
        x._data,
        this.gamma._data,
        this.beta._data,
        rows,
        cols,
        this.epsilon,
        this.resultBuffer._data,
        this.normalized._data,
        this.mean._data,
        this.std._data
      );
      return this.resultBuffer;
    }

    const result = this.resultBuffer._data;
    const normalizedData = this.normalized._data;
    const means = this.mean._data;
    const stds = this.std._data;

    const xData = x._data;

    // Hitung mean dan variance per kolom (per token)
    for (let j = 0; j < cols; j++) {
      let sum = 0;
      for (let i = 0; i < rows; i++) {
        sum += xData[i * cols + j];
      }
      const m = sum / rows;
      means[j] = m;

      let sumSq = 0;
      for (let i = 0; i < rows; i++) {
        const diff = xData[i * cols + j] - m;
        sumSq += diff * diff;
      }
      stds[j] = Math.sqrt(sumSq / rows + this.epsilon);
    }

    // Normalize
    const gData = this.gamma._data;
    const bData = this.beta._data;

    for (let j = 0; j < cols; j++) {
      const s = stds[j];
      const m = means[j];
      for (let i = 0; i < rows; i++) {
        const idx = i * cols + j;
        const norm = (xData[idx] - m) / s;
        normalizedData[idx] = norm;
        result[idx] = norm * gData[i] + bData[i];
      }
    }

    return this.resultBuffer;
  }

  backward(_y: Matrix, err: Matrix): Matrix {
    const [rows, cols] = err._shape;
    const [fwdRows, fwdCols] = this.inputShape;
    if (rows !== fwdRows || cols !== fwdCols) {
      throw new Error(`LayerNormalization.backward: err shape [${rows}x${cols}] does not match forward input shape [${fwdRows}x${fwdCols}]`);
    }
    this.ensureBackwardBuffers(rows, cols);

    const dGamma = this.dGammaBuffer._data;
    const dBeta = this.dBetaBuffer._data;
    const dx = this.dxBuffer._data;

    const errData = err._data;
    const normData = this.normalized._data;
    const gData = this.gamma._data;
    const stdData = this.std._data;

    if (isNativeAvailable()) {
      layerNormBackwardNative(
        errData,
        normData,
        gData,
        rows,
        cols,
        stdData,
        dGamma,
        dBeta,
        dx
      );
    } else {
      // 1. Hitung gradien untuk gamma dan beta
      for (let i = 0; i < rows; i++) {
        let sumG = 0;
        let sumB = 0;
        for (let j = 0; j < cols; j++) {
          const idx = i * cols + j;
          sumG += errData[idx] * normData[idx];
          sumB += errData[idx];
        }
        dGamma[i] = sumG;
        dBeta[i] = sumB;
      }

      // 2. Hitung gradien ke input (dx)
      for (let j = 0; j < cols; j++) {
        const s = stdData[j];
        let sum1 = 0;
        let sum2 = 0;
        for (let i = 0; i < rows; i++) {
          const idx = i * cols + j;
          const e = errData[idx] * gData[i];
          sum1 += e;
          sum2 += e * normData[idx];
        }

        for (let i = 0; i < rows; i++) {
          const idx = i * cols + j;
          dx[idx] = (gData[i] * errData[idx] - (sum1 / rows) - (normData[idx] * sum2 / rows)) / s;
        }
      }
    }

    // [Update]: Update gamma dan beta menggunakan optimizer
    const gGrad = this.dGammaBuffer;
    const bGrad = this.dBetaBuffer;

    // Gradient clipping untuk LN parameters
    if (this.clipGradient !== false) {
      const limit = typeof this.clipGradient === "number" ? this.clipGradient : 5.0;
      this.clipGradients(gGrad, limit);
      this.clipGradients(bGrad, limit);
    }

    const gUpdate = this.optimizerGamma.calculate(gGrad, this.alpha);
    const bUpdate = this.optimizerBeta.calculate(bGrad, this.alpha);

    this.gamma.subInPlace(gUpdate);
    this.beta.subInPlace(bUpdate);

    return this.dxBuffer;
  }

  private clipGradients(m: Matrix, limit: number) {
    const data = m._data;
    for (let i = 0; i < data.length; i++) {
      if (data[i] > limit) data[i] = limit;
      else if (data[i] < -limit) data[i] = -limit;
    }
  }

  private ensureForwardBuffers(rows: number, cols: number, workspace: "train" | "eval" = "train"): void {
    const size = rows * cols;
    if (workspace === "eval") {
      this.evalBuffers.resultData = MemoryManager.ensureCapacity(this.evalBuffers.resultData || new Float32Array(0), size, this.memoryConfig) as any;
      this.evalBuffers.normalizedData = MemoryManager.ensureCapacity(this.evalBuffers.normalizedData || new Float32Array(0), size, this.memoryConfig) as any;
      this.evalBuffers.meanData = MemoryManager.ensureCapacity(this.evalBuffers.meanData || new Float32Array(0), cols, this.memoryConfig) as any;
      this.evalBuffers.stdData = MemoryManager.ensureCapacity(this.evalBuffers.stdData || new Float32Array(0), cols, this.memoryConfig) as any;

      this.resultData = this.evalBuffers.resultData;
      this.normalizedData = this.evalBuffers.normalizedData;
      this.meanData = this.evalBuffers.meanData;
      this.stdData = this.evalBuffers.stdData;
    } else {
      this.resultData = MemoryManager.ensureCapacity(this.resultData, size, this.memoryConfig) as any;
      this.normalizedData = MemoryManager.ensureCapacity(this.normalizedData, size, this.memoryConfig) as any;
      this.meanData = MemoryManager.ensureCapacity(this.meanData, cols, this.memoryConfig) as any;
      this.stdData = MemoryManager.ensureCapacity(this.stdData, cols, this.memoryConfig) as any;
    }

    this.resultBuffer = Matrix.fromFlat(this.resultData.subarray(0, size) as any, [rows, cols]);
    this.normalized = Matrix.fromFlat(this.normalizedData.subarray(0, size) as any, [rows, cols]);
    this.mean = Matrix.fromFlat(this.meanData.subarray(0, cols) as any, [1, cols]);
    this.std = Matrix.fromFlat(this.stdData.subarray(0, cols) as any, [1, cols]);
  }

  private ensureBackwardBuffers(rows: number, cols: number): void {
    const size = rows * cols;
    this.dGammaData = MemoryManager.ensureCapacity(this.dGammaData, this.units, this.memoryConfig) as any;
    this.dBetaData = MemoryManager.ensureCapacity(this.dBetaData, this.units, this.memoryConfig) as any;
    this.dxData = MemoryManager.ensureCapacity(this.dxData, size, this.memoryConfig) as any;

    this.dGammaBuffer = Matrix.fromFlat(this.dGammaData.subarray(0, this.units) as any, [this.units, 1]);
    this.dBetaBuffer = Matrix.fromFlat(this.dBetaData.subarray(0, this.units) as any, [this.units, 1]);
    this.dxBuffer = Matrix.fromFlat(this.dxData.subarray(0, size) as any, [rows, cols]);
  }

  compile({ alpha, optimizer, error, clipGradient }: { alpha?: number; optimizer?: Optimzier; error?: Cost; clipGradient?: number | boolean }) {
    if (alpha !== undefined) this.alpha = alpha;
    if (optimizer !== undefined) {
      this.optimizerName = optimizer;
      this.optimizerGamma = setOptimizer(optimizer, this.gamma._shape, this.alpha);
      this.optimizerBeta = setOptimizer(optimizer, this.beta._shape, this.alpha);
    }
    if (clipGradient !== undefined) this.clipGradient = clipGradient;
  }

  releaseWorkspace(): void {
    this.evalBuffers = {};
    this.resultData = new Float32Array(0);
    this.normalizedData = new Float32Array(0);
    this.meanData = new Float32Array(0);
    this.stdData = new Float32Array(0);
    this.dGammaData = new Float32Array(0);
    this.dBetaData = new Float32Array(0);
    this.dxData = new Float32Array(0);

    this.resultBuffer = mj.matrix([]);
    this.normalized = mj.matrix([]);
    this.mean = mj.matrix([]);
    this.std = mj.matrix([]);
    this.dGammaBuffer = mj.matrix([]);
    this.dBetaBuffer = mj.matrix([]);
    this.dxBuffer = mj.matrix([]);
  }

  dispose(): void {
    this.releaseWorkspace();
    this.optimizerGamma?.dispose?.();
    this.optimizerBeta?.dispose?.();
    (this as any).gamma = null;
    (this as any).beta = null;
    (this as any).optimizerGamma = null;
    (this as any).optimizerBeta = null;
  }
}
