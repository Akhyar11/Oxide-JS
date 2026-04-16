import { Cost, Optimzier, OptimzierType, StatusLayer } from "../@types/type";
import mj from "../math";
import Matrix from "../matrix";
import { isNativeAvailable, layerNormNative } from "../math/rust_backend";
import setOptimizer from "../utils/setOptimizer";

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

  private epsilon = 1e-5;
  private input: Matrix = mj.matrix([]);
  private normalized: Matrix = mj.matrix([]);
  private std: Matrix = mj.matrix([]);
  private mean: Matrix = mj.matrix([]);
  private alpha: number = 0.01;
  private optimizerName: Optimzier = "sgd";
  private optimizerGamma: OptimzierType;
  private optimizerBeta: OptimzierType;

  constructor({
    units,
    status = "norm",
    alpha = 0.01,
    optimizer = "sgd"
  }: {
    units: number;
    status?: StatusLayer;
    alpha?: number;
    optimizer?: Optimzier;
  }) {
    this.units = units;
    this.status = status;
    this.alpha = alpha;
    this.optimizerName = optimizer;
    this.gamma = mj.ones([units, 1]);
    this.beta = mj.zeros([units, 1]);
    this.params = units * 2;
    this.optimizerGamma = setOptimizer(this.optimizerName, this.gamma._shape, this.alpha);
    this.optimizerBeta = setOptimizer(this.optimizerName, this.beta._shape, this.alpha);
  }

  save() {
    return {
      name: this.name,
      status: this.status,
      units: this.units,
      gamma: this.gamma._value,
      beta: this.beta._value,
    };
  }

  load(gamma: number[][], beta: number[][]): void {
    this.gamma._value = gamma;
    this.gamma._shape = [gamma.length, gamma[0]?.length ?? 0];
    this.beta._value = beta;
    this.beta._shape = [beta.length, beta[0]?.length ?? 0];
  }

  forward(x: Matrix): Matrix {
    const [rows, cols] = x._shape;
    this.input = x;
    this.inputShape = [rows, cols];
    this.outputShape = [rows, cols];

    if (isNativeAvailable()) {
      const res = new Float64Array(rows * cols);
      const norm = new Float64Array(rows * cols);
      const m = new Float64Array(cols);
      const s = new Float64Array(cols);
      layerNormNative(x._data, this.gamma._data, this.beta._data, rows, cols, this.epsilon, res, norm, m, s);
      this.normalized = Matrix.fromFlat(norm, [rows, cols]);
      this.mean = Matrix.fromFlat(m, [1, cols]);
      this.std = Matrix.fromFlat(s, [1, cols]);
      return Matrix.fromFlat(res, [rows, cols]);
    }

    const result = new Float64Array(rows * cols);
    const normalizedData = new Float64Array(rows * cols);
    const means = new Float64Array(cols);
    const vars = new Float64Array(cols);

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
      vars[j] = sumSq / rows;
    }

    this.mean = Matrix.fromFlat(means, [1, cols]);
    this.std = Matrix.fromFlat(vars.map(v => Math.sqrt(v + this.epsilon)), [1, cols]);

    // Normalize
    const gData = this.gamma._data;
    const bData = this.beta._data;
    const stdData = this.std._data;

    for (let j = 0; j < cols; j++) {
      const s = stdData[j];
      const m = means[j];
      for (let i = 0; i < rows; i++) {
        const idx = i * cols + j;
        const norm = (xData[idx] - m) / s;
        normalizedData[idx] = norm;
        result[idx] = norm * gData[i] + bData[i];
      }
    }

    this.normalized = Matrix.fromFlat(normalizedData, [rows, cols]);
    return Matrix.fromFlat(result, [rows, cols]);
  }

  backward(_y: Matrix, err: Matrix): Matrix {
    const [rows, cols] = err._shape;
    const dGamma = new Float64Array(this.units);
    const dBeta = new Float64Array(this.units);
    const dx = new Float64Array(rows * cols);

    const errData = err._data;
    const normData = this.normalized._data;
    const gData = this.gamma._data;
    const stdData = this.std._data;

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

    // [Update]: Update gamma dan beta menggunakan optimizer
    const gGrad = Matrix.fromFlat(dGamma, [this.units, 1]);
    const bGrad = Matrix.fromFlat(dBeta, [this.units, 1]);
    
    // Gradient clipping untuk LN parameters
    this.clipGradients(gGrad, 1.0);
    this.clipGradients(bGrad, 1.0);

    const gUpdate = this.optimizerGamma.calculate(gGrad, this.alpha);
    const bUpdate = this.optimizerBeta.calculate(bGrad, this.alpha);
    
    this.gamma.subInPlace(gUpdate);
    this.beta.subInPlace(bUpdate);

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

    return Matrix.fromFlat(dx, [rows, cols]);
  }

  private clipGradients(m: Matrix, limit: number) {
    const data = m._data;
    for (let i = 0; i < data.length; i++) {
      if (data[i] > limit) data[i] = limit;
      else if (data[i] < -limit) data[i] = -limit;
    }
  }

  compile({ alpha, optimizer, error }: { alpha?: number; optimizer?: Optimzier; error?: Cost }) {
    if (alpha !== undefined) this.alpha = alpha;
    if (optimizer !== undefined) {
      this.optimizerName = optimizer;
      this.optimizerGamma = setOptimizer(optimizer, this.gamma._shape, this.alpha);
      this.optimizerBeta = setOptimizer(optimizer, this.beta._shape, this.alpha);
    }
  }

  resetLoss(): void {
    this.loss = 0;
  }
}
