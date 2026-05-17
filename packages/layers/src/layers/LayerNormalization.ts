import { BaseLayer, LayerConfig } from "../base/BaseLayer.js";
import { Matrix, mj, engine } from "@oxide-js/core";

export interface LayerNormalizationConfig extends LayerConfig {
  epsilon?: number;
}

/**
 * Helper to broadcast a column vector [rows, 1] to [rows, cols]
 */
function broadcastColumn(colVec: Matrix, shape: [number, number]): Matrix {
  const [rows, cols] = shape;
  if (colVec._shape[0] !== rows || colVec._shape[1] !== 1) {
    throw new Error(`[LayerNormalization] Invalid shape for column broadcasting: expected [${rows}, 1], got [${colVec._shape}]`);
  }
  const resultData = new Float32Array(rows * cols);
  const colData = colVec._data;
  for (let i = 0; i < rows; i++) {
    const val = colData[i];
    for (let j = 0; j < cols; j++) {
      resultData[i * cols + j] = val;
    }
  }
  const res = Matrix.fromFlat(resultData, shape);

  engine.record(
    [colVec],
    [res],
    (grad: Matrix) => {
      return [mj.sumAxis(grad, 1)];
    },
    { saveInput: false, saveOutput: false }
  );

  return res;
}

/**
 * Helper to broadcast a row vector [1, cols] to [rows, cols]
 */
function broadcastRow(rowVec: Matrix, shape: [number, number]): Matrix {
  const [rows, cols] = shape;
  if (rowVec._shape[0] !== 1 || rowVec._shape[1] !== cols) {
    throw new Error(`[LayerNormalization] Invalid shape for row broadcasting: expected [1, ${cols}], got [${rowVec._shape}]`);
  }
  const resultData = new Float32Array(rows * cols);
  const rowData = rowVec._data;
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      resultData[i * cols + j] = rowData[j];
    }
  }
  const res = Matrix.fromFlat(resultData, shape);

  engine.record(
    [rowVec],
    [res],
    (grad: Matrix) => {
      return [mj.sumAxis(grad, 0)];
    },
    { saveInput: false, saveOutput: false }
  );

  return res;
}

export class LayerNormalization extends BaseLayer {
  public epsilon: number;

  constructor(config?: LayerNormalizationConfig) {
    super(config || {});
    this.epsilon = config?.epsilon ?? 1e-5;
  }

  /**
   * Menghitung output shape logis [batch, features]
   */
  public computeOutputShape(inputShape: number[]): number[] {
    return [...inputShape];
  }

  /**
   * Menginisialisasi parameter gamma dan beta
   */
  public build(inputShape: number[]): void {
    if (this.isBuilt) return;

    this.inputShape = [...inputShape];
    this.outputShape = this.computeOutputShape(inputShape);

    const features = inputShape[inputShape.length - 1] ?? 1;

    // gamma diinisialisasi dengan angka 1, beta diinisialisasi dengan angka 0
    const gamma = mj.ones([1, features]);
    const beta = mj.zeros([1, features]);

    this.addParameter("gamma", gamma, true);
    this.addParameter("beta", beta, true);

    this.isBuilt = true;
  }

  /**
   * Forward Pass matematika Layer Normalization
   */
  protected compute(inputs: Matrix, isTraining?: boolean): Matrix {
    const [rows, cols] = inputs._shape;

    const gamma = this.getParameter("gamma");
    const beta = this.getParameter("beta");

    if (!gamma || !beta) {
      throw new Error("[LayerNormalization] Bobot gamma atau beta tidak terinisialisasi. Pastikan build() sudah dijalankan.");
    }

    // 1. Mean tiap baris (axis 1)
    const sum = mj.sumAxis(inputs, 1);
    const mean = mj.mul(sum, 1 / cols); // shape: [rows, 1]

    // 2. Centered input: inputs - mean
    const meanBC = broadcastColumn(mean, [rows, cols]);
    const xCentered = mj.sub(inputs, meanBC);

    // 3. Variance tiap baris
    const xCenteredSq = mj.mul(xCentered, xCentered);
    const varSum = mj.sumAxis(xCenteredSq, 1);
    const variance = mj.mul(varSum, 1 / cols); // shape: [rows, 1]

    // 4. Standar deviasi terbalik: 1 / sqrt(var + eps) = (var + eps) ^ -0.5
    const invStd = mj.pow(mj.add(variance, this.epsilon), -0.5); // shape: [rows, 1]
    const invStdBC = broadcastColumn(invStd, [rows, cols]);

    // 5. Normalisasi inputs
    const xNorm = mj.mul(xCentered, invStdBC);

    // 6. Skala dan geser menggunakan gamma dan beta
    const gammaBC = broadcastRow(gamma, [rows, cols]);
    const betaBC = broadcastRow(beta, [rows, cols]);

    // y = xNorm * gamma + beta
    return mj.add(mj.mul(xNorm, gammaBC), betaBC);
  }

  /**
   * Konfigurasi Keras
   */
  public getConfig(): Record<string, any> {
    return {
      ...super.getConfig(),
      epsilon: this.epsilon
    };
  }
}
