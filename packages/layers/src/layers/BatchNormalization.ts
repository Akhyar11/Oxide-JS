import { BaseLayer, LayerConfig } from "../base/BaseLayer.js";
import { Matrix, mj, engine } from "@oxide-js/core";

export interface BatchNormalizationConfig extends LayerConfig {
  epsilon?: number;
  momentum?: number;
}

/**
 * Helper to broadcast a row vector [1, cols] to [rows, cols]
 */
function broadcastRow(rowVec: Matrix, shape: [number, number]): Matrix {
  const [rows, cols] = shape;
  if (rowVec._shape[0] !== 1 || rowVec._shape[1] !== cols) {
    throw new Error(`[BatchNormalization] Invalid shape for row broadcasting: expected [1, ${cols}], got [${rowVec._shape}]`);
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

export class BatchNormalization extends BaseLayer {
  public epsilon: number;
  public momentum: number;

  constructor(config?: BatchNormalizationConfig) {
    super(config || {});
    this.epsilon = config?.epsilon ?? 1e-5;
    this.momentum = config?.momentum ?? 0.99;
  }

  /**
   * Menghitung output shape logis [batch, features]
   */
  public computeOutputShape(inputShape: number[]): number[] {
    return [...inputShape];
  }

  /**
   * Menginisialisasi parameter gamma, beta, movingMean, dan movingVariance
   */
  public build(inputShape: number[]): void {
    if (this.isBuilt) return;

    this.inputShape = [...inputShape];
    this.outputShape = this.computeOutputShape(inputShape);

    const features = inputShape[inputShape.length - 1] ?? 1;

    // Trainable Parameters
    const gamma = mj.ones([1, features]);
    const beta = mj.zeros([1, features]);

    // Non-Trainable Parameters (Moving Statistics)
    const movingMean = mj.zeros([1, features]);
    const movingVariance = mj.ones([1, features]);

    this.addParameter("gamma", gamma, true);
    this.addParameter("beta", beta, true);
    this.addParameter("movingMean", movingMean, false);
    this.addParameter("movingVariance", movingVariance, false);

    this.isBuilt = true;
  }

  /**
   * Forward Pass matematika Batch Normalization
   */
  protected compute(inputs: Matrix, isTraining?: boolean): Matrix {
    const [rows, cols] = inputs._shape;
    const training = isTraining ?? this.training;

    const gamma = this.getParameter("gamma");
    const beta = this.getParameter("beta");
    const movingMean = this.getParameter("movingMean");
    const movingVariance = this.getParameter("movingVariance");

    if (!gamma || !beta || !movingMean || !movingVariance) {
      throw new Error("[BatchNormalization] Parameter belum diinisialisasi. Jalankan build() terlebih dahulu.");
    }

    let mean: Matrix;
    let variance: Matrix;

    if (training) {
      // 1. Hitung mean batch (axis 0)
      const sum = mj.sumAxis(inputs, 0);
      mean = mj.mul(sum, 1 / rows); // shape: [1, cols]

      // 2. Centered inputs: inputs - mean
      const meanBC = broadcastRow(mean, [rows, cols]);
      const xCentered = mj.sub(inputs, meanBC);

      // 3. Hitung variance batch
      const xCenteredSq = mj.mul(xCentered, xCentered);
      const varSum = mj.sumAxis(xCenteredSq, 0);
      variance = mj.mul(varSum, 1 / rows); // shape: [1, cols]

      // 4. Update Moving Statistics in-place (tanpa masuk graph autodiff)
      for (let i = 0; i < cols; i++) {
        movingMean._data[i] = movingMean._data[i] * this.momentum + mean._data[i] * (1 - this.momentum);
        movingVariance._data[i] = movingVariance._data[i] * this.momentum + variance._data[i] * (1 - this.momentum);
      }
    } else {
      // Evaluasi/Inference menggunakan Moving Statistics yang sudah di-track
      mean = movingMean;
      variance = movingVariance;
    }

    // 5. Normalisasi
    const meanBC = broadcastRow(mean, [rows, cols]);
    const xCentered = mj.sub(inputs, meanBC);

    const invStd = mj.pow(mj.add(variance, this.epsilon), -0.5); // shape: [1, cols]
    const invStdBC = broadcastRow(invStd, [rows, cols]);
    const xNorm = mj.mul(xCentered, invStdBC);

    // 6. Skala dan geser
    const gammaBC = broadcastRow(gamma, [rows, cols]);
    const betaBC = broadcastRow(beta, [rows, cols]);

    return mj.add(mj.mul(xNorm, gammaBC), betaBC);
  }

  /**
   * Konfigurasi Keras
   */
  public getConfig(): Record<string, any> {
    return {
      ...super.getConfig(),
      epsilon: this.epsilon,
      momentum: this.momentum
    };
  }
}
