import { BaseLayer, LayerConfig, type ForwardOptions } from "../base/BaseLayer.js";
import { Matrix, mj, engine } from "@oxide-js/core";
import { isNativeAvailable, batchNormalizationForwardNative, batchNormalizationBackwardNative } from "../rust_backend.js";

export interface BatchNormalizationConfig extends LayerConfig {
  epsilon?: number;
  momentum?: number;
}

/**
 * Helper function to perform batch normalization with custom backward pass recorded on tape
 */
function batchNorm(
  inputs: Matrix,
  gamma: Matrix,
  beta: Matrix,
  movingMean: Matrix,
  movingVariance: Matrix,
  epsilon: number,
  momentum: number,
  training: boolean
): Matrix {
  const [rows, cols] = inputs._shape;
  const resultData = new Float32Array(rows * cols);
  const meanData = new Float32Array(cols);
  const invStdData = new Float32Array(cols);

  if (isNativeAvailable()) {
    batchNormalizationForwardNative(
      inputs._data,
      gamma._data,
      beta._data,
      movingMean._data,
      movingVariance._data,
      epsilon,
      momentum,
      training,
      resultData,
      meanData,
      invStdData
    );
  } else {
    const inputsData = inputs._data;
    const gammaData = gamma._data;
    const betaData = beta._data;
    const movingMeanData = movingMean._data;
    const movingVarianceData = movingVariance._data;

    if (training) {
      for (let j = 0; j < cols; j++) {
        let sum = 0;
        for (let i = 0; i < rows; i++) {
          sum += inputsData[i * cols + j];
        }
        const m = sum / rows;
        meanData[j] = m;

        let varSum = 0;
        for (let i = 0; i < rows; i++) {
          const diff = inputsData[i * cols + j] - m;
          varSum += diff * diff;
        }
        const variance = varSum / rows;
        const istd = 1 / Math.sqrt(variance + epsilon);
        invStdData[j] = istd;

        // Update moving statistics in-place
        movingMeanData[j] = movingMeanData[j] * momentum + m * (1 - momentum);
        movingVarianceData[j] = movingVarianceData[j] * momentum + variance * (1 - momentum);
      }
    } else {
      for (let j = 0; j < cols; j++) {
        meanData[j] = movingMeanData[j];
        const variance = movingVarianceData[j];
        invStdData[j] = 1 / Math.sqrt(variance + epsilon);
      }
    }

    for (let i = 0; i < rows; i++) {
      const rowOffset = i * cols;
      for (let j = 0; j < cols; j++) {
        const m = meanData[j];
        const istd = invStdData[j];
        const xCentered = inputsData[rowOffset + j] - m;
        const xNorm = xCentered * istd;
        resultData[rowOffset + j] = xNorm * gammaData[j] + betaData[j];
      }
    }
  }

  const res = Matrix.fromFlat(resultData, [rows, cols]);

  engine.record(
    [inputs, gamma, beta],
    [res],
    (grad: Matrix) => {
      const gradIn = new Float32Array(rows * cols);
      const gradGamma = new Float32Array(cols);
      const gradBeta = new Float32Array(cols);

      if (isNativeAvailable()) {
        batchNormalizationBackwardNative(
          grad._data,
          inputs._data,
          meanData,
          invStdData,
          gamma._data,
          training,
          gradIn,
          gradGamma,
          gradBeta
        );
      } else {
        const gradOutData = grad._data;
        const inputsData = inputs._data;
        const gammaData = gamma._data;

        if (training) {
          for (let j = 0; j < cols; j++) {
            const m = meanData[j];
            const istd = invStdData[j];

            let sumDhat = 0;
            let sumDhatXhat = 0;

            for (let i = 0; i < rows; i++) {
              const dy = gradOutData[i * cols + j];
              const xhat = (inputsData[i * cols + j] - m) * istd;
              const dhat = dy * gammaData[j];

              sumDhat += dhat;
              sumDhatXhat += dhat * xhat;

              gradGamma[j] += dy * xhat;
              gradBeta[j] += dy;
            }

            const meanDhat = sumDhat / rows;
            const meanDhatXhat = sumDhatXhat / rows;

            for (let i = 0; i < rows; i++) {
              const xhat = (inputsData[i * cols + j] - m) * istd;
              const dhat = gradOutData[i * cols + j] * gammaData[j];
              gradIn[i * cols + j] = istd * (dhat - meanDhat - xhat * meanDhatXhat);
            }
          }
        } else {
          for (let j = 0; j < cols; j++) {
            const m = meanData[j];
            const istd = invStdData[j];

            for (let i = 0; i < rows; i++) {
              const dy = gradOutData[i * cols + j];
              const xhat = (inputsData[i * cols + j] - m) * istd;

              gradGamma[j] += dy * xhat;
              gradBeta[j] += dy;

              gradIn[i * cols + j] = dy * gammaData[j] * istd;
            }
          }
        }
      }

      return [
        Matrix.fromFlat(gradIn, [rows, cols]),
        Matrix.fromFlat(gradGamma, [1, cols]),
        Matrix.fromFlat(gradBeta, [1, cols])
      ];
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
  protected compute(inputs: Matrix, options?: ForwardOptions): Matrix {
    const training = options?.training ?? this.training;

    const gamma = this.getParameter("gamma");
    const beta = this.getParameter("beta");
    const movingMean = this.getParameter("movingMean");
    const movingVariance = this.getParameter("movingVariance");

    if (!gamma || !beta || !movingMean || !movingVariance) {
      throw new Error("[BatchNormalization] Parameter belum diinisialisasi. Jalankan build() terlebih dahulu.");
    }

    return batchNorm(inputs, gamma, beta, movingMean, movingVariance, this.epsilon, this.momentum, training);
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
