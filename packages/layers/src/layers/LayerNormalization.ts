import { BaseLayer, LayerConfig, type ForwardOptions } from "../base/BaseLayer.js";
import { Matrix, mj, engine } from "@oxide-js/core";
import { isNativeAvailable, layerNormalizationForwardNative, layerNormalizationBackwardNative } from "../rust_backend.js";

export interface LayerNormalizationConfig extends LayerConfig {
  epsilon?: number;
}

/**
 * Helper function to perform layer normalization with custom backward pass recorded on tape
 */
function layerNorm(inputs: Matrix, gamma: Matrix, beta: Matrix, epsilon: number): Matrix {
  const [rows, cols] = inputs._shape;
  const resultData = new Float32Array(rows * cols);
  const meanData = new Float32Array(rows);
  const invStdData = new Float32Array(rows);

  if (isNativeAvailable()) {
    layerNormalizationForwardNative(inputs._data, gamma._data, beta._data, epsilon, resultData, meanData, invStdData);
  } else {
    const inputsData = inputs._data;
    const gammaData = gamma._data;
    const betaData = beta._data;

    for (let i = 0; i < rows; i++) {
      const rowOffset = i * cols;
      let sum = 0;
      for (let j = 0; j < cols; j++) {
        sum += inputsData[rowOffset + j];
      }
      const m = sum / cols;
      meanData[i] = m;

      let varSum = 0;
      for (let j = 0; j < cols; j++) {
        const diff = inputsData[rowOffset + j] - m;
        varSum += diff * diff;
      }
      const variance = varSum / cols;
      const istd = 1 / Math.sqrt(variance + epsilon);
      invStdData[i] = istd;

      for (let j = 0; j < cols; j++) {
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
        layerNormalizationBackwardNative(
          grad._data,
          inputs._data,
          meanData,
          invStdData,
          gamma._data,
          gradIn,
          gradGamma,
          gradBeta
        );
      } else {
        const gradOutData = grad._data;
        const inputsData = inputs._data;
        const gammaData = gamma._data;

        for (let i = 0; i < rows; i++) {
          const rowOffset = i * cols;
          const m = meanData[i];
          const istd = invStdData[i];

          let sumDhat = 0;
          let sumDhatXhat = 0;

          for (let j = 0; j < cols; j++) {
            const dy = gradOutData[rowOffset + j];
            const xhat = (inputsData[rowOffset + j] - m) * istd;
            const dhat = dy * gammaData[j];

            sumDhat += dhat;
            sumDhatXhat += dhat * xhat;

            gradGamma[j] += dy * xhat;
            gradBeta[j] += dy;
          }

          const meanDhat = sumDhat / cols;
          const meanDhatXhat = sumDhatXhat / cols;

          for (let j = 0; j < cols; j++) {
            const xhat = (inputsData[rowOffset + j] - m) * istd;
            const dhat = gradOutData[rowOffset + j] * gammaData[j];
            gradIn[rowOffset + j] = istd * (dhat - meanDhat - xhat * meanDhatXhat);
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
  protected compute(inputs: Matrix, options?: ForwardOptions): Matrix {
    const gamma = this.getParameter("gamma");
    const beta = this.getParameter("beta");

    if (!gamma || !beta) {
      throw new Error("[LayerNormalization] Bobot gamma atau beta tidak terinisialisasi. Pastikan build() sudah dijalankan.");
    }

    return layerNorm(inputs, gamma, beta, this.epsilon);
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
