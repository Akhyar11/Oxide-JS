import { BaseLayer, LayerConfig } from "../base/BaseLayer.js";
import { Matrix, engine } from "@oxide-js/core";
import { isNativeAvailable, averagePooling1DForwardNative, averagePooling1DBackwardNative } from "../rust_backend.js";

export interface AveragePooling1DConfig extends LayerConfig {
  poolSize?: number;
  strides?: number;
  padding?: "valid" | "same";
  sequenceLength?: number;
  inputDim?: number;
}

/**
 * Helper function to perform 1D average pooling with custom backward pass recorded on tape
 */
function averagePooling1DForward(
  inputs: Matrix,
  batchSize: number,
  sequenceLength: number,
  inputDim: number,
  poolSize: number,
  strides: number,
  padding: "valid" | "same"
): Matrix {
  const B = batchSize;
  const L = sequenceLength;
  const C = inputDim;

  let L_out: number;
  let padLeft = 0;

  if (padding === "valid") {
    L_out = Math.floor((L - poolSize) / strides) + 1;
  } else {
    L_out = Math.ceil(L / strides);
    const totalPadding = (L_out - 1) * strides + poolSize - L;
    padLeft = Math.floor(totalPadding / 2);
  }

  const outRows = B * L_out;
  const outCols = C;
  const outData = new Float32Array(outRows * outCols);
  const inputsData = inputs._data;

  // Pre-compute count of valid entries per window to support exact padding division
  const windowCounts = new Int32Array(outRows);

  if (isNativeAvailable()) {
    averagePooling1DForwardNative(
      inputsData,
      B,
      L,
      C,
      poolSize,
      strides,
      padLeft,
      outData,
      windowCounts
    );
  } else {
    for (let b = 0; b < B; b++) {
      for (let i = 0; i < L_out; i++) {
        const outRowIdx = b * L_out + i;
        const tStart = i * strides - padLeft;

        let count = 0;
        for (let k = 0; k < poolSize; k++) {
          const t = tStart + k;
          if (t >= 0 && t < L) {
            count++;
          }
        }
        windowCounts[outRowIdx] = count;

        for (let c = 0; c < C; c++) {
          let sum = 0;
          for (let k = 0; k < poolSize; k++) {
            const t = tStart + k;
            if (t >= 0 && t < L) {
              sum += inputsData[(b * L + t) * C + c];
            }
          }
          outData[outRowIdx * C + c] = count > 0 ? sum / count : 0;
        }
      }
    }
  }

  const result = Matrix.fromFlat(outData, [outRows, outCols]);

  engine.record(
    [inputs],
    [result],
    (grad: Matrix) => {
      const gradInputsData = new Float32Array(B * L * C);
      const gradOutData = grad._data;

      if (isNativeAvailable()) {
        averagePooling1DBackwardNative(
          gradOutData,
          windowCounts,
          B,
          L,
          C,
          poolSize,
          strides,
          padLeft,
          gradInputsData
        );
      } else {
        for (let b = 0; b < B; b++) {
          for (let i = 0; i < L_out; i++) {
            const outRowIdx = b * L_out + i;
            const tStart = i * strides - padLeft;
            const count = windowCounts[outRowIdx];

            if (count > 0) {
              for (let c = 0; c < C; c++) {
                const val = gradOutData[outRowIdx * C + c] / count;
                for (let k = 0; k < poolSize; k++) {
                  const t = tStart + k;
                  if (t >= 0 && t < L) {
                    gradInputsData[(b * L + t) * C + c] += val;
                  }
                }
              }
            }
          }
        }
      }

      return [Matrix.fromFlat(gradInputsData, [B * L, C])];
    },
    { saveInput: false, saveOutput: false }
  );

  return result;
}

export class AveragePooling1D extends BaseLayer {
  public poolSize: number;
  public strides: number;
  public padding: "valid" | "same";
  public sequenceLength?: number;
  public inputDim?: number;

  constructor(config: AveragePooling1DConfig = {}) {
    super(config);
    this.poolSize = config.poolSize ?? 2;
    this.strides = config.strides ?? this.poolSize;
    this.padding = config.padding ?? "valid";
    this.sequenceLength = config.sequenceLength;
    this.inputDim = config.inputDim;
  }

  /**
   * Validasi bentuk input spesifik untuk 2D matriks sekuensial AveragePooling1D
   */
  public validateInputShape(inputs: Matrix): void {
    if (!this.isBuilt) return;

    const actualRows = inputs._shape[0];
    const actualCols = inputs._shape[1] ?? 1;

    if (actualCols !== this.inputDim) {
      throw new Error(
        `[${this.name}] Input shape mismatch. Expected input channels (inputDim) to be ${this.inputDim}, got ${actualCols}.`
      );
    }

    if (actualRows % this.sequenceLength! !== 0) {
      throw new Error(
        `[${this.name}] Input shape mismatch. Total rows (${actualRows}) must be a multiple of sequenceLength (${this.sequenceLength}).`
      );
    }
  }

  /**
   * Menghitung output shape logis [batch * L_out, inputDim]
   */
  public computeOutputShape(inputShape: number[]): number[] {
    let batch = inputShape[0] ?? -1;
    let L = inputShape.length > 1 ? inputShape[1] : 1;

    if (inputShape.length === 3) {
      batch = inputShape[0] ?? -1;
      L = inputShape[1] ?? 1;
    } else if (inputShape.length === 2) {
      if (this.sequenceLength) {
        L = this.sequenceLength;
        batch = batch === -1 ? -1 : Math.floor(inputShape[0] / L);
      } else {
        throw new Error("[AveragePooling1D] 'sequenceLength' harus ditentukan dalam config jika inputShape 2D.");
      }
    }

    const L_out = this.padding === "valid"
      ? Math.floor((L - this.poolSize) / this.strides) + 1
      : Math.ceil(L / this.strides);

    const channels = inputShape[inputShape.length - 1];

    if (batch === -1) {
      return [-1, channels];
    }
    return [batch * L_out, channels];
  }

  /**
   * Menginisialisasi parameter
   */
  public build(inputShape: number[]): void {
    if (this.isBuilt) return;

    this.inputShape = [...inputShape];

    let seqLen = this.sequenceLength;
    let inCols = inputShape[inputShape.length - 1] ?? 1;

    if (inputShape.length === 3) {
      seqLen = inputShape[1];
      inCols = inputShape[2];
    } else if (inputShape.length === 2) {
      if (seqLen === undefined) {
        throw new Error("[AveragePooling1D] 'sequenceLength' harus ditentukan dalam config atau inputShape harus berupa 3D [batch, sequenceLength, inputDim].");
      }
    }

    this.sequenceLength = seqLen;
    this.inputDim = inCols;

    this.outputShape = this.computeOutputShape(inputShape);
    this.isBuilt = true;
  }

  /**
   * Forward Pass matematika layer AveragePooling1D
   */
  protected compute(inputs: Matrix, isTraining?: boolean): Matrix {
    const totalRows = inputs._shape[0];
    const L = this.sequenceLength!;
    const C = this.inputDim!;
    const B = Math.floor(totalRows / L);

    return averagePooling1DForward(
      inputs,
      B,
      L,
      C,
      this.poolSize,
      this.strides,
      this.padding
    );
  }

  /**
   * Konfigurasi spesifik Keras
   */
  public getConfig(): Record<string, any> {
    return {
      ...super.getConfig(),
      poolSize: this.poolSize,
      strides: this.strides,
      padding: this.padding,
      sequenceLength: this.sequenceLength,
      inputDim: this.inputDim
    };
  }
}
