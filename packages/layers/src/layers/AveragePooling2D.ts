import { BaseLayer, LayerConfig } from "../base/BaseLayer.js";
import { Matrix, engine } from "@oxide-js/core";
import { isNativeAvailable, averagePooling2DForwardNative, averagePooling2DBackwardNative } from "../rust_backend.js";

export interface AveragePooling2DConfig extends LayerConfig {
  poolSize?: number | [number, number];
  strides?: number | [number, number];
  padding?: "valid" | "same";
  imageShape?: [number, number]; // [height, width]
  inputDim?: number; // number of channels
}

/**
 * Helper function to perform 2D average pooling with custom backward pass recorded on tape
 */
function averagePooling2DForward(
  inputs: Matrix,
  batchSize: number,
  height: number,
  width: number,
  channels: number,
  poolRows: number,
  poolCols: number,
  strideRows: number,
  strideCols: number,
  padding: "valid" | "same"
): Matrix {
  const B = batchSize;
  const H = height;
  const W = width;
  const C = channels;

  let H_out: number;
  let W_out: number;
  let padTop = 0;
  let padLeft = 0;

  if (padding === "valid") {
    H_out = Math.floor((H - poolRows) / strideRows) + 1;
    W_out = Math.floor((W - poolCols) / strideCols) + 1;
  } else {
    H_out = Math.ceil(H / strideRows);
    W_out = Math.ceil(W / strideCols);
    
    const totalPaddingRows = (H_out - 1) * strideRows + poolRows - H;
    padTop = Math.floor(totalPaddingRows / 2);

    const totalPaddingCols = (W_out - 1) * strideCols + poolCols - W;
    padLeft = Math.floor(totalPaddingCols / 2);
  }

  const outRows = B * H_out * W_out;
  const outCols = C;
  const outData = new Float32Array(outRows * outCols);
  const inputsData = inputs._data;

  // Pre-compute count of valid entries per window to support exact padding division
  const windowCounts = new Int32Array(outRows);

  if (isNativeAvailable()) {
    averagePooling2DForwardNative(
      inputsData,
      B,
      H,
      W,
      C,
      poolRows,
      poolCols,
      strideRows,
      strideCols,
      padTop,
      padLeft,
      H_out,
      W_out,
      outData,
      windowCounts
    );
  } else {
    for (let b = 0; b < B; b++) {
      for (let i = 0; i < H_out; i++) {
        for (let j = 0; j < W_out; j++) {
          const outRowIdx = b * H_out * W_out + i * W_out + j;
          const hStart = i * strideRows - padTop;
          const wStart = j * strideCols - padLeft;

          let count = 0;
          for (let pr = 0; pr < poolRows; pr++) {
            const hIdx = hStart + pr;
            if (hIdx >= 0 && hIdx < H) {
              for (let pc = 0; pc < poolCols; pc++) {
                const wIdx = wStart + pc;
                if (wIdx >= 0 && wIdx < W) {
                  count++;
                }
              }
            }
          }
          windowCounts[outRowIdx] = count;

          for (let c = 0; c < C; c++) {
            let sum = 0;
            for (let pr = 0; pr < poolRows; pr++) {
              const hIdx = hStart + pr;
              if (hIdx >= 0 && hIdx < H) {
                for (let pc = 0; pc < poolCols; pc++) {
                  const wIdx = wStart + pc;
                  if (wIdx >= 0 && wIdx < W) {
                    sum += inputsData[(b * H * W + hIdx * W + wIdx) * C + c];
                  }
                }
              }
            }
            outData[outRowIdx * C + c] = count > 0 ? sum / count : 0;
          }
        }
      }
    }
  }

  const result = Matrix.fromFlat(outData, [outRows, outCols]);

  engine.record(
    [inputs],
    [result],
    (grad: Matrix) => {
      const gradInputsData = new Float32Array(B * H * W * C);
      const gradOutData = grad._data;

      if (isNativeAvailable()) {
        averagePooling2DBackwardNative(
          gradOutData,
          windowCounts,
          B,
          H,
          W,
          C,
          poolRows,
          poolCols,
          strideRows,
          strideCols,
          padTop,
          padLeft,
          H_out,
          W_out,
          gradInputsData
        );
      } else {
        for (let b = 0; b < B; b++) {
          for (let i = 0; i < H_out; i++) {
            for (let j = 0; j < W_out; j++) {
              const outRowIdx = b * H_out * W_out + i * W_out + j;
              const hStart = i * strideRows - padTop;
              const wStart = j * strideCols - padLeft;
              const count = windowCounts[outRowIdx];

              if (count > 0) {
                for (let c = 0; c < C; c++) {
                  const val = gradOutData[outRowIdx * C + c] / count;
                  for (let pr = 0; pr < poolRows; pr++) {
                    const hIdx = hStart + pr;
                    if (hIdx >= 0 && hIdx < H) {
                      for (let pc = 0; pc < poolCols; pc++) {
                        const wIdx = wStart + pc;
                        if (wIdx >= 0 && wIdx < W) {
                          gradInputsData[(b * H * W + hIdx * W + wIdx) * C + c] += val;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }

      return [Matrix.fromFlat(gradInputsData, [B * H * W, C])];
    },
    { saveInput: false, saveOutput: false }
  );

  return result;
}

export class AveragePooling2D extends BaseLayer {
  public poolSize: [number, number];
  public strides: [number, number];
  public padding: "valid" | "same";
  public imageShape?: [number, number];
  public inputDim?: number;

  constructor(config: AveragePooling2DConfig = {}) {
    super(config);
    
    const pSize = config.poolSize ?? 2;
    this.poolSize = Array.isArray(pSize) ? pSize : [pSize, pSize];

    const str = config.strides ?? this.poolSize;
    this.strides = Array.isArray(str) ? str : [str, str];

    this.padding = config.padding ?? "valid";
    this.imageShape = config.imageShape;
    this.inputDim = config.inputDim;
  }

  /**
   * Validasi bentuk input spesifik untuk 2D matriks sekuensial AveragePooling2D
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

    const pixelsPerSample = this.imageShape![0] * this.imageShape![1];
    if (actualRows % pixelsPerSample !== 0) {
      throw new Error(
        `[${this.name}] Input shape mismatch. Total rows (${actualRows}) must be a multiple of H * W (${pixelsPerSample}).`
      );
    }
  }

  /**
   * Menghitung output shape logis [batch * H_out * W_out, inputDim]
   */
  public computeOutputShape(inputShape: number[]): number[] {
    let batch = inputShape[0] ?? -1;
    let H = 1;
    let W = 1;

    if (inputShape.length === 4) {
      batch = inputShape[0] ?? -1;
      H = inputShape[1] ?? 1;
      W = inputShape[2] ?? 1;
    } else if (inputShape.length === 2) {
      if (this.imageShape) {
        H = this.imageShape[0];
        W = this.imageShape[1];
        batch = batch === -1 ? -1 : Math.floor(inputShape[0] / (H * W));
      } else {
        throw new Error("[AveragePooling2D] 'imageShape' harus ditentukan dalam config jika inputShape 2D.");
      }
    }

    const poolRows = this.poolSize[0];
    const poolCols = this.poolSize[1];
    const strideRows = this.strides[0];
    const strideCols = this.strides[1];

    const H_out = this.padding === "valid"
      ? Math.floor((H - poolRows) / strideRows) + 1
      : Math.ceil(H / strideRows);

    const W_out = this.padding === "valid"
      ? Math.floor((W - poolCols) / strideCols) + 1
      : Math.ceil(W / strideCols);

    const channels = inputShape[inputShape.length - 1];

    if (batch === -1) {
      return [-1, channels];
    }
    return [batch * H_out * W_out, channels];
  }

  /**
   * Menginisialisasi parameter
   */
  public build(inputShape: number[]): void {
    if (this.isBuilt) return;

    this.inputShape = [...inputShape];

    let h = this.imageShape ? this.imageShape[0] : undefined;
    let w = this.imageShape ? this.imageShape[1] : undefined;
    let inCols = inputShape[inputShape.length - 1] ?? 1;

    if (inputShape.length === 4) {
      h = inputShape[1];
      w = inputShape[2];
      inCols = inputShape[3];
    } else if (inputShape.length === 2) {
      if (h === undefined || w === undefined) {
        throw new Error("[AveragePooling2D] 'imageShape' (sebagai [height, width]) harus ditentukan dalam config atau inputShape harus berupa 4D [batch, height, width, channels].");
      }
    }

    this.imageShape = [h!, w!];
    this.inputDim = inCols;

    this.outputShape = this.computeOutputShape(inputShape);
    this.isBuilt = true;
  }

  /**
   * Forward Pass matematika layer AveragePooling2D
   */
  protected compute(inputs: Matrix, isTraining?: boolean): Matrix {
    const totalRows = inputs._shape[0];
    const H = this.imageShape![0];
    const W = this.imageShape![1];
    const C = this.inputDim!;
    const B = Math.floor(totalRows / (H * W));

    const poolRows = this.poolSize[0];
    const poolCols = this.poolSize[1];
    const strideRows = this.strides[0];
    const strideCols = this.strides[1];

    return averagePooling2DForward(
      inputs,
      B,
      H,
      W,
      C,
      poolRows,
      poolCols,
      strideRows,
      strideCols,
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
      imageShape: this.imageShape,
      inputDim: this.inputDim
    };
  }
}
