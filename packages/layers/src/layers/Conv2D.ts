import { BaseLayer, LayerConfig } from "../base/BaseLayer.js";
import { Matrix, mj, engine } from "@oxide-js/core";
import { isNativeAvailable, grid2ColNative, col2GridNative } from "../rust_backend.js";

export interface Conv2DConfig extends LayerConfig {
  filters: number;
  kernelSize: number | [number, number];
  strides?: number | [number, number];
  padding?: "valid" | "same";
  activation?: string;
  useBias?: boolean;
  kernelInitializer?: string;
  biasInitializer?: string;
  imageShape?: [number, number]; // [height, width]
  inputDim?: number; // number of input channels
}

/**
 * Helper function to perform grid2col mapping with its custom backward pass recorded on tape
 */
function grid2col(
  inputs: Matrix,
  batchSize: number,
  height: number,
  width: number,
  channels: number,
  kernelRows: number,
  kernelCols: number,
  strideRows: number,
  strideCols: number,
  padding: "valid" | "same"
): { patchMatrix: Matrix; H_out: number; W_out: number } {
  const B = batchSize;
  const H = height;
  const W = width;
  const C = channels;

  let H_out: number;
  let W_out: number;
  let padTop = 0;
  let padLeft = 0;

  if (padding === "valid") {
    H_out = Math.floor((H - kernelRows) / strideRows) + 1;
    W_out = Math.floor((W - kernelCols) / strideCols) + 1;
  } else {
    H_out = Math.ceil(H / strideRows);
    W_out = Math.ceil(W / strideCols);
    
    const totalPaddingRows = (H_out - 1) * strideRows + kernelRows - H;
    padTop = Math.floor(totalPaddingRows / 2);

    const totalPaddingCols = (W_out - 1) * strideCols + kernelCols - W;
    padLeft = Math.floor(totalPaddingCols / 2);
  }

  const patchRows = B * H_out * W_out;
  const patchCols = kernelRows * kernelCols * C;
  const patchData = new Float32Array(patchRows * patchCols);
  const inputsData = inputs._data;

  if (isNativeAvailable()) {
    grid2ColNative(
      inputsData,
      B,
      H,
      W,
      C,
      kernelRows,
      kernelCols,
      strideRows,
      strideCols,
      padTop,
      padLeft,
      H_out,
      W_out,
      patchData
    );
  } else {
    for (let b = 0; b < B; b++) {
      for (let i = 0; i < H_out; i++) {
        for (let j = 0; j < W_out; j++) {
          const outRowIdx = b * H_out * W_out + i * W_out + j;
          const destOffset = outRowIdx * patchCols;

          const hStart = i * strideRows - padTop;
          const wStart = j * strideCols - padLeft;

          for (let kr = 0; kr < kernelRows; kr++) {
            const hIdx = hStart + kr;
            const kernelRowOffset = kr * kernelCols * C;

            if (hIdx >= 0 && hIdx < H) {
              for (let kc = 0; kc < kernelCols; kc++) {
                const wIdx = wStart + kc;
                const kernelColOffset = kc * C;

                if (wIdx >= 0 && wIdx < W) {
                  const srcOffset = (b * H * W + hIdx * W + wIdx) * C;
                  const kernelOffset = kernelRowOffset + kernelColOffset;
                  for (let c = 0; c < C; c++) {
                    patchData[destOffset + kernelOffset + c] = inputsData[srcOffset + c];
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  const patchMatrix = Matrix.fromFlat(patchData, [patchRows, patchCols]);

  engine.record(
    [inputs],
    [patchMatrix],
    (grad: Matrix) => {
      const gradInputsData = new Float32Array(B * H * W * C);
      const gradOutData = grad._data;

      if (isNativeAvailable()) {
        col2GridNative(
          gradOutData,
          B,
          H,
          W,
          C,
          kernelRows,
          kernelCols,
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
              const srcOffset = outRowIdx * patchCols;

              const hStart = i * strideRows - padTop;
              const wStart = j * strideCols - padLeft;

              for (let kr = 0; kr < kernelRows; kr++) {
                const hIdx = hStart + kr;
                const kernelRowOffset = kr * kernelCols * C;

                if (hIdx >= 0 && hIdx < H) {
                  for (let kc = 0; kc < kernelCols; kc++) {
                    const wIdx = wStart + kc;
                    const kernelColOffset = kc * C;

                    if (wIdx >= 0 && wIdx < W) {
                      const destOffset = (b * H * W + hIdx * W + wIdx) * C;
                      const kernelOffset = kernelRowOffset + kernelColOffset;
                      for (let c = 0; c < C; c++) {
                        gradInputsData[destOffset + c] += gradOutData[srcOffset + kernelOffset + c];
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

  return { patchMatrix, H_out, W_out };
}

export class Conv2D extends BaseLayer {
  public filters: number;
  public kernelSize: [number, number];
  public strides: [number, number];
  public padding: "valid" | "same";
  public activation: string;
  public useBias: boolean;
  public kernelInitializer: string;
  public biasInitializer: string;
  public imageShape?: [number, number];
  public inputDim?: number;

  constructor(config: Conv2DConfig) {
    super(config);
    if (config.filters === undefined || config.filters <= 0) {
      throw new Error("[Conv2D] 'filters' wajib berupa angka positif.");
    }
    if (config.kernelSize === undefined) {
      throw new Error("[Conv2D] 'kernelSize' wajib ditentukan.");
    }

    this.filters = config.filters;
    this.kernelSize = Array.isArray(config.kernelSize)
      ? config.kernelSize
      : [config.kernelSize, config.kernelSize];
    this.strides = Array.isArray(config.strides)
      ? config.strides
      : (config.strides !== undefined ? [config.strides, config.strides] : [1, 1]);

    this.padding = config.padding ?? "valid";
    this.activation = config.activation ?? "linear";
    this.useBias = config.useBias ?? true;
    this.kernelInitializer = config.kernelInitializer ?? "glorot_normal";
    this.biasInitializer = config.biasInitializer ?? "zeros";
    this.imageShape = config.imageShape;
    this.inputDim = config.inputDim;
  }

  /**
   * Validasi bentuk input spesifik untuk 2D matriks sekuensial Conv2D
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
   * Menghitung output shape logis [batch * H_out * W_out, filters]
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
        throw new Error("[Conv2D] 'imageShape' harus ditentukan dalam config jika inputShape 2D.");
      }
    }

    const kernelRows = this.kernelSize[0];
    const kernelCols = this.kernelSize[1];
    const strideRows = this.strides[0];
    const strideCols = this.strides[1];

    const H_out = this.padding === "valid"
      ? Math.floor((H - kernelRows) / strideRows) + 1
      : Math.ceil(H / strideRows);

    const W_out = this.padding === "valid"
      ? Math.floor((W - kernelCols) / strideCols) + 1
      : Math.ceil(W / strideCols);

    if (batch === -1) {
      return [-1, this.filters];
    }
    return [batch * H_out * W_out, this.filters];
  }

  /**
   * Menginisialisasi parameter 'kernel' dan 'bias'
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
        throw new Error("[Conv2D] 'imageShape' (sebagai [height, width]) harus ditentukan dalam config atau inputShape harus berupa 4D [batch, height, width, channels].");
      }
    }

    this.imageShape = [h!, w!];
    this.inputDim = inCols;

    this.outputShape = this.computeOutputShape(inputShape);

    const kernelRows = this.kernelSize[0];
    const kernelCols = this.kernelSize[1];

    // Initializer untuk kernel [kernelRows * kernelCols * inputDim, filters]
    const kernelVal = this.createInitializer(this.kernelInitializer, [kernelRows * kernelCols * this.inputDim, this.filters]);
    this.addParameter("kernel", kernelVal, true, [kernelRows * kernelCols * this.inputDim, this.filters]);

    // Initializer untuk bias [filters, 1]
    if (this.useBias) {
      const biasVal = this.createInitializer(this.biasInitializer, [this.filters, 1]);
      this.addParameter("bias", biasVal, true, [this.filters, 1]);
    }

    this.isBuilt = true;
  }

  /**
   * Getter untuk kernel parameter
   */
  public get kernel(): Matrix | undefined {
    return this.getParameter("kernel");
  }

  /**
   * Getter untuk bias parameter
   */
  public get bias(): Matrix | undefined {
    return this.getParameter("bias");
  }

  /**
   * Forward Pass matematika layer Conv2D
   */
  protected compute(inputs: Matrix, isTraining?: boolean): Matrix {
    const kernel = this.kernel;
    if (!kernel) {
      throw new Error("[Conv2D] Bobot 'kernel' tidak terinisialisasi. Pastikan build() sudah dijalankan.");
    }

    const totalRows = inputs._shape[0];
    const H = this.imageShape![0];
    const W = this.imageShape![1];
    const C = this.inputDim!;
    const B = Math.floor(totalRows / (H * W));

    const kernelRows = this.kernelSize[0];
    const kernelCols = this.kernelSize[1];
    const strideRows = this.strides[0];
    const strideCols = this.strides[1];

    // 1. Dapatkan patch matrix menggunakan grid2col
    const { patchMatrix } = grid2col(
      inputs,
      B,
      H,
      W,
      C,
      kernelRows,
      kernelCols,
      strideRows,
      strideCols,
      this.padding
    );

    // 2. dot = patchMatrix * kernel
    let dot = mj.dotProduct(patchMatrix, kernel);

    // 3. Tambahkan bias jika digunakan
    if (this.useBias && this.bias) {
      const dotT = mj.transpose(dot);
      mj.addBias(dotT, this.bias);
      dot = mj.transpose(dotT);
    }

    // 4. Aplikasikan fungsi aktivasi
    let output = dot;
    if (this.activation !== "linear") {
      try {
        const actFn = this.resolveActivation(this.activation);
        output = actFn(dot);
      } catch (err) {
        console.warn(
          `[Conv2D] Activation '${this.activation}' tidak ditemukan atau error: ${(err as Error).message}. Menggunakan 'linear'.`
        );
      }
    }

    return output;
  }

  /**
   * Konfigurasi spesifik Keras
   */
  public getConfig(): Record<string, any> {
    return {
      ...super.getConfig(),
      filters: this.filters,
      kernelSize: this.kernelSize,
      strides: this.strides,
      padding: this.padding,
      activation: this.activation,
      useBias: this.useBias,
      kernelInitializer: this.kernelInitializer,
      biasInitializer: this.biasInitializer,
      imageShape: this.imageShape,
      inputDim: this.inputDim
    };
  }
}
