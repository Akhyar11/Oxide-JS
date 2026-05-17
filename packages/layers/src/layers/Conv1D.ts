import { BaseLayer, LayerConfig } from "../base/BaseLayer.js";
import { Matrix, mj, engine } from "@oxide-js/core";
import { isNativeAvailable, seq2ColNative, col2SeqNative } from "../rust_backend.js";

export interface Conv1DConfig extends LayerConfig {
  filters: number;
  kernelSize: number;
  strides?: number;
  padding?: "valid" | "same";
  activation?: string;
  useBias?: boolean;
  kernelInitializer?: string;
  biasInitializer?: string;
  sequenceLength?: number;
  inputDim?: number;
}

/**
 * Helper function to perform seq2col mapping with its custom backward pass recorded on tape
 */
function seq2col(
  inputs: Matrix,
  batchSize: number,
  sequenceLength: number,
  inputDim: number,
  kernelSize: number,
  strides: number,
  padding: "valid" | "same"
): { patchMatrix: Matrix; L_out: number } {
  const B = batchSize;
  const L = sequenceLength;
  const C = inputDim;

  let L_out: number;
  let padLeft = 0;

  if (padding === "valid") {
    L_out = Math.floor((L - kernelSize) / strides) + 1;
  } else {
    L_out = Math.ceil(L / strides);
    const totalPadding = (L_out - 1) * strides + kernelSize - L;
    padLeft = Math.floor(totalPadding / 2);
  }

  const patchRows = B * L_out;
  const patchCols = kernelSize * C;
  const patchData = new Float32Array(patchRows * patchCols);
  const inputsData = inputs._data;

  if (isNativeAvailable()) {
    seq2ColNative(inputsData, B, L, C, kernelSize, strides, padLeft, patchData);
  } else {
    for (let b = 0; b < B; b++) {
      for (let i = 0; i < L_out; i++) {
        const outRowIdx = b * L_out + i;
        const destOffset = outRowIdx * patchCols;

        // Start sequence index for the window
        const tStart = i * strides - padLeft;

        for (let k = 0; k < kernelSize; k++) {
          const t = tStart + k;
          const kernelOffset = k * C;

          if (t >= 0 && t < L) {
            const srcOffset = (b * L + t) * C;
            for (let c = 0; c < C; c++) {
              patchData[destOffset + kernelOffset + c] = inputsData[srcOffset + c];
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
      const gradInputsData = new Float32Array(B * L * C);
      const gradOutData = grad._data;

      if (isNativeAvailable()) {
        col2SeqNative(gradOutData, B, L, C, kernelSize, strides, padLeft, gradInputsData);
      } else {
        for (let b = 0; b < B; b++) {
          for (let i = 0; i < L_out; i++) {
            const outRowIdx = b * L_out + i;
            const srcOffset = outRowIdx * patchCols;
            const tStart = i * strides - padLeft;

            for (let k = 0; k < kernelSize; k++) {
              const t = tStart + k;
              const kernelOffset = k * C;

              if (t >= 0 && t < L) {
                const destOffset = (b * L + t) * C;
                for (let c = 0; c < C; c++) {
                  gradInputsData[destOffset + c] += gradOutData[srcOffset + kernelOffset + c];
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

  return { patchMatrix, L_out };
}

export class Conv1D extends BaseLayer {
  public filters: number;
  public kernelSize: number;
  public strides: number;
  public padding: "valid" | "same";
  public activation: string;
  public useBias: boolean;
  public kernelInitializer: string;
  public biasInitializer: string;
  public sequenceLength?: number;
  public inputDim?: number;

  constructor(config: Conv1DConfig) {
    super(config);
    if (config.filters === undefined || config.filters <= 0) {
      throw new Error("[Conv1D] 'filters' wajib berupa angka positif.");
    }
    if (config.kernelSize === undefined || config.kernelSize <= 0) {
      throw new Error("[Conv1D] 'kernelSize' wajib berupa angka positif.");
    }
    this.filters = config.filters;
    this.kernelSize = config.kernelSize;
    this.strides = config.strides ?? 1;
    this.padding = config.padding ?? "valid";
    this.activation = config.activation ?? "linear";
    this.useBias = config.useBias ?? true;
    this.kernelInitializer = config.kernelInitializer ?? "glorot_normal";
    this.biasInitializer = config.biasInitializer ?? "zeros";
    this.sequenceLength = config.sequenceLength;
    this.inputDim = config.inputDim;
  }

  /**
   * Validasi bentuk input spesifik untuk 2D matriks sekuensial Conv1D
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
   * Menghitung output shape logis [batch * L_out, filters]
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
        throw new Error("[Conv1D] 'sequenceLength' harus ditentukan dalam config jika inputShape 2D.");
      }
    }

    const L_out = this.padding === "valid"
      ? Math.floor((L - this.kernelSize) / this.strides) + 1
      : Math.ceil(L / this.strides);

    if (batch === -1) {
      return [-1, this.filters];
    }
    return [batch * L_out, this.filters];
  }

  /**
   * Menginisialisasi parameter 'kernel' dan 'bias'
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
        throw new Error("[Conv1D] 'sequenceLength' harus ditentukan dalam config atau inputShape harus berupa 3D [batch, sequenceLength, inputDim].");
      }
    }

    this.sequenceLength = seqLen;
    this.inputDim = inCols;

    this.outputShape = this.computeOutputShape(inputShape);

    // Initializer untuk kernel [kernelSize * inputDim, filters]
    const kernelVal = this.createInitializer(this.kernelInitializer, [this.kernelSize * this.inputDim, this.filters]);
    this.addParameter("kernel", kernelVal, true, [this.kernelSize * this.inputDim, this.filters]);

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
   * Forward Pass matematika layer Conv1D
   */
  protected compute(inputs: Matrix, isTraining?: boolean): Matrix {
    const kernel = this.kernel;
    if (!kernel) {
      throw new Error("[Conv1D] Bobot 'kernel' tidak terinisialisasi. Pastikan build() sudah dijalankan.");
    }

    const totalRows = inputs._shape[0];
    const L = this.sequenceLength!;
    const C = this.inputDim!;
    const B = Math.floor(totalRows / L);

    // 1. Dapatkan patch matrix menggunakan seq2col
    const { patchMatrix } = seq2col(
      inputs,
      B,
      L,
      C,
      this.kernelSize,
      this.strides,
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
          `[Conv1D] Activation '${this.activation}' tidak ditemukan atau error: ${(err as Error).message}. Menggunakan 'linear'.`
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
      sequenceLength: this.sequenceLength,
      inputDim: this.inputDim
    };
  }
}
