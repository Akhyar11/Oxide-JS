import { BaseLayer, LayerConfig, type ForwardOptions } from "../base/BaseLayer.js";
import { Matrix, mj, engine } from "@oxide-js/core";
import { isNativeAvailable, rnnForwardNative, rnnBackwardNative } from "../rust_backend.js";

export interface SimpleRNNConfig extends LayerConfig {
  units: number;
  activation?: "tanh" | "relu" | "sigmoid" | "linear";
  useBias?: boolean;
  returnSequences?: boolean;
  kernelInitializer?: string;
  recurrentInitializer?: string;
  biasInitializer?: string;
  sequenceLength?: number;
  inputDim?: number;
}

function activateJS(x: number, act: string): number {
  switch (act) {
    case "tanh":
      return Math.tanh(x);
    case "relu":
      return x > 0 ? x : 0;
    case "sigmoid":
      return 1 / (1 + Math.exp(-x));
    default:
      return x;
  }
}

function activateGradJS(y: number, act: string): number {
  switch (act) {
    case "tanh":
      return 1 - y * y;
    case "relu":
      return y > 0 ? 1 : 0;
    case "sigmoid":
      return y * (1 - y);
    default:
      return 1;
  }
}

/**
 * Helper function to run SimpleRNN forward pass and record it on the tape
 */
function rnnForward(
  inputs: Matrix,
  kernel: Matrix,
  recurrentKernel: Matrix,
  bias: Matrix | undefined,
  batchSize: number,
  sequenceLength: number,
  inputDim: number,
  units: number,
  activation: string,
  returnSequences: boolean
): { out: Matrix; hiddenStates: Matrix } {
  const B = batchSize;
  const T = sequenceLength;
  const C = inputDim;
  const H = units;

  const outRows = returnSequences ? B * T : B;
  const outCols = H;

  const outData = new Float32Array(outRows * outCols);
  const hiddenStatesData = new Float32Array(B * T * H);

  const biasParam = bias ?? mj.zeros([H, 1]);

  if (isNativeAvailable()) {
    rnnForwardNative(
      inputs._data,
      kernel._data,
      recurrentKernel._data,
      biasParam._data,
      B,
      T,
      C,
      H,
      activation,
      returnSequences,
      outData,
      hiddenStatesData
    );
  } else {
    const inputsData = inputs._data;
    const kernelData = kernel._data;
    const recurrentKernelData = recurrentKernel._data;
    const biasData = biasParam._data;

    for (let b = 0; b < B; b++) {
      const a = new Float32Array(H);

      for (let t = 0; t < T; t++) {
        const stepOffset = (b * T + t) * C;
        const hiddenOffset = (b * T + t) * H;

        // 1. Calculate x_t * W_x + bias
        for (let j = 0; j < H; j++) {
          let sum = biasData[j] ?? 0.0;
          for (let c = 0; c < C; c++) {
            sum += inputsData[stepOffset + c] * kernelData[c * H + j];
          }
          a[j] = sum;
        }

        // 2. Add h_{t-1} * W_h if t > 0
        if (t > 0) {
          const prevHiddenOffset = (b * T + t - 1) * H;
          for (let j = 0; j < H; j++) {
            let sum = 0;
            for (let k = 0; k < H; k++) {
              sum += hiddenStatesData[prevHiddenOffset + k] * recurrentKernelData[k * H + j];
            }
            a[j] += sum;
          }
        }

        // 3. Apply activation and save hidden state
        for (let j = 0; j < H; j++) {
          const hVal = activateJS(a[j], activation);
          hiddenStatesData[hiddenOffset + j] = hVal;
        }

        // 4. Save to out if returnSequences is true
        if (returnSequences) {
          const outOffset = (b * T + t) * H;
          for (let j = 0; j < H; j++) {
            outData[outOffset + j] = hiddenStatesData[hiddenOffset + j];
          }
        }
      }

      // 5. If returnSequences is false, save the last hidden state to out
      if (!returnSequences) {
        const lastHiddenOffset = (b * T + T - 1) * H;
        const outOffset = b * H;
        for (let j = 0; j < H; j++) {
          outData[outOffset + j] = hiddenStatesData[lastHiddenOffset + j];
        }
      }
    }
  }

  const out = Matrix.fromFlat(outData, [outRows, outCols]);
  const hiddenStates = Matrix.fromFlat(hiddenStatesData, [B * T, H]);

  engine.record(
    bias ? [inputs, kernel, recurrentKernel, bias] : [inputs, kernel, recurrentKernel],
    [out],
    (grad: Matrix) => {
      const gradIn = new Float32Array(B * T * C);
      const gradKernel = new Float32Array(C * H);
      const gradRecurrentKernel = new Float32Array(H * H);
      const gradBias = new Float32Array(H);

      if (isNativeAvailable()) {
        rnnBackwardNative(
          grad._data,
          inputs._data,
          hiddenStatesData,
          kernel._data,
          recurrentKernel._data,
          B,
          T,
          C,
          H,
          activation,
          returnSequences,
          gradIn,
          gradKernel,
          gradRecurrentKernel,
          gradBias
        );
      } else {
        const gradOutData = grad._data;
        const inputsData = inputs._data;
        const kernelData = kernel._data;
        const recurrentKernelData = recurrentKernel._data;

        for (let b = 0; b < B; b++) {
          const dhNext = new Float32Array(H);

          for (let t = T - 1; t >= 0; t--) {
            const stepOffset = (b * T + t) * C;
            const hiddenOffset = (b * T + t) * H;

            // 1. Get gradient from output layer (d_out)
            const dh = new Float32Array(H);
            for (let j = 0; j < H; j++) {
              const dOut = returnSequences
                ? gradOutData[(b * T + t) * H + j]
                : (t === T - 1 ? gradOutData[b * H + j] : 0.0);
              dh[j] = dOut + dhNext[j];
            }

            // 2. Pre-activation gradient (da_t)
            const da = new Float32Array(H);
            for (let j = 0; j < H; j++) {
              const hVal = hiddenStatesData[hiddenOffset + j];
              da[j] = dh[j] * activateGradJS(hVal, activation);
            }

            // 3. Accumulate weight and bias gradients
            for (let j = 0; j < H; j++) {
              gradBias[j] += da[j];

              for (let c = 0; c < C; c++) {
                gradKernel[c * H + j] += inputsData[stepOffset + c] * da[j];
              }

              if (t > 0) {
                const prevHiddenOffset = (b * T + t - 1) * H;
                for (let k = 0; k < H; k++) {
                  gradRecurrentKernel[k * H + j] += hiddenStatesData[prevHiddenOffset + k] * da[j];
                }
              }
            }

            // 4. Input gradient (dx_t)
            for (let c = 0; c < C; c++) {
              let sum = 0.0;
              for (let j = 0; j < H; j++) {
                sum += da[j] * kernelData[c * H + j];
              }
              gradIn[stepOffset + c] = sum;
            }

            // 5. Update dhNext for the next step (t-1)
            for (let k = 0; k < H; k++) {
              let sum = 0.0;
              for (let j = 0; j < H; j++) {
                sum += da[j] * recurrentKernelData[k * H + j];
              }
              dhNext[k] = sum;
            }
          }
        }
      }

      const gradInputsMatrix = Matrix.fromFlat(gradIn, [B * T, C]);
      const gradKernelMatrix = Matrix.fromFlat(gradKernel, [C, H]);
      const gradRecurrentKernelMatrix = Matrix.fromFlat(gradRecurrentKernel, [H, H]);

      if (bias) {
        const gradBiasMatrix = Matrix.fromFlat(gradBias, [H, 1]);
        return [gradInputsMatrix, gradKernelMatrix, gradRecurrentKernelMatrix, gradBiasMatrix];
      }

      return [gradInputsMatrix, gradKernelMatrix, gradRecurrentKernelMatrix];
    },
    { saveInput: false, saveOutput: false }
  );

  return { out, hiddenStates };
}

export class SimpleRNN extends BaseLayer {
  public units: number;
  public activation: "tanh" | "relu" | "sigmoid" | "linear";
  public useBias: boolean;
  public returnSequences: boolean;
  public kernelInitializer: string;
  public recurrentInitializer: string;
  public biasInitializer: string;
  public sequenceLength?: number;
  public inputDim?: number;

  constructor(config: SimpleRNNConfig) {
    super(config);
    if (config.units === undefined || config.units <= 0) {
      throw new Error("[SimpleRNN] 'units' wajib berupa angka positif.");
    }
    this.units = config.units;
    this.activation = config.activation ?? "tanh";
    this.useBias = config.useBias ?? true;
    this.returnSequences = config.returnSequences ?? false;
    this.kernelInitializer = config.kernelInitializer ?? "glorot_normal";
    this.recurrentInitializer = config.recurrentInitializer ?? "glorot_normal";
    this.biasInitializer = config.biasInitializer ?? "zeros";
    this.sequenceLength = config.sequenceLength;
    this.inputDim = config.inputDim;
  }

  public get kernel(): Matrix | undefined {
    return this.getParameter("kernel");
  }

  public get recurrentKernel(): Matrix | undefined {
    return this.getParameter("recurrentKernel");
  }

  public get bias(): Matrix | undefined {
    return this.getParameter("bias");
  }

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
        throw new Error("[SimpleRNN] 'sequenceLength' harus ditentukan dalam config jika inputShape 2D.");
      }
    }

    if (batch === -1) {
      return [-1, this.units];
    }

    return this.returnSequences ? [batch * L, this.units] : [batch, this.units];
  }

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
        throw new Error(
          "[SimpleRNN] 'sequenceLength' harus ditentukan dalam config atau inputShape harus berupa 3D [batch, sequenceLength, inputDim]."
        );
      }
    }

    this.sequenceLength = seqLen;
    this.inputDim = inCols;

    this.outputShape = this.computeOutputShape(inputShape);

    // Initializer untuk W_x (kernel)
    const kernelVal = this.createInitializer(this.kernelInitializer, [this.inputDim, this.units]);
    this.addParameter("kernel", kernelVal, true, [this.inputDim, this.units]);

    // Initializer untuk W_h (recurrentKernel)
    const recurrentKernelVal = this.createInitializer(this.recurrentInitializer, [this.units, this.units]);
    this.addParameter("recurrentKernel", recurrentKernelVal, true, [this.units, this.units]);

    // Initializer untuk bias
    if (this.useBias) {
      const biasVal = this.createInitializer(this.biasInitializer, [this.units, 1]);
      this.addParameter("bias", biasVal, true, [this.units, 1]);
    }

    this.isBuilt = true;
  }

  protected compute(inputs: Matrix, options?: ForwardOptions): Matrix {
    const kernel = this.kernel;
    const recurrentKernel = this.recurrentKernel;
    if (!kernel || !recurrentKernel) {
      throw new Error("[SimpleRNN] Bobot 'kernel' atau 'recurrentKernel' tidak terinisialisasi.");
    }

    const totalRows = inputs._shape[0];
    const L = this.sequenceLength!;
    const C = this.inputDim!;
    const B = Math.floor(totalRows / L);

    const { out } = rnnForward(
      inputs,
      kernel,
      recurrentKernel,
      this.useBias ? this.bias : undefined,
      B,
      L,
      C,
      this.units,
      this.activation,
      this.returnSequences
    );

    return out;
  }

  public getConfig(): Record<string, any> {
    return {
      ...super.getConfig(),
      units: this.units,
      activation: this.activation,
      useBias: this.useBias,
      returnSequences: this.returnSequences,
      kernelInitializer: this.kernelInitializer,
      recurrentInitializer: this.recurrentInitializer,
      biasInitializer: this.biasInitializer,
      sequenceLength: this.sequenceLength,
      inputDim: this.inputDim
    };
  }
}
