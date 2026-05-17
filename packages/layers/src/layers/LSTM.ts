import { BaseLayer, LayerConfig } from "../base/BaseLayer.js";
import { Matrix, mj, engine } from "@oxide-js/core";
import { isNativeAvailable, lstmForwardNative, lstmBackwardNative } from "../rust_backend.js";

export interface LSTMConfig extends LayerConfig {
  units: number;
  useBias?: boolean;
  returnSequences?: boolean;
  kernelInitializer?: string;
  recurrentInitializer?: string;
  biasInitializer?: string;
  sequenceLength?: number;
  inputDim?: number;
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/**
 * Helper function to run LSTM forward pass and record it on the tape
 */
function lstmForward(
  inputs: Matrix,
  kernel: Matrix,
  recurrentKernel: Matrix,
  bias: Matrix | undefined,
  batchSize: number,
  sequenceLength: number,
  inputDim: number,
  units: number,
  returnSequences: boolean
): { out: Matrix; hiddenStates: Matrix; cellStates: Matrix; gateValues: Matrix } {
  const B = batchSize;
  const T = sequenceLength;
  const C = inputDim;
  const H = units;

  const outRows = returnSequences ? B * T : B;
  const outCols = H;

  const outData = new Float32Array(outRows * outCols);
  const hiddenStatesData = new Float32Array(B * T * H);
  const cellStatesData = new Float32Array(B * T * H);
  const gateValuesData = new Float32Array(B * T * 4 * H);

  const biasParam = bias ?? mj.zeros([4 * H, 1]);

  if (isNativeAvailable()) {
    lstmForwardNative(
      inputs._data,
      kernel._data,
      recurrentKernel._data,
      biasParam._data,
      B,
      T,
      C,
      H,
      returnSequences,
      outData,
      hiddenStatesData,
      cellStatesData,
      gateValuesData
    );
  } else {
    const inputsData = inputs._data;
    const kernelData = kernel._data;
    const recurrentKernelData = recurrentKernel._data;
    const biasData = biasParam._data;

    for (let b = 0; b < B; b++) {
      const h = new Float32Array(H);
      const c = new Float32Array(H);
      const a = new Float32Array(4 * H);

      for (let t = 0; t < T; t++) {
        const stepOffset = (b * T + t) * C;
        const hiddenOffset = (b * T + t) * H;
        const gateOffset = (b * T + t) * 4 * H;

        // 1. Calculate x_t * W + bias
        for (let j = 0; j < 4 * H; j++) {
          let sum = biasData[j] ?? 0.0;
          for (let cIdx = 0; cIdx < C; cIdx++) {
            sum += inputsData[stepOffset + cIdx] * kernelData[cIdx * 4 * H + j];
          }
          a[j] = sum;
        }

        // 2. Add h_{t-1} * U if t > 0
        if (t > 0) {
          const prevHiddenOffset = (b * T + t - 1) * H;
          for (let j = 0; j < 4 * H; j++) {
            let sum = 0;
            for (let k = 0; k < H; k++) {
              sum += hiddenStatesData[prevHiddenOffset + k] * recurrentKernelData[k * 4 * H + j];
            }
            a[j] += sum;
          }
        } else {
          for (let j = 0; j < 4 * H; j++) {
            let sum = 0;
            for (let k = 0; k < H; k++) {
              sum += h[k] * recurrentKernelData[k * 4 * H + j];
            }
            a[j] += sum;
          }
        }

        // 3. Compute gates
        // 0..H -> input gate i
        // H..2*H -> forget gate f
        // 2*H..3*H -> candidate cell c_tilde
        // 3*H..4*H -> output gate o
        for (let j = 0; j < H; j++) {
          const gateI = sigmoid(a[j]);
          const gateF = sigmoid(a[H + j]);
          const gateC = Math.tanh(a[2 * H + j]);
          const gateO = sigmoid(a[3 * H + j]);

          gateValuesData[gateOffset + j] = gateI;
          gateValuesData[gateOffset + H + j] = gateF;
          gateValuesData[gateOffset + 2 * H + j] = gateC;
          gateValuesData[gateOffset + 3 * H + j] = gateO;

          // Update cell state
          const prevCVal = t > 0 ? cellStatesData[(b * T + t - 1) * H + j] : c[j];
          const newC = gateF * prevCVal + gateI * gateC;
          cellStatesData[hiddenOffset + j] = newC;

          // Update hidden state
          const newH = gateO * Math.tanh(newC);
          hiddenStatesData[hiddenOffset + j] = newH;
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
  const cellStates = Matrix.fromFlat(cellStatesData, [B * T, H]);
  const gateValues = Matrix.fromFlat(gateValuesData, [B * T, 4 * H]);

  engine.record(
    bias ? [inputs, kernel, recurrentKernel, bias] : [inputs, kernel, recurrentKernel],
    [out],
    (grad: Matrix) => {
      const gradIn = new Float32Array(B * T * C);
      const gradKernel = new Float32Array(C * 4 * H);
      const gradRecurrentKernel = new Float32Array(H * 4 * H);
      const gradBias = new Float32Array(4 * H);

      if (isNativeAvailable()) {
        lstmBackwardNative(
          grad._data,
          inputs._data,
          hiddenStatesData,
          cellStatesData,
          gateValuesData,
          kernel._data,
          recurrentKernel._data,
          B,
          T,
          C,
          H,
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
          const dcNext = new Float32Array(H);
          const da = new Float32Array(4 * H);

          for (let t = T - 1; t >= 0; t--) {
            const stepOffset = (b * T + t) * C;
            const hiddenOffset = (b * T + t) * H;
            const gateOffset = (b * T + t) * 4 * H;

            // 1. Get gradient from output layer (d_out)
            const dh = new Float32Array(H);
            for (let j = 0; j < H; j++) {
              const dOut = returnSequences
                ? gradOutData[(b * T + t) * H + j]
                : (t === T - 1 ? gradOutData[b * H + j] : 0.0);
              dh[j] = dOut + dhNext[j];
            }

            // 2. Compute gate pre-activation gradients
            for (let j = 0; j < H; j++) {
              const cVal = cellStatesData[hiddenOffset + j];
              const cTanh = Math.tanh(cVal);

              const gateI = gateValuesData[gateOffset + j];
              const gateF = gateValuesData[gateOffset + H + j];
              const gateC = gateValuesData[gateOffset + 2 * H + j];
              const gateO = gateValuesData[gateOffset + 3 * H + j];

              const dO = dh[j] * cTanh;
              const daO = dO * gateO * (1 - gateO);

              const dc = dh[j] * gateO * (1 - cTanh * cTanh) + dcNext[j];

              const dCTilde = dc * gateI;
              const daC = dCTilde * (1 - gateC * gateC);

              const prevCVal = t > 0 ? cellStatesData[(b * T + t - 1) * H + j] : 0.0;
              const dF = dc * prevCVal;
              const daF = dF * gateF * (1 - gateF);

              const dI = dc * gateC;
              const daI = dI * gateI * (1 - gateI);

              da[j] = daI;
              da[H + j] = daF;
              da[2 * H + j] = daC;
              da[3 * H + j] = daO;

              dcNext[j] = dc * gateF;
            }

            // 3. Accumulate weight and bias gradients
            for (let j = 0; j < 4 * H; j++) {
              gradBias[j] += da[j];

              for (let cIdx = 0; cIdx < C; cIdx++) {
                gradKernel[cIdx * 4 * H + j] += inputsData[stepOffset + cIdx] * da[j];
              }

              if (t > 0) {
                const prevHiddenOffset = (b * T + t - 1) * H;
                for (let k = 0; k < H; k++) {
                  gradRecurrentKernel[k * 4 * H + j] += hiddenStatesData[prevHiddenOffset + k] * da[j];
                }
              }
            }

            // 4. Input gradient (dx_t)
            for (let cIdx = 0; cIdx < C; cIdx++) {
              let sum = 0.0;
              for (let g = 0; g < 4 * H; g++) {
                sum += da[g] * kernelData[cIdx * 4 * H + g];
              }
              gradIn[stepOffset + cIdx] = sum;
            }

            // 5. Update dhNext for the next step (t-1)
            for (let k = 0; k < H; k++) {
              let sum = 0.0;
              for (let g = 0; g < 4 * H; g++) {
                sum += da[g] * recurrentKernelData[k * 4 * H + g];
              }
              dhNext[k] = sum;
            }
          }
        }
      }

      const gradInputsMatrix = Matrix.fromFlat(gradIn, [B * T, C]);
      const gradKernelMatrix = Matrix.fromFlat(gradKernel, [C, 4 * H]);
      const gradRecurrentKernelMatrix = Matrix.fromFlat(gradRecurrentKernel, [H, 4 * H]);

      if (bias) {
        const gradBiasMatrix = Matrix.fromFlat(gradBias, [4 * H, 1]);
        return [gradInputsMatrix, gradKernelMatrix, gradRecurrentKernelMatrix, gradBiasMatrix];
      }

      return [gradInputsMatrix, gradKernelMatrix, gradRecurrentKernelMatrix];
    },
    { saveInput: false, saveOutput: false }
  );

  return { out, hiddenStates, cellStates, gateValues };
}

export class LSTM extends BaseLayer {
  public units: number;
  public useBias: boolean;
  public returnSequences: boolean;
  public kernelInitializer: string;
  public recurrentInitializer: string;
  public biasInitializer: string;
  public sequenceLength?: number;
  public inputDim?: number;

  constructor(config: LSTMConfig) {
    super(config);
    if (config.units === undefined || config.units <= 0) {
      throw new Error("[LSTM] 'units' wajib berupa angka positif.");
    }
    this.units = config.units;
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
        throw new Error("[LSTM] 'sequenceLength' harus ditentukan dalam config jika inputShape 2D.");
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
          "[LSTM] 'sequenceLength' harus ditentukan dalam config atau inputShape harus berupa 3D [batch, sequenceLength, inputDim]."
        );
      }
    }

    this.sequenceLength = seqLen;
    this.inputDim = inCols;

    this.outputShape = this.computeOutputShape(inputShape);

    // Initializer untuk W (kernel) -> [inputDim, 4 * units]
    const kernelVal = this.createInitializer(this.kernelInitializer, [this.inputDim, 4 * this.units]);
    this.addParameter("kernel", kernelVal, true, [this.inputDim, 4 * this.units]);

    // Initializer untuk U (recurrentKernel) -> [units, 4 * units]
    const recurrentKernelVal = this.createInitializer(this.recurrentInitializer, [this.units, 4 * this.units]);
    this.addParameter("recurrentKernel", recurrentKernelVal, true, [this.units, 4 * this.units]);

    // Initializer untuk bias -> [4 * units, 1]
    if (this.useBias) {
      const biasVal = this.createInitializer(this.biasInitializer, [4 * this.units, 1]);
      this.addParameter("bias", biasVal, true, [4 * this.units, 1]);
    }

    this.isBuilt = true;
  }

  protected compute(inputs: Matrix, isTraining?: boolean): Matrix {
    const kernel = this.kernel;
    const recurrentKernel = this.recurrentKernel;
    if (!kernel || !recurrentKernel) {
      throw new Error("[LSTM] Bobot 'kernel' atau 'recurrentKernel' tidak terinisialisasi.");
    }

    const totalRows = inputs._shape[0];
    const L = this.sequenceLength!;
    const C = this.inputDim!;
    const B = Math.floor(totalRows / L);

    const { out } = lstmForward(
      inputs,
      kernel,
      recurrentKernel,
      this.useBias ? this.bias : undefined,
      B,
      L,
      C,
      this.units,
      this.returnSequences
    );

    return out;
  }

  public getConfig(): Record<string, any> {
    return {
      ...super.getConfig(),
      units: this.units,
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
