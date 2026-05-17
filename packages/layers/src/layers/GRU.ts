import { BaseLayer, LayerConfig, type ForwardOptions } from "../base/BaseLayer.js";
import { Matrix, mj, engine } from "@oxide-js/core";
import { isNativeAvailable, gruForwardNative, gruBackwardNative } from "../rust_backend.js";

export interface GRUConfig extends LayerConfig {
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
 * Helper function to run GRU forward pass and record it on the tape
 */
function gruForward(
  inputs: Matrix,
  kernel: Matrix,
  recurrentKernel: Matrix,
  bias: Matrix | undefined,
  batchSize: number,
  sequenceLength: number,
  inputDim: number,
  units: number,
  returnSequences: boolean
): { out: Matrix; hiddenStates: Matrix; gateValues: Matrix } {
  const B = batchSize;
  const T = sequenceLength;
  const C = inputDim;
  const H = units;

  const outRows = returnSequences ? B * T : B;
  const outCols = H;

  const outData = new Float32Array(outRows * outCols);
  const hiddenStatesData = new Float32Array(B * T * H);
  const gateValuesData = new Float32Array(B * T * 3 * H);

  const biasParam = bias ?? mj.zeros([3 * H, 1]);

  if (isNativeAvailable()) {
    gruForwardNative(
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
      gateValuesData
    );
  } else {
    const inputsData = inputs._data;
    const kernelData = kernel._data;
    const recurrentKernelData = recurrentKernel._data;
    const biasData = biasParam._data;

    for (let b = 0; b < B; b++) {
      const h = new Float32Array(H);
      const a = new Float32Array(3 * H);

      for (let t = 0; t < T; t++) {
        const stepOffset = (b * T + t) * C;
        const hiddenOffset = (b * T + t) * H;
        const gateOffset = (b * T + t) * 3 * H;

        // 1. Calculate x_t * W + bias
        for (let j = 0; j < 3 * H; j++) {
          let sum = biasData[j] ?? 0.0;
          for (let cIdx = 0; cIdx < C; cIdx++) {
            sum += inputsData[stepOffset + cIdx] * kernelData[cIdx * 3 * H + j];
          }
          a[j] = sum;
        }

        // 2. Add h_{t-1} * U for z and r gates
        const prevHiddenOffset = (b * T + t - 1) * H;

        for (let j = 0; j < 2 * H; j++) {
          let sum = 0;
          for (let k = 0; k < H; k++) {
            const hPrevVal = t > 0 ? hiddenStatesData[prevHiddenOffset + k] : h[k];
            sum += hPrevVal * recurrentKernelData[k * 3 * H + j];
          }
          a[j] += sum;
        }

        // Update gate (z) and Reset gate (r) activations
        for (let j = 0; j < H; j++) {
          const gateZ = sigmoid(a[j]);
          const gateR = sigmoid(a[H + j]);

          gateValuesData[gateOffset + j] = gateZ;
          gateValuesData[gateOffset + H + j] = gateR;
        }

        // 3. Candidate hidden state (h_tilde)
        // a_h = x_t * W_h + b_h + (r_t * h_{t-1}) * U_h
        for (let j = 0; j < H; j++) {
          let sum = 0;
          for (let k = 0; k < H; k++) {
            const rVal = gateValuesData[gateOffset + H + k];
            const hPrevVal = t > 0 ? hiddenStatesData[prevHiddenOffset + k] : h[k];
            sum += (rVal * hPrevVal) * recurrentKernelData[k * 3 * H + 2 * H + j];
          }
          const candidateA = a[2 * H + j] + sum;
          const gateH = Math.tanh(candidateA);

          gateValuesData[gateOffset + 2 * H + j] = gateH;

          // Compute new hidden state: h_t = (1 - z) * h_{t-1} + z * h_tilde
          const zVal = gateValuesData[gateOffset + j];
          const hPrevVal = t > 0 ? hiddenStatesData[prevHiddenOffset + j] : h[j];
          const newH = (1.0 - zVal) * hPrevVal + zVal * gateH;

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
  const gateValues = Matrix.fromFlat(gateValuesData, [B * T, 3 * H]);

  engine.record(
    bias ? [inputs, kernel, recurrentKernel, bias] : [inputs, kernel, recurrentKernel],
    [out],
    (grad: Matrix) => {
      const gradIn = new Float32Array(B * T * C);
      const gradKernel = new Float32Array(C * 3 * H);
      const gradRecurrentKernel = new Float32Array(H * 3 * H);
      const gradBias = new Float32Array(3 * H);

      if (isNativeAvailable()) {
        gruBackwardNative(
          grad._data,
          inputs._data,
          hiddenStatesData,
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
          const da = new Float32Array(3 * H);

          for (let t = T - 1; t >= 0; t--) {
            const stepOffset = (b * T + t) * C;
            const hiddenOffset = (b * T + t) * H;
            const gateOffset = (b * T + t) * 3 * H;

            // 1. Get gradient from output layer (d_out)
            const dh = new Float32Array(H);
            for (let j = 0; j < H; j++) {
              const dOut = returnSequences
                ? gradOutData[(b * T + t) * H + j]
                : (t === T - 1 ? gradOutData[b * H + j] : 0.0);
              dh[j] = dOut + dhNext[j];
            }

            // 2. Compute gate pre-activation gradients
            const prevHiddenOffset = (b * T + t - 1) * H;
            const dhReset = new Float32Array(H);

            for (let j = 0; j < H; j++) {
              const gateZ = gateValuesData[gateOffset + j];
              const gateH = gateValuesData[gateOffset + 2 * H + j];

              const hPrevVal = t > 0 ? hiddenStatesData[prevHiddenOffset + j] : 0.0;

              const dHTilde = dh[j] * gateZ;
              const daH = dHTilde * (1 - gateH * gateH);

              const dZ = dh[j] * (gateH - hPrevVal);
              const daZ = dZ * gateZ * (1 - gateZ);

              da[j] = daZ;
              da[2 * H + j] = daH;
            }

            // Reseted gradient contribution from Candidate preactivation da_H
            for (let k = 0; k < H; k++) {
              let sum = 0.0;
              for (let j = 0; j < H; j++) {
                sum += da[2 * H + j] * recurrentKernelData[k * 3 * H + 2 * H + j];
              }
              dhReset[k] = sum;
            }

            for (let j = 0; j < H; j++) {
              const gateR = gateValuesData[gateOffset + H + j];
              const hPrevVal = t > 0 ? hiddenStatesData[prevHiddenOffset + j] : 0.0;

              const dR = dhReset[j] * hPrevVal;
              const daR = dR * gateR * (1 - gateR);

              da[H + j] = daR;
            }

            // 3. Accumulate weight and bias gradients
            for (let j = 0; j < 3 * H; j++) {
              gradBias[j] += da[j];

              for (let cIdx = 0; cIdx < C; cIdx++) {
                gradKernel[cIdx * 3 * H + j] += inputsData[stepOffset + cIdx] * da[j];
              }

              if (t > 0) {
                // For U_z (0..H) and U_r (H..2*H), input is hidden_states_{t-1}
                for (let k = 0; k < H; k++) {
                  const hPrevVal = hiddenStatesData[prevHiddenOffset + k];
                  if (j < 2 * H) {
                    gradRecurrentKernel[k * 3 * H + j] += hPrevVal * da[j];
                  } else {
                    // For U_h, input is r_t * h_{t-1}
                    const rVal = gateValuesData[gateOffset + H + k];
                    gradRecurrentKernel[k * 3 * H + j] += (rVal * hPrevVal) * da[j];
                  }
                }
              }
            }

            // 4. Input gradient (dx_t)
            for (let cIdx = 0; cIdx < C; cIdx++) {
              let sum = 0.0;
              for (let g = 0; g < 3 * H; g++) {
                sum += da[g] * kernelData[cIdx * 3 * H + g];
              }
              gradIn[stepOffset + cIdx] = sum;
            }

            // 5. Update dhNext for the next step (t-1)
            for (let k = 0; k < H; k++) {
              let sum = 0.0;

              // Direct recurrence z gate contribution
              const gateZ = gateValuesData[gateOffset + k];
              sum += dh[k] * (1.0 - gateZ);

              // Reseted contribution
              const gateR = gateValuesData[gateOffset + H + k];
              sum += dhReset[k] * gateR;

              // U_z and U_r recurrent contribution
              for (let j = 0; j < 2 * H; j++) {
                sum += da[j] * recurrentKernelData[k * 3 * H + j];
              }

              dhNext[k] = sum;
            }
          }
        }
      }

      const gradInputsMatrix = Matrix.fromFlat(gradIn, [B * T, C]);
      const gradKernelMatrix = Matrix.fromFlat(gradKernel, [C, 3 * H]);
      const gradRecurrentKernelMatrix = Matrix.fromFlat(gradRecurrentKernel, [H, 3 * H]);

      if (bias) {
        const gradBiasMatrix = Matrix.fromFlat(gradBias, [3 * H, 1]);
        return [gradInputsMatrix, gradKernelMatrix, gradRecurrentKernelMatrix, gradBiasMatrix];
      }

      return [gradInputsMatrix, gradKernelMatrix, gradRecurrentKernelMatrix];
    },
    { saveInput: false, saveOutput: false }
  );

  return { out, hiddenStates, gateValues };
}

export class GRU extends BaseLayer {
  public units: number;
  public useBias: boolean;
  public returnSequences: boolean;
  public kernelInitializer: string;
  public recurrentInitializer: string;
  public biasInitializer: string;
  public sequenceLength?: number;
  public inputDim?: number;

  constructor(config: GRUConfig) {
    super(config);
    if (config.units === undefined || config.units <= 0) {
      throw new Error("[GRU] 'units' wajib berupa angka positif.");
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
        throw new Error("[GRU] 'sequenceLength' harus ditentukan dalam config jika inputShape 2D.");
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
          "[GRU] 'sequenceLength' harus ditentukan dalam config atau inputShape harus berupa 3D [batch, sequenceLength, inputDim]."
        );
      }
    }

    this.sequenceLength = seqLen;
    this.inputDim = inCols;

    this.outputShape = this.computeOutputShape(inputShape);

    // Initializer untuk W (kernel) -> [inputDim, 3 * units]
    const kernelVal = this.createInitializer(this.kernelInitializer, [this.inputDim, 3 * this.units]);
    this.addParameter("kernel", kernelVal, true, [this.inputDim, 3 * this.units]);

    // Initializer untuk U (recurrentKernel) -> [units, 3 * units]
    const recurrentKernelVal = this.createInitializer(this.recurrentInitializer, [this.units, 3 * this.units]);
    this.addParameter("recurrentKernel", recurrentKernelVal, true, [this.units, 3 * this.units]);

    // Initializer untuk bias -> [3 * units, 1]
    if (this.useBias) {
      const biasVal = this.createInitializer(this.biasInitializer, [3 * this.units, 1]);
      this.addParameter("bias", biasVal, true, [3 * this.units, 1]);
    }

    this.isBuilt = true;
  }

  protected compute(inputs: Matrix, options?: ForwardOptions): Matrix {
    const kernel = this.kernel;
    const recurrentKernel = this.recurrentKernel;
    if (!kernel || !recurrentKernel) {
      throw new Error("[GRU] Bobot 'kernel' atau 'recurrentKernel' tidak terinisialisasi.");
    }

    const totalRows = inputs._shape[0];
    const L = this.sequenceLength!;
    const C = this.inputDim!;
    const B = Math.floor(totalRows / L);

    const { out } = gruForward(
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
