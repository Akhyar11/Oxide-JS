import { BaseLayer, LayerConfig, type ForwardOptions } from "../base/BaseLayer.js";
import { Matrix, mj, engine } from "@oxide-js/core";
import { isNativeAvailable, attentionForwardNative, attentionBackwardNative } from "../rust_backend.js";

export interface AttentionConfig extends LayerConfig {
  units: number;
  useBias?: boolean;
  kernelInitializer?: string;
  biasInitializer?: string;
  sequenceLength?: number;
  inputDim?: number;
}

function attentionForward(
  inputsQ: Matrix,
  inputsK: Matrix,
  wQ: Matrix,
  wK: Matrix,
  wV: Matrix,
  bQ: Matrix | undefined,
  bK: Matrix | undefined,
  bV: Matrix | undefined,
  batchSize: number,
  seqLenQ: number,
  seqLenK: number,
  inputDim: number,
  units: number,
  useBias: boolean
): { out: Matrix; q: Matrix; k: Matrix; v: Matrix; scores: Matrix; probs: Matrix } {
  const B = batchSize;
  const Lq = seqLenQ;
  const Lk = seqLenK;
  const C = inputDim;
  const H = units;

  const outData = new Float32Array(B * Lq * H);
  const qData = new Float32Array(B * Lq * H);
  const kData = new Float32Array(B * Lk * H);
  const vData = new Float32Array(B * Lk * H);
  const scoresData = new Float32Array(B * Lq * Lk);
  const probsData = new Float32Array(B * Lq * Lk);

  const bQParam = bQ ?? mj.zeros([H, 1]);
  const bKParam = bK ?? mj.zeros([H, 1]);
  const bVParam = bV ?? mj.zeros([H, 1]);

  if (isNativeAvailable()) {
    attentionForwardNative(
      inputsQ._data,
      inputsK._data,
      wQ._data,
      wK._data,
      wV._data,
      bQParam._data,
      bKParam._data,
      bVParam._data,
      B,
      Lq,
      Lk,
      C,
      H,
      useBias,
      outData,
      qData,
      kData,
      vData,
      scoresData,
      probsData
    );
  } else {
    const inputsQData = inputsQ._data;
    const inputsKData = inputsK._data;
    const wQData = wQ._data;
    const wKData = wK._data;
    const wVData = wV._data;
    const bQData = bQParam._data;
    const bKData = bKParam._data;
    const bVData = bVParam._data;

    const scale = 1.0 / Math.sqrt(H);

    for (let b = 0; b < B; b++) {
      // 1. Project inputsQ to Q
      for (let t = 0; t < Lq; t++) {
        const rowOffset = (b * Lq + t) * C;
        const outOffset = (b * Lq + t) * H;

        for (let j = 0; j < H; j++) {
          let sumQ = useBias ? (bQData[j] ?? 0.0) : 0.0;
          for (let c = 0; c < C; c++) {
            sumQ += inputsQData[rowOffset + c] * wQData[c * H + j];
          }
          qData[outOffset + j] = sumQ;
        }
      }

      // 2. Project inputsK to K and V
      for (let t = 0; t < Lk; t++) {
        const rowOffset = (b * Lk + t) * C;
        const outOffset = (b * Lk + t) * H;

        for (let j = 0; j < H; j++) {
          let sumK = useBias ? (bKData[j] ?? 0.0) : 0.0;
          let sumV = useBias ? (bVData[j] ?? 0.0) : 0.0;

          for (let c = 0; c < C; c++) {
            const val = inputsKData[rowOffset + c];
            sumK += val * wKData[c * H + j];
            sumV += val * wVData[c * H + j];
          }

          kData[outOffset + j] = sumK;
          vData[outOffset + j] = sumV;
        }
      }

      // 3. Compute scores & Softmax
      for (let i = 0; i < Lq; i++) {
        const scoreOffset = (b * Lq + i) * Lk;
        const qOffset = (b * Lq + i) * H;

        let maxVal = -Infinity;

        for (let j = 0; j < Lk; j++) {
          const kOffset = (b * Lk + j) * H;
          let sum = 0.0;
          for (let h = 0; h < H; h++) {
            sum += qData[qOffset + h] * kData[kOffset + h];
          }
          const scoreVal = sum * scale;
          scoresData[scoreOffset + j] = scoreVal;
          if (scoreVal > maxVal) {
            maxVal = scoreVal;
          }
        }

        // Softmax
        let sumExp = 0.0;
        for (let j = 0; j < Lk; j++) {
          const sVal = scoresData[scoreOffset + j];
          const expVal = Math.exp(sVal - maxVal);
          probsData[scoreOffset + j] = expVal;
          sumExp += expVal;
        }

        for (let j = 0; j < Lk; j++) {
          probsData[scoreOffset + j] /= sumExp;
        }
      }

      // 4. Compute Output
      for (let i = 0; i < Lq; i++) {
        const probOffset = (b * Lq + i) * Lk;
        const outOffset = (b * Lq + i) * H;

        for (let h = 0; h < H; h++) {
          let sum = 0.0;
          for (let j = 0; j < Lk; j++) {
            const vOffset = (b * Lk + j) * H;
            sum += probsData[probOffset + j] * vData[vOffset + h];
          }
          outData[outOffset + h] = sum;
        }
      }
    }
  }

  const out = Matrix.fromFlat(outData, [B * Lq, H]);
  const q = Matrix.fromFlat(qData, [B * Lq, H]);
  const k = Matrix.fromFlat(kData, [B * Lk, H]);
  const v = Matrix.fromFlat(vData, [B * Lk, H]);
  const scores = Matrix.fromFlat(scoresData, [B * Lq, Lk]);
  const probs = Matrix.fromFlat(probsData, [B * Lq, Lk]);

  engine.record(
    useBias && bQ && bK && bV
      ? [inputsQ, inputsK, wQ, wK, wV, bQ, bK, bV]
      : [inputsQ, inputsK, wQ, wK, wV],
    [out],
    (grad: Matrix) => {
      const gradInQ = new Float32Array(B * Lq * C);
      const gradInK = new Float32Array(B * Lk * C);
      const gradWQ = new Float32Array(C * H);
      const gradWK = new Float32Array(C * H);
      const gradWV = new Float32Array(C * H);
      const gradBQ = new Float32Array(H);
      const gradBK = new Float32Array(H);
      const gradBV = new Float32Array(H);

      if (isNativeAvailable()) {
        attentionBackwardNative(
          grad._data,
          inputsQ._data,
          inputsK._data,
          qData,
          kData,
          vData,
          probsData,
          wQ._data,
          wK._data,
          wV._data,
          B,
          Lq,
          Lk,
          C,
          H,
          useBias,
          gradInQ,
          gradInK,
          gradWQ,
          gradWK,
          gradWV,
          gradBQ,
          gradBK,
          gradBV
        );
      } else {
        const gradOutData = grad._data;
        const inputsQData = inputsQ._data;
        const inputsKData = inputsK._data;
        const wQData = wQ._data;
        const wKData = wK._data;
        const wVData = wV._data;

        const scale = 1.0 / Math.sqrt(H);

        for (let b = 0; b < B; b++) {
          const dq = new Float32Array(Lq * H);
          const dk = new Float32Array(Lk * H);
          const dv = new Float32Array(Lk * H);
          const dprobs = new Float32Array(Lq * Lk);
          const dscores = new Float32Array(Lq * Lk);

          // 1. Output backprop
          for (let i = 0; i < Lq; i++) {
            const outOffset = (b * Lq + i) * H;
            const probOffset = (b * Lq + i) * Lk;

            for (let h = 0; h < H; h++) {
              const dOut = gradOutData[outOffset + h];

              for (let j = 0; j < Lk; j++) {
                const vOffset = (b * Lk + j) * H;
                dprobs[i * Lk + j] += dOut * vData[vOffset + h];
                dv[j * H + h] += dOut * probsData[probOffset + j];
              }
            }
          }

          // 2. Softmax backward
          for (let i = 0; i < Lq; i++) {
            const probOffset = (b * Lq + i) * Lk;

            let dotProd = 0.0;
            for (let kVal = 0; kVal < Lk; kVal++) {
              dotProd += dprobs[i * Lk + kVal] * probsData[probOffset + kVal];
            }

            for (let j = 0; j < Lk; j++) {
              const pVal = probsData[probOffset + j];
              dscores[i * Lk + j] = pVal * (dprobs[i * Lk + j] - dotProd);
            }
          }

          // 3. Score backprop
          for (let i = 0; i < Lq; i++) {
            const qOffset = (b * Lq + i) * H;

            for (let j = 0; j < Lk; j++) {
              const kOffset = (b * Lk + j) * H;
              const dsVal = dscores[i * Lk + j] * scale;

              for (let h = 0; h < H; h++) {
                dq[i * H + h] += dsVal * kData[kOffset + h];
                dk[j * H + h] += dsVal * qData[qOffset + h];
              }
            }
          }

          // 4. Linear projection backward & input gradient accumulation
          for (let t = 0; t < Lq; t++) {
            const rowOffset = (b * Lq + t) * C;

            for (let j = 0; j < H; j++) {
              const dqVal = dq[t * H + j];

              if (useBias) {
                gradBQ[j] += dqVal;
              }

              for (let c = 0; c < C; c++) {
                gradWQ[c * H + j] += inputsQData[rowOffset + c] * dqVal;
              }
            }

            // Gradient of inputsQ
            for (let c = 0; c < C; c++) {
              let sumIn = 0.0;
              for (let h = 0; h < H; h++) {
                sumIn += dq[t * H + h] * wQData[c * H + h];
              }
              gradInQ[rowOffset + c] = sumIn;
            }
          }

          for (let t = 0; t < Lk; t++) {
            const rowOffset = (b * Lk + t) * C;

            for (let j = 0; j < H; j++) {
              const dkVal = dk[t * H + j];
              const dvVal = dv[t * H + j];

              if (useBias) {
                gradBK[j] += dkVal;
                gradBV[j] += dvVal;
              }

              for (let c = 0; c < C; c++) {
                const inputVal = inputsKData[rowOffset + c];
                gradWK[c * H + j] += inputVal * dkVal;
                gradWV[c * H + j] += inputVal * dvVal;
              }
            }

            // Gradient of inputsK
            for (let c = 0; c < C; c++) {
              let sumIn = 0.0;
              for (let h = 0; h < H; h++) {
                sumIn += dk[t * H + h] * wKData[c * H + h];
                sumIn += dv[t * H + h] * wVData[c * H + h];
              }
              gradInK[rowOffset + c] = sumIn;
            }
          }
        }
      }

      const gradInputsQMatrix = Matrix.fromFlat(gradInQ, [B * Lq, C]);
      const gradInputsKMatrix = Matrix.fromFlat(gradInK, [B * Lk, C]);
      const gradWQMatrix = Matrix.fromFlat(gradWQ, [C, H]);
      const gradWKMatrix = Matrix.fromFlat(gradWK, [C, H]);
      const gradWVMatrix = Matrix.fromFlat(gradWV, [C, H]);

      if (useBias && bQ && bK && bV) {
        const gradBQMatrix = Matrix.fromFlat(gradBQ, [H, 1]);
        const gradBKMatrix = Matrix.fromFlat(gradBK, [H, 1]);
        const gradBVMatrix = Matrix.fromFlat(gradBV, [H, 1]);
        return [
          gradInputsQMatrix,
          gradInputsKMatrix,
          gradWQMatrix,
          gradWKMatrix,
          gradWVMatrix,
          gradBQMatrix,
          gradBKMatrix,
          gradBVMatrix,
        ];
      }

      return [gradInputsQMatrix, gradInputsKMatrix, gradWQMatrix, gradWKMatrix, gradWVMatrix];
    },
    { saveInput: false, saveOutput: false }
  );

  return { out, q, k, v, scores, probs };
}

export class Attention extends BaseLayer {
  public units: number;
  public useBias: boolean;
  public kernelInitializer: string;
  public biasInitializer: string;
  public sequenceLength?: number;
  public inputDim?: number;

  // External query and key matrices
  public externalQuery?: Matrix;
  public externalKey?: Matrix;

  constructor(config: AttentionConfig) {
    super(config);
    if (config.units === undefined || config.units <= 0) {
      throw new Error("[Attention] 'units' wajib berupa angka positif.");
    }
    this.units = config.units;
    this.useBias = config.useBias ?? true;
    this.kernelInitializer = config.kernelInitializer ?? "glorot_normal";
    this.biasInitializer = config.biasInitializer ?? "zeros";
    this.sequenceLength = config.sequenceLength;
    this.inputDim = config.inputDim;
  }

  public setExternal(external: {
    query?: Matrix;
    key?: Matrix;
    trainableQuery?: boolean;
    trainableKey?: boolean;
  }): void {
    if (external.query) {
      this.addParameter("externalQuery", external.query, external.trainableQuery ?? false);
      this.externalQuery = external.query;
    } else {
      this.parameters.delete("externalQuery");
      this.externalQuery = undefined;
    }

    if (external.key) {
      this.addParameter("externalKey", external.key, external.trainableKey ?? false);
      this.externalKey = external.key;
    } else {
      this.parameters.delete("externalKey");
      this.externalKey = undefined;
    }
  }


  public get wQ(): Matrix | undefined {
    return this.getParameter("wQ");
  }

  public get wK(): Matrix | undefined {
    return this.getParameter("wK");
  }

  public get wV(): Matrix | undefined {
    return this.getParameter("wV");
  }

  public get bQ(): Matrix | undefined {
    return this.getParameter("bQ");
  }

  public get bK(): Matrix | undefined {
    return this.getParameter("bK");
  }

  public get bV(): Matrix | undefined {
    return this.getParameter("bV");
  }

  public validateInputShape(inputs: Matrix): void {
    if (!this.isBuilt) return;

    const qSource = this.externalQuery ?? inputs;
    const actualRows = qSource._shape[0];
    const actualCols = qSource._shape[1] ?? 1;

    if (actualCols !== this.inputDim) {
      throw new Error(
        `[${this.name}] Input shape mismatch. Expected input channels (inputDim) to be ${this.inputDim}, got ${actualCols}.`
      );
    }

    const seqLenQ = this.sequenceLength!;
    if (actualRows % seqLenQ !== 0) {
      throw new Error(
        `[${this.name}] Input shape mismatch. Total query rows (${actualRows}) must be a multiple of sequenceLength (${seqLenQ}).`
      );
    }

    if (this.externalKey) {
      const kRows = this.externalKey._shape[0];
      const kCols = this.externalKey._shape[1] ?? 1;
      const B = Math.floor(actualRows / seqLenQ);
      if (kCols !== this.inputDim) {
        throw new Error(
          `[${this.name}] External key shape mismatch. Expected input channels to be ${this.inputDim}, got ${kCols}.`
        );
      }
      if (kRows % B !== 0) {
        throw new Error(
          `[${this.name}] External key shape mismatch. Total key rows (${kRows}) must be a multiple of batch size (${B}).`
        );
      }
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
        throw new Error("[Attention] 'sequenceLength' harus ditentukan dalam config jika inputShape 2D.");
      }
    }

    if (batch === -1) {
      return [-1, this.units];
    }

    return [batch * L, this.units];
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
          "[Attention] 'sequenceLength' harus ditentukan dalam config atau inputShape harus berupa 3D [batch, sequenceLength, inputDim]."
        );
      }
    }

    this.sequenceLength = seqLen;
    this.inputDim = inCols;

    this.outputShape = this.computeOutputShape(inputShape);

    // Initializer untuk W_q, W_k, W_v -> [inputDim, units]
    const wQVal = this.createInitializer(this.kernelInitializer, [this.inputDim, this.units]);
    const wKVal = this.createInitializer(this.kernelInitializer, [this.inputDim, this.units]);
    const wVVal = this.createInitializer(this.kernelInitializer, [this.inputDim, this.units]);

    this.addParameter("wQ", wQVal, true, [this.inputDim, this.units]);
    this.addParameter("wK", wKVal, true, [this.inputDim, this.units]);
    this.addParameter("wV", wVVal, true, [this.inputDim, this.units]);

    // Initializer untuk bias -> [units, 1]
    if (this.useBias) {
      const bQVal = this.createInitializer(this.biasInitializer, [this.units, 1]);
      const bKVal = this.createInitializer(this.biasInitializer, [this.units, 1]);
      const bVVal = this.createInitializer(this.biasInitializer, [this.units, 1]);

      this.addParameter("bQ", bQVal, true, [this.units, 1]);
      this.addParameter("bK", bKVal, true, [this.units, 1]);
      this.addParameter("bV", bVVal, true, [this.units, 1]);
    }

    this.isBuilt = true;
  }

  protected compute(inputs: Matrix, options?: ForwardOptions): Matrix {
    const wQ = this.wQ;
    const wK = this.wK;
    const wV = this.wV;
    if (!wQ || !wK || !wV) {
      throw new Error("[Attention] Bobot 'wQ', 'wK', atau 'wV' tidak terinisialisasi.");
    }

    const qSource = this.externalQuery ?? inputs;
    const kSource = this.externalKey ?? inputs;

    const totalRowsQ = qSource._shape[0];
    const Lq = this.sequenceLength!;
    const B = Math.floor(totalRowsQ / Lq);
    const Lk = Math.floor(kSource._shape[0] / B);
    const C = this.inputDim!;

    const { out } = attentionForward(
      qSource,
      kSource,
      wQ,
      wK,
      wV,
      this.useBias ? this.bQ : undefined,
      this.useBias ? this.bK : undefined,
      this.useBias ? this.bV : undefined,
      B,
      Lq,
      Lk,
      C,
      this.units,
      this.useBias
    );

    return out;
  }

  public getConfig(): Record<string, any> {
    return {
      ...super.getConfig(),
      units: this.units,
      useBias: this.useBias,
      kernelInitializer: this.kernelInitializer,
      biasInitializer: this.biasInitializer,
      sequenceLength: this.sequenceLength,
      inputDim: this.inputDim,
    };
  }
}
