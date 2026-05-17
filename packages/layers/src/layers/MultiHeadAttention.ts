import { BaseLayer, LayerConfig, type ForwardOptions } from "../base/BaseLayer.js";
import { Matrix, mj, engine } from "@oxide-js/core";
import { isNativeAvailable, multiHeadAttentionForwardNative, multiHeadAttentionBackwardNative } from "../rust_backend.js";

export interface MultiHeadAttentionConfig extends LayerConfig {
  numHeads: number;
  keyDim: number;
  valueDim?: number;
  outputDim?: number;
  useBias?: boolean;
  kernelInitializer?: string;
  biasInitializer?: string;
  sequenceLength?: number;
  inputDim?: number;
}

function multiHeadAttentionForward(
  inputsQ: Matrix,
  inputsK: Matrix,
  inputsV: Matrix,
  wQ: Matrix,
  wK: Matrix,
  wV: Matrix,
  wO: Matrix,
  bQ: Matrix | undefined,
  bK: Matrix | undefined,
  bV: Matrix | undefined,
  bO: Matrix | undefined,
  batchSize: number,
  seqLenQ: number,
  seqLenK: number,
  inputDimQ: number,
  inputDimK: number,
  inputDimV: number,
  numHeads: number,
  keyDim: number,
  valueDim: number,
  outputDim: number,
  useBias: boolean
): { out: Matrix; q: Matrix; k: Matrix; v: Matrix; scores: Matrix; probs: Matrix; outConcat: Matrix } {
  const B = batchSize;
  const Lq = seqLenQ;
  const Lk = seqLenK;
  const Cq = inputDimQ;
  const Ck = inputDimK;
  const Cv = inputDimV;
  const H = numHeads;
  const Kd = keyDim;
  const Vd = valueDim;
  const Od = outputDim;

  const outData = new Float32Array(B * Lq * Od);
  const qData = new Float32Array(B * Lq * H * Kd);
  const kData = new Float32Array(B * Lk * H * Kd);
  const vData = new Float32Array(B * Lk * H * Vd);
  const scoresData = new Float32Array(B * Lq * H * Lk);
  const probsData = new Float32Array(B * Lq * H * Lk);
  const outConcatData = new Float32Array(B * Lq * H * Vd);

  const bQParam = bQ ?? mj.zeros([H * Kd, 1]);
  const bKParam = bK ?? mj.zeros([H * Kd, 1]);
  const bVParam = bV ?? mj.zeros([H * Vd, 1]);
  const bOParam = bO ?? mj.zeros([Od, 1]);

  if (isNativeAvailable()) {
    multiHeadAttentionForwardNative(
      inputsQ._data,
      inputsK._data,
      inputsV._data,
      wQ._data,
      wK._data,
      wV._data,
      wO._data,
      bQParam._data,
      bKParam._data,
      bVParam._data,
      bOParam._data,
      B,
      Lq,
      Lk,
      Cq,
      Ck,
      Cv,
      H,
      Kd,
      Vd,
      Od,
      useBias,
      outData,
      qData,
      kData,
      vData,
      scoresData,
      probsData,
      outConcatData
    );
  } else {
    const inputsQData = inputsQ._data;
    const inputsKData = inputsK._data;
    const inputsVData = inputsV._data;
    const wQData = wQ._data;
    const wKData = wK._data;
    const wVData = wV._data;
    const wOData = wO._data;
    const bQData = bQParam._data;
    const bKData = bKParam._data;
    const bVData = bVParam._data;
    const bOData = bOParam._data;

    const scale = 1.0 / Math.sqrt(Kd);

    // 1. Projections Q, K, V
    for (let b = 0; b < B; b++) {
      // Project Q
      for (let t = 0; t < Lq; t++) {
        const rowOffset = (b * Lq + t) * Cq;
        const outOffset = (b * Lq + t) * H * Kd;

        for (let j = 0; j < H * Kd; j++) {
          let sum = useBias ? (bQData[j] ?? 0.0) : 0.0;
          for (let c = 0; c < Cq; c++) {
            sum += inputsQData[rowOffset + c] * wQData[c * H * Kd + j];
          }
          qData[outOffset + j] = sum;
        }
      }

      // Project K
      for (let t = 0; t < Lk; t++) {
        const rowOffset = (b * Lk + t) * Ck;
        const outOffset = (b * Lk + t) * H * Kd;

        for (let j = 0; j < H * Kd; j++) {
          let sum = useBias ? (bKData[j] ?? 0.0) : 0.0;
          for (let c = 0; c < Ck; c++) {
            sum += inputsKData[rowOffset + c] * wKData[c * H * Kd + j];
          }
          kData[outOffset + j] = sum;
        }
      }

      // Project V
      for (let t = 0; t < Lk; t++) {
        const rowOffset = (b * Lk + t) * Cv;
        const outOffset = (b * Lk + t) * H * Vd;

        for (let j = 0; j < H * Vd; j++) {
          let sum = useBias ? (bVData[j] ?? 0.0) : 0.0;
          for (let c = 0; c < Cv; c++) {
            sum += inputsVData[rowOffset + c] * wVData[c * H * Vd + j];
          }
          vData[outOffset + j] = sum;
        }
      }
    }

    // 2. Head Attention calculations
    for (let b = 0; b < B; b++) {
      for (let h = 0; h < H; h++) {
        // Calculate scores & Softmax
        for (let i = 0; i < Lq; i++) {
          const qRowOffset = (b * Lq + i) * H * Kd + h * Kd;
          const scoreRowOffset = (b * Lq + i) * H * Lk + h * Lk;

          let maxVal = -Infinity;

          for (let j = 0; j < Lk; j++) {
            const kRowOffset = (b * Lk + j) * H * Kd + h * Kd;
            let sum = 0.0;
            for (let d = 0; d < Kd; d++) {
              sum += qData[qRowOffset + d] * kData[kRowOffset + d];
            }
            const scoreVal = sum * scale;
            scoresData[scoreRowOffset + j] = scoreVal;
            if (scoreVal > maxVal) {
              maxVal = scoreVal;
            }
          }

          // Softmax
          let sumExp = 0.0;
          for (let j = 0; j < Lk; j++) {
            const sVal = scoresData[scoreRowOffset + j];
            const expVal = Math.exp(sVal - maxVal);
            probsData[scoreRowOffset + j] = expVal;
            sumExp += expVal;
          }

          for (let j = 0; j < Lk; j++) {
            probsData[scoreRowOffset + j] /= sumExp;
          }
        }

        // Context Value representation
        for (let i = 0; i < Lq; i++) {
          const probRowOffset = (b * Lq + i) * H * Lk + h * Lk;
          const concatRowOffset = (b * Lq + i) * H * Vd + h * Vd;

          for (let d = 0; d < Vd; d++) {
            let sum = 0.0;
            for (let j = 0; j < Lk; j++) {
              const vRowOffset = (b * Lk + j) * H * Vd + h * Vd;
              sum += probsData[probRowOffset + j] * vData[vRowOffset + d];
            }
            outConcatData[concatRowOffset + d] = sum;
          }
        }
      }
    }

    // 3. Final Output Projection
    const concatDim = H * Vd;
    for (let i = 0; i < B * Lq; i++) {
      for (let j = 0; j < Od; j++) {
        let sum = useBias ? (bOData[j] ?? 0.0) : 0.0;
        for (let c = 0; c < concatDim; c++) {
          sum += outConcatData[i * concatDim + c] * wOData[c * Od + j];
        }
        outData[i * Od + j] = sum;
      }
    }
  }

  const out = Matrix.fromFlat(outData, [B * Lq, Od]);
  const q = Matrix.fromFlat(qData, [B * Lq, H * Kd]);
  const k = Matrix.fromFlat(kData, [B * Lk, H * Kd]);
  const v = Matrix.fromFlat(vData, [B * Lk, H * Vd]);
  const scores = Matrix.fromFlat(scoresData, [B * Lq, H * Lk]);
  const probs = Matrix.fromFlat(probsData, [B * Lq, H * Lk]);
  const outConcat = Matrix.fromFlat(outConcatData, [B * Lq, H * Vd]);

  engine.record(
    useBias && bQ && bK && bV && bO
      ? [inputsQ, inputsK, inputsV, wQ, wK, wV, wO, bQ, bK, bV, bO]
      : [inputsQ, inputsK, inputsV, wQ, wK, wV, wO],
    [out],
    (grad: Matrix) => {
      const gradInQ = new Float32Array(B * Lq * Cq);
      const gradInK = new Float32Array(B * Lk * Ck);
      const gradInV = new Float32Array(B * Lk * Cv);
      const gradWQ = new Float32Array(Cq * H * Kd);
      const gradWK = new Float32Array(Ck * H * Kd);
      const gradWV = new Float32Array(Cv * H * Vd);
      const gradWO = new Float32Array(H * Vd * Od);
      const gradBQ = new Float32Array(H * Kd);
      const gradBK = new Float32Array(H * Kd);
      const gradBV = new Float32Array(H * Vd);
      const gradBO = new Float32Array(Od);

      if (isNativeAvailable()) {
        multiHeadAttentionBackwardNative(
          grad._data,
          inputsQ._data,
          inputsK._data,
          inputsV._data,
          qData,
          kData,
          vData,
          probsData,
          outConcatData,
          wQ._data,
          wK._data,
          wV._data,
          wO._data,
          B,
          Lq,
          Lk,
          Cq,
          Ck,
          Cv,
          H,
          Kd,
          Vd,
          Od,
          useBias,
          gradInQ,
          gradInK,
          gradInV,
          gradWQ,
          gradWK,
          gradWV,
          gradWO,
          gradBQ,
          gradBK,
          gradBV,
          gradBO
        );
      } else {
        const gradOutData = grad._data;
        const inputsQData = inputsQ._data;
        const inputsKData = inputsK._data;
        const inputsVData = inputsV._data;
        const wQData = wQ._data;
        const wKData = wK._data;
        const wVData = wV._data;
        const wOData = wO._data;

        const scale = 1.0 / Math.sqrt(Kd);
        const concatDim = H * Vd;

        // 1. Output Projector Backpropagation
        const gradOutConcatData = new Float32Array(B * Lq * concatDim);
        for (let i = 0; i < B * Lq; i++) {
          for (let j = 0; j < Od; j++) {
            const dOut = gradOutData[i * Od + j];
            if (useBias) {
              gradBO[j] += dOut;
            }
            for (let c = 0; c < concatDim; c++) {
              gradWO[c * Od + j] += outConcatData[i * concatDim + c] * dOut;
              gradOutConcatData[i * concatDim + c] += dOut * wOData[c * Od + j];
            }
          }
        }

        // 2. Attention backward to Q_h, K_h, V_h
        const gradQData = new Float32Array(B * Lq * H * Kd);
        const gradKData = new Float32Array(B * Lk * H * Kd);
        const gradVData = new Float32Array(B * Lk * H * Vd);

        for (let b = 0; b < B; b++) {
          for (let h = 0; h < H; h++) {
            const dprobs = new Float32Array(Lq * Lk);
            const dscores = new Float32Array(Lq * Lk);

            // Output backprop to probs and local V
            for (let i = 0; i < Lq; i++) {
              const probRowOffset = (b * Lq + i) * H * Lk + h * Lk;
              const concatRowOffset = (b * Lq + i) * H * Vd + h * Vd;

              for (let d = 0; d < Vd; d++) {
                const dOutConcat = gradOutConcatData[concatRowOffset + d];

                for (let j = 0; j < Lk; j++) {
                  const vRowOffset = (b * Lk + j) * H * Vd + h * Vd;
                  dprobs[i * Lk + j] += dOutConcat * vData[vRowOffset + d];
                  gradVData[vRowOffset + d] += dOutConcat * probsData[probRowOffset + j];
                }
              }
            }

            // Softmax backward
            for (let i = 0; i < Lq; i++) {
              const probRowOffset = (b * Lq + i) * H * Lk + h * Lk;

              let dotProd = 0.0;
              for (let kIdx = 0; kIdx < Lk; kIdx++) {
                dotProd += dprobs[i * Lk + kIdx] * probsData[probRowOffset + kIdx];
              }

              for (let j = 0; j < Lk; j++) {
                const pVal = probsData[probRowOffset + j];
                dscores[i * Lk + j] = pVal * (dprobs[i * Lk + j] - dotProd);
              }
            }

            // Scores backprop to local Q and K
            for (let i = 0; i < Lq; i++) {
              const qRowOffset = (b * Lq + i) * H * Kd + h * Kd;

              for (let j = 0; j < Lk; j++) {
                const kRowOffset = (b * Lk + j) * H * Kd + h * Kd;
                const dsVal = dscores[i * Lk + j] * scale;

                for (let d = 0; d < Kd; d++) {
                  gradQData[qRowOffset + d] += dsVal * kData[kRowOffset + d];
                  gradKData[kRowOffset + d] += dsVal * qData[qRowOffset + d];
                }
              }
            }
          }
        }

        // 3. Projections Backward Pass
        // Q Projection Backward
        for (let b = 0; b < B; b++) {
          for (let t = 0; t < Lq; t++) {
            const rowOffset = (b * Lq + t) * Cq;
            const outOffset = (b * Lq + t) * H * Kd;

            for (let j = 0; j < H * Kd; j++) {
              const dqVal = gradQData[outOffset + j];
              if (useBias) {
                gradBQ[j] += dqVal;
              }
              for (let c = 0; c < Cq; c++) {
                gradWQ[c * H * Kd + j] += inputsQData[rowOffset + c] * dqVal;
              }
            }

            // Gradient inputsQ
            for (let c = 0; c < Cq; c++) {
              let sum = 0.0;
              for (let j = 0; j < H * Kd; j++) {
                sum += gradQData[outOffset + j] * wQData[c * H * Kd + j];
              }
              gradInQ[rowOffset + c] = sum;
            }
          }
        }

        // K Projection Backward
        for (let b = 0; b < B; b++) {
          for (let t = 0; t < Lk; t++) {
            const rowOffset = (b * Lk + t) * Ck;
            const outOffset = (b * Lk + t) * H * Kd;

            for (let j = 0; j < H * Kd; j++) {
              const dkVal = gradKData[outOffset + j];
              if (useBias) {
                gradBK[j] += dkVal;
              }
              for (let c = 0; c < Ck; c++) {
                gradWK[c * H * Kd + j] += inputsKData[rowOffset + c] * dkVal;
              }
            }

            // Gradient inputsK
            for (let c = 0; c < Ck; c++) {
              let sum = 0.0;
              for (let j = 0; j < H * Kd; j++) {
                sum += gradKData[outOffset + j] * wKData[c * H * Kd + j];
              }
              gradInK[rowOffset + c] = sum;
            }
          }
        }

        // V Projection Backward
        for (let b = 0; b < B; b++) {
          for (let t = 0; t < Lk; t++) {
            const rowOffset = (b * Lk + t) * Cv;
            const outOffset = (b * Lk + t) * H * Vd;

            for (let j = 0; j < H * Vd; j++) {
              const dvVal = gradVData[outOffset + j];
              if (useBias) {
                gradBV[j] += dvVal;
              }
              for (let c = 0; c < Cv; c++) {
                gradWV[c * H * Vd + j] += inputsVData[rowOffset + c] * dvVal;
              }
            }

            // Gradient inputsV
            for (let c = 0; c < Cv; c++) {
              let sum = 0.0;
              for (let j = 0; j < H * Vd; j++) {
                sum += gradVData[outOffset + j] * wVData[c * H * Vd + j];
              }
              gradInV[rowOffset + c] = sum;
            }
          }
        }
      }

      const gradInputsQMatrix = Matrix.fromFlat(gradInQ, [B * Lq, Cq]);
      const gradInputsKMatrix = Matrix.fromFlat(gradInK, [B * Lk, Ck]);
      const gradInputsVMatrix = Matrix.fromFlat(gradInV, [B * Lk, Cv]);
      const gradWQMatrix = Matrix.fromFlat(gradWQ, [Cq, H * Kd]);
      const gradWKMatrix = Matrix.fromFlat(gradWK, [Ck, H * Kd]);
      const gradWVMatrix = Matrix.fromFlat(gradWV, [Cv, H * Vd]);
      const gradWOMatrix = Matrix.fromFlat(gradWO, [H * Vd, Od]);

      if (useBias && bQ && bK && bV && bO) {
        const gradBQMatrix = Matrix.fromFlat(gradBQ, [H * Kd, 1]);
        const gradBKMatrix = Matrix.fromFlat(gradBK, [H * Kd, 1]);
        const gradBVMatrix = Matrix.fromFlat(gradBV, [H * Vd, 1]);
        const gradBOMatrix = Matrix.fromFlat(gradBO, [Od, 1]);
        return [
          gradInputsQMatrix,
          gradInputsKMatrix,
          gradInputsVMatrix,
          gradWQMatrix,
          gradWKMatrix,
          gradWVMatrix,
          gradWOMatrix,
          gradBQMatrix,
          gradBKMatrix,
          gradBVMatrix,
          gradBOMatrix,
        ];
      }

      return [
        gradInputsQMatrix,
        gradInputsKMatrix,
        gradInputsVMatrix,
        gradWQMatrix,
        gradWKMatrix,
        gradWVMatrix,
        gradWOMatrix,
      ];
    },
    { saveInput: false, saveOutput: false }
  );

  return { out, q, k, v, scores, probs, outConcat };
}

export class MultiHeadAttention extends BaseLayer {
  public numHeads: number;
  public keyDim: number;
  public valueDim: number;
  public outputDim?: number;
  public useBias: boolean;
  public kernelInitializer: string;
  public biasInitializer: string;
  public sequenceLength?: number;
  public inputDim?: number;

  // External query, key, and value matrices
  public externalQuery?: Matrix;
  public externalKey?: Matrix;
  public externalValue?: Matrix;

  constructor(config: MultiHeadAttentionConfig) {
    super(config);
    if (config.numHeads === undefined || config.numHeads <= 0) {
      throw new Error("[MultiHeadAttention] 'numHeads' wajib berupa angka positif.");
    }
    if (config.keyDim === undefined || config.keyDim <= 0) {
      throw new Error("[MultiHeadAttention] 'keyDim' wajib berupa angka positif.");
    }
    this.numHeads = config.numHeads;
    this.keyDim = config.keyDim;
    this.valueDim = config.valueDim ?? config.keyDim;
    this.outputDim = config.outputDim;
    this.useBias = config.useBias ?? true;
    this.kernelInitializer = config.kernelInitializer ?? "glorot_normal";
    this.biasInitializer = config.biasInitializer ?? "zeros";
    this.sequenceLength = config.sequenceLength;
    this.inputDim = config.inputDim;
  }

  public setExternal(external: {
    query?: Matrix;
    key?: Matrix;
    value?: Matrix;
    trainableQuery?: boolean;
    trainableKey?: boolean;
    trainableValue?: boolean;
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

    if (external.value) {
      this.addParameter("externalValue", external.value, external.trainableValue ?? false);
      this.externalValue = external.value;
    } else {
      this.parameters.delete("externalValue");
      this.externalValue = undefined;
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

  public get wO(): Matrix | undefined {
    return this.getParameter("wO");
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

  public get bO(): Matrix | undefined {
    return this.getParameter("bO");
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
        throw new Error("[MultiHeadAttention] 'sequenceLength' harus ditentukan dalam config jika inputShape 2D.");
      }
    }

    const oDim = this.outputDim ?? (this.inputDim ?? 1);
    if (batch === -1) {
      return [-1, oDim];
    }

    return [batch * L, oDim];
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
          "[MultiHeadAttention] 'sequenceLength' harus ditentukan dalam config atau inputShape harus berupa 3D [batch, sequenceLength, inputDim]."
        );
      }
    }

    this.sequenceLength = seqLen;
    this.inputDim = inCols;

    if (this.outputDim === undefined) {
      this.outputDim = this.inputDim;
    }

    this.outputShape = this.computeOutputShape(inputShape);

    // Initializer untuk W_q, W_k, W_v
    const wQVal = this.createInitializer(this.kernelInitializer, [this.inputDim, this.numHeads * this.keyDim]);
    const wKVal = this.createInitializer(this.kernelInitializer, [this.inputDim, this.numHeads * this.keyDim]);
    const wVVal = this.createInitializer(this.kernelInitializer, [this.inputDim, this.numHeads * this.valueDim]);
    const wOVal = this.createInitializer(this.kernelInitializer, [this.numHeads * this.valueDim, this.outputDim]);

    this.addParameter("wQ", wQVal, true, [this.inputDim, this.numHeads * this.keyDim]);
    this.addParameter("wK", wKVal, true, [this.inputDim, this.numHeads * this.keyDim]);
    this.addParameter("wV", wVVal, true, [this.inputDim, this.numHeads * this.valueDim]);
    this.addParameter("wO", wOVal, true, [this.numHeads * this.valueDim, this.outputDim]);

    // Initializer untuk bias
    if (this.useBias) {
      const bQVal = this.createInitializer(this.biasInitializer, [this.numHeads * this.keyDim, 1]);
      const bKVal = this.createInitializer(this.biasInitializer, [this.numHeads * this.keyDim, 1]);
      const bVVal = this.createInitializer(this.biasInitializer, [this.numHeads * this.valueDim, 1]);
      const bOVal = this.createInitializer(this.biasInitializer, [this.outputDim, 1]);

      this.addParameter("bQ", bQVal, true, [this.numHeads * this.keyDim, 1]);
      this.addParameter("bK", bKVal, true, [this.numHeads * this.keyDim, 1]);
      this.addParameter("bV", bVVal, true, [this.numHeads * this.valueDim, 1]);
      this.addParameter("bO", bOVal, true, [this.outputDim, 1]);
    }

    this.isBuilt = true;
  }

  protected compute(inputs: Matrix, options?: ForwardOptions): Matrix {
    const wQ = this.wQ;
    const wK = this.wK;
    const wV = this.wV;
    const wO = this.wO;
    if (!wQ || !wK || !wV || !wO) {
      throw new Error("[MultiHeadAttention] Bobot 'wQ', 'wK', 'wV', atau 'wO' tidak terinisialisasi.");
    }

    const qSource = this.externalQuery ?? inputs;
    const kSource = this.externalKey ?? this.externalQuery ?? inputs;
    const vSource = this.externalValue ?? this.externalKey ?? this.externalQuery ?? inputs;

    const totalRowsQ = qSource._shape[0];
    const Lq = this.sequenceLength!;
    const B = Math.floor(totalRowsQ / Lq);
    const Lk = Math.floor(kSource._shape[0] / B);

    const Cq = qSource._shape[1] ?? 1;
    const Ck = kSource._shape[1] ?? 1;
    const Cv = vSource._shape[1] ?? 1;

    const { out } = multiHeadAttentionForward(
      qSource,
      kSource,
      vSource,
      wQ,
      wK,
      wV,
      wO,
      this.useBias ? this.bQ : undefined,
      this.useBias ? this.bK : undefined,
      this.useBias ? this.bV : undefined,
      this.useBias ? this.bO : undefined,
      B,
      Lq,
      Lk,
      Cq,
      Ck,
      Cv,
      this.numHeads,
      this.keyDim,
      this.valueDim,
      this.outputDim!,
      this.useBias
    );

    return out;
  }

  public getConfig(): Record<string, any> {
    return {
      ...super.getConfig(),
      numHeads: this.numHeads,
      keyDim: this.keyDim,
      valueDim: this.valueDim,
      outputDim: this.outputDim,
      useBias: this.useBias,
      kernelInitializer: this.kernelInitializer,
      biasInitializer: this.biasInitializer,
      sequenceLength: this.sequenceLength,
      inputDim: this.inputDim,
    };
  }
}
