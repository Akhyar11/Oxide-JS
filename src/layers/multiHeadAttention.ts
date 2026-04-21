import { Cost, Optimzier, OptimzierType, StatusLayer } from "../@types/type";
import mj from "../math";
import { isNativeAvailable, multiHeadAttentionBackwardNative, multiHeadAttentionForwardNative } from "../math/rust_backend";
import Matrix from "../matrix";
import setOptimizer from "../utils/setOptimizer";
import Dense from "./dense";

interface MultiHeadAttentionLayer {
  units: number;
  heads: number;
  seqLen: number;
  alpha?: number;
  status?: StatusLayer;
  clipGradient?: number | boolean;
}

export default class MultiHeadAttention {
  name = "multi head attention layer";
  units: number;
  heads: number;
  headUnits: number;
  seqLen: number;
  alpha: number;
  status: StatusLayer;
  clipGradient: number | boolean;

  q: Matrix;
  k: Matrix;
  v: Matrix;
  wo: Dense;

  inputShape: [number, number];
  outputShape: [number, number];
  params: number;
  loss: number = 0;

  private input: Matrix = mj.matrix([]);
  private padMask: boolean[] = [];
  private hasExternalPadMask: boolean = false;
  private padMaskSourceRef: Float32Array | null = null;

  private optimizerQ: OptimzierType;
  private optimizerK: OptimzierType;
  private optimizerV: OptimzierType;
  private optimizerName: Optimzier = "sgd";

  private Q: Matrix;
  private K: Matrix;
  private V: Matrix;
  private concatenated: Matrix;

  private gradInputBuffer: Matrix;
  private gradContributionBuffer: Matrix;
  private dQAll: Matrix;
  private dKAll: Matrix;
  private dVAll: Matrix;

  private gradQBuffer: Matrix;
  private gradKBuffer: Matrix;
  private gradVBuffer: Matrix;

  private oldQBuffer: Matrix;
  private oldKBuffer: Matrix;
  private oldVBuffer: Matrix;

  private attentionData: Float32Array = new Float32Array(0);
  private errAttentionScratch: Float32Array;
  private errScoreScratch: Float32Array;

  constructor({ units, heads, seqLen, alpha = 0.1, status = "input", clipGradient = 5.0 }: MultiHeadAttentionLayer) {
    this.units = units;
    this.heads = heads;
    this.seqLen = seqLen;
    this.alpha = alpha;
    this.status = status;
    this.clipGradient = clipGradient;

    this.inputShape = [units, seqLen];
    this.outputShape = [units, seqLen];

    if (this.units % this.heads !== 0) {
      throw new Error(`units (${units}) must be divisible by heads (${heads})`);
    }
    this.headUnits = this.units / this.heads;

    this.q = mj.xavier([this.units, this.units]);
    this.k = mj.xavier([this.units, this.units]);
    this.v = mj.xavier([this.units, this.units]);

    this.wo = new Dense({
      units: this.units,
      outputUnits: this.units,
      activation: "linear",
      alpha,
      clipGradient,
    });

    this.optimizerQ = setOptimizer(this.optimizerName, this.q._shape, alpha);
    this.optimizerK = setOptimizer(this.optimizerName, this.k._shape, alpha);
    this.optimizerV = setOptimizer(this.optimizerName, this.v._shape, alpha);

    this.Q = mj.zeros([this.units, seqLen]);
    this.K = mj.zeros([this.units, seqLen]);
    this.V = mj.zeros([this.units, seqLen]);
    this.concatenated = mj.zeros([this.units, seqLen]);

    this.gradInputBuffer = mj.zeros([this.units, seqLen]);
    this.gradContributionBuffer = mj.zeros([this.units, seqLen]);
    this.dQAll = mj.zeros([this.units, seqLen]);
    this.dKAll = mj.zeros([this.units, seqLen]);
    this.dVAll = mj.zeros([this.units, seqLen]);

    this.gradQBuffer = mj.zeros([this.units, this.units]);
    this.gradKBuffer = mj.zeros([this.units, this.units]);
    this.gradVBuffer = mj.zeros([this.units, this.units]);

    this.oldQBuffer = mj.zeros([this.units, this.units]);
    this.oldKBuffer = mj.zeros([this.units, this.units]);
    this.oldVBuffer = mj.zeros([this.units, this.units]);

    this.params = 3 * this.units * this.units + this.wo.params;
    this.errAttentionScratch = new Float32Array(this.seqLen * this.seqLen);
    this.errScoreScratch = new Float32Array(this.seqLen * this.seqLen);
    this.ensureSequenceBuffersForBatch(seqLen);
  }

  compile({ alpha, optimizer, error, clipGradient }: { alpha?: number; optimizer?: Optimzier; error?: Cost; clipGradient?: number | boolean }) {
    if (alpha !== undefined) this.alpha = alpha;
    if (optimizer !== undefined) {
      this.optimizerName = optimizer;
      this.optimizerQ = setOptimizer(optimizer, this.q._shape, this.alpha);
      this.optimizerK = setOptimizer(optimizer, this.k._shape, this.alpha);
      this.optimizerV = setOptimizer(optimizer, this.v._shape, this.alpha);
    }
    this.wo.compile({ alpha, optimizer, error });
  }

  setPadMask(padMask: boolean[]): void {
    this.padMask = padMask;
    this.hasExternalPadMask = true;
    this.padMaskSourceRef = null;
  }

  forward(x: Matrix): Matrix {
    const totalCols = x._shape[1];
    const seqLen = this.seqLen;
    if (totalCols % seqLen !== 0) {
      throw new Error(`MultiHeadAttention.forward: totalCols (${totalCols}) is not divisible by seqLen (${seqLen})`);
    }
    const batchSize = totalCols / seqLen;

    this.ensureSequenceBuffersForBatch(totalCols);

    this.input = x;
    if (this.hasExternalPadMask && this.padMask.length === totalCols) {
      // Gunakan mask yang sudah divalidasi caller, hindari scan ulang input.
    } else if (this.padMask.length !== totalCols || this.padMaskSourceRef !== x._data) {
      this.padMask = MultiHeadAttention.detectPadColumns(x, this.padMask);
      this.hasExternalPadMask = false;
      this.padMaskSourceRef = x._data;
    }

    mj.dotProduct(this.q, x, this.Q);
    mj.dotProduct(this.k, x, this.K);
    mj.dotProduct(this.v, x, this.V);

    const scale = 1 / Math.sqrt(this.headUnits);

    if (isNativeAvailable()) {
      multiHeadAttentionForwardNative(
        this.Q._data,
        this.K._data,
        this.V._data,
        this.padMask,
        this.heads,
        this.headUnits,
        seqLen,
        batchSize,
        scale,
        this.concatenated._data,
        this.attentionData
      );
    } else {
      MultiHeadAttention.forwardFallback(
        this.Q._data,
        this.K._data,
        this.V._data,
        this.padMask,
        this.heads,
        this.headUnits,
        seqLen,
        batchSize,
        scale,
        this.concatenated._data,
        this.attentionData
      );
    }

    this.outputShape = [this.concatenated._shape[0], this.concatenated._shape[1]];
    return this.wo.forward(this.concatenated);
  }

  backward(y: Matrix, err: Matrix): Matrix {
    const dCat = this.wo.backward(y, err);
    const totalCols = dCat._shape[1];
    const seqLen = this.seqLen;
    if (totalCols % seqLen !== 0) {
      throw new Error(`MultiHeadAttention.backward: totalCols (${totalCols}) is not divisible by seqLen (${seqLen})`);
    }
    const batchSize = totalCols / seqLen;
    const scale = 1 / Math.sqrt(this.headUnits);

    if (isNativeAvailable()) {
      multiHeadAttentionBackwardNative(
        this.Q._data,
        this.K._data,
        this.V._data,
        this.attentionData,
        dCat._data,
        this.padMask,
        this.heads,
        this.headUnits,
        seqLen,
        batchSize,
        scale,
        this.dQAll._data,
        this.dKAll._data,
        this.dVAll._data
      );
    } else {
      MultiHeadAttention.backwardFallback(
        this.Q._data,
        this.K._data,
        this.V._data,
        this.attentionData,
        dCat._data,
        this.padMask,
        this.heads,
        this.headUnits,
        seqLen,
        batchSize,
        scale,
        this.dQAll._data,
        this.dKAll._data,
        this.dVAll._data,
        this.errAttentionScratch,
        this.errScoreScratch
      );
    }

    const gradQ = mj.dotProduct(this.dQAll, this.input, this.gradQBuffer, false, true);
    const gradK = mj.dotProduct(this.dKAll, this.input, this.gradKBuffer, false, true);
    const gradV = mj.dotProduct(this.dVAll, this.input, this.gradVBuffer, false, true);

    if (this.clipGradient !== false) {
      const limit = typeof this.clipGradient === "number" ? this.clipGradient : 5.0;
      this.clipGradients(gradQ, limit);
      this.clipGradients(gradK, limit);
      this.clipGradients(gradV, limit);
    }

    this.oldQBuffer.copyFrom(this.q);
    this.oldKBuffer.copyFrom(this.k);
    this.oldVBuffer.copyFrom(this.v);

    this.q.subInPlace(this.optimizerQ.calculate(gradQ, this.alpha));
    this.k.subInPlace(this.optimizerK.calculate(gradK, this.alpha));
    this.v.subInPlace(this.optimizerV.calculate(gradV, this.alpha));

    const gradInput = mj.dotProduct(this.oldQBuffer, this.dQAll, this.gradInputBuffer, true, false);
    mj.dotProduct(this.oldKBuffer, this.dKAll, this.gradContributionBuffer, true, false);
    gradInput.addInPlace(this.gradContributionBuffer);
    mj.dotProduct(this.oldVBuffer, this.dVAll, this.gradContributionBuffer, true, false);
    gradInput.addInPlace(this.gradContributionBuffer);

    return gradInput;
  }

  save() {
    return {
      name: this.name,
      status: this.status,
      units: this.units,
      heads: this.heads,
      seqLen: this.seqLen,
      alpha: this.alpha,
      clipGradient: this.clipGradient,
      q: this.q._value,
      k: this.k._value,
      v: this.v._value,
      wo: this.wo.save(),
    };
  }

  load(data: any) {
    if (data.q && data.k && data.v) {
      this.q._value = data.q;
      this.k._value = data.k;
      this.v._value = data.v;
    } else if (data.attentionHeads) {
      this.loadLegacyHeads(data.attentionHeads);
    }

    if (data.wo) {
      this.wo.load(data.wo.weight, data.wo.bias, data.wo.clipGradient);
    }
    if (data.clipGradient !== undefined) this.clipGradient = data.clipGradient;
    this.optimizerQ = setOptimizer(this.optimizerName, this.q._shape, this.alpha);
    this.optimizerK = setOptimizer(this.optimizerName, this.k._shape, this.alpha);
    this.optimizerV = setOptimizer(this.optimizerName, this.v._shape, this.alpha);
  }

  private ensureSequenceBuffersForBatch(totalCols: number) {
    const batchSize = Math.floor(totalCols / this.seqLen);
    const expectedAttentionLen = this.heads * batchSize * this.seqLen * this.seqLen;
    // `attentionData.length` dipakai sebagai cache validity signal karena buffer ini
    // bergantung langsung pada kombinasi [heads, batchSize, seqLen].
    if (this.Q._shape[1] === totalCols && this.attentionData.length === expectedAttentionLen) {
      return;
    }

    this.inputShape = [this.units, totalCols];
    this.outputShape = [this.units, totalCols];

    this.Q = mj.zeros([this.units, totalCols]);
    this.K = mj.zeros([this.units, totalCols]);
    this.V = mj.zeros([this.units, totalCols]);
    this.concatenated = mj.zeros([this.units, totalCols]);

    this.gradInputBuffer = mj.zeros([this.units, totalCols]);
    this.gradContributionBuffer = mj.zeros([this.units, totalCols]);
    this.dQAll = mj.zeros([this.units, totalCols]);
    this.dKAll = mj.zeros([this.units, totalCols]);
    this.dVAll = mj.zeros([this.units, totalCols]);

    this.attentionData = new Float32Array(this.heads * batchSize * this.seqLen * this.seqLen);
  }

  private loadLegacyHeads(headsData: Array<{ q: number[][]; k: number[][]; v: number[][] }>) {
    const fusedQ = new Float32Array(this.units * this.units);
    const fusedK = new Float32Array(this.units * this.units);
    const fusedV = new Float32Array(this.units * this.units);

    for (let head = 0; head < this.heads; head++) {
      const legacyHead = headsData[head];
      if (!legacyHead) continue;

      for (let row = 0; row < this.headUnits; row++) {
        const targetRow = head * this.headUnits + row;
        const qRow = legacyHead.q[row] ?? [];
        const kRow = legacyHead.k[row] ?? [];
        const vRow = legacyHead.v[row] ?? [];
        const offset = targetRow * this.units;

        for (let col = 0; col < this.units; col++) {
          fusedQ[offset + col] = qRow[col] ?? 0;
          fusedK[offset + col] = kRow[col] ?? 0;
          fusedV[offset + col] = vRow[col] ?? 0;
        }
      }
    }

    this.q = Matrix.fromFlat(fusedQ, [this.units, this.units]);
    this.k = Matrix.fromFlat(fusedK, [this.units, this.units]);
    this.v = Matrix.fromFlat(fusedV, [this.units, this.units]);
  }

  private clipGradients(m: Matrix, limit: number) {
    mj.clipGradients(m, limit);
  }

  private static detectPadColumns(matrix: Matrix, reuse?: boolean[]): boolean[] {
    const [rows, cols] = matrix._shape;
    const mask = reuse && reuse.length === cols ? reuse : new Array<boolean>(cols);
    mask.fill(true);
    for (let j = 0; j < cols; j++) {
      for (let i = 0; i < rows; i++) {
        if (matrix._data[i * cols + j] !== 0) {
          mask[j] = false;
          break;
        }
      }
    }
    return mask;
  }


  private static forwardFallback(
    qData: Float32Array,
    kData: Float32Array,
    vData: Float32Array,
    padMask: boolean[],
    heads: number,
    headUnits: number,
    seqLen: number,
    batchSize: number,
    scale: number,
    outData: Float32Array,
    attentionData: Float32Array
  ): void {
    const totalCols = seqLen * batchSize;
    for (let head = 0; head < heads; head++) {
      const rowStart = head * headUnits;
      for (let batch = 0; batch < batchSize; batch++) {
        const sampleOffset = batch * seqLen;
        const attnOffset = (head * batchSize + batch) * seqLen * seqLen;

        for (let qPos = 0; qPos < seqLen; qPos++) {
          const qCol = sampleOffset + qPos;
          if (padMask[qCol]) {
            for (let kPos = 0; kPos < seqLen; kPos++) {
              attentionData[attnOffset + kPos * seqLen + qPos] = 0;
            }
            for (let i = 0; i < headUnits; i++) {
              outData[(rowStart + i) * totalCols + qCol] = 0;
            }
            continue;
          }

          let maxScore = -Infinity;
          for (let kPos = 0; kPos < seqLen; kPos++) {
            const kCol = sampleOffset + kPos;
            const scoreIdx = attnOffset + kPos * seqLen + qPos;
            if (padMask[kCol] || kPos > qPos) {
              attentionData[scoreIdx] = Number.NEGATIVE_INFINITY;
              continue;
            }
            let score = 0;
            for (let i = 0; i < headUnits; i++) {
              const row = rowStart + i;
              score += kData[row * totalCols + kCol] * qData[row * totalCols + qCol];
            }
            score *= scale;
            attentionData[scoreIdx] = score;
            if (score > maxScore) maxScore = score;
          }

          if (!Number.isFinite(maxScore)) continue;

          let sumExp = 0;
          for (let kPos = 0; kPos < seqLen; kPos++) {
            const scoreIdx = attnOffset + kPos * seqLen + qPos;
            const score = attentionData[scoreIdx];
            if (!Number.isFinite(score)) {
              attentionData[scoreIdx] = 0;
              continue;
            }
            const expValue = Math.exp(score - maxScore);
            attentionData[scoreIdx] = expValue;
            sumExp += expValue;
          }

          if (!Number.isFinite(sumExp) || sumExp <= 0) {
            for (let kPos = 0; kPos < seqLen; kPos++) {
              attentionData[attnOffset + kPos * seqLen + qPos] = 0;
            }
            continue;
          }

          for (let kPos = 0; kPos < seqLen; kPos++) {
            attentionData[attnOffset + kPos * seqLen + qPos] /= sumExp;
          }

          for (let i = 0; i < headUnits; i++) {
            const row = rowStart + i;
            let sum = 0;
            for (let kPos = 0; kPos < seqLen; kPos++) {
              const kCol = sampleOffset + kPos;
              sum += vData[row * totalCols + kCol] * attentionData[attnOffset + kPos * seqLen + qPos];
            }
            outData[row * totalCols + qCol] = sum;
          }
        }
      }
    }
  }

  private static backwardFallback(
    qData: Float32Array,
    kData: Float32Array,
    vData: Float32Array,
    attentionData: Float32Array,
    dOutData: Float32Array,
    padMask: boolean[],
    heads: number,
    headUnits: number,
    seqLen: number,
    batchSize: number,
    scale: number,
    dQOut: Float32Array,
    dKOut: Float32Array,
    dVOut: Float32Array,
    errAttention: Float32Array,
    errScore: Float32Array
  ): void {
    const totalCols = seqLen * batchSize;
    dQOut.fill(0);
    dKOut.fill(0);
    dVOut.fill(0);

    for (let head = 0; head < heads; head++) {
      const rowStart = head * headUnits;
      for (let batch = 0; batch < batchSize; batch++) {
        const sampleOffset = batch * seqLen;
        const attnOffset = (head * batchSize + batch) * seqLen * seqLen;
        errAttention.fill(0);
        errScore.fill(0);

        for (let qPos = 0; qPos < seqLen; qPos++) {
          const qCol = sampleOffset + qPos;
          if (padMask[qCol]) {
            for (let i = 0; i < headUnits; i++) {
              dQOut[(rowStart + i) * totalCols + qCol] = 0;
            }
            continue;
          }

          for (let i = 0; i < headUnits; i++) {
            const row = rowStart + i;
            const dOutVal = dOutData[row * totalCols + qCol];
            for (let kPos = 0; kPos < seqLen; kPos++) {
              const attnIdx = attnOffset + kPos * seqLen + qPos;
              dVOut[row * totalCols + sampleOffset + kPos] += dOutVal * attentionData[attnIdx];
              errAttention[kPos * seqLen + qPos] += vData[row * totalCols + sampleOffset + kPos] * dOutVal;
            }
          }

          let dot = 0;
          for (let kPos = 0; kPos < seqLen; kPos++) {
            const localIdx = kPos * seqLen + qPos;
            dot += attentionData[attnOffset + localIdx] * errAttention[localIdx];
          }

          for (let kPos = 0; kPos < seqLen; kPos++) {
            const localIdx = kPos * seqLen + qPos;
            errScore[localIdx] = attentionData[attnOffset + localIdx] * (errAttention[localIdx] - dot) * scale;
          }

          for (let i = 0; i < headUnits; i++) {
            const row = rowStart + i;
            let dqSum = 0;
            for (let kPos = 0; kPos < seqLen; kPos++) {
              const scoreGrad = errScore[kPos * seqLen + qPos];
              dqSum += kData[row * totalCols + sampleOffset + kPos] * scoreGrad;
              dKOut[row * totalCols + sampleOffset + kPos] += qData[row * totalCols + qCol] * scoreGrad;
            }
            dQOut[row * totalCols + qCol] = dqSum;
          }
        }
      }
    }
  }
}
