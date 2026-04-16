import { Cost, Optimzier, OptimzierType, StatusLayer } from "../@types/type";
import { softmaxBackwardInto, softmaxInto } from "../activation";
import mj from "../math";
import { applyAttentionMaskNative, isNativeAvailable } from "../math/rust_backend";
import Matrix from "../matrix";
import setOptimizer from "../utils/setOptimizer";
import Dense from "./dense";

interface MultiHeadAttentionLayer {
  units: number;
  heads: number;
  seqLen: number;
  alpha?: number;
  status?: StatusLayer;
}

export default class MultiHeadAttention {
  name = "multi head attention layer";
  units: number;
  heads: number;
  headUnits: number;
  seqLen: number;
  alpha: number;
  status: StatusLayer;

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

  private qHeadViews: Matrix[] = [];
  private kHeadViews: Matrix[] = [];
  private vHeadViews: Matrix[] = [];
  private outHeadViews: Matrix[] = [];
  private dQHeadViews: Matrix[] = [];
  private dKHeadViews: Matrix[] = [];
  private dVHeadViews: Matrix[] = [];
  private scoreBuffers: Matrix[] = [];
  private attentionBuffers: Matrix[] = [];
  private errAttentionBuffers: Matrix[] = [];
  private errScoreBuffers: Matrix[] = [];

  constructor({ units, heads, seqLen, alpha = 0.1, status = "input" }: MultiHeadAttentionLayer) {
    this.units = units;
    this.heads = heads;
    this.seqLen = seqLen;
    this.alpha = alpha;
    this.status = status;

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
    this.ensureSequenceBuffers(seqLen);
  }

  compile({ alpha, optimizer, error }: { alpha?: number; optimizer?: Optimzier; error?: Cost }) {
    if (alpha !== undefined) this.alpha = alpha;
    if (optimizer !== undefined) {
      this.optimizerName = optimizer;
      this.optimizerQ = setOptimizer(optimizer, this.q._shape, this.alpha);
      this.optimizerK = setOptimizer(optimizer, this.k._shape, this.alpha);
      this.optimizerV = setOptimizer(optimizer, this.v._shape, this.alpha);
    }
    this.wo.compile({ alpha, optimizer, error });
  }

  forward(x: Matrix): Matrix {
    const seqLen = x._shape[1];
    this.ensureSequenceBuffers(seqLen);

    this.input = x;
    this.padMask = MultiHeadAttention.detectPadColumns(x, this.padMask);

    mj.dotProduct(this.q, x, this.Q);
    mj.dotProduct(this.k, x, this.K);
    mj.dotProduct(this.v, x, this.V);

    const scale = 1 / Math.sqrt(this.headUnits);
    for (let i = 0; i < this.heads; i++) {
      const score = mj.dotProduct(this.kHeadViews[i], this.qHeadViews[i], this.scoreBuffers[i], true, false);

      if (isNativeAvailable()) {
        applyAttentionMaskNative(score._data, this.padMask, score._shape[0], score._shape[1], scale);
      } else {
        const scoreData = score._data;
        for (let j = 0; j < scoreData.length; j++) {
          scoreData[j] *= scale;
        }
        MultiHeadAttention.applyMasks(scoreData, score._shape[0], score._shape[1], this.padMask);
      }

      softmaxInto(score, this.attentionBuffers[i], false);
      mj.dotProduct(this.vHeadViews[i], this.attentionBuffers[i], this.outHeadViews[i]);
      MultiHeadAttention.zeroMaskedColumnsInPlace(this.outHeadViews[i], this.padMask);
    }

    this.outputShape = [this.concatenated._shape[0], this.concatenated._shape[1]];
    return this.wo.forward(this.concatenated);
  }

  backward(y: Matrix, err: Matrix): Matrix {
    const dCat = this.wo.backward(y, err);
    const seqLen = dCat._shape[1];
    this.ensureSequenceBuffers(seqLen);

    const scale = 1 / Math.sqrt(this.headUnits);
    const headSpan = this.headUnits * seqLen;

    for (let i = 0; i < this.heads; i++) {
      const start = i * headSpan;
      const errHead = Matrix.fromFlat(dCat._data.subarray(start, start + headSpan), [this.headUnits, seqLen]);

      mj.dotProduct(errHead, this.attentionBuffers[i], this.dVHeadViews[i], false, true);
      mj.dotProduct(this.vHeadViews[i], errHead, this.errAttentionBuffers[i], true, false);
      softmaxBackwardInto(this.attentionBuffers[i], this.errAttentionBuffers[i], this.errScoreBuffers[i], false);
      this.errScoreBuffers[i].mulInPlace(scale);

      mj.dotProduct(this.kHeadViews[i], this.errScoreBuffers[i], this.dQHeadViews[i]);
      mj.dotProduct(this.qHeadViews[i], this.errScoreBuffers[i], this.dKHeadViews[i], false, true);
    }

    const gradQ = mj.dotProduct(this.dQAll, this.input, this.gradQBuffer, false, true);
    const gradK = mj.dotProduct(this.dKAll, this.input, this.gradKBuffer, false, true);
    const gradV = mj.dotProduct(this.dVAll, this.input, this.gradVBuffer, false, true);

    this.clipGradients(gradQ, 1.0);
    this.clipGradients(gradK, 1.0);
    this.clipGradients(gradV, 1.0);

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
      this.wo.load(data.wo.weight, data.wo.bias);
    }
  }

  private ensureSequenceBuffers(seqLen: number) {
    if (this.seqLen === seqLen && this.Q._shape[1] === seqLen && this.qHeadViews.length === this.heads) {
      return;
    }

    this.seqLen = seqLen;
    this.inputShape = [this.units, seqLen];
    this.outputShape = [this.units, seqLen];

    this.Q = mj.zeros([this.units, seqLen]);
    this.K = mj.zeros([this.units, seqLen]);
    this.V = mj.zeros([this.units, seqLen]);
    this.concatenated = mj.zeros([this.units, seqLen]);

    this.gradInputBuffer = mj.zeros([this.units, seqLen]);
    this.gradContributionBuffer = mj.zeros([this.units, seqLen]);
    this.dQAll = mj.zeros([this.units, seqLen]);
    this.dKAll = mj.zeros([this.units, seqLen]);
    this.dVAll = mj.zeros([this.units, seqLen]);

    this.qHeadViews = this.createHeadViews(this.Q, seqLen);
    this.kHeadViews = this.createHeadViews(this.K, seqLen);
    this.vHeadViews = this.createHeadViews(this.V, seqLen);
    this.outHeadViews = this.createHeadViews(this.concatenated, seqLen);
    this.dQHeadViews = this.createHeadViews(this.dQAll, seqLen);
    this.dKHeadViews = this.createHeadViews(this.dKAll, seqLen);
    this.dVHeadViews = this.createHeadViews(this.dVAll, seqLen);

    this.scoreBuffers = new Array<Matrix>(this.heads);
    this.attentionBuffers = new Array<Matrix>(this.heads);
    this.errAttentionBuffers = new Array<Matrix>(this.heads);
    this.errScoreBuffers = new Array<Matrix>(this.heads);
    for (let i = 0; i < this.heads; i++) {
      this.scoreBuffers[i] = mj.zeros([seqLen, seqLen]);
      this.attentionBuffers[i] = mj.zeros([seqLen, seqLen]);
      this.errAttentionBuffers[i] = mj.zeros([seqLen, seqLen]);
      this.errScoreBuffers[i] = mj.zeros([seqLen, seqLen]);
    }
  }

  private createHeadViews(matrix: Matrix, seqLen: number): Matrix[] {
    const views: Matrix[] = new Array(this.heads);
    const headSpan = this.headUnits * seqLen;
    for (let i = 0; i < this.heads; i++) {
      const start = i * headSpan;
      views[i] = Matrix.fromFlat(matrix._data.subarray(start, start + headSpan), [this.headUnits, seqLen]);
    }
    return views;
  }

  private loadLegacyHeads(headsData: Array<{ q: number[][]; k: number[][]; v: number[][] }>) {
    const fusedQ = new Float64Array(this.units * this.units);
    const fusedK = new Float64Array(this.units * this.units);
    const fusedV = new Float64Array(this.units * this.units);

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
    const data = m._data;
    for (let i = 0; i < data.length; i++) {
      if (data[i] > limit) data[i] = limit;
      else if (data[i] < -limit) data[i] = -limit;
    }
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

  private static applyMasks(
    scoreData: Float64Array,
    rows: number,
    cols: number,
    padMask: boolean[]
  ): void {
    const maskedValue = -1e9;
    for (let query = 0; query < cols; query++) {
      if (padMask[query]) {
        for (let key = 0; key < rows; key++) {
          scoreData[key * cols + query] = maskedValue;
        }
        scoreData[query * cols + query] = 0;
        continue;
      }

      for (let key = 0; key < rows; key++) {
        if (padMask[key] || key > query) {
          scoreData[key * cols + query] = maskedValue;
        }
      }
    }
  }

  private static zeroMaskedColumnsInPlace(matrix: Matrix, padMask: boolean[]): void {
    const [rows, cols] = matrix._shape;
    const out = matrix._data;
    for (let j = 0; j < cols; j++) {
      if (!padMask[j]) continue;
      for (let i = 0; i < rows; i++) {
        out[i * cols + j] = 0;
      }
    }
  }
}
