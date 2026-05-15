import { Cost, Optimizer, OptimizerType, StatusLayer, engine } from "@oxide-js/core";
import { softmaxBackward, softmaxOnly } from "@oxide-js/core";
import { mj } from "@oxide-js/core";
import { Matrix } from "@oxide-js/core";
import { setLoss } from "@oxide-js/core";
import { setOptimizer } from "@oxide-js/core";
import { isNativeAvailable, applyAttentionMaskNative } from "@oxide-js/core";

interface SelfAttentionLayer {
  units: number;
  outputUnits?: number;
  seqLen?: number;
  alpha?: number;
  loss?: Cost;
  status?: StatusLayer;
  clipGradient?: number | boolean;
}

export default class SelfAttention {
  name = "self attention layer";
  units: number;
  outputUnits: number;
  inputShape: [number, number];
  outputShape: [number, number];
  params: number;
  q: Matrix;
  k: Matrix;
  v: Matrix;
  alpha: number;
  loss: number = 0;
  status: StatusLayer = "input";
  clipGradient: number | boolean = 5.0;
  private lossFunc: Function;
  private input: Matrix = mj.matrix([]);
  private output: Matrix = mj.matrix([]);
  private attention: Matrix = mj.matrix([]);
  private padMask: boolean[] = [];
  private Q: Matrix = mj.matrix([]);
  private K: Matrix = mj.matrix([]);
  private V: Matrix = mj.matrix([]);
  private optimizerQ: OptimizerType;
  private optimizerK: OptimizerType;
  private optimizerV: OptimizerType;
  private optimizerName: Optimizer = "sgd";
  
  // Buffers untuk mengurangi load GC
  private oldQBuffer: Matrix | null = null;
  private oldKBuffer: Matrix | null = null;
  private oldVBuffer: Matrix | null = null;
  private qkBuffer: Matrix | null = null;
  
  constructor({
    units,
    outputUnits,
    seqLen = 1,
    alpha = 0.1,
    loss = "mse",
    status = "input",
    clipGradient = 5.0,
  }: SelfAttentionLayer) {
    this.units = units;
    this.outputUnits = outputUnits ?? units;
    this.inputShape = [units, seqLen];
    this.outputShape = [this.outputUnits, seqLen];
    // params: 3 bobot matrix (Q, K, V) masing-masing [outputUnits x units]
    this.params = 3 * this.outputUnits * this.units;
    this.q = mj.xavier([this.outputUnits, this.units]);
    this.k = mj.xavier([this.outputUnits, this.units]);
    this.v = mj.xavier([this.outputUnits, this.units]);
    this.lossFunc = setLoss(loss);
    this.status = status;
    this.alpha = alpha;
    this.clipGradient = clipGradient;

    // Initialize optimizers
    this.optimizerQ = setOptimizer(this.optimizerName, this.q._shape, alpha);
    this.optimizerK = setOptimizer(this.optimizerName, this.k._shape, alpha);
    this.optimizerV = setOptimizer(this.optimizerName, this.v._shape, alpha);
  }

  compile({ alpha, optimizer, clipGradient }: { alpha?: number; optimizer?: Optimizer; clipGradient?: number | boolean }) {
    if (alpha !== undefined) this.alpha = alpha;
    if (optimizer !== undefined) {
      this.optimizerName = optimizer;
      this.optimizerQ = setOptimizer(optimizer, this.q._shape, this.alpha);
      this.optimizerK = setOptimizer(optimizer, this.k._shape, this.alpha);
      this.optimizerV = setOptimizer(optimizer, this.v._shape, this.alpha);
    }
    if (clipGradient !== undefined) this.clipGradient = clipGradient;
  }

  save() {
    return {
      name: this.name,
      status: this.status,
      units: this.units,
      alpha: this.alpha,
      clipGradient: this.clipGradient,
      q: this.q._value,
      k: this.k._value,
      v: this.v._value,
    };
  }

  toKerasConfig() {
    return {
      class_name: "SelfAttention",
      config: {
        units: this.units,
        name: `self_attention_${Math.floor(Math.random() * 1000)}`,
        trainable: true,
      }
    };
  }

  getWeightsManifest(): { name: string; shape: [number, number]; data: Float32Array }[] {
    return [
      { name: "q", shape: this.q._shape, data: this.q._data },
      { name: "k", shape: this.k._shape, data: this.k._data },
      { name: "v", shape: this.v._shape, data: this.v._data },
    ];
  }

  setWeightsFromBinary(weights: Record<string, Float32Array>): void {
    if (weights.q) this.q._data.set(weights.q);
    if (weights.k) this.k._data.set(weights.k);
    if (weights.v) this.v._data.set(weights.v);
  }

  load(q?: number[][], k?: number[][], v?: number[][], clipGradient?: number | boolean): void {
    if (q) {
      this.q._value = q;
      this.q._shape = [q.length, q[0]?.length ?? 0];
    }
    if (k) {
      this.k._value = k;
      this.k._shape = [k.length, k[0]?.length ?? 0];
    }
    if (v) {
      this.v._value = v;
      this.v._shape = [v.length, v[0]?.length ?? 0];
    }
    if (clipGradient !== undefined) this.clipGradient = clipGradient;
  }

  getParams(): Matrix[] {
    return [this.q, this.k, this.v];
  }

  update(alpha?: number): void {
    const a = alpha || this.alpha;
    this.optimizerQ.apply(this.q, a);
    this.optimizerK.apply(this.k, a);
    this.optimizerV.apply(this.v, a);
  }

  forward(x: Matrix): Matrix {
    this.inputShape = [x._shape[0], x._shape[1]];
    this.padMask = SelfAttention.detectPadColumns(x, this.padMask);
    
    if (this.Q._shape[0] !== this.q._shape[0] || this.Q._shape[1] !== x._shape[1]) {
        this.Q = mj.zeros([this.q._shape[0], x._shape[1]]);
        this.K = mj.zeros([this.k._shape[0], x._shape[1]]);
        this.V = mj.zeros([this.v._shape[0], x._shape[1]]);
    }
    
    const wq = mj.dotProduct(this.q, x, this.Q);
    const wk = mj.dotProduct(this.k, x, this.K);
    const wv = mj.dotProduct(this.v, x, this.V);

    if (!this.qkBuffer || this.qkBuffer._shape[0] !== wk._shape[1] || this.qkBuffer._shape[1] !== wq._shape[1]) {
        this.qkBuffer = mj.zeros([wk._shape[1], wq._shape[1]]);
    }
    const qk = mj.dotProduct(wk, wq, this.qkBuffer, true, false);
    const scale = 1 / Math.sqrt(this.outputUnits);
    if (isNativeAvailable()) {
      applyAttentionMaskNative(qk._data, this.padMask, qk._shape[0], qk._shape[1], scale);
      this.attention = softmaxOnly(qk);
    } else {
      const qkData = qk._data;
      for (let i = 0; i < qkData.length; i++) {
        qkData[i] *= scale;
      }
      SelfAttention.applyMasks(qkData, qk._shape[0], qk._shape[1], this.padMask);
      this.attention = softmaxOnly(qk);
    }
    if (this.output._shape[0] !== wv._shape[0] || this.output._shape[1] !== this.attention._shape[1]) {
      this.output = mj.zeros([wv._shape[0], this.attention._shape[1]]);
    }
    const output = mj.dotProduct(wv, this.attention, this.output);
    SelfAttention.zeroMaskedColumnsInPlace(output, this.padMask);

    this.outputShape = [output._shape[0], output._shape[1]];

    this.input = x;
    this.output = output;

    const tape = engine.tape;
    if (tape) {
      tape.record([x, this.q, this.k, this.v], [output], (grad: Matrix) => {
        this.calculateGradients(grad);
      });
    }

    return output;
  }

  private calculateGradients(grad: Matrix): Matrix {
    let backwardInput = grad;
    if (grad._shape[1] === 1) {
      backwardInput = mj.reshape(grad, this.output._shape);
    }

    const errV = mj.dotProduct(backwardInput, this.attention, undefined, false, true);
    const errAttention = mj.dotProduct(this.V, backwardInput, undefined, true, false);

    const errQKMatrix = softmaxBackward(this.attention, errAttention, false);
    const scale = 1 / Math.sqrt(this.outputUnits);
    const errQK = mj.mul(errQKMatrix, scale);

    const errQ = mj.dotProduct(this.K, errQK);
    const errK = mj.dotProduct(this.Q, errQK, undefined, false, true);

    const gradQ = mj.dotProduct(errQ, this.input, undefined, false, true);
    const gradK = mj.dotProduct(errK, this.input, undefined, false, true);
    const gradV = mj.dotProduct(errV, this.input, undefined, false, true);

    if (this.clipGradient !== false) {
      const limit = typeof this.clipGradient === "number" ? this.clipGradient : 5.0;
      this.clipGradients(gradQ, limit);
      this.clipGradients(gradK, limit);
      this.clipGradients(gradV, limit);
    }

    const accumulate = (p: Matrix, g: Matrix) => {
      if (p.grad) p.grad.addInPlace(g);
      else p.grad = g;
    };
    accumulate(this.q, gradQ);
    accumulate(this.k, gradK);
    accumulate(this.v, gradV);

    const gradQOutput = mj.dotProduct(this.q, errQ, undefined, true, false);
    const gradKOutput = mj.dotProduct(this.k, errK, undefined, true, false);
    const gradVOutput = mj.dotProduct(this.v, errV, undefined, true, false);

    gradQOutput.addInPlace(gradKOutput);
    gradQOutput.addInPlace(gradVOutput);
    
    if (this.input.grad) this.input.grad.addInPlace(gradQOutput);
    else this.input.grad = gradQOutput;

    return gradQOutput;
  }

  backward(y: Matrix, err: Matrix, gradOnly = false) {
    let backwardInput = err;
    if (this.status === "output") {
      [, backwardInput] = this.lossFunc(y, this.output);
    }

    const gradInput = this.calculateGradients(backwardInput);
    if (!gradOnly) this.update(this.alpha);
    return gradInput;
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
    scoreData: Float32Array | Float64Array,
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
