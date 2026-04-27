import { Cost, Optimzier, OptimzierType, StatusLayer } from "../@types/type";
import { softmaxBackward, softmaxOnly } from "../activation";
import mj from "../math";
import Matrix from "../matrix";
import { setLoss } from "../utils";
import setOptimizer from "../utils/setOptimizer";
import { isNativeAvailable, applyAttentionMaskNative } from "../math/rust_backend";
import { MemoryManager, WorkspaceConfig } from "../utils/memory";

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
  memoryConfig: WorkspaceConfig;
  private lossFunc: Function;
  private input: Matrix = mj.matrix([]);
  private attention: Matrix = mj.matrix([]);
  private padMask: boolean[] = [];
  private optimizerQ: OptimzierType;
  private optimizerK: OptimzierType;
  private optimizerV: OptimzierType;
  private optimizerName: Optimzier = "sgd";

  private QData: any = new Float32Array(0);
  private KData: any = new Float32Array(0);
  private VData: any = new Float32Array(0);
  private qkData: any = new Float32Array(0);
  private outputData: any = new Float32Array(0);
  private evalBuffers: Record<string, any> = {};

  private oldQData: any = new Float32Array(0);
  private oldKData: any = new Float32Array(0);
  private oldVData: any = new Float32Array(0);

  private Q: Matrix;
  private K: Matrix;
  private V: Matrix;
  private output: Matrix;
  private qkBuffer: Matrix;
  private oldQBuffer: Matrix;
  private oldKBuffer: Matrix;
  private oldVBuffer: Matrix;
  
  constructor({
    units,
    outputUnits,
    seqLen = 1,
    alpha = 0.1,
    loss = "mse",
    status = "input",
    clipGradient = 5.0,
    memoryConfig = {},
  }: SelfAttentionLayer & { memoryConfig?: WorkspaceConfig }) {
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

    this.memoryConfig = memoryConfig;
    // Initialize optimizers
    this.optimizerQ = setOptimizer(this.optimizerName, this.q._shape, alpha);
    this.optimizerK = setOptimizer(this.optimizerName, this.k._shape, alpha);
    this.optimizerV = setOptimizer(this.optimizerName, this.v._shape, alpha);

    this.Q = mj.matrix([]);
    this.K = mj.matrix([]);
    this.V = mj.matrix([]);
    this.output = mj.matrix([]);
    this.qkBuffer = mj.matrix([]);
    this.oldQBuffer = mj.matrix([]);
    this.oldKBuffer = mj.matrix([]);
    this.oldVBuffer = mj.matrix([]);
  }

  compile({ alpha, optimizer, clipGradient }: { alpha?: number; optimizer?: Optimzier; clipGradient?: number | boolean }) {
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

  load(q: number[][], k: number[][], v: number[][], clipGradient?: number | boolean): void {
    this.q._value = q;
    this.q._shape = [q.length, q[0]?.length ?? 0];
    this.k._value = k;
    this.k._shape = [k.length, k[0]?.length ?? 0];
    this.v._value = v;
    this.v._shape = [v.length, v[0]?.length ?? 0];
    if (clipGradient !== undefined) this.clipGradient = clipGradient;
  }

  forward(x: Matrix, options?: { workspace?: "train" | "eval" }): Matrix {
    this.inputShape = [x._shape[0], x._shape[1]];
    this.padMask = SelfAttention.detectPadColumns(x, this.padMask);
    
    this.ensureForwardBuffers(this.outputUnits, x._shape[1], options?.workspace);
    
    const wq = mj.dotProduct(this.q, x, this.Q);
    const wk = mj.dotProduct(this.k, x, this.K);
    const wv = mj.dotProduct(this.v, x, this.V);

    // qkBuffer is already ensured in ensureForwardBuffers
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
    // output buffer is already ensured in ensureForwardBuffers
    const output = mj.dotProduct(wv, this.attention, this.output);
    SelfAttention.zeroMaskedColumnsInPlace(output, this.padMask);

    this.outputShape = [output._shape[0], output._shape[1]];

    this.input = x;
    return output;
  }

  backward(y: Matrix, err: Matrix) {
    let backwardInput = err;
    let loss = 0;
    if (this.status === "output") {
      [loss, backwardInput] = this.lossFunc(y, this.output);
    } else {
      if (err._shape[1] === 1) {
        backwardInput = mj.reshape(err, this.output._shape);
      }
    }

    const errV = mj.dotProduct(backwardInput, this.attention, undefined, false, true);
    const errAttention = mj.dotProduct(this.V, backwardInput, undefined, true, false);

    // [CORRECTED] Use centralized Softmax Jacobian Backprop
    const errQKMatrix = softmaxBackward(this.attention, errAttention, false);

    const scale = 1 / Math.sqrt(this.outputUnits);
    const errQK = mj.mul(errQKMatrix, scale);

    const errQ = mj.dotProduct(this.K, errQK);
    const errK = mj.dotProduct(this.Q, errQK, undefined, false, true);

    const gradQ = mj.dotProduct(errQ, this.input, undefined, false, true);
    const gradK = mj.dotProduct(errK, this.input, undefined, false, true);
    const gradV = mj.dotProduct(errV, this.input, undefined, false, true);

    // Simpan bobot lama SEBELUM update menggunakan pre-allocated buffer
    this.ensureBackwardBuffers();
    
    this.oldQBuffer.copyFrom(this.q);
    this.oldKBuffer.copyFrom(this.k);
    this.oldVBuffer.copyFrom(this.v);

    const oldQ = this.oldQBuffer;
    const oldK = this.oldKBuffer;
    const oldV = this.oldVBuffer;

    // [New] Gradient Clipping
    if (this.clipGradient !== false) {
      const limit = typeof this.clipGradient === "number" ? this.clipGradient : 5.0;
      this.clipGradients(gradQ, limit);
      this.clipGradients(gradK, limit);
      this.clipGradients(gradV, limit);
    }

    // Update bobot In-Place!
    this.q.subInPlace(this.optimizerQ.calculate(gradQ, this.alpha));
    this.k.subInPlace(this.optimizerK.calculate(gradK, this.alpha));
    this.v.subInPlace(this.optimizerV.calculate(gradV, this.alpha));

    // Gunakan bobot LAMA untuk meneruskan gradient ke input
    const gradQOutput = mj.dotProduct(oldQ, errQ, undefined, true, false);
    const gradKOutput = mj.dotProduct(oldK, errK, undefined, true, false);
    const gradVOutput = mj.dotProduct(oldV, errV, undefined, true, false);

    // Gradient ke input adalah jumlah gradient dari ketiga path Q, K, V
    gradQOutput.addInPlace(gradKOutput);
    gradQOutput.addInPlace(gradVOutput);
    
    return gradQOutput;
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
    scoreData: any,
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

  private ensureForwardBuffers(outputUnits: number, seqLen: number, workspace: "train" | "eval" = "train"): void {
    const size = outputUnits * seqLen;
    const qkSize = seqLen * seqLen;

    if (workspace === "eval") {
      this.evalBuffers.QData = MemoryManager.ensureCapacity(this.evalBuffers.QData || new Float32Array(0), size, this.memoryConfig) as any;
      this.evalBuffers.KData = MemoryManager.ensureCapacity(this.evalBuffers.KData || new Float32Array(0), size, this.memoryConfig) as any;
      this.evalBuffers.VData = MemoryManager.ensureCapacity(this.evalBuffers.VData || new Float32Array(0), size, this.memoryConfig) as any;
      this.evalBuffers.qkData = MemoryManager.ensureCapacity(this.evalBuffers.qkData || new Float32Array(0), qkSize, this.memoryConfig) as any;
      this.evalBuffers.outputData = MemoryManager.ensureCapacity(this.evalBuffers.outputData || new Float32Array(0), size, this.memoryConfig) as any;

      this.QData = this.evalBuffers.QData;
      this.KData = this.evalBuffers.KData;
      this.VData = this.evalBuffers.VData;
      this.qkData = this.evalBuffers.qkData;
      this.outputData = this.evalBuffers.outputData;
    } else {
      this.QData = MemoryManager.ensureCapacity(this.QData, size, this.memoryConfig) as any;
      this.KData = MemoryManager.ensureCapacity(this.KData, size, this.memoryConfig) as any;
      this.VData = MemoryManager.ensureCapacity(this.VData, size, this.memoryConfig) as any;
      this.qkData = MemoryManager.ensureCapacity(this.qkData, qkSize, this.memoryConfig) as any;
      this.outputData = MemoryManager.ensureCapacity(this.outputData, size, this.memoryConfig) as any;
    }

    this.Q = Matrix.fromFlat(this.QData.subarray(0, size) as any, [outputUnits, seqLen]);
    this.K = Matrix.fromFlat(this.KData.subarray(0, size) as any, [outputUnits, seqLen]);
    this.V = Matrix.fromFlat(this.VData.subarray(0, size) as any, [outputUnits, seqLen]);
    this.qkBuffer = Matrix.fromFlat(this.qkData.subarray(0, qkSize) as any, [seqLen, seqLen]);
    this.output = Matrix.fromFlat(this.outputData.subarray(0, size) as any, [outputUnits, seqLen]);
  }

  private ensureBackwardBuffers(): void {
    const size = this.outputUnits * this.units;
    this.oldQData = MemoryManager.ensureCapacity(this.oldQData, size, this.memoryConfig) as any;
    this.oldKData = MemoryManager.ensureCapacity(this.oldKData, size, this.memoryConfig) as any;
    this.oldVData = MemoryManager.ensureCapacity(this.oldVData, size, this.memoryConfig) as any;

    this.oldQBuffer = Matrix.fromFlat(this.oldQData.subarray(0, size) as any, this.q._shape);
    this.oldKBuffer = Matrix.fromFlat(this.oldKData.subarray(0, size) as any, this.k._shape);
    this.oldVBuffer = Matrix.fromFlat(this.oldVData.subarray(0, size) as any, this.v._shape);
  }

  releaseWorkspace(): void {
    this.evalBuffers = {};
    this.QData = new Float32Array(0);
    this.KData = new Float32Array(0);
    this.VData = new Float32Array(0);
    this.qkData = new Float32Array(0);
    this.outputData = new Float32Array(0);
    this.oldQData = new Float32Array(0);
    this.oldKData = new Float32Array(0);
    this.oldVData = new Float32Array(0);

    this.Q = mj.matrix([]);
    this.K = mj.matrix([]);
    this.V = mj.matrix([]);
    this.output = mj.matrix([]);
    this.qkBuffer = mj.matrix([]);
    this.oldQBuffer = mj.matrix([]);
    this.oldKBuffer = mj.matrix([]);
    this.oldVBuffer = mj.matrix([]);
  }

  dispose(): void {
    this.releaseWorkspace();
    this.optimizerQ?.dispose?.();
    this.optimizerK?.dispose?.();
    this.optimizerV?.dispose?.();
    (this as any).q = null;
    (this as any).k = null;
    (this as any).v = null;
  }
}
