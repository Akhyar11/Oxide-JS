import { readFileSync } from "fs";
import mj from "../math";
import Matrix from "../matrix";
import Sequential from "./sequential";
import { MultiHeadAttention, Dense, PositionalEncoding, LayerNormalization, Embedding, Dropout } from "../layers";

interface TransformersConfig {
  units: number;          // d_model (embedding size)
  seqLen: number;         // sequence length
  vocabSize: number;      // vocabulary size
  heads?: number;         // number of attention heads (default 8)
  dropoutRate?: number;   // dropout rate (default 0.1)
  alpha?: number;         // learning rate
  padTokenId?: number;
}

/**
 * Improved Transformer Model
 * 
 * Arsitektur:
 * Input (Indices) -> Embedding -> PositionalEncoding
 * Block:
 *   1. Pre-Norm: LayerNorm1 -> MultiHeadAttention -> Dropout1 -> Add (Residual 1)
 *   2. FFN: LayerNorm2 -> Dense(4x units, relu) -> DropoutFFN -> Dense(units, linear) -> Dropout2 -> Add (Residual 2)
 * Output: Dense (applied to each token independently) -> Output
 */
export default class Transformers extends Sequential {
  public vocabSize: number;
  private embedding: Embedding;
  private pe: PositionalEncoding;
  private ln1: LayerNormalization;
  private mha: MultiHeadAttention;
  private drop1: Dropout;
  private ln2: LayerNormalization;
  private ffn1: Dense;
  private dropFfn: Dropout;
  private ffn2: Dense;
  private drop2: Dropout;
  private dense: Dense;

  private xInput: Matrix = mj.matrix([]);
  private xEmb: Matrix = mj.matrix([]);
  private xPe: Matrix = mj.matrix([]);
  
  private xLn1: Matrix = mj.matrix([]);
  private xRes1: Matrix;
  private xLn2: Matrix = mj.matrix([]);
  private xRes2: Matrix;

  private errRes1Buf: Matrix;
  private errRes2Buf: Matrix;
  private lastTokenBuffer: Matrix;
  private emptyErr: Matrix = mj.matrix([[]]);
  private lastTokenIndex: number = 0;
  private padMaskBuffer: boolean[] = [];
  private profilerEnabled: boolean = false;
  private profileStats: { [key: string]: { totalMs: number; count: number } } = Object.create(null);

  constructor({ units, seqLen, vocabSize, heads = 8, dropoutRate = 0.1, alpha = 0.01, padTokenId }: TransformersConfig) {
    const embedding = new Embedding({ vocabSize, embeddingDim: units, alpha, padTokenId });
    const pe = new PositionalEncoding({ dModel: units, maxSeqLen: seqLen });
    
    // Block
    const ln1 = new LayerNormalization({ units });
    const mha = new MultiHeadAttention({ units, heads, seqLen, alpha });
    const drop1 = new Dropout({ rate: dropoutRate });
    
    const ln2 = new LayerNormalization({ units });
    const ffn1 = new Dense({ units, outputUnits: units * 4, activation: "relu", alpha });
    const dropFfn = new Dropout({ rate: dropoutRate });
    const ffn2 = new Dense({ units: units * 4, outputUnits: units, activation: "linear", alpha });
    const drop2 = new Dropout({ rate: dropoutRate });
    
    // Output Projector (applied independently to sequence length)
    const dense = new Dense({
      units: units, 
      outputUnits: vocabSize, 
      activation: "linear",
      alpha,
      status: "output",
      loss: "softmaxCrossEntropy" // Paksa gunakan Cross Entropy dari awal
    });

    super({ layers: [embedding, pe, ln1, mha, drop1, ln2, ffn1, dropFfn, ffn2, drop2, dense] });
    
    this.embedding = embedding;
    this.pe = pe;
    this.ln1 = ln1;
    this.mha = mha;
    this.drop1 = drop1;
    this.ln2 = ln2;
    this.ffn1 = ffn1;
    this.dropFfn = dropFfn;
    this.ffn2 = ffn2;
    this.drop2 = drop2;
    this.dense = dense;

    // Pre-allocate buffers
    this.xRes1 = mj.zeros([units, seqLen]);
    this.xRes2 = mj.zeros([units, seqLen]);
    this.errRes1Buf = mj.zeros([units, seqLen]);
    this.errRes2Buf = mj.zeros([units, seqLen]);
    this.lastTokenBuffer = mj.zeros([units, 1]);
    this.vocabSize = vocabSize;
    this.train();
  }

  forward(x: Matrix): Matrix {
    const [seqLen, batchSize] = x._shape;
    const units = this.embedding.embeddingDim;
    const totalTokens = seqLen * batchSize;

    const padMaskStart = this.profileStart();
    if (this.padMaskBuffer.length !== totalTokens) {
      this.padMaskBuffer = new Array<boolean>(totalTokens);
    }
    let maskIdx = 0;
    for (let b = 0; b < batchSize; b++) {
      for (let pos = 0; pos < seqLen; pos++) {
        this.padMaskBuffer[maskIdx++] = x._data[pos * batchSize + b] === this.embedding.padTokenId;
      }
    }
    this.profileEnd("pad mask creation", padMaskStart);

    // 1. Embedding Forward
    const embeddingForwardStart = this.profileStart();
    const xEmb = this.embedding.forward(x); // returns [Units, totalTokens]
    this.profileEnd("embedding forward", embeddingForwardStart);
    
    // 2. Positional Encoding
    const xPe = this.pe.forward(xEmb);

    // 3. Block Forward
    let h = xPe;
    
    // Residual 1: Norm -> Attention -> Dropout -> Add
    const layerNorm1ForwardStart = this.profileStart();
    const xLn1 = this.ln1.forward(h);
    this.profileEnd("layer norm forward", layerNorm1ForwardStart);
    this.mha.setPadMask(this.padMaskBuffer);
    const mhaForwardStart = this.profileStart();
    const xMhaOut = this.mha.forward(xLn1);
    this.profileEnd("MHA forward", mhaForwardStart);
    const xDrop1Out = this.drop1.forward(xMhaOut);
    const res1 = mj.add(h, xDrop1Out);
    
    // Residual 2: Norm -> FFN -> Dropout -> Add
    const layerNorm2ForwardStart = this.profileStart();
    const xLn2 = this.ln2.forward(res1);
    this.profileEnd("layer norm forward", layerNorm2ForwardStart);
    const ffnForwardStart = this.profileStart();
    const xFfn1Out = this.ffn1.forward(xLn2);
    const xDropFfnOut = this.dropFfn.forward(xFfn1Out);
    const xFfn2Out = this.ffn2.forward(xDropFfnOut);
    const xDrop2Out = this.drop2.forward(xFfn2Out);
    this.profileEnd("FFN forward", ffnForwardStart);
    const res2 = mj.add(res1, xDrop2Out);

    // 4. Extract Last Token Embeddings for Output
    // We only want the embedding of the LAST token for each sample in the batch
    if (this.lastTokenBuffer._shape[1] !== batchSize) {
      this.lastTokenBuffer = mj.zeros([units, batchSize]);
    }
    
    const res2Data = res2._data;
    const lastTokenData = this.lastTokenBuffer._data;
    const totalCols = res2._shape[1];
    
    for (let b = 0; b < batchSize; b++) {
      const lastTokenCol = (b + 1) * seqLen - 1;
      for (let i = 0; i < units; i++) {
        lastTokenData[i * batchSize + b] = res2Data[i * totalCols + lastTokenCol];
      }
    }

    // 5. Output Dense Layer
    const outputDenseForwardStart = this.profileStart();
    const out = this.dense.forward(this.lastTokenBuffer);
    this.profileEnd("output dense forward", outputDenseForwardStart);
    return out;
  }

  backward(y: Matrix) {
    // 1. Output Dense Backward
    const outputDenseBackwardStart = this.profileStart();
    const errDense = this.dense.backward(y, this.emptyErr);
    this.profileEnd("output dense backward", outputDenseBackwardStart);
    this.loss = this.dense.loss;
    const batchSize = errDense._shape[1];
    const seqLen = this.pe.maxSeqLen;
    const units = this.embedding.embeddingDim;
    const totalTokens = seqLen * batchSize;

    // 2. Map Dense Error back to the full sequence length matrix
    const mapDenseErrStart = this.profileStart();
    if (this.errRes2Buf._shape[0] !== units || this.errRes2Buf._shape[1] !== totalTokens) {
      this.errRes2Buf = mj.zeros([units, totalTokens]);
    } else {
      this.errRes2Buf._data.fill(0);
    }
    const res2Err = this.errRes2Buf;
    const res2ErrData = res2Err._data;
    const errDenseData = errDense._data;
    
    for (let b = 0; b < batchSize; b++) {
      const lastTokenCol = (b + 1) * seqLen - 1;
      for (let i = 0; i < units; i++) {
        res2ErrData[i * totalTokens + lastTokenCol] = errDenseData[i * batchSize + b];
      }
    }
    this.profileEnd("mapping dense error", mapDenseErrStart);

    // 3. Block Backward
    const ffnBackwardStart = this.profileStart();
    const errDrop2 = this.drop2.backward(this.emptyErr, res2Err);
    const errFfn2 = this.ffn2.backward(this.emptyErr, errDrop2);
    const errDropFfn = this.dropFfn.backward(this.emptyErr, errFfn2);
    const errFfn1 = this.ffn1.backward(this.emptyErr, errDropFfn);
    this.profileEnd("FFN backward", ffnBackwardStart);
    const layerNorm2BackwardStart = this.profileStart();
    const errLn2 = this.ln2.backward(this.emptyErr, errFfn1);
    this.profileEnd("layer norm backward", layerNorm2BackwardStart);
    
    const res1Err = mj.add(res2Err, errLn2);
    
    const errDrop1 = this.drop1.backward(this.emptyErr, res1Err);
    const mhaBackwardStart = this.profileStart();
    const errMha = this.mha.backward(this.emptyErr, errDrop1);
    this.profileEnd("MHA backward", mhaBackwardStart);
    const layerNorm1BackwardStart = this.profileStart();
    const errLn1 = this.ln1.backward(this.emptyErr, errMha);
    this.profileEnd("layer norm backward", layerNorm1BackwardStart);
    
    const peErr = mj.add(res1Err, errLn1);
    
    // 4. PE & Embedding Backward
    const embeddingBackwardStart = this.profileStart();
    const embErr = this.pe.backward(this.emptyErr, peErr);
    this.embedding.backward(this.emptyErr, embErr);
    this.profileEnd("embedding backward", embeddingBackwardStart);
  }

  load(path: string) {
    const dataJson = readFileSync(path, "utf-8");
    const data = JSON.parse(dataJson);

    if (!Array.isArray(data) || data.length < 11) {
      throw new Error(`Invalid transformer model file: ${path}`);
    }

    const [embedding, _pe, ln1, mha, drop1, ln2, ffn1, dropFfn, ffn2, drop2, dense] = data;

    if (embedding?.weight) {
      this.embedding.load(embedding.weight);
      if ("padTokenId" in embedding) {
        this.embedding.padTokenId = embedding.padTokenId;
      }
    }

    if (ln1?.gamma && ln1?.beta) this.ln1.load(ln1.gamma, ln1.beta);
    if (mha) this.mha.load(mha);
    if (drop1?.rate !== undefined) this.drop1.load({ rate: drop1.rate, status: drop1.status ?? this.drop1.status });
    if (ln2?.gamma && ln2?.beta) this.ln2.load(ln2.gamma, ln2.beta);
    if (ffn1?.weight && ffn1?.bias) this.ffn1.load(ffn1.weight, ffn1.bias);
    if (dropFfn?.rate !== undefined) this.dropFfn.load({ rate: dropFfn.rate, status: dropFfn.status ?? this.dropFfn.status });
    if (ffn2?.weight && ffn2?.bias) this.ffn2.load(ffn2.weight, ffn2.bias);
    if (drop2?.rate !== undefined) this.drop2.load({ rate: drop2.rate, status: drop2.status ?? this.drop2.status });
    if (dense?.weight && dense?.bias) this.dense.load(dense.weight, dense.bias);

    this.vocabSize = this.embedding.vocabSize;
  }

  resizeVocab(newVocabSize: number) {
    this.embedding.resize(newVocabSize);
    this.dense.resize(newVocabSize);
    this.vocabSize = newVocabSize; // SINKRONKAN
  }

  fit(X: Matrix[], y: Matrix[], epochs: number, cb: (loss: number) => any = (_) => { }) {
    this.train();
    for (let i = 0; i < epochs; i++) {
      this.dense.resetLoss();
      let epochLoss = 0;

      for (let j = 0; j < X.length; j++) {
        this.forward(X[j]);
        this.backward(y[j]);
        epochLoss = this.dense.loss;
      }

      this.loss = epochLoss;
      cb(this.loss);
      if (this.loss < 0.01) return 0;
    }
  }

  enableProfiling(enabled: boolean = true): this {
    this.profilerEnabled = enabled;
    return this;
  }

  disableProfiling(): this {
    this.profilerEnabled = false;
    return this;
  }

  resetProfiling(): void {
    this.profileStats = Object.create(null);
  }

  getProfilingReport(reset: boolean = false): { [key: string]: { totalMs: number; avgMs: number; count: number } } {
    const report: { [key: string]: { totalMs: number; avgMs: number; count: number } } = {};
    for (const key of Object.keys(this.profileStats)) {
      const stat = this.profileStats[key];
      report[key] = {
        totalMs: stat.totalMs,
        avgMs: stat.count > 0 ? stat.totalMs / stat.count : 0,
        count: stat.count,
      };
    }
    if (reset) this.resetProfiling();
    return report;
  }

  private profileStart(): number {
    if (!this.profilerEnabled) return 0;
    if (typeof performance !== "undefined" && typeof performance.now === "function") {
      return performance.now();
    }
    return Date.now();
  }

  private profileEnd(label: string, start: number): void {
    if (!this.profilerEnabled) return;
    const end = typeof performance !== "undefined" && typeof performance.now === "function"
      ? performance.now()
      : Date.now();
    const elapsed = end - start;
    if (!this.profileStats[label]) {
      this.profileStats[label] = { totalMs: 0, count: 0 };
    }
    this.profileStats[label].totalMs += elapsed;
    this.profileStats[label].count += 1;
  }
}
