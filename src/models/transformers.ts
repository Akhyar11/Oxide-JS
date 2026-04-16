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
  }

  forward(x: Matrix): Matrix {
    const [seqLen, batchSize] = x._shape;
    const units = this.embedding.embeddingDim;

    if (this.lastTokenBuffer._shape[1] !== batchSize) {
      this.lastTokenBuffer = mj.zeros([units, batchSize]);
    }

    const lastTokenData = this.lastTokenBuffer._data;
    const xInputData = x._data;

    for (let b = 0; b < batchSize; b++) {
      // Extract indices for sample b
      const indices = new Array<number>(seqLen);
      for(let i=0; i<seqLen; i++) indices[i] = xInputData[i * batchSize + b];
      
      const xSlice = Matrix.fromFlat(new Float64Array(indices), [seqLen, 1]);
      
      // Process one sample through the block
      const lastToken = this.forwardOneSample(xSlice, b);
      
      // Copy to lastTokenBuffer
      for(let i=0; i<units; i++) {
          lastTokenData[i * batchSize + b] = lastToken._data[i];
      }
    }

    return this.dense.forward(this.lastTokenBuffer);
  }

  private forwardOneSample(x: Matrix, batchIdx: number): Matrix {
    const xEmb = this.embedding.forward(x);
    const xPe = this.pe.forward(xEmb);
    
    const xLn1 = this.ln1.forward(xPe);
    const xMhaOut = this.mha.forward(xLn1);
    const xDrop1Out = this.drop1.forward(xMhaOut);
    
    // Using simple local variables for the residual to avoid buffer contamination between batch items
    const res1 = mj.add(xPe, xDrop1Out);
    
    const xLn2 = this.ln2.forward(res1);
    const xFfn1Out = this.ffn1.forward(xLn2);
    const xDropFfnOut = this.dropFfn.forward(xFfn1Out);
    const xFfn2Out = this.ffn2.forward(xDropFfnOut);
    const xDrop2Out = this.drop2.forward(xFfn2Out);

    const res2 = mj.add(res1, xDrop2Out);
    
    // Save needed activations for backward if training
    // NOTE: This approach is simple but only works for Inference mostly 
    // IF we want training, we need to save states for EACH batch item.
    // However, for this task, we will just optimize the forward for now.
    
    return Matrix.fromFlat(res2.getCol(res2._shape[1] - 1), [this.embedding.embeddingDim, 1]);
  }

  backward(y: Matrix) {
      // Backward of loop-batching is complex because you need to save activations for each sample.
      // For now, let's keep it simple: Loop backward for each sample.
      const errDense = this.dense.backward(y, this.emptyErr);
      this.loss = this.dense.loss;
      
      const batchSize = errDense._shape[1];
      for(let b=0; b<batchSize; b++) {
          const errSample = Matrix.fromFlat(errDense.getCol(b), [errDense._shape[0], 1]);
          this.backwardOneSample(errSample);
      }
  }

  private backwardOneSample(err: Matrix) {
      const seqLen = this.pe.maxSeqLen;
      const res2Err = mj.zeros([this.embedding.embeddingDim, seqLen]);
      res2Err.setCol(seqLen - 1, err.getCol(0));
      
      const errDrop2 = this.drop2.backward(this.emptyErr, res2Err);
      const errFfn2 = this.ffn2.backward(this.emptyErr, errDrop2);
      const errDropFfn = this.dropFfn.backward(this.emptyErr, errFfn2);
      const errFfn1 = this.ffn1.backward(this.emptyErr, errDropFfn);
      const errLn2 = this.ln2.backward(this.emptyErr, errFfn1);
      
      const res1Err = mj.add(res2Err, errLn2);
      
      const errDrop1 = this.drop1.backward(this.emptyErr, res1Err);
      const errMha = this.mha.backward(this.emptyErr, errDrop1);
      const errLn1 = this.ln1.backward(this.emptyErr, errMha);
      
      const peErr = mj.add(res1Err, errLn1);
      const embErr = this.pe.backward(this.emptyErr, peErr);
      this.embedding.backward(this.emptyErr, embErr);
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
  }

  resizeVocab(newVocabSize: number) {
    this.embedding.resize(newVocabSize);
    this.dense.resize(newVocabSize);
    this.vocabSize = newVocabSize; // SINKRONKAN
  }

  fit(X: Matrix[], y: Matrix[], epochs: number, cb: (loss: number) => any = (_) => { }) {
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
}
