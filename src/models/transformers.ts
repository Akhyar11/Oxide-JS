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
      activation: "linear",  // Note: softmax can be applied in loss or another layer if needed
      alpha,
      status: "output",
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
  }

  forward(x: Matrix): Matrix {
    this.xInput = x;

    // Embedding & Positional Encoding
    this.xEmb = this.embedding.forward(x);
    this.xPe = this.pe.forward(this.xEmb);
    
    // --- Transformer Block ---
    // 1. Attention part (Pre-Norm)
    this.xLn1 = this.ln1.forward(this.xPe);
    const xMhaOut = this.mha.forward(this.xLn1);
    const xDrop1Out = this.drop1.forward(xMhaOut);
    
    // Residual 1
    this.xRes1.copyFrom(this.xPe);
    this.xRes1.addInPlace(xDrop1Out);
    
    // 2. FFN part (Pre-Norm)
    this.xLn2 = this.ln2.forward(this.xRes1);
    const xFfn1Out = this.ffn1.forward(this.xLn2);
    const xDropFfnOut = this.dropFfn.forward(xFfn1Out);
    const xFfn2Out = this.ffn2.forward(xDropFfnOut);
    const xDrop2Out = this.drop2.forward(xFfn2Out);

    // Residual 2
    this.xRes2.copyFrom(this.xRes1);
    this.xRes2.addInPlace(xDrop2Out);

    // --- Output Projection ---
    // Instead of flatten, we apply Dense to every token in sequence: [vocabSize, seqLen]
    return this.dense.forward(this.xRes2);
  }

  backward(y: Matrix) {
    // 1. Backward Output Dense -> grad shape: [units, seqLen]
    const errDense = this.dense.backward(y, mj.matrix([[]]));
    this.loss = this.dense.loss;

    // Gradient that flows into Residual 2 (from the output projection)
    this.errRes2Buf.copyFrom(errDense);
    
    // 2. Backward FFN block
    const errDrop2 = this.drop2.backward(y, errDense);
    const errFfn2 = this.ffn2.backward(y, errDrop2);
    const errDropFfn = this.dropFfn.backward(y, errFfn2);
    const errFfn1 = this.ffn1.backward(y, errDropFfn);
    const errLn2 = this.ln2.backward(y, errFfn1);

    // Gradient that flows into Residual 1 is: (Grad from Ln2) + (Grad flowing bypass in Res 2)
    this.errRes1Buf.copyFrom(this.errRes2Buf);
    this.errRes1Buf.addInPlace(errLn2);

    // 3. Backward Attention block
    const errDrop1 = this.drop1.backward(y, this.errRes1Buf);
    const errMha = this.mha.backward(y, errDrop1);
    const errLn1 = this.ln1.backward(y, errMha);

    // Gradient into PE is: (Grad from Ln1) + (Grad flowing bypass in Res 1)
    // we reuse errRes1Buf since we don't need it anymore
    this.errRes1Buf.addInPlace(errLn1);
    const totalErrPe = this.errRes1Buf;

    // 4. Backward Positional Encoding & Embedding
    const errEmb = this.pe.backward(y, totalErrPe);
    this.embedding.backward(y, errEmb);
  }

  load(path: string) {
    super.load(path);
    for (const layer of this.layers) {
      if (layer instanceof Embedding) this.embedding = layer;
      else if (layer instanceof PositionalEncoding) this.pe = layer;
      else if (layer instanceof MultiHeadAttention) this.mha = layer;
      else if (layer instanceof Dropout) {
         // assign based on their indices if needed, standard load does layers anyway
      }
      // re-mapping might require strict index checking if multiple of same layer exist
    }
  }

  resizeVocab(newVocabSize: number): void {
      this.embedding.resize(newVocabSize);
      this.dense.resize(newVocabSize);
  }

  fit(
    X: Matrix[],
    y: Matrix[],
    epochs: number,
    cb: (loss: number) => any = (_) => { }
  ) {
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
