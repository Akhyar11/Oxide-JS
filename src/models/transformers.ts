import { readFileSync } from "fs";
import { softmaxInto } from "../activation";
import mj from "../math";
import Matrix from "../matrix";
import Sequential from "./sequential";
import { MultiHeadAttention, Dense, PositionalEncoding, LayerNormalization, Embedding, Dropout } from "../layers";
import { FitConfig, FitResult } from "../@types/fitConfig";
import { isNativeAvailable, maskedSparseSoftmaxCrossEntropyNative } from "../math/rust_backend";
import { MemoryManager } from "../utils/memory";

export type TransformersPredictMode = "next-token" | "full-sequence";

interface TransformersConfig {
  units: number;          // d_model (embedding size)
  seqLen: number;         // sequence length
  vocabSize: number;      // vocabulary size
  heads?: number;         // number of attention heads (default 8)
  numBlocks?: number;     // number of transformer blocks (default 1)
  dropoutRate?: number;   // dropout rate (default 0.1)
  alpha?: number;         // learning rate
  padTokenId?: number;
  clipGradient?: number | boolean;
  predictMode?: TransformersPredictMode;
}

interface TransformerBlock {
  ln1: LayerNormalization;
  mha: MultiHeadAttention;
  drop1: Dropout;
  ln2: LayerNormalization;
  ffn1: Dense;
  dropFfn: Dropout;
  ffn2: Dense;
  drop2: Dropout;
  
  // Capacity-based residual buffers
  xRes1Data: any;
  xRes2Data: any;
  xRes1: Matrix;
  xRes2: Matrix;

  errRes1Data: any;
  errRes2Data: any;
  errRes1Buf: Matrix;
  errRes2Buf: Matrix;
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
  public numBlocks: number;
  private embedding: Embedding;
  private pe: PositionalEncoding;
  private blocks: TransformerBlock[];
  private dense: Dense;

  // Alias ke block pertama untuk kompatibilitas internal/test lama.
  private ln1: LayerNormalization;
  private mha: MultiHeadAttention;
  private drop1: Dropout;
  private ln2: LayerNormalization;
  private ffn1: Dense;
  private dropFfn: Dropout;
  private ffn2: Dense;
  private drop2: Dropout;

  private lastTokenBuffer: Matrix;
  private lossGradientBuffer: Matrix;
  private lastInputTokens: Matrix = mj.matrix([]);
  private emptyErr: Matrix = mj.matrix([[]]);
  private padMaskBuffer: boolean[] = [];
  private profilerEnabled: boolean = false;
  private profileStats: { [key: string]: { totalMs: number; count: number } } = Object.create(null);
  private positionOffset: number = 0;
  private predictMode: TransformersPredictMode;

  constructor({
    units,
    seqLen,
    vocabSize,
    heads = 8,
    numBlocks = 1,
    dropoutRate = 0.1,
    alpha = 0.01,
    padTokenId,
    clipGradient = 5.0,
    predictMode = "next-token",
  }: TransformersConfig) {
    if (!Number.isInteger(numBlocks) || numBlocks < 1) {
      throw new Error(`Transformers: numBlocks harus integer >= 1, got ${numBlocks}`);
    }
    if (predictMode !== "next-token" && predictMode !== "full-sequence") {
      throw new Error(`Transformers: predictMode harus "next-token" atau "full-sequence", got ${predictMode}`);
    }

    const embedding = new Embedding({ vocabSize, embeddingDim: units, alpha, padTokenId });
    const pe = new PositionalEncoding({ dModel: units, maxSeqLen: seqLen });
    const blocks: TransformerBlock[] = [];
    const flatLayers: Array<Embedding | PositionalEncoding | LayerNormalization | MultiHeadAttention | Dropout | Dense> = [embedding, pe];

    for (let blockIndex = 0; blockIndex < numBlocks; blockIndex++) {
      const ln1 = new LayerNormalization({ units, clipGradient });
      const mha = new MultiHeadAttention({ units, heads, seqLen, alpha, clipGradient });
      const drop1 = new Dropout({ rate: dropoutRate });

      const ln2 = new LayerNormalization({ units, clipGradient });
      const ffn1 = new Dense({ units, outputUnits: units * 4, activation: "relu", alpha, clipGradient });
      const dropFfn = new Dropout({ rate: dropoutRate });
      const ffn2 = new Dense({ units: units * 4, outputUnits: units, activation: "linear", alpha, clipGradient });
      const drop2 = new Dropout({ rate: dropoutRate });

      blocks.push({
        ln1,
        mha,
        drop1,
        ln2,
        ffn1,
        dropFfn,
        ffn2,
        drop2,
        xRes1Data: new Float32Array(0),
        xRes2Data: new Float32Array(0),
        xRes1: mj.matrix([]),
        xRes2: mj.matrix([]),
        errRes1Data: new Float32Array(0),
        errRes2Data: new Float32Array(0),
        errRes1Buf: mj.matrix([]),
        errRes2Buf: mj.matrix([]),
      });

      flatLayers.push(ln1, mha, drop1, ln2, ffn1, dropFfn, ffn2, drop2);
    }

    // Output Projector (applied independently to sequence length)
    const dense = new Dense({
      units: units,
      outputUnits: vocabSize,
      activation: "linear",
      alpha,
      status: "output",
      loss: "softmaxCrossEntropy", // Paksa gunakan Cross Entropy dari awal
      clipGradient
    });
    flatLayers.push(dense);

    super({ layers: flatLayers });

    this.embedding = embedding;
    this.pe = pe;
    this.blocks = blocks;
    this.numBlocks = numBlocks;
    this.ln1 = blocks[0].ln1;
    this.mha = blocks[0].mha;
    this.drop1 = blocks[0].drop1;
    this.ln2 = blocks[0].ln2;
    this.ffn1 = blocks[0].ffn1;
    this.dropFfn = blocks[0].dropFfn;
    this.ffn2 = blocks[0].ffn2;
    this.drop2 = blocks[0].drop2;
    this.dense = dense;
    this.predictMode = predictMode;

    // Pre-allocate buffers (will be managed by capacity ensures)
    this.lastTokenBuffer = mj.matrix([]);
    this.lossGradientBuffer = mj.matrix([]);
    this.vocabSize = vocabSize;
    this.train();
  }

  forward(x: Matrix, batchSize = 1, options?: { workspace?: "train" | "eval" }): Matrix {
    this.lastInputTokens = x;
    const res2 = this.forwardTransformerBlock(x, options);
    if (this.isTrainingMode) {
      const outputDenseForwardStart = this.profileStart();
      const out = this.dense.forward(res2, options);
      this.profileEnd("output dense forward", outputDenseForwardStart);
      return out;
    }
    return this.projectLastToken(res2, x._shape[0], x._shape[1], options);
  }

  forwardFullSequence(x: Matrix, options?: { workspace?: "train" | "eval" }): Matrix {
    this.lastInputTokens = x;
    const res2 = this.forwardTransformerBlock(x, options);
    const outputDenseForwardStart = this.profileStart();
    const out = this.dense.forward(res2, options);
    this.profileEnd("output dense forward", outputDenseForwardStart);
    return out;
  }

  forwardNextToken(x: Matrix, options?: { workspace?: "train" | "eval" }): Matrix {
    this.lastInputTokens = x;
    const res2 = this.forwardTransformerBlock(x, options);
    if (this.isTrainingMode) {
      const lastTokenState = this.extractLastTokenState(res2, x._shape[0], x._shape[1]);
      const outputDenseForwardStart = this.profileStart();
      const out = this.dense.forward(lastTokenState, options);
      this.profileEnd("output dense forward", outputDenseForwardStart);
      return out;
    }
    return this.projectLastToken(res2, x._shape[0], x._shape[1], options);
  }

  predict(x: Matrix): Matrix {
    const wasTraining = this.isTrainingMode;
    this.eval();
    const out = this.predictMode === "full-sequence"
      ? this.forwardFullSequence(x, { workspace: "eval" })
      : this.forwardNextToken(x, { workspace: "eval" });
    if (wasTraining) this.train();
    return out;
  }

  setPredictMode(mode: TransformersPredictMode): this {
    if (mode !== "next-token" && mode !== "full-sequence") {
      throw new Error(`Transformers.setPredictMode: mode harus "next-token" atau "full-sequence", got ${mode}`);
    }
    this.predictMode = mode;
    return this;
  }

  getPredictMode(): TransformersPredictMode {
    return this.predictMode;
  }

  backward(y: Matrix) {
    const batchSize = this.lastInputTokens._shape[1];
    const seqLen = this.lastInputTokens._shape[0];
    const units = this.embedding.embeddingDim;
    const totalTokens = seqLen * batchSize;
    this.ensureBlockBuffers(units, totalTokens);

    const outputDenseBackwardStart = this.profileStart();
    let errDense: Matrix;
    if (y._shape[0] === seqLen) {
      const lossGradientBuildStart = this.profileStart();
      const lossState = this.buildShiftedLossGradient(y);
      this.profileEnd("loss gradient build", lossGradientBuildStart);
      const denseBackwardStart = this.profileStart();
      errDense = this.dense.backward(this.emptyErr, lossState.gradient);
      this.profileEnd("output dense backward", denseBackwardStart);
      this.loss = lossState.loss;
      this.dense.loss = lossState.loss;
    } else if (y._shape[0] === 1) {
      errDense = this.dense.backward(y, this.emptyErr);
      this.profileEnd("output dense backward", outputDenseBackwardStart);
      this.loss = this.dense.loss;
    } else {
      this.profileEnd("output dense backward", outputDenseBackwardStart);
      throw new Error(
        `Transformers.backward: expected target shape [${seqLen}, batch] for full-sequence training or [1, batch] for legacy last-token training, got [${y._shape[0]}, ${y._shape[1]}]`
      );
    }

    // 2. Map Dense Error back to the full sequence length matrix
    const mapDenseErrStart = this.profileStart();
    let res2Err = errDense;
    if (y._shape[0] === 1) {
      const lastBlock = this.blocks[this.blocks.length - 1];
      if (lastBlock.errRes2Buf._shape[0] !== units || lastBlock.errRes2Buf._shape[1] !== totalTokens) {
        lastBlock.errRes2Buf = mj.zeros([units, totalTokens]);
      } else {
        lastBlock.errRes2Buf._data.fill(0);
      }
      res2Err = lastBlock.errRes2Buf;
      const res2ErrData = res2Err._data;
      const errDenseData = errDense._data;

      for (let b = 0; b < batchSize; b++) {
        const lastTokenCol = (b + 1) * seqLen - 1;
        for (let i = 0; i < units; i++) {
          res2ErrData[i * totalTokens + lastTokenCol] = errDenseData[i * batchSize + b];
        }
      }
    }
    this.profileEnd("mapping dense error", mapDenseErrStart);

    // 3. Block Backward (reverse order)
    let blockErr = res2Err;
    for (let blockIndex = this.blocks.length - 1; blockIndex >= 0; blockIndex--) {
      const block = this.blocks[blockIndex];

      const ffnBackwardStart = this.profileStart();
      const errDrop2 = block.drop2.backward(this.emptyErr, blockErr);
      const errFfn2 = block.ffn2.backward(this.emptyErr, errDrop2);
      const errDropFfn = block.dropFfn.backward(this.emptyErr, errFfn2);
      const errFfn1 = block.ffn1.backward(this.emptyErr, errDropFfn);
      this.profileEnd(`FFN backward [block ${blockIndex}]`, ffnBackwardStart);

      const layerNorm2BackwardStart = this.profileStart();
      const errLn2 = block.ln2.backward(this.emptyErr, errFfn1);
      this.profileEnd(`layer norm backward [block ${blockIndex}]`, layerNorm2BackwardStart);

      const res1Err = mj.addInto(blockErr, errLn2, block.errRes1Buf);

      const errDrop1 = block.drop1.backward(this.emptyErr, res1Err);
      const mhaBackwardStart = this.profileStart();
      const errMha = block.mha.backward(this.emptyErr, errDrop1);
      this.profileEnd(`MHA backward [block ${blockIndex}]`, mhaBackwardStart);

      const layerNorm1BackwardStart = this.profileStart();
      const errLn1 = block.ln1.backward(this.emptyErr, errMha);
      this.profileEnd(`layer norm backward [block ${blockIndex}]`, layerNorm1BackwardStart);

      // Reuse errRes2Buf block-local setelah blockErr sebelumnya tidak lagi dipakai.
      blockErr = mj.addInto(res1Err, errLn1, block.errRes2Buf);
    }

    // 4. PE & Embedding Backward
    const embeddingBackwardStart = this.profileStart();
    const embErr = this.pe.backward(this.emptyErr, blockErr);
    this.embedding.backward(this.emptyErr, embErr);
    this.profileEnd("embedding backward", embeddingBackwardStart);
  }

  releaseWorkspace(): void {
    super.releaseWorkspace();
    this.lastTokenBuffer = mj.matrix([]);
    this.lossGradientBuffer = mj.matrix([]);
    for (const block of this.blocks) {
      block.xRes1Data = new Float32Array(0);
      block.xRes2Data = new Float32Array(0);
      block.xRes1 = mj.matrix([]);
      block.xRes2 = mj.matrix([]);
      block.errRes1Data = new Float32Array(0);
      block.errRes2Data = new Float32Array(0);
      block.errRes1Buf = mj.matrix([]);
      block.errRes2Buf = mj.matrix([]);
    }
  }

  dispose(): void {
    super.dispose();
    this.blocks = [];
  }

  private forwardTransformerBlock(x: Matrix, options?: { workspace?: "train" | "eval" }): Matrix {
    const [seqLen, batchSize] = x._shape;
    const units = this.embedding.embeddingDim;
    const totalTokens = seqLen * batchSize;
    this.ensureBlockBuffers(units, totalTokens);

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
    const xEmb = this.embedding.forward(x, options); // returns [Units, totalTokens]
    this.profileEnd("embedding forward", embeddingForwardStart);

    // 2. Positional Encoding
    const xPe = this.pe.forward(xEmb, this.positionOffset, seqLen, undefined, options);

    // 3. Transformer Blocks
    let h = xPe;
    for (let blockIndex = 0; blockIndex < this.blocks.length; blockIndex++) {
      const block = this.blocks[blockIndex];
      const layerNorm1ForwardStart = this.profileStart();
      const xLn1 = block.ln1.forward(h, options);
      this.profileEnd(`layer norm forward [block ${blockIndex}]`, layerNorm1ForwardStart);

      block.mha.setPadMask(this.padMaskBuffer);
      block.mha.setEffectiveSeqLen(seqLen);
      const mhaForwardStart = this.profileStart();
      const xMhaOut = block.mha.forward(xLn1, options);
      this.profileEnd(`MHA forward [block ${blockIndex}]`, mhaForwardStart);
      const xDrop1Out = block.drop1.forward(xMhaOut, options);
      const res1 = mj.addInto(h, xDrop1Out, block.xRes1);

      const layerNorm2ForwardStart = this.profileStart();
      const xLn2 = block.ln2.forward(res1, options);
      this.profileEnd(`layer norm forward [block ${blockIndex}]`, layerNorm2ForwardStart);
      const ffnForwardStart = this.profileStart();
      const xFfn1Out = block.ffn1.forward(xLn2, options);
      const xDropFfnOut = block.dropFfn.forward(xFfn1Out, options);
      const xFfn2Out = block.ffn2.forward(xDropFfnOut, options);
      const xDrop2Out = block.drop2.forward(xFfn2Out, options);
      this.profileEnd(`FFN forward [block ${blockIndex}]`, ffnForwardStart);
      h = mj.addInto(res1, xDrop2Out, block.xRes2);
    }

    return h;
  }

  private projectLastToken(res2: Matrix, seqLen: number, batchSize: number, options?: { workspace?: "train" | "eval" }): Matrix {
    const outputDenseForwardStart = this.profileStart();
    const out = this.dense.projectLastTokenFromSequence(res2, seqLen, batchSize, options);
    this.profileEnd("output dense forward", outputDenseForwardStart);
    return out;
  }

  private extractLastTokenState(res2: Matrix, seqLen: number, batchSize: number): Matrix {
    const units = res2._shape[0];
    if (this.lastTokenBuffer._shape[0] !== units || this.lastTokenBuffer._shape[1] !== batchSize) {
      this.lastTokenBuffer = mj.zeros([units, batchSize]);
    }

    const sourceData = res2._data;
    const totalCols = res2._shape[1];
    const outData = this.lastTokenBuffer._data;

    for (let b = 0; b < batchSize; b++) {
      const tokenCol = (b + 1) * seqLen - 1;
      for (let i = 0; i < units; i++) {
        outData[i * batchSize + b] = sourceData[i * totalCols + tokenCol];
      }
    }

    return this.lastTokenBuffer;
  }

  private buildShiftedLossGradient(targets: Matrix): { loss: number; gradient: Matrix; validTokens: number } {
    const [seqLen, batchSize] = targets._shape;
    const totalTokens = seqLen * batchSize;
    const logits = this.dense.getLastOutput();

    if (this.lossGradientBuffer._shape[0] !== this.vocabSize || this.lossGradientBuffer._shape[1] !== totalTokens) {
      this.lossGradientBuffer = mj.zeros([this.vocabSize, totalTokens]);
    }
    if (isNativeAvailable()) {
      const result = maskedSparseSoftmaxCrossEntropyNative(
        logits._data,
        this.lastInputTokens._data,
        targets._data,
        seqLen,
        batchSize,
        this.vocabSize,
        this.embedding.padTokenId,
        this.lossGradientBuffer._data
      );
      return {
        loss: result.loss,
        gradient: this.lossGradientBuffer,
        validTokens: result.validTokens,
      };
    }

    const probs = softmaxInto(logits, this.lossGradientBuffer, false);
    const gradData = this.lossGradientBuffer._data;
    const probsData = probs._data;
    const targetData = targets._data;
    const inputData = this.lastInputTokens._data;
    const epsilon = 1e-15;
    const padTokenId = this.embedding.padTokenId;
    let totalLoss = 0;
    let validTokens = 0;
    for (let b = 0; b < batchSize; b++) {
      for (let pos = 0; pos < seqLen; pos++) {
        const sourceIndex = pos * batchSize + b;
        const tokenIndex = b * seqLen + pos;
        const sourceToken = Math.floor(inputData[sourceIndex]);
        const targetToken = Math.floor(targetData[sourceIndex]);
        const canTrainOnPosition =
          pos < seqLen - 1 &&
          (padTokenId === null || (sourceToken !== padTokenId && targetToken !== padTokenId));

        if (!canTrainOnPosition) {
          for (let vocabIndex = 0; vocabIndex < this.vocabSize; vocabIndex++) {
            gradData[vocabIndex * totalTokens + tokenIndex] = 0;
          }
          continue;
        }
        if (targetToken < 0 || targetToken >= this.vocabSize) {
          throw new Error(
            `Transformers.backward: target token '${targetToken}' di posisi ${pos} batch ${b} berada di luar vocab (0 - ${this.vocabSize - 1})`
          );
        }

        validTokens++;
        const targetOffset = targetToken * totalTokens + tokenIndex;
        totalLoss -= Math.log(Math.max(epsilon, probsData[targetOffset]));
        gradData[targetOffset] -= 1;
      }
    }

    if (validTokens === 0) {
      throw new Error("Transformers.backward: tidak ada token valid untuk full-sequence causal LM loss.");
    }

    return {
      loss: totalLoss / validTokens,
      gradient: this.lossGradientBuffer,
      validTokens,
    };
  }

  protected computeSampleLoss(yTrue: Matrix, yPred: Matrix): number {
    const seqLen = this.lastInputTokens._shape[0];
    const batchSize = this.lastInputTokens._shape[1];
    const isFullSequenceTarget =
      yTrue._shape[0] === seqLen &&
      yTrue._shape[1] === batchSize &&
      yPred._shape[0] === this.vocabSize &&
      yPred._shape[1] === seqLen * batchSize;

    if (!isFullSequenceTarget) {
      return super.computeSampleLoss(yTrue, yPred);
    }

    if (isNativeAvailable()) {
      if (this.lossGradientBuffer._shape[0] !== this.vocabSize || this.lossGradientBuffer._shape[1] !== seqLen * batchSize) {
        this.lossGradientBuffer = mj.zeros([this.vocabSize, seqLen * batchSize]);
      }
      const result = maskedSparseSoftmaxCrossEntropyNative(
        yPred._data,
        this.lastInputTokens._data,
        yTrue._data,
        seqLen,
        batchSize,
        this.vocabSize,
        this.embedding.padTokenId,
        this.lossGradientBuffer._data
      );
      if (result.validTokens === 0) {
        throw new Error("Transformers.computeSampleLoss: tidak ada token valid untuk full-sequence causal LM loss.");
      }
      return result.loss;
    }

    const logitsData = yPred._data;
    const targetData = yTrue._data;
    const inputData = this.lastInputTokens._data;
    const padTokenId = this.embedding.padTokenId;

    let totalLoss = 0;
    let validTokens = 0;
    const totalTokens = seqLen * batchSize;

    for (let b = 0; b < batchSize; b++) {
      for (let pos = 0; pos < seqLen; pos++) {
        const sourceIndex = pos * batchSize + b;
        const tokenIndex = b * seqLen + pos;
        const sourceToken = Math.floor(inputData[sourceIndex]);
        const targetToken = Math.floor(targetData[sourceIndex]);
        const canTrainOnPosition =
          pos < seqLen - 1 &&
          (padTokenId === null || (sourceToken !== padTokenId && targetToken !== padTokenId));

        if (!canTrainOnPosition) {
          continue;
        }

        if (targetToken < 0 || targetToken >= this.vocabSize) {
          throw new Error(
            `Transformers.computeSampleLoss: target token '${targetToken}' di posisi ${pos} batch ${b} berada di luar vocab (0 - ${this.vocabSize - 1})`
          );
        }

        validTokens++;
        let maxLogit = -Infinity;
        for (let vocabIndex = 0; vocabIndex < this.vocabSize; vocabIndex++) {
          const value = logitsData[vocabIndex * totalTokens + tokenIndex];
          if (value > maxLogit) maxLogit = value;
        }

        let sumExp = 0;
        for (let vocabIndex = 0; vocabIndex < this.vocabSize; vocabIndex++) {
          sumExp += Math.exp(logitsData[vocabIndex * totalTokens + tokenIndex] - maxLogit);
        }

        totalLoss += Math.log(sumExp) + maxLogit - logitsData[targetToken * totalTokens + tokenIndex];
      }
    }

    if (validTokens === 0) {
      throw new Error("Transformers.computeSampleLoss: tidak ada token valid untuk full-sequence causal LM loss.");
    }

    return totalLoss / validTokens;
  }

  protected useBackwardLossForTrainingBatch(yTrue: Matrix, yPred: Matrix): boolean {
    const seqLen = this.lastInputTokens._shape[0];
    const batchSize = this.lastInputTokens._shape[1];
    return (
      yTrue._shape[0] === seqLen &&
      yTrue._shape[1] === batchSize &&
      yPred._shape[0] === this.vocabSize &&
      yPred._shape[1] === seqLen * batchSize
    );
  }

  load(path: string) {
    const dataJson = readFileSync(path, "utf-8");
    const data = JSON.parse(dataJson);

    if (!Array.isArray(data) || data.length < 11) {
      throw new Error(`Invalid transformer model file: ${path}`);
    }

    const inferredNumBlocks = this.inferNumBlocksFromSerializedLayers(data);
    if (inferredNumBlocks !== this.blocks.length) {
      throw new Error(
        `Transformers.load: model memiliki ${inferredNumBlocks} block, tetapi instance saat ini dibuat dengan ${this.blocks.length} block.`
      );
    }

    const [embedding, _pe] = data;
    const dense = data[data.length - 1];

    if (embedding?.weight) {
      this.embedding.load(embedding.weight);
      if ("padTokenId" in embedding) {
        this.embedding.padTokenId = embedding.padTokenId;
      }
    }

    let offset = 2;
    for (const block of this.blocks) {
      const [ln1, mha, drop1, ln2, ffn1, dropFfn, ffn2, drop2] = data.slice(offset, offset + 8);
      if (ln1?.gamma && ln1?.beta) block.ln1.load(ln1.gamma, ln1.beta, ln1.clipGradient);
      if (mha) block.mha.load(mha);
      if (drop1?.rate !== undefined) block.drop1.load({ rate: drop1.rate, status: drop1.status ?? block.drop1.status });
      if (ln2?.gamma && ln2?.beta) block.ln2.load(ln2.gamma, ln2.beta, ln2.clipGradient);
      if (ffn1?.weight && ffn1?.bias) block.ffn1.load(ffn1.weight, ffn1.bias, ffn1.clipGradient);
      if (dropFfn?.rate !== undefined) block.dropFfn.load({ rate: dropFfn.rate, status: dropFfn.status ?? block.dropFfn.status });
      if (ffn2?.weight && ffn2?.bias) block.ffn2.load(ffn2.weight, ffn2.bias, ffn2.clipGradient);
      if (drop2?.rate !== undefined) block.drop2.load({ rate: drop2.rate, status: drop2.status ?? block.drop2.status });
      offset += 8;
    }

    if (dense?.weight && dense?.bias) this.dense.load(dense.weight, dense.bias, dense.clipGradient);

    this.vocabSize = this.embedding.vocabSize;
  }

  resizeVocab(newVocabSize: number) {
    this.embedding.resize(newVocabSize);
    this.dense.resize(newVocabSize);
    this.vocabSize = newVocabSize; // SINKRONKAN
  }

  fit(
    X: Matrix[],
    y: Matrix[],
    epochs: number,
    config?: FitConfig
  ): FitResult;
  fit(
    X: Matrix[],
    y: Matrix[],
    epochs: number,
    cb?: (loss: number) => any
  ): FitResult;
  fit(
    X: Matrix[],
    y: Matrix[],
    epochs: number,
    configOrCb: FitConfig | ((loss: number) => any) = {}
  ): FitResult {
    return super.fit(X, y, epochs, configOrCb as any);
  }

  /**
   * Sets the position offset applied to Positional Encoding during the next
   * forward pass. Use before training a trimmed left-padded batch so that
   * real tokens retain their original absolute positions in the PE table.
   */
  setPositionOffset(offset: number): this {
    this.positionOffset = Math.max(0, Math.floor(offset));
    return this;
  }

  /** Resets the position offset back to 0 (default). */
  resetPositionOffset(): this {
    this.positionOffset = 0;
    for (const block of this.blocks) {
      block.mha.resetEffectiveSeqLen();
    }
    return this;
  }

  /** Returns the padTokenId used by the embedding layer. */
  getPadTokenId(): number | null {
    return this.embedding.padTokenId;
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

  private ensureBlockBuffers(units: number, totalTokens: number): void {
    const required = units * totalTokens;
    for (const block of this.blocks) {
      block.xRes1Data = MemoryManager.ensureCapacity(block.xRes1Data, required, this.layers[0].memoryConfig);
      block.xRes2Data = MemoryManager.ensureCapacity(block.xRes2Data, required, this.layers[0].memoryConfig);
      block.errRes1Data = MemoryManager.ensureCapacity(block.errRes1Data, required, this.layers[0].memoryConfig);
      block.errRes2Data = MemoryManager.ensureCapacity(block.errRes2Data, required, this.layers[0].memoryConfig);

      block.xRes1 = Matrix.fromFlat(block.xRes1Data.subarray(0, required), [units, totalTokens]);
      block.xRes2 = Matrix.fromFlat(block.xRes2Data.subarray(0, required), [units, totalTokens]);
      block.errRes1Buf = Matrix.fromFlat(block.errRes1Data.subarray(0, required), [units, totalTokens]);
      block.errRes2Buf = Matrix.fromFlat(block.errRes2Data.subarray(0, required), [units, totalTokens]);
    }
  }

  private inferNumBlocksFromSerializedLayers(layers: Array<Record<string, unknown>>): number {
    const mhaCount = layers.filter((layer) => layer.name === "multi head attention layer").length;
    if (mhaCount > 0) {
      return mhaCount;
    }

    const inferred = (layers.length - 3) / 8;
    if (Number.isInteger(inferred) && inferred >= 1) {
      return inferred;
    }
    return 1;
  }
}
