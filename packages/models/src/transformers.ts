import { readFileSync } from "fs";
import { softmaxInto } from "@oxide-js/core";
import { mj } from "@oxide-js/core";
import { Matrix } from "@oxide-js/core";
import Sequential from "./sequential.js";
import { MultiHeadAttention, Dense, PositionalEncoding, LayerNormalization, Embedding, Dropout } from "@oxide-js/layers";
import { FitConfig, FitResult } from "@oxide-js/core";
import { isNativeAvailable, maskedSparseSoftmaxCrossEntropyNative, trimPaddingBatch, formatLoss, formatProgressBar, formatTime, shuffleInPlace, splitTrainValidation } from "@oxide-js/core";

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
  embeddingTrainable?: boolean;
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
  xRes1: Matrix;
  xRes2: Matrix;
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
  private lastFullSequenceLossWeight: number = 0;
  private fitBatchInputBufferData: Float32Array = new Float32Array(0);
  private fitBatchTargetBufferData: Float32Array = new Float32Array(0);
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
    embeddingTrainable = true,
    clipGradient = 5.0,
    predictMode = "next-token",
  }: TransformersConfig) {
    if (!Number.isInteger(numBlocks) || numBlocks < 1) {
      throw new Error(`Transformers: numBlocks harus integer >= 1, got ${numBlocks}`);
    }
    if (predictMode !== "next-token" && predictMode !== "full-sequence") {
      throw new Error(`Transformers: predictMode harus "next-token" atau "full-sequence", got ${predictMode}`);
    }

    const embedding = new Embedding({ vocabSize, embeddingDim: units, alpha, padTokenId, trainable: embeddingTrainable });
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
        xRes1: mj.zeros([units, seqLen]),
        xRes2: mj.zeros([units, seqLen]),
        errRes1Buf: mj.zeros([units, seqLen]),
        errRes2Buf: mj.zeros([units, seqLen]),
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

    // Pre-allocate buffers
    this.lastTokenBuffer = mj.zeros([units, 1]);
    this.lossGradientBuffer = mj.zeros([vocabSize, seqLen]);
    this.vocabSize = vocabSize;
    this.train();
  }

  forward(x: Matrix): Matrix {
    this.lastInputTokens = x;
    const res2 = this.forwardTransformerBlock(x);
    if (this.isTrainingMode) {
      const outputDenseForwardStart = this.profileStart();
      const out = this.dense.forward(res2);
      this.profileEnd("output dense forward", outputDenseForwardStart);
      return out;
    }
    return this.projectLastToken(res2, x._shape[0], x._shape[1]);
  }

  forwardFullSequence(x: Matrix): Matrix {
    this.lastInputTokens = x;
    const res2 = this.forwardTransformerBlock(x);
    const outputDenseForwardStart = this.profileStart();
    const out = this.dense.forward(res2);
    this.profileEnd("output dense forward", outputDenseForwardStart);
    return out;
  }

  forwardNextToken(x: Matrix): Matrix {
    this.lastInputTokens = x;
    const res2 = this.forwardTransformerBlock(x);
    if (this.isTrainingMode) {
      const lastTokenState = this.extractLastTokenState(res2, x._shape[0], x._shape[1]);
      const outputDenseForwardStart = this.profileStart();
      const out = this.dense.forward(lastTokenState);
      this.profileEnd("output dense forward", outputDenseForwardStart);
      return out;
    }
    return this.projectLastToken(res2, x._shape[0], x._shape[1]);
  }

  predict(x: Matrix): Matrix {
    const wasTraining = this.isTrainingMode;
    this.eval();
    const out = this.predictMode === "full-sequence"
      ? this.forwardFullSequence(x)
      : this.forwardNextToken(x);
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

  backward(y: Matrix, batchSizeArg: number = 1, gradOnly = false) {
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
      errDense = this.dense.backward(this.emptyErr, lossState.gradient, gradOnly);
      this.profileEnd("output dense backward", denseBackwardStart);
      this.loss = lossState.loss;
      this.dense.loss = lossState.loss;
      this.lastFullSequenceLossWeight = lossState.validTokens;
    } else if (y._shape[0] === 1) {
      errDense = this.dense.backward(y, this.emptyErr, gradOnly);
      this.profileEnd("output dense backward", outputDenseBackwardStart);
      this.loss = this.dense.loss;
      this.lastFullSequenceLossWeight = y._shape[1];
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
      const errDrop2 = block.drop2.backward(this.emptyErr, blockErr, gradOnly);
      const errFfn2 = block.ffn2.backward(this.emptyErr, errDrop2, gradOnly);
      const errDropFfn = block.dropFfn.backward(this.emptyErr, errFfn2, gradOnly);
      const errFfn1 = block.ffn1.backward(this.emptyErr, errDropFfn, gradOnly);
      this.profileEnd(`FFN backward [block ${blockIndex}]`, ffnBackwardStart);

      const layerNorm2BackwardStart = this.profileStart();
      const errLn2 = block.ln2.backward(this.emptyErr, errFfn1, gradOnly);
      this.profileEnd(`layer norm backward [block ${blockIndex}]`, layerNorm2BackwardStart);

      const res1Err = mj.addInto(blockErr, errLn2, block.errRes1Buf);

      const errDrop1 = block.drop1.backward(this.emptyErr, res1Err, gradOnly);
      const mhaBackwardStart = this.profileStart();
      const errMha = block.mha.backward(this.emptyErr, errDrop1, gradOnly);
      this.profileEnd(`MHA backward [block ${blockIndex}]`, mhaBackwardStart);

      const layerNorm1BackwardStart = this.profileStart();
      const errLn1 = block.ln1.backward(this.emptyErr, errMha, gradOnly);
      this.profileEnd(`layer norm backward [block ${blockIndex}]`, layerNorm1BackwardStart);

      // Reuse errRes2Buf block-local setelah blockErr sebelumnya tidak lagi dipakai.
      blockErr = mj.addInto(res1Err, errLn1, block.errRes2Buf);
    }

    // 4. PE & Embedding Backward
    const embeddingBackwardStart = this.profileStart();
    const embErr = this.pe.backward(this.emptyErr, blockErr, gradOnly);
    this.embedding.backward(this.emptyErr, embErr, gradOnly);
    this.profileEnd("embedding backward", embeddingBackwardStart);
  }

  private forwardTransformerBlock(x: Matrix): Matrix {
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
    const xEmb = this.embedding.forward(x); // returns [Units, totalTokens]
    this.profileEnd("embedding forward", embeddingForwardStart);

    // 2. Positional Encoding
    const xPe = this.pe.forward(xEmb, this.positionOffset, seqLen);

    // 3. Transformer Blocks
    let h = xPe;
    for (let blockIndex = 0; blockIndex < this.blocks.length; blockIndex++) {
      const block = this.blocks[blockIndex];
      const layerNorm1ForwardStart = this.profileStart();
      const xLn1 = block.ln1.forward(h);
      this.profileEnd(`layer norm forward [block ${blockIndex}]`, layerNorm1ForwardStart);

      block.mha.setPadMask(this.padMaskBuffer);
      block.mha.setEffectiveSeqLen(seqLen);
      const mhaForwardStart = this.profileStart();
      const xMhaOut = block.mha.forward(xLn1);
      this.profileEnd(`MHA forward [block ${blockIndex}]`, mhaForwardStart);
      const xDrop1Out = block.drop1.forward(xMhaOut);
      const res1 = mj.addInto(h, xDrop1Out, block.xRes1);

      const layerNorm2ForwardStart = this.profileStart();
      const xLn2 = block.ln2.forward(res1);
      this.profileEnd(`layer norm forward [block ${blockIndex}]`, layerNorm2ForwardStart);
      const ffnForwardStart = this.profileStart();
      const xFfn1Out = block.ffn1.forward(xLn2);
      const xDropFfnOut = block.dropFfn.forward(xFfn1Out);
      const xFfn2Out = block.ffn2.forward(xDropFfnOut);
      const xDrop2Out = block.drop2.forward(xFfn2Out);
      this.profileEnd(`FFN forward [block ${blockIndex}]`, ffnForwardStart);
      h = mj.addInto(res1, xDrop2Out, block.xRes2);
    }

    return h;
  }

  private projectLastToken(res2: Matrix, seqLen: number, batchSize: number): Matrix {
    const outputDenseForwardStart = this.profileStart();
    const out = this.dense.projectLastTokenFromSequence(res2, seqLen, batchSize);
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

    // Loss dilaporkan sebagai mean per valid token, maka gradien logits juga harus
    // dinormalisasi dengan jumlah token valid agar skala update konsisten.
    for (let i = 0; i < gradData.length; i++) {
      gradData[i] /= validTokens;
    }

    return {
      loss: totalLoss / validTokens,
      gradient: this.lossGradientBuffer,
      validTokens,
    };
  }

  protected computeSampleLoss(yTrue: Matrix, yPred: Matrix): number {
    return this.computeLossAndWeight(yTrue, yPred).loss;
  }

  protected computeLossAndWeight(yTrue: Matrix, yPred: Matrix): { loss: number; weight: number } {
    const seqLen = this.lastInputTokens._shape[0];
    const batchSize = this.lastInputTokens._shape[1];
    const isFullSequenceTarget =
      yTrue._shape[0] === seqLen &&
      yTrue._shape[1] === batchSize &&
      yPred._shape[0] === this.vocabSize &&
      yPred._shape[1] === seqLen * batchSize;

    if (!isFullSequenceTarget) {
      return super.computeLossAndWeight(yTrue, yPred);
    }

    return this.computeFullSequenceLossAndValidTokens(yTrue, yPred);
  }

  protected computeLossWeight(yTrue: Matrix, yPred: Matrix): number {
    const seqLen = this.lastInputTokens._shape[0];
    const batchSize = this.lastInputTokens._shape[1];
    const isFullSequenceTarget =
      yTrue._shape[0] === seqLen &&
      yTrue._shape[1] === batchSize &&
      yPred._shape[0] === this.vocabSize &&
      yPred._shape[1] === seqLen * batchSize;

    if (!isFullSequenceTarget) {
      return super.computeLossWeight(yTrue, yPred);
    }

    return this.computeFullSequenceLossAndValidTokens(yTrue, yPred).weight;
  }

  protected computeLossAndWeightFromBackward(yTrue: Matrix, yPred: Matrix): { loss: number; weight: number } {
    const seqLen = this.lastInputTokens._shape[0];
    const batchSize = this.lastInputTokens._shape[1];
    const isFullSequenceTarget =
      yTrue._shape[0] === seqLen &&
      yTrue._shape[1] === batchSize &&
      yPred._shape[0] === this.vocabSize &&
      yPred._shape[1] === seqLen * batchSize;

    if (!isFullSequenceTarget) {
      return super.computeLossAndWeightFromBackward(yTrue, yPred);
    }

    return {
      loss: this.loss,
      weight: this.lastFullSequenceLossWeight,
    };
  }

  private computeFullSequenceLossAndValidTokens(yTrue: Matrix, yPred: Matrix): { loss: number; weight: number } {
    const seqLen = this.lastInputTokens._shape[0];
    const batchSize = this.lastInputTokens._shape[1];
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
      this.lastFullSequenceLossWeight = result.validTokens;
      return {
        loss: result.loss,
        weight: result.validTokens,
      };
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

    this.lastFullSequenceLossWeight = validTokens;
    return {
      loss: totalLoss / validTokens,
      weight: validTokens,
    };
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

  private loadKerasLayersModel(modelJson: any, jsonPath: string): void {
    const layersConfig = modelJson.modelTopology?.config?.layers;
    if (!Array.isArray(layersConfig) || layersConfig.length !== this.layers.length) {
      throw new Error(`Invalid transformer model file: ${jsonPath}`);
    }

    const weightsManifest = modelJson.weightsManifest?.[0];
    const weights = weightsManifest?.weights ?? [];
    const binFilename = weightsManifest?.paths?.[0];
    if (!binFilename) {
      throw new Error(`Transformers.load: weights manifest missing for ${jsonPath}`);
    }

    const jsonDir = jsonPath.substring(0, jsonPath.lastIndexOf("/") + 1);
    const binBuffer = readFileSync(`${jsonDir}${binFilename}`);
    const combinedWeights = new Float32Array(binBuffer.buffer, binBuffer.byteOffset, binBuffer.byteLength / 4);
    let weightOffset = 0;

    for (let i = 0; i < this.layers.length; i++) {
      const layer = this.layers[i] as any;
      const config = layersConfig[i]?.config ?? {};

      if (layer instanceof Embedding && config.trainable !== undefined) {
        layer.load({
          trainable: config.trainable,
          padTokenId: config.mask_zero ? 0 : layer.padTokenId,
        });
      } else if (layer instanceof Dropout && config.rate !== undefined) {
        layer.load({ rate: config.rate, status: layer.status });
      }

      if (typeof layer.setWeightsFromBinary !== "function") continue;

      const layerName = config.name;
      const layerWeights = layerName
        ? weights.filter((w: any) => w.name.startsWith(`${layerName}/`))
        : [];
      if (layerWeights.length === 0) continue;

      const binaryWeights: Record<string, Float32Array> = {};
      for (const weightInfo of layerWeights) {
        const weightSize = weightInfo.shape.reduce((a: number, b: number) => a * b, 1);
        const paramName = weightInfo.name.split("/").pop();
        binaryWeights[paramName] = combinedWeights.subarray(weightOffset, weightOffset + weightSize);
        weightOffset += weightSize;
      }

      layer.setWeightsFromBinary(binaryWeights);
    }

    this.vocabSize = this.embedding.vocabSize;
  }

  load(path: string) {
    const dataJson = readFileSync(path, "utf-8");
    let data = JSON.parse(dataJson);

    if (data?.format === "layers-model") {
      this.loadKerasLayersModel(data, path);
      return;
    }

    // Support standardized oxide-v1 format
    if (data && data.format === "oxide-v1" && data.modelTopology) {
      data = data.modelTopology.layers;
    }

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
      this.embedding.load(embedding);
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
    if (!Array.isArray(X) || !Array.isArray(y) || X.length === 0 || X.length !== y.length) {
      throw new Error("X dan y harus memiliki jumlah sample yang sama dan tidak kosong");
    }
    if (!Number.isFinite(epochs) || epochs < 1) {
      throw new Error("epochs harus >= 1");
    }

    const legacyCallback = typeof configOrCb === "function" ? configOrCb : undefined;
    const config = typeof configOrCb === "function" ? {} : configOrCb;
    const {
      batchSize = Math.max(1, Math.floor(X.length / 10)),
      validationSplit = 0,
      earlyStoppingPatience = Infinity,
      shuffle = true,
      verbose = false,
      onEpochEnd = () => { },
      monitorMetric = validationSplit > 0 ? "valLoss" : "loss",
      minDelta = 0,
      mode = "min",
      trimPadding = true,
      paddingSide = "right",
    } = config;

    if (!Number.isInteger(batchSize) || batchSize < 1) {
      throw new Error("batchSize harus >= 1");
    }
    if (validationSplit < 0 || validationSplit >= 1) {
      throw new Error("validationSplit harus antara 0 dan 1");
    }
    if (earlyStoppingPatience < 0) {
      throw new Error("earlyStoppingPatience harus >= 0");
    }

    this.assertTransformerFitSupported(X, y);

    const [trainX, valX] = splitTrainValidation(X, validationSplit);
    const [trainY, valY] = splitTrainValidation(y, validationSplit);
    if (trainX.length === 0) {
      throw new Error("Data train kosong setelah validationSplit");
    }

    const history: FitResult["history"] = {
      loss: [],
      ...(validationSplit > 0 ? { valLoss: [] } : {}),
    };

    let bestLoss = mode === "min" ? Infinity : -Infinity;
    let bestEpoch = 0;
    let noImprovementCount = 0;
    let stoppedEarly = false;
    let stoppingEpoch: number | undefined;
    let valLoss: number | undefined;

    const trainIndices = Array.from({ length: trainX.length }, (_, i) => i);
    this.train();

    for (let epoch = 0; epoch < epochs; epoch++) {
      const epochStartTime = Date.now();
      this.resetPositionOffset();

      if (verbose) {
        const progress = formatProgressBar(0, trainX.length);
        const valStr = validationSplit > 0 ? ` | Val Loss: ${valLoss !== undefined ? formatLoss(valLoss) : "...."}` : "";
        process.stdout.write(
          `\rEpoch ${epoch + 1}/${epochs} ${progress} | Loss: ....${valStr} | 0.0 samples/s | ETA: --:--`
        );
      }

      for (const layer of this.layers) {
        if (typeof (layer as any).resetLoss === "function") {
          (layer as any).resetLoss();
        }
      }

      if (shuffle) {
        shuffleInPlace(trainIndices);
      }

      let totalEpochLoss = 0;
      let totalEpochWeight = 0;

      for (let start = 0; start < trainX.length; start += batchSize) {
        const end = Math.min(start + batchSize, trainX.length);
        const currentBatchSize = end - start;
        const batch = this.buildTransformerBatch(trainX, trainY, trainIndices, start, currentBatchSize, trimPadding, paddingSide);
        const isFullSequenceTarget = batch.y._shape[0] === batch.x._shape[0];
        const pred = isFullSequenceTarget
          ? this.forwardFullSequence(batch.x)
          : this.forwardNextToken(batch.x);
        this.backward(batch.y);
        const batchLossState = this.useBackwardLossForTrainingBatch(batch.y, pred)
          ? this.computeLossAndWeightFromBackward(batch.y, pred)
          : this.computeLossAndWeight(batch.y, pred);

        totalEpochLoss += batchLossState.loss * batchLossState.weight;
        totalEpochWeight += batchLossState.weight;
        this.resetPositionOffset();

        if (verbose) {
          const elapsed = (Date.now() - epochStartTime) / 1000;
          const samplesProcessed = end;
          const speed = samplesProcessed / Math.max(elapsed, 0.001);
          const eta = (trainX.length - samplesProcessed) / speed;
          const progress = formatProgressBar(end, trainX.length);
          const valStr = validationSplit > 0 ? ` | Val Loss: ${valLoss !== undefined ? formatLoss(valLoss) : "...."}` : "";
          process.stdout.write(
            `\rEpoch ${epoch + 1}/${epochs} ${progress} | Loss: ${formatLoss(batchLossState.loss)}${valStr} | ${speed.toFixed(1)} samples/s | ETA: ${formatTime(eta)}`
          );
        }
      }

      const epochLoss = totalEpochLoss / totalEpochWeight;
      this.loss = epochLoss;
      history.loss.push(epochLoss);

      if (validationSplit > 0 && valX.length > 0) {
        valLoss = this.runTransformerValidation(valX, valY, batchSize, verbose, trimPadding, paddingSide);
        (history.valLoss as number[]).push(valLoss);
        this.train();
      }

      if (verbose) {
        const progress = formatProgressBar(trainX.length, trainX.length);
        const valStr = validationSplit > 0 ? ` | Val Loss: ${valLoss !== undefined ? formatLoss(valLoss) : "...."}` : "";
        const elapsed = (Date.now() - epochStartTime) / 1000;
        const speed = trainX.length / Math.max(elapsed, 0.001);
        process.stdout.write(
          `\rEpoch ${epoch + 1}/${epochs} ${progress} | Loss: ${formatLoss(epochLoss)}${valStr} | ${speed.toFixed(1)} samples/s | ETA: 00:00\n`
        );
      }

      const metricValue = monitorMetric === "valLoss" && valLoss !== undefined ? valLoss : epochLoss;
      const isImprovement = mode === "min"
        ? metricValue < bestLoss - minDelta
        : metricValue > bestLoss + minDelta;

      if (isImprovement) {
        bestLoss = metricValue;
        bestEpoch = epoch;
        noImprovementCount = 0;
      } else {
        noImprovementCount++;
      }

      legacyCallback?.(epochLoss);
      onEpochEnd(epoch, epochLoss, valLoss);

      if (noImprovementCount >= earlyStoppingPatience) {
        stoppedEarly = true;
        stoppingEpoch = epoch;
        if (verbose) console.log(`Early stopping di epoch ${epoch + 1}.`);
        break;
      }
    }

    this.eval();
    this.resetPositionOffset();
    return {
      history,
      bestEpoch,
      bestLoss,
      stoppedEarly,
      stoppingEpoch,
    };
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
    for (const block of this.blocks) {
      if (block.xRes1._shape[0] !== units || block.xRes1._shape[1] !== totalTokens) {
        block.xRes1 = mj.zeros([units, totalTokens]);
      }
      if (block.xRes2._shape[0] !== units || block.xRes2._shape[1] !== totalTokens) {
        block.xRes2 = mj.zeros([units, totalTokens]);
      }
      if (block.errRes1Buf._shape[0] !== units || block.errRes1Buf._shape[1] !== totalTokens) {
        block.errRes1Buf = mj.zeros([units, totalTokens]);
      }
      if (block.errRes2Buf._shape[0] !== units || block.errRes2Buf._shape[1] !== totalTokens) {
        block.errRes2Buf = mj.zeros([units, totalTokens]);
      }
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

  private runTransformerValidation(
    valX: Matrix[],
    valY: Matrix[],
    batchSize: number,
    verbose: boolean,
    trimPadding: boolean,
    paddingSide: "left" | "right"
  ): number {
    this.eval();
    this.resetPositionOffset();
    let totalValLoss = 0;
    let totalValWeight = 0;
    const valStartTime = Date.now();
    const valIndices = Array.from({ length: valX.length }, (_, i) => i);

    for (let start = 0; start < valX.length; start += batchSize) {
      const end = Math.min(start + batchSize, valX.length);
      const currentBatchSize = end - start;
      const batch = this.buildTransformerBatch(valX, valY, valIndices, start, currentBatchSize, trimPadding, paddingSide);
      const isFullSequenceTarget = batch.y._shape[0] === batch.x._shape[0];
      const pred = isFullSequenceTarget
        ? this.forwardFullSequence(batch.x)
        : this.forwardNextToken(batch.x);
      const lossState = this.computeLossAndWeight(batch.y, pred);
      totalValLoss += lossState.loss * lossState.weight;
      totalValWeight += lossState.weight;
      this.resetPositionOffset();

      if (verbose) {
        const elapsed = (Date.now() - valStartTime) / 1000;
        const samplesProcessed = end;
        const speed = samplesProcessed / Math.max(elapsed, 0.001);
        const eta = (valX.length - samplesProcessed) / speed;
        process.stdout.write(`\rValidating  ${formatProgressBar(samplesProcessed, valX.length)} | ${speed.toFixed(1)} samples/s | ETA: ${formatTime(eta)}`);
      }
    }

    if (verbose) process.stdout.write("\n");
    return totalValLoss / totalValWeight;
  }

  private buildTransformerBatch(
    X: Matrix[],
    y: Matrix[],
    indices: number[],
    start: number,
    currentBatchSize: number,
    trimPadding: boolean,
    paddingSide: "left" | "right"
  ): { x: Matrix; y: Matrix } {
    if (currentBatchSize === 1) {
      let sampleX = X[indices[start]];
      let sampleY = y[indices[start]];
      if (trimPadding && sampleY._shape[0] === sampleX._shape[0]) {
        const trimResult = trimPaddingBatch(sampleX, sampleY, this.embedding.padTokenId as number, paddingSide);
        sampleX = trimResult.x;
        sampleY = trimResult.y;
        this.setPositionOffset(trimResult.positionOffset);
      }
      return { x: sampleX, y: sampleY };
    }

    const isFullSequenceTarget = y[indices[start]]._shape[0] === X[indices[start]]._shape[0];
    const supportsTrimPadding = trimPadding && isFullSequenceTarget;
    const trimWindow = supportsTrimPadding
      ? this.computeTrimWindow(X, y, indices, start, currentBatchSize, paddingSide)
      : { startRow: 0, rowCount: X[indices[start]]._shape[0], positionOffset: 0 };
    const batchX = this.createFitBatchMatrix("x", trimWindow.rowCount, currentBatchSize);
    const targetRows = isFullSequenceTarget ? trimWindow.rowCount : 1;
    const batchY = this.createFitBatchMatrix("y", targetRows, currentBatchSize);

    for (let j = 0; j < currentBatchSize; j++) {
      const sampleX = X[indices[start + j]];
      const sampleY = y[indices[start + j]];
      batchX.setCol(j, sampleX._data.subarray(trimWindow.startRow, trimWindow.startRow + trimWindow.rowCount));
      if (isFullSequenceTarget) {
        batchY.setCol(j, sampleY._data.subarray(trimWindow.startRow, trimWindow.startRow + trimWindow.rowCount));
      } else {
        batchY.setCol(j, sampleY._data);
      }
    }

    this.setPositionOffset(trimWindow.positionOffset);
    return { x: batchX, y: batchY };
  }

  private createFitBatchMatrix(kind: "x" | "y", rows: number, cols: number): Matrix {
    const requiredLength = rows * cols;
    let nextBuffer = kind === "x" ? this.fitBatchInputBufferData : this.fitBatchTargetBufferData;
    if (nextBuffer.length < requiredLength) {
      nextBuffer = new Float32Array(Math.max(requiredLength, Math.max(1, nextBuffer.length * 2)));
      if (kind === "x") {
        this.fitBatchInputBufferData = nextBuffer;
      } else {
        this.fitBatchTargetBufferData = nextBuffer;
      }
    }
    return Matrix.fromFlat(nextBuffer.subarray(0, requiredLength), [rows, cols]);
  }

  private computeTrimWindow(
    X: Matrix[],
    y: Matrix[],
    indices: number[],
    start: number,
    currentBatchSize: number,
    paddingSide: "left" | "right"
  ): { startRow: number; rowCount: number; positionOffset: number } {
    const seqLen = X[indices[start]]._shape[0];
    const padId = this.embedding.padTokenId;
    if (paddingSide === "right") {
      let lastUsefulPos = -1;
      for (let j = 0; j < currentBatchSize; j++) {
        const xData = X[indices[start + j]]._data;
        const yData = y[indices[start + j]]._data;
        for (let pos = 0; pos < seqLen; pos++) {
          if (xData[pos] !== padId || yData[pos] !== padId) {
            lastUsefulPos = Math.max(lastUsefulPos, pos);
          }
        }
      }
      if (lastUsefulPos < 0 || lastUsefulPos + 1 >= seqLen) {
        return { startRow: 0, rowCount: seqLen, positionOffset: 0 };
      }
      return { startRow: 0, rowCount: lastUsefulPos + 1, positionOffset: 0 };
    }

    let firstUsefulPos = seqLen;
    for (let j = 0; j < currentBatchSize; j++) {
      const xData = X[indices[start + j]]._data;
      const yData = y[indices[start + j]]._data;
      for (let pos = 0; pos < seqLen; pos++) {
        if (xData[pos] !== padId || yData[pos] !== padId) {
          firstUsefulPos = Math.min(firstUsefulPos, pos);
          break;
        }
      }
    }

    if (firstUsefulPos <= 0 || firstUsefulPos >= seqLen) {
      return { startRow: 0, rowCount: seqLen, positionOffset: 0 };
    }
    return {
      startRow: firstUsefulPos,
      rowCount: seqLen - firstUsefulPos,
      positionOffset: firstUsefulPos,
    };
  }

  private assertTransformerFitSupported(X: Matrix[], y: Matrix[]): void {
    for (let i = 0; i < X.length; i++) {
      if (X[i]._shape[1] !== 1) {
        throw new Error(`Transformers.fit: expected token input shape [seqLen, 1] per sample, got [${X[i]._shape[0]}, ${X[i]._shape[1]}]`);
      }
      if (y[i]._shape[1] !== 1) {
        throw new Error(`Transformers.fit: expected target shape [seqLen, 1] or [1, 1] per sample, got [${y[i]._shape[0]}, ${y[i]._shape[1]}]`);
      }
      const isFullSequenceTarget = y[i]._shape[0] === X[i]._shape[0];
      const isLegacyNextTokenTarget = y[i]._shape[0] === 1;
      if (!isFullSequenceTarget && !isLegacyNextTokenTarget) {
        throw new Error(
          `Transformers.fit: expected target shape [${X[i]._shape[0]}, 1] for full-sequence causal LM or [1, 1] for next-token training, got [${y[i]._shape[0]}, ${y[i]._shape[1]}]`
        );
      }
    }
  }
}
