import { FitConfig, FitResult } from "@oxide-js/core";
import { Cost, Matrix as MatrixType, Optimizer } from "@oxide-js/core";
import { CompileDenseLayers, Dense, Embedding, GRU, LSTM, RNN } from "@oxide-js/layers";
import { mj } from "@oxide-js/core";
import { Matrix } from "@oxide-js/core";
import { formatLoss, formatProgressBar, formatTime, shuffleInPlace, splitTrainValidation } from "@oxide-js/core";
import Sequential, { SequentialLayers } from "./sequential.js";

export type RecurrentKind = "rnn" | "lstm" | "gru";
export type RecurrentTrainingMode = "many-to-one" | "many-to-many";
export type RecurrentPooling = "last" | "mean" | "max";

export interface RecurrentModelConfig {
  kind: RecurrentKind;
  inputSize?: number;
  vocabSize?: number;
  embeddingDim?: number;
  embeddingTrainable?: boolean;
  hiddenSize?: number;
  hiddenSizes?: number[];
  numLayers?: number;
  outputSize: number;
  seqLen: number;
  mode?: RecurrentTrainingMode;
  pooling?: RecurrentPooling;
  padTokenId?: number | null;
  stateful?: boolean;
  alpha?: number;
  loss?: Cost;
  optimizer?: Optimizer;
  clipGradient?: number | boolean;
}

type RecurrentLayer = RNN | LSTM | GRU;

export default class RecurrentModel extends Sequential {
  readonly kind: RecurrentKind;
  readonly mode: RecurrentTrainingMode;
  readonly pooling: RecurrentPooling;
  readonly padTokenId: number | null;
  readonly seqLen: number;
  readonly outputSize: number;
  readonly stateful: boolean;
  readonly hiddenSizes: number[];
  private readonly embeddingLayer: Embedding | null;
  private readonly recurrentLayers: RecurrentLayer[];
  private readonly outputLayer: Dense;
  private poolingCache:
    | {
      batchSize: number;
      mask: Uint8Array;
      counts: Int32Array;
      argmax: Int32Array | null;
    }
    | null = null;
  private batchSequenceInputBufferData: Float32Array = new Float32Array(0);
  private batchSequenceTargetBufferData: Float32Array = new Float32Array(0);

  constructor({
    kind,
    inputSize,
    vocabSize,
    embeddingDim,
    embeddingTrainable = true,
    hiddenSize,
    hiddenSizes,
    numLayers = 1,
    outputSize,
    seqLen,
    mode = "many-to-one",
    pooling = "last",
    padTokenId = null,
    stateful = false,
    alpha = 0.01,
    loss = "mse",
    optimizer = "adam",
    clipGradient = 5.0,
  }: RecurrentModelConfig) {
    if (!Number.isInteger(seqLen) || seqLen < 1) {
      throw new Error(`RecurrentModel: seqLen harus integer >= 1, got ${seqLen}`);
    }
    if (!Number.isInteger(outputSize) || outputSize < 1) {
      throw new Error(`RecurrentModel: outputSize harus integer >= 1, got ${outputSize}`);
    }

    const normalizedHiddenSizes = RecurrentModel.resolveHiddenSizes(hiddenSize, hiddenSizes, numLayers);
    if (pooling !== "last" && pooling !== "mean" && pooling !== "max") {
      throw new Error(`RecurrentModel: pooling harus "last" | "mean" | "max", got ${pooling}`);
    }
    const useEmbedding = vocabSize !== undefined || embeddingDim !== undefined;
    if (useEmbedding) {
      if (!Number.isInteger(vocabSize) || (vocabSize as number) < 1) {
        throw new Error(`RecurrentModel: vocabSize harus integer >= 1, got ${vocabSize}`);
      }
      if (!Number.isInteger(embeddingDim) || (embeddingDim as number) < 1) {
        throw new Error(`RecurrentModel: embeddingDim harus integer >= 1, got ${embeddingDim}`);
      }
    } else if (!Number.isInteger(inputSize) || (inputSize as number) < 1) {
      throw new Error(`RecurrentModel: inputSize harus integer >= 1 saat Embedding tidak dipakai, got ${inputSize}`);
    }

    const layers: SequentialLayers = [];
    let currentUnits = useEmbedding ? (embeddingDim as number) : (inputSize as number);
    let embeddingLayer: Embedding | null = null;

    if (useEmbedding) {
      embeddingLayer = new Embedding({
        vocabSize: vocabSize as number,
        embeddingDim: embeddingDim as number,
        alpha,
        optimizer,
        padTokenId,
        trainable: embeddingTrainable,
      });
      layers.push(embeddingLayer);
    }

    const recurrentLayers: RecurrentLayer[] = [];
    for (let i = 0; i < normalizedHiddenSizes.length; i++) {
      const nextHiddenSize = normalizedHiddenSizes[i];
      const returnSequences = mode === "many-to-many"
        || i < normalizedHiddenSizes.length - 1
        || (mode === "many-to-one" && pooling !== "last");
      const status = i === 0 ? "input" : "train";
      const layer = RecurrentModel.createRecurrentLayer(kind, {
        units: currentUnits,
        hiddenUnits: nextHiddenSize,
        returnSequences,
        stateful,
        alpha,
        optimizer,
        status,
        clipGradient,
        loss,
      });
      recurrentLayers.push(layer);
      layers.push(layer);
      currentUnits = kind === "gru" && (layer as GRU).bidirectional ? nextHiddenSize * 2 : nextHiddenSize;
    }

    const outputLayer = new Dense({
      units: currentUnits,
      outputUnits: outputSize,
      activation: "linear",
      status: "output",
      alpha,
      optimizer,
      loss,
      clipGradient,
    });
    layers.push(outputLayer);

    super({ layers });

    this.kind = kind;
    this.mode = mode;
    this.pooling = pooling;
    this.padTokenId = padTokenId;
    this.seqLen = seqLen;
    this.outputSize = outputSize;
    this.stateful = stateful;
    this.hiddenSizes = normalizedHiddenSizes;
    this.embeddingLayer = embeddingLayer;
    this.recurrentLayers = recurrentLayers;
    this.outputLayer = outputLayer;

    const compileConfig: CompileDenseLayers = {
      alpha,
      optimizer,
      error: loss,
      clipGradient,
    };
    this.compile(compileConfig);
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

    this.assertFitSupported(X, y, batchSize, shuffle, validationSplit);

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
      this.resetStates();

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
        const currentBatchX = this.buildInputBatch(trainX, trainIndices, start, currentBatchSize);
        const currentBatchY = this.buildTargetBatch(trainY, trainIndices, start, currentBatchSize);

        const pred = this.forward(currentBatchX, currentBatchSize);
        this.backward(currentBatchY, currentBatchSize);
        const batchLossState = this.useBackwardLossForTrainingBatch(currentBatchY, pred)
          ? this.computeLossAndWeightFromBackward(currentBatchY, pred)
          : this.computeLossAndWeight(currentBatchY, pred);

        totalEpochLoss += batchLossState.loss * batchLossState.weight;
        totalEpochWeight += batchLossState.weight;

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
        valLoss = this.runValidation(valX, valY, verbose, batchSize);
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
    return {
      history,
      bestEpoch,
      bestLoss,
      stoppedEarly,
      stoppingEpoch,
    };
  }

  resetState(): void {
    this.resetStates();
  }

  resetStates(): void {
    for (const layer of this.recurrentLayers) {
      if (typeof (layer as any).resetState === "function") {
        (layer as any).resetState();
      }
    }
  }

  getDenseOutputLayer(): Dense {
    return this.outputLayer;
  }

  forward(x: Matrix, batchSize: number = 1): Matrix {
    let input = x;
    if (this.embeddingLayer) {
      input = this.embeddingLayer.forward(input);
    }

    for (const layer of this.recurrentLayers) {
      input = batchSize > 1 && typeof (layer as any).forwardBatch === "function"
        ? (layer as any).forwardBatch(input, batchSize)
        : layer.forward(input);
    }

    if (this.mode === "many-to-one" && this.pooling !== "last") {
      input = this.poolSequenceOutput(input, x, batchSize);
    } else {
      this.poolingCache = null;
    }

    return this.outputLayer.forward(input);
  }

  backward(y: Matrix, batchSize: number = 1, gradOnly = false) {
    let err = this.outputLayer.backward(y, mj.matrix([[]]), gradOnly);
    if (this.outputLayer.status === "output") {
      this.loss = this.outputLayer.loss;
    }

    if (this.mode === "many-to-one" && this.pooling !== "last") {
      err = this.expandPooledErrorToSequence(err);
    }

    for (let i = this.recurrentLayers.length - 1; i >= 0; i--) {
      const layer = this.recurrentLayers[i];
      err = batchSize > 1 && typeof (layer as any).backwardBatch === "function"
        ? (layer as any).backwardBatch(y, err, batchSize, gradOnly)
        : (layer as any).backward(y, err, gradOnly);
    }

    if (this.embeddingLayer) {
      this.embeddingLayer.backward(y, err, gradOnly);
    }
  }

  protected runValidation(
    valX: Matrix[],
    valY: Matrix[],
    verbose: boolean,
    batchSize: number = 1
  ): number {
    this.eval();
    this.resetStates();
    let totalValLoss = 0;
    let totalValWeight = 0;
    const valStartTime = Date.now();
    const valIndices = Array.from({ length: valX.length }, (_, i) => i);

    for (let start = 0; start < valX.length; start += batchSize) {
      const end = Math.min(start + batchSize, valX.length);
      const currentBatchSize = end - start;
      const batchX = this.buildInputBatch(valX, valIndices, start, currentBatchSize);
      const batchY = this.buildTargetBatch(valY, valIndices, start, currentBatchSize);
      const pred = this.forward(batchX, currentBatchSize);
      const lossState = this.computeLossAndWeight(batchY, pred);
      totalValLoss += lossState.loss * lossState.weight;
      totalValWeight += lossState.weight;

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

  private poolSequenceOutput(sequence: Matrix, inputTokensOrMask: Matrix, batchSize: number): Matrix {
    const [hiddenUnits, totalCols] = sequence._shape;
    const expectedCols = this.seqLen * batchSize;
    if (totalCols !== expectedCols) {
      throw new Error(
        `RecurrentModel pooling: expected sequence output cols ${expectedCols}, got ${totalCols}`
      );
    }

    const pooled = this.createReusableBatchMatrix("x", hiddenUnits, batchSize);
    pooled._data.fill(0);
    const mask = new Uint8Array(expectedCols);
    const counts = new Int32Array(batchSize);
    const argmax = this.pooling === "max" ? new Int32Array(hiddenUnits * batchSize).fill(-1) : null;

    for (let sample = 0; sample < batchSize; sample++) {
      for (let t = 0; t < this.seqLen; t++) {
        const col = t * batchSize + sample;
        const valid = this.isValidPoolingStep(inputTokensOrMask, t, sample, batchSize);
        mask[col] = valid ? 1 : 0;
        if (valid) counts[sample]++;
      }
      if (counts[sample] === 0) {
        throw new Error("RecurrentModel pooling: sample has no valid non-pad tokens.");
      }
    }

    for (let row = 0; row < hiddenUnits; row++) {
      for (let sample = 0; sample < batchSize; sample++) {
        let acc = 0;
        let bestValue = -Infinity;
        let bestCol = -1;
        for (let t = 0; t < this.seqLen; t++) {
          const col = t * batchSize + sample;
          if (mask[col] === 0) continue;
          const value = sequence._data[row * totalCols + col];
          if (this.pooling === "mean") {
            acc += value;
          } else {
            if (bestCol === -1 || value > bestValue) {
              bestValue = value;
              bestCol = col;
            }
          }
        }
        if (this.pooling === "mean") {
          pooled._data[row * batchSize + sample] = acc / counts[sample];
        } else {
          pooled._data[row * batchSize + sample] = bestValue;
          (argmax as Int32Array)[row * batchSize + sample] = bestCol;
        }
      }
    }

    this.poolingCache = { batchSize, mask, counts, argmax };
    return pooled;
  }

  private expandPooledErrorToSequence(err: Matrix): Matrix {
    if (!this.poolingCache) {
      throw new Error("RecurrentModel pooling backward: missing forward pooling cache.");
    }

    const hiddenUnits = err._shape[0];
    const batchSize = this.poolingCache.batchSize;
    if (err._shape[1] !== batchSize) {
      throw new Error(
        `RecurrentModel pooling backward: expected pooled error cols ${batchSize}, got ${err._shape[1]}`
      );
    }

    const totalCols = this.seqLen * batchSize;
    const expanded = this.createSequenceBatchMatrix(hiddenUnits, totalCols);
    expanded._data.fill(0);

    if (this.pooling === "mean") {
      for (let row = 0; row < hiddenUnits; row++) {
        for (let sample = 0; sample < batchSize; sample++) {
          const grad = err._data[row * batchSize + sample] / this.poolingCache.counts[sample];
          for (let t = 0; t < this.seqLen; t++) {
            const col = t * batchSize + sample;
            if (this.poolingCache.mask[col] === 1) {
              expanded._data[row * totalCols + col] = grad;
            }
          }
        }
      }
      return expanded;
    }

    const argmax = this.poolingCache.argmax;
    if (!argmax) {
      throw new Error("RecurrentModel pooling backward: missing max-pooling argmax cache.");
    }
    for (let row = 0; row < hiddenUnits; row++) {
      for (let sample = 0; sample < batchSize; sample++) {
        const col = argmax[row * batchSize + sample];
        expanded._data[row * totalCols + col] = err._data[row * batchSize + sample];
      }
    }
    return expanded;
  }

  private isValidPoolingStep(input: Matrix, timeIndex: number, batchIndex: number, batchSize: number): boolean {
    if (!this.embeddingLayer || this.padTokenId === null) return true;
    if (input._shape[0] !== this.seqLen) return true;
    const cols = input._shape[1];
    if (cols !== 1 && cols !== batchSize) return true;
    const colIndex = cols === 1 ? 0 : batchIndex;
    return input._data[timeIndex * cols + colIndex] !== this.padTokenId;
  }

  private static resolveHiddenSizes(
    hiddenSize: number | undefined,
    hiddenSizes: number[] | undefined,
    numLayers: number
  ): number[] {
    if (hiddenSizes !== undefined) {
      if (!Array.isArray(hiddenSizes) || hiddenSizes.length === 0) {
        throw new Error("RecurrentModel: hiddenSizes harus array non-empty.");
      }
      for (const size of hiddenSizes) {
        if (!Number.isInteger(size) || size < 1) {
          throw new Error(`RecurrentModel: setiap hidden size harus integer >= 1, got ${size}`);
        }
      }
      return hiddenSizes.slice();
    }

    if (!Number.isInteger(numLayers) || numLayers < 1) {
      throw new Error(`RecurrentModel: numLayers harus integer >= 1, got ${numLayers}`);
    }
    if (!Number.isInteger(hiddenSize) || (hiddenSize as number) < 1) {
      throw new Error(`RecurrentModel: hiddenSize harus integer >= 1 saat hiddenSizes tidak diberikan, got ${hiddenSize}`);
    }
    return Array.from({ length: numLayers }, () => hiddenSize as number);
  }

  private static createRecurrentLayer(kind: RecurrentKind, config: {
    units: number;
    hiddenUnits: number;
    returnSequences: boolean;
    stateful: boolean;
    alpha: number;
    optimizer: Optimizer;
    status: "input" | "train";
    clipGradient: number | boolean;
    loss: Cost;
  }): RecurrentLayer {
    if (kind === "rnn") {
      return new RNN(config);
    }
    if (kind === "lstm") {
      return new LSTM(config);
    }
    return new GRU(config);
  }

  private buildInputBatch(X: Matrix[], indices: number[], start: number, currentBatchSize: number): Matrix {
    if (this.embeddingLayer) {
      const batch = this.createReusableBatchMatrix("x", this.seqLen, currentBatchSize);
      for (let j = 0; j < currentBatchSize; j++) {
        const sample = X[indices[start + j]];
        this.assertEmbeddedInputSampleShape(sample);
        batch.setCol(j, sample._data);
      }
      return batch;
    }

    const rows = this.recurrentLayers[0].units;
    const totalCols = this.seqLen * currentBatchSize;
    const batch = this.createSequenceBatchMatrix(rows, totalCols);
    const batchData = batch._data;

    for (let j = 0; j < currentBatchSize; j++) {
      const sample = X[indices[start + j]];
      this.assertRawInputSampleShape(sample, rows);
      for (let t = 0; t < this.seqLen; t++) {
        const targetCol = t * currentBatchSize + j;
        for (let row = 0; row < rows; row++) {
          batchData[row * totalCols + targetCol] = sample._data[row * this.seqLen + t];
        }
      }
    }

    return batch;
  }

  private buildTargetBatch(y: Matrix[], indices: number[], start: number, currentBatchSize: number): Matrix {
    if (this.mode === "many-to-many") {
      return this.buildManyToManyTargetBatch(y, indices, start, currentBatchSize);
    }

    const targetRows = y[indices[start]]._shape[0];
    const batch = this.createReusableBatchMatrix("y", targetRows, currentBatchSize);
    for (let j = 0; j < currentBatchSize; j++) {
      const sample = y[indices[start + j]];
      if (!this.isSupportedManyToOneTarget(sample)) {
        throw new Error(
          `RecurrentModel.fit: expected many-to-one target shape [1, 1] (sparse) or [${this.outputSize}, 1] (dense), got [${sample._shape[0]}, ${sample._shape[1]}]`
        );
      }
      if (sample._shape[0] !== targetRows) {
        throw new Error(
          `RecurrentModel.fit: semua target dalam satu batch harus memiliki jumlah row yang sama. Expected ${targetRows}, got ${sample._shape[0]}`
        );
      }
      batch.setCol(j, sample._data);
    }
    return batch;
  }

  private buildManyToManyTargetBatch(y: Matrix[], indices: number[], start: number, currentBatchSize: number): Matrix {
    const firstTarget = y[indices[start]];
    const targetRows = firstTarget._shape[0];
    const totalCols = this.seqLen * currentBatchSize;
    const batch = this.createSequenceBatchMatrix(targetRows, totalCols, "y");
    const batchData = batch._data;

    for (let j = 0; j < currentBatchSize; j++) {
      const sample = y[indices[start + j]];
      if (!this.isSupportedManyToManyTarget(sample)) {
        throw new Error(
          `RecurrentModel.fit: expected many-to-many target shape [1, ${this.seqLen}] (sparse) or [${this.outputSize}, ${this.seqLen}] (dense), got [${sample._shape[0]}, ${sample._shape[1]}]`
        );
      }
      if (sample._shape[0] !== targetRows) {
        throw new Error(
          `RecurrentModel.fit: semua target many-to-many dalam satu batch harus memiliki jumlah row yang sama. Expected ${targetRows}, got ${sample._shape[0]}`
        );
      }
      for (let t = 0; t < this.seqLen; t++) {
        const targetCol = t * currentBatchSize + j;
        for (let row = 0; row < targetRows; row++) {
          batchData[row * totalCols + targetCol] = sample._data[row * this.seqLen + t];
        }
      }
    }

    return batch;
  }

  private createSequenceBatchMatrix(rows: number, cols: number, kind: "x" | "y" = "x"): MatrixType {
    const requiredLength = rows * cols;
    let nextBuffer = kind === "x" ? this.batchSequenceInputBufferData : this.batchSequenceTargetBufferData;
    if (nextBuffer.length < requiredLength) {
      nextBuffer = new Float32Array(Math.max(requiredLength, Math.max(1, nextBuffer.length * 2)));
      if (kind === "x") {
        this.batchSequenceInputBufferData = nextBuffer;
      } else {
        this.batchSequenceTargetBufferData = nextBuffer;
      }
    }
    return Matrix.fromFlat(nextBuffer.subarray(0, requiredLength), [rows, cols]);
  }

  private assertFitSupported(
    X: Matrix[],
    y: Matrix[],
    batchSize: number,
    shuffle: boolean,
    validationSplit: number
  ): void {
    if (this.stateful && batchSize !== 1) {
      throw new Error("RecurrentModel.fit: stateful=true hanya mendukung batchSize=1.");
    }
    if (this.stateful && shuffle) {
      throw new Error("RecurrentModel.fit: stateful=true tidak boleh dipakai bersama shuffle=true.");
    }
    if (this.stateful && validationSplit > 0) {
      throw new Error("RecurrentModel.fit: stateful=true tidak mendukung validationSplit > 0 pada loop training saat ini.");
    }
    if (batchSize > 1) {
      for (const layer of this.recurrentLayers) {
        if (typeof (layer as any).forwardBatch !== "function" || typeof (layer as any).backwardBatch !== "function") {
          throw new Error(`RecurrentModel.fit: ${layer.name} tidak mendukung batchSize > 1.`);
        }
      }
    }
    for (let i = 0; i < X.length; i++) {
      if (this.embeddingLayer) {
        this.assertEmbeddedInputSampleShape(X[i]);
      } else {
        this.assertRawInputSampleShape(X[i], this.recurrentLayers[0].units);
      }
      const targetSupported = this.mode === "many-to-one"
        ? this.isSupportedManyToOneTarget(y[i])
        : this.isSupportedManyToManyTarget(y[i]);
      if (!targetSupported) {
        throw new Error(
          this.mode === "many-to-one"
            ? `RecurrentModel.fit: expected many-to-one target shape [1, 1] (sparse) or [${this.outputSize}, 1] (dense), got [${y[i]._shape[0]}, ${y[i]._shape[1]}]`
            : `RecurrentModel.fit: expected many-to-many target shape [1, ${this.seqLen}] (sparse) or [${this.outputSize}, ${this.seqLen}] (dense), got [${y[i]._shape[0]}, ${y[i]._shape[1]}]`
        );
      }
    }
  }

  private assertEmbeddedInputSampleShape(sample: Matrix): void {
    if (sample._shape[0] !== this.seqLen || sample._shape[1] !== 1) {
      throw new Error(
        `RecurrentModel.fit: expected token input shape [${this.seqLen}, 1] saat Embedding dipakai, got [${sample._shape[0]}, ${sample._shape[1]}]`
      );
    }
  }

  private assertRawInputSampleShape(sample: Matrix, inputRows: number): void {
    if (sample._shape[0] !== inputRows || sample._shape[1] !== this.seqLen) {
      throw new Error(
        `RecurrentModel.fit: expected raw sequence input shape [${inputRows}, ${this.seqLen}], got [${sample._shape[0]}, ${sample._shape[1]}]`
      );
    }
  }

  private isSupportedManyToOneTarget(sample: Matrix): boolean {
    return sample._shape[1] === 1 && (sample._shape[0] === 1 || sample._shape[0] === this.outputSize);
  }

  private isSupportedManyToManyTarget(sample: Matrix): boolean {
    return sample._shape[1] === this.seqLen && (sample._shape[0] === 1 || sample._shape[0] === this.outputSize);
  }

  protected computeLossWeight(yTrue: Matrix, yPred: Matrix): number {
    if (this.mode === "many-to-many") {
      return yPred._shape[1];
    }
    return super.computeLossWeight(yTrue, yPred);
  }
}
