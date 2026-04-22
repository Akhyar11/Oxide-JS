import { readFileSync, writeFileSync } from "fs";
import { Cost, Layers, Matrix } from "../@types/type";
import { setLayers, setLoss, splitTrainValidation, shuffleInPlace, formatLoss, formatProgressBar, formatTime } from "../utils";
import { CompileDenseLayers, Dense, Convolution } from "../layers";
import mj from "../math";
import { FitConfig, FitResult } from "../@types/fitConfig";

export type SequentialLayers = Layers[];

export default class Sequential {
  layers: SequentialLayers;
  loss = 0;
  protected isTrainingMode = true;
  constructor({ layers = [] }: { layers?: SequentialLayers } = {}) {
    this.layers = layers;
  }

  summary() {
    console.log("========== Model Info ==========");
    let totalParams = 0;
    for (let layer of this.layers) {
      console.log(`Layer name   : ${layer.name}`);
      console.log(`Layer input  : [${layer.inputShape}]`);
      console.log(`Layer output : [${layer.outputShape}]`);
      console.log(`Layer param  : ${layer.params}`);
      console.log("");
      totalParams += layer.params;
    }

    console.log("Total params =", totalParams);
    console.log("========== End Info ==========");
  }

  add(layer: Layers) {
    this.layers.push(layer);
  }

  save(path: string) {
    const data = [];
    for (let layer of this.layers) {
      data.push(layer.save());
    }
    const dataJson = JSON.stringify(data);
    writeFileSync(path, dataJson);
  }

  load(path: string) {
    const dataJson = readFileSync(path, "utf-8");
    const data = JSON.parse(dataJson);
    this.layers = [];
    this.layers = setLayers(data);
  }

  compile(config: CompileDenseLayers) {
    for (let layer of this.layers) {
      if (typeof (layer as any).compile === "function") {
        (layer as any).compile(config);
      }
    }
  }

  forward(x: Matrix): Matrix {
    let input = x;
    for (let layer of this.layers) {
      input = layer.forward(input);
    }
    return input;
  }

  backward(y: Matrix) {
    let err = mj.matrix([[]]);
    for (let i = this.layers.length - 1; i >= 0; i--) {
      err = this.layers[i].backward(y, err);
      if (this.layers[i].status === "output") this.loss = (this.layers[i] as any).loss;
    }
  }

  train(): this {
    this.isTrainingMode = true;
    for (const layer of this.layers) {
      if (typeof (layer as any).setTrainingMode === "function") {
        (layer as any).setTrainingMode(true);
      }
    }
    return this;
  }

  eval(): this {
    this.isTrainingMode = false;
    for (const layer of this.layers) {
      if (typeof (layer as any).setTrainingMode === "function") {
        (layer as any).setTrainingMode(false);
      }
    }
    return this;
  }

  predict(x: Matrix): Matrix {
    const wasTraining = this.isTrainingMode;
    this.eval();
    const out = this.forward(x);
    if (wasTraining) this.train();
    return out;
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

    this.assertRecurrentFitSupported(X, batchSize, shuffle, validationSplit);

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

      for (let start = 0; start < trainX.length; start += batchSize) {
        const end = Math.min(start + batchSize, trainX.length);
        const currentBatchSize = end - start;

        let currentBatchX: Matrix;
        let currentBatchY: Matrix;

        if (currentBatchSize === 1) {
          const idx = trainIndices[start];
          currentBatchX = trainX[idx];
          currentBatchY = trainY[idx];
        } else {
          // Buat batch matrix [rows, currentBatchSize]
          const [rowsX] = trainX[0]._shape;
          const [rowsY] = trainY[0]._shape;
          currentBatchX = mj.zeros([rowsX, currentBatchSize]);
          currentBatchY = mj.zeros([rowsY, currentBatchSize]);

          for (let j = 0; j < currentBatchSize; j++) {
            const idx = trainIndices[start + j];
            currentBatchX.setCol(j, trainX[idx]._data);
            currentBatchY.setCol(j, trainY[idx]._data);
          }
        }

        const pred = this.forward(currentBatchX);
        const batchLossValue = this.computeSampleLoss(currentBatchY, pred);
        this.backward(currentBatchY);

        totalEpochLoss += batchLossValue * currentBatchSize;

        if (verbose) {
          const elapsed = (Date.now() - epochStartTime) / 1000;
          const samplesProcessed = end;
          const speed = samplesProcessed / Math.max(elapsed, 0.001);
          const eta = (trainX.length - samplesProcessed) / speed;

          const progress = formatProgressBar(end, trainX.length);
          const valStr = validationSplit > 0 ? ` | Val Loss: ${valLoss !== undefined ? formatLoss(valLoss) : "...."}` : "";
          const speedStr = ` | ${speed.toFixed(1)} samples/s`;
          const etaStr = ` | ETA: ${formatTime(eta)}`;

          process.stdout.write(
            `\rEpoch ${epoch + 1}/${epochs} ${progress} | Loss: ${formatLoss(batchLossValue)}${valStr}${speedStr}${etaStr}`
          );
        }
      }

      const epochLoss = totalEpochLoss / trainX.length;
      this.loss = epochLoss;
      history.loss.push(epochLoss);

      if (verbose) {
        const progress = formatProgressBar(trainX.length, trainX.length);
        const valStr = validationSplit > 0 ? ` | Val Loss: ${valLoss !== undefined ? formatLoss(valLoss) : "...."}` : "";
        const elapsed = (Date.now() - epochStartTime) / 1000;
        const speed = trainX.length / Math.max(elapsed, 0.001);
        const speedStr = ` | ${speed.toFixed(1)} samples/s`;
        process.stdout.write(
          `\rEpoch ${epoch + 1}/${epochs} ${progress} | Loss: ${formatLoss(epochLoss)}${valStr}${speedStr} | ETA: 00:00\n`
        );
      }

      if (validationSplit > 0 && valX.length > 0) {
        valLoss = this.runValidation(valX, valY, verbose);
        (history.valLoss as number[]).push(valLoss);
        this.train();
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
        if (verbose) {
          console.log(`Early stopping di epoch ${epoch + 1}.`);
        }
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

  protected runValidation(
    valX: Matrix[],
    valY: Matrix[],
    verbose: boolean
  ): number {
    this.eval();
    let totalValLoss = 0;
    const valStartTime = Date.now();

    for (let i = 0; i < valX.length; i++) {
      const pred = this.forward(valX[i]);
      totalValLoss += this.computeSampleLoss(valY[i], pred);

      if (verbose) {
        const elapsed = (Date.now() - valStartTime) / 1000;
        const samplesProcessed = i + 1;
        const speed = samplesProcessed / Math.max(elapsed, 0.001);
        const eta = (valX.length - samplesProcessed) / speed;

        const progress = formatProgressBar(samplesProcessed, valX.length);
        const speedStr = ` | ${speed.toFixed(1)} samples/s`;
        const etaStr = ` | ETA: ${formatTime(eta)}`;

        process.stdout.write(
          `\rValidating  ${progress}${speedStr}${etaStr}`
        );
      }
    }

    if (verbose) process.stdout.write("\n");
    return totalValLoss / valX.length;
  }

  protected resolveLossName(): Cost {
    let outputLayer: any = this.layers[this.layers.length - 1] as any;
    for (let i = this.layers.length - 1; i >= 0; i--) {
      if ((this.layers[i] as any).status === "output") {
        outputLayer = this.layers[i] as any;
        break;
      }
    }
    if (!outputLayer || typeof outputLayer.save !== "function") return "mse";
    const saved = outputLayer.save?.();
    const lossName = saved?.loss;
    if (
      lossName === "mse" ||
      lossName === "crossEntropy" ||
      lossName === "binaryCrossEntropy" ||
      lossName === "softmaxCrossEntropy"
    ) {
      return lossName;
    }
    return "mse";
  }

  protected computeSampleLoss(yTrue: Matrix, yPred: Matrix): number {
    const isSparseTarget = yTrue._shape[0] === 1 && yPred._shape[0] > 1;
    const lossName = this.resolveLossName();
    const selectedLoss: Cost = isSparseTarget && lossName === "mse" ? "softmaxCrossEntropy" : lossName;
    const lossFn = setLoss(selectedLoss);
    const [loss] = lossFn(yTrue, yPred);
    return loss;
  }

  private assertRecurrentFitSupported(
    X: Matrix[],
    batchSize: number,
    shuffle: boolean,
    validationSplit: number
  ): void {
    const recurrentLayers = this.layers.filter((layer) => this.isRecurrentLayer(layer));
    if (recurrentLayers.length === 0) return;
    const firstRecurrentLayer = recurrentLayers[0];
    const statefulRecurrentLayers = recurrentLayers.filter((layer) => (layer as any).stateful === true);

    if (batchSize !== 1) {
      throw new Error(
        `Sequential.fit: ${firstRecurrentLayer.name} hanya mendukung training per-sample (batchSize=1). ` +
        "Generic batching saat ini menggabungkan sample menjadi kolom matrix dan tidak valid untuk sequence input recurrent."
      );
    }

    if (statefulRecurrentLayers.length > 0 && shuffle) {
      throw new Error(
        `Sequential.fit: ${statefulRecurrentLayers[0].name} dengan stateful=true tidak boleh dipakai bersama shuffle=true ` +
        "karena hidden state dapat bocor ke sample acak berikutnya."
      );
    }

    if (statefulRecurrentLayers.length > 0 && validationSplit > 0) {
      throw new Error(
        `Sequential.fit: ${statefulRecurrentLayers[0].name} dengan stateful=true tidak mendukung validationSplit > 0 ` +
        "karena state training dan validation akan saling memengaruhi dalam loop generic saat ini."
      );
    }

    for (let i = 0; i < X.length; i++) {
      if (X[i]._shape[1] < 1) {
        throw new Error(`Sequential.fit: sample sequence pada index ${i} harus memiliki panjang >= 1.`);
      }
    }
  }

  private isRecurrentLayer(layer: SequentialLayers[number]): boolean {
    return layer.name === "rnn layer" || layer.name === "lstm layer" || layer.name === "gru layer";
  }
}
