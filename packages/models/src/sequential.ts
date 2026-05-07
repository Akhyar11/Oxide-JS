import { readFileSync, writeFileSync } from "fs";
import { Cost, Matrix, setLoss, splitTrainValidation, shuffleInPlace, formatLoss, formatProgressBar, formatTime, mj } from "@oxide-js/core";
import { setLayers, Layers, CompileDenseLayers, Embedding, MemoryBank } from "@oxide-js/layers";
import { FitConfig, FitResult } from "@oxide-js/core";

export type SequentialLayers = Layers[];

export default class Sequential {
  layers: SequentialLayers;
  loss = 0;
  protected isTrainingMode = true;
  private batchInputBufferData: Float32Array = new Float32Array(0);
  private batchTargetBufferData: Float32Array = new Float32Array(0);
  constructor({ layers = [] }: { layers?: SequentialLayers } = {}) {
    this.assertSequentialCompatibleLayers(layers);
    this.layers = layers;
  }

  private assertSequentialCompatibleLayers(layers: SequentialLayers): void {
    for (const layer of layers) {
      if (layer instanceof MemoryBank) {
        throw new Error(
          "Sequential: MemoryBank tidak didukung di arsitektur Sequential. " +
            "Gunakan model manual/custom dan panggil forward/backward MemoryBank secara eksplisit."
        );
      }
    }
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
    this.assertSequentialCompatibleLayers([layer]);
    this.layers.push(layer);
  }

  fillEmbeddingWeight(source: string | Matrix | number[][] | Float32Array | {
    weight?: number[][];
    layers?: any[];
    name?: string;
    vocabSize?: number;
    embeddingDim?: number;
    trainable?: boolean;
  }): this {
    const embeddingLayer = this.layers.find((layer): layer is Embedding => layer instanceof Embedding);
    if (!embeddingLayer) {
      throw new Error("Sequential.fillEmbeddingWeight: model ini tidak memiliki Embedding layer.");
    }
    embeddingLayer.fillWeight(source);
    return this;
  }

  /**
   * Saves model architecture and weights in a Keras-compatible format (model.json + binary weights).
   */
  save(basePath: string) {
    const weightsData: Float32Array[] = [];
    const weightEntries: any[] = [];
    const layerConfigs: any[] = [];
    let currentOffset = 0;

    for (const layer of this.layers) {
      const layerConfig = (layer as any).toKerasConfig ? (layer as any).toKerasConfig() : {
        class_name: layer.name,
        config: (layer as any).save()
      };
      layerConfigs.push(layerConfig);

      const manifest = (layer as any).getWeightsManifest ? (layer as any).getWeightsManifest() : [];
      
      for (const w of manifest) {
        const length = w.data.length;
        // Map "weight" to "kernel" and "bias" to "bias" for Keras naming
        const kerasName = w.name === "weight" ? "kernel" : w.name;
        
        weightEntries.push({
          name: `${layerConfig.config.name}/${kerasName}`,
          shape: w.shape,
          dtype: "float32"
        });
        weightsData.push(w.data);
        currentOffset += length;
      }
    }

    const jsonPath = basePath.endsWith(".json") ? basePath : `${basePath}.json`;
    const binPath = basePath.endsWith(".json") ? basePath.replace(/\.json$/, ".weights.bin") : `${basePath}.weights.bin`;
    const binFilename = binPath.substring(binPath.lastIndexOf("/") + 1);
    const modelJson = {
      format: "layers-model",
      generatedBy: "Oxide-JS v2.3.1",
      convertedBy: null,
      modelTopology: {
        class_name: "Sequential",
        config: {
          layers: layerConfigs,
          name: "sequential_model"
        },
        keras_version: "2.8.0",
        backend: "tensorflow"
      },
      weightsManifest: [
        {
          paths: [binFilename],
          weights: weightEntries
        }
      ]
    };

    // Save JSON
    writeFileSync(jsonPath, JSON.stringify(modelJson, null, 2));

    // Save Binary Weights
    const combinedBuffer = new Float32Array(currentOffset);
    let offset = 0;
    for (const data of weightsData) {
      combinedBuffer.set(data, offset);
      offset += data.length;
    }
    writeFileSync(binPath, Buffer.from(combinedBuffer.buffer));

    console.log(`[Sequential] Keras-compatible model saved: ${jsonPath} and ${binPath}`);
  }

  /**
   * Loads model architecture and binary weights from a Keras-compatible model.json.
   */
  load(jsonPath: string) {
    const dataJson = readFileSync(jsonPath, "utf-8");
    const modelJson = JSON.parse(dataJson);
    
    let layersConfig: any[] = [];
    let weightsManifest: any[] = [];

    // Detect format
    if (modelJson.format === "layers-model") {
      // Keras/TFJS format
      layersConfig = modelJson.modelTopology.config.layers;
      weightsManifest = modelJson.weightsManifest[0].weights;
    } else {
      // Legacy oxide-v1 format (Fallback)
      layersConfig = modelJson.modelTopology.layers;
    }

    const binFilename = modelJson.weightsManifest ? modelJson.weightsManifest[0].paths[0] : null;
    const jsonDir = jsonPath.substring(0, jsonPath.lastIndexOf("/") + 1);
    const binPath = binFilename ? `${jsonDir}${binFilename}` : jsonPath.replace(".json", ".weights.bin");
    
    const binBuffer = readFileSync(binPath);
    const combinedWeights = new Float32Array(binBuffer.buffer, binBuffer.byteOffset, binBuffer.byteLength / 4);

    // Normalize layersConfig to Oxide-JS format for setLayers
    const normalizedLayers = layersConfig.map(l => {
      if (l.class_name && l.config) {
        // Map Keras class_name to Oxide-JS expected names if necessary
        return {
          ...l.config,
          name: l.class_name
        };
      }
      return l;
    });

    this.layers = [];
    const layers = setLayers(normalizedLayers);
    
    // Restore weights
    let weightOffset = 0;
    
    // For Keras format, we use weightsManifest. For Oxide-V1, we use config.weights
    const isKeras = modelJson.format === "layers-model";

    for (let i = 0; i < layers.length; i++) {
      const layer = layers[i];
      const rawConfig = layersConfig[i];
      
      if ((layer as any).setWeightsFromBinary) {
        const layerWeights: Record<string, Float32Array> = {};
        
        if (isKeras) {
          // Extract expected weights for this layer based on its Keras name
          const kerasName = rawConfig.config?.name || rawConfig.name;
          const layerWeightInfos = weightsManifest.filter((w: any) => 
            w.name.startsWith(kerasName + "/")
          );
          
          if (layerWeightInfos.length > 0) {
            for (const wInfo of layerWeightInfos) {
              // Calculate size from shape
              const weightSize = wInfo.shape.reduce((a: number, b: number) => a * b, 1);
              const paramName = wInfo.name.split("/").pop();
              layerWeights[paramName] = combinedWeights.subarray(weightOffset, weightOffset + weightSize);
              weightOffset += weightSize;
            }
            (layer as any).setWeightsFromBinary(layerWeights);
          } else {
            console.warn(`[WARN] No weights found in manifest for ${kerasName}`);
          }
        } else {
          // Legacy oxide-v1 format
          if (rawConfig.weights) {
            for (const wInfo of rawConfig.weights) {
              layerWeights[wInfo.name] = combinedWeights.subarray(wInfo.offset, wInfo.offset + wInfo.length);
            }
            (layer as any).setWeightsFromBinary(layerWeights);
          }
        }
      }
    }

    this.assertSequentialCompatibleLayers(layers);
    this.layers = layers;
    console.log(`[Sequential] Model loaded successfully from ${jsonPath}`);
  }

  compile(config: CompileDenseLayers) {
    for (let layer of this.layers) {
      if (typeof (layer as any).compile === "function") {
        (layer as any).compile(config);
      }
    }
  }

  forward(x: Matrix, batchSize: number = 1): Matrix {
    let input = x;
    for (let layer of this.layers) {
      if (batchSize > 1 && typeof (layer as any).forwardBatch === "function") {
        input = (layer as any).forwardBatch(input, batchSize);
      } else {
        input = layer.forward(input);
      }
    }
    return input;
  }

  backward(y: Matrix, batchSize: number = 1, gradOnly = false) {
    let err = mj.matrix([[]]);
    for (let i = this.layers.length - 1; i >= 0; i--) {
      const layer = this.layers[i] as any;
      if (batchSize > 1 && typeof layer.backwardBatch === "function") {
        err = layer.backwardBatch(y, err, batchSize, gradOnly);
      } else {
        err = layer.backward(y, err, gradOnly);
      }
      if (layer.status === "output") this.loss = layer.loss;
    }
  }

  /**
   * Menerapkan gradien hasil Tape ke seluruh layer.
   */
  applyGradients(alpha: number) {
    for (const layer of this.layers) {
      if (typeof (layer as any).update === "function") {
        (layer as any).update(alpha);
      }
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

    this.assertPerSampleFitSupported(X, y);

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
      let totalEpochWeight = 0;

      for (let start = 0; start < trainX.length; start += batchSize) {
        const end = Math.min(start + batchSize, trainX.length);
        const currentBatchSize = end - start;
        const currentBatchX = this.buildColumnBatch(trainX, trainIndices, start, currentBatchSize, "input");
        const currentBatchY = this.buildColumnBatch(trainY, trainIndices, start, currentBatchSize, "target");

        const pred = this.forward(currentBatchX, currentBatchSize);
        this.backward(currentBatchY, currentBatchSize);
        const batchLossState = this.useBackwardLossForTrainingBatch(currentBatchY, pred)
          ? this.computeLossAndWeightFromBackward(currentBatchY, pred)
          : this.computeLossAndWeight(currentBatchY, pred);
        const batchLossValue = batchLossState.loss;

        totalEpochLoss += batchLossValue * batchLossState.weight;
        totalEpochWeight += batchLossState.weight;

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

      const epochLoss = totalEpochLoss / totalEpochWeight;
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
        valLoss = this.runValidation(valX, valY, verbose, batchSize);
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
    verbose: boolean,
    batchSize: number = 1
  ): number {
    this.eval();
    let totalValLoss = 0;
    let totalValWeight = 0;
    const valStartTime = Date.now();
    const valIndices = Array.from({ length: valX.length }, (_, i) => i);

    for (let start = 0; start < valX.length; start += batchSize) {
      const end = Math.min(start + batchSize, valX.length);
      const currentBatchSize = end - start;
      const valBatchX = this.buildColumnBatch(valX, valIndices, start, currentBatchSize, "input");
      const valBatchY = this.buildColumnBatch(valY, valIndices, start, currentBatchSize, "target");
      const pred = this.forward(valBatchX, currentBatchSize);
      const lossState = this.computeLossAndWeight(valBatchY, pred);
      totalValLoss += lossState.loss * lossState.weight;
      totalValWeight += lossState.weight;

      if (verbose) {
        const elapsed = (Date.now() - valStartTime) / 1000;
        const samplesProcessed = end;
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
    return totalValLoss / totalValWeight;
  }

  protected resolveLossName(): Cost {
    let outputLayer: any = this.layers[this.layers.length - 1] as any;
    for (let i = this.layers.length - 1; i >= 0; i--) {
      if ((this.layers[i] as any).status === "output") {
        outputLayer = this.layers[i] as any;
        break;
      }
    }
    if (!outputLayer) return "mse";
    const lossName =
      typeof outputLayer.getLossName === "function"
        ? outputLayer.getLossName()
        : outputLayer.lossName;
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
    this.assertSparseSoftmaxAutoSwitchSupported(isSparseTarget, lossName);
    const selectedLoss: Cost = isSparseTarget && lossName === "mse" ? "softmaxCrossEntropy" : lossName;
    const lossFn = setLoss(selectedLoss);
    const [loss] = lossFn(yTrue, yPred);
    return loss;
  }

  protected computeLossAndWeight(yTrue: Matrix, yPred: Matrix): { loss: number; weight: number } {
    return {
      loss: this.computeSampleLoss(yTrue, yPred),
      weight: this.computeLossWeight(yTrue, yPred),
    };
  }

  protected computeLossAndWeightFromBackward(yTrue: Matrix, yPred: Matrix): { loss: number; weight: number } {
    return {
      loss: this.loss,
      weight: this.computeLossWeight(yTrue, yPred),
    };
  }

  protected computeLossWeight(yTrue: Matrix, _yPred: Matrix): number {
    return yTrue._shape[1];
  }

  protected useBackwardLossForTrainingBatch(_yTrue: Matrix, _yPred: Matrix): boolean {
    return false;
  }

  private assertSparseSoftmaxAutoSwitchSupported(isSparseTarget: boolean, lossName: Cost): void {
    if (!isSparseTarget || lossName !== "mse") return;
    const outputLayer = this.getOutputLayer();
    if (!outputLayer || typeof outputLayer.getActivationName !== "function") return;
    if (outputLayer.getActivationName() !== "softmax") return;
    throw new Error(
      "Sparse multiclass target requires activation='linear' with loss='softmaxCrossEntropy', or use one-hot target with loss='crossEntropy'. Do not use activation='softmax' with implicit softmaxCrossEntropy."
    );
  }

  private getOutputLayer(): any {
    let outputLayer: any = this.layers[this.layers.length - 1] as any;
    for (let i = this.layers.length - 1; i >= 0; i--) {
      if ((this.layers[i] as any).status === "output") {
        outputLayer = this.layers[i] as any;
        break;
      }
    }
    return outputLayer;
  }

  protected createReusableBatchMatrix(kind: "x" | "y", rows: number, cols: number): Matrix {
    const requiredLength = rows * cols;
    const currentBuffer = kind === "x" ? this.batchInputBufferData : this.batchTargetBufferData;
    let nextBuffer = currentBuffer;

    if (nextBuffer.length < requiredLength) {
      const nextCapacity = Math.max(requiredLength, Math.max(1, nextBuffer.length * 2));
      nextBuffer = new Float32Array(nextCapacity);
      if (kind === "x") {
        this.batchInputBufferData = nextBuffer;
      } else {
        this.batchTargetBufferData = nextBuffer;
      }
    }

    return Matrix.fromFlat(nextBuffer.subarray(0, requiredLength), [rows, cols]);
  }

  protected isRecurrentLayer(layer: SequentialLayers[number]): boolean {
    return layer.name === "rnn layer" || layer.name === "lstm layer" || layer.name === "gru layer";
  }

  private assertPerSampleFitSupported(X: Matrix[], y: Matrix[]): void {
    if (this.layers.some((layer) => this.isRecurrentLayer(layer))) {
      throw new Error(
        "Sequential.fit only supports per-sample supervised loss. Use Transformers.fit for causal LM or RecurrentModel.fit for sequence/recurrent training."
      );
    }

    for (let i = 0; i < X.length; i++) {
      if (X[i]._shape[1] < 1 || y[i]._shape[1] < 1) {
        throw new Error(`Sequential.fit: sample pada index ${i} harus memiliki kolom >= 1.`);
      }
      if (y[i]._shape[1] > 1) {
        throw new Error(
          "Sequential.fit only supports per-sample supervised loss. Use Transformers.fit for causal LM or RecurrentModel.fit for sequence/recurrent training."
        );
      }
    }
  }

  private buildColumnBatch(
    samples: Matrix[],
    indices: number[],
    start: number,
    currentBatchSize: number,
    kind: "input" | "target"
  ): Matrix {
    if (currentBatchSize === 1) {
      return samples[indices[start]];
    }

    const [rows, cols] = samples[indices[start]]._shape;
    if (cols !== 1) {
      throw new Error(
        `Sequential.fit: batchSize > 1 saat ini hanya mendukung sample ${kind} berbentuk [rows, 1]. Ditemukan [${rows}, ${cols}].`
      );
    }

    const batch = this.createReusableBatchMatrix(kind === "input" ? "x" : "y", rows, currentBatchSize);
    for (let j = 0; j < currentBatchSize; j++) {
      const sample = samples[indices[start + j]];
      if (sample._shape[0] !== rows || sample._shape[1] !== 1) {
        throw new Error(
          `Sequential.fit: semua sample ${kind} dalam satu batch harus memiliki shape [${rows}, 1]. Ditemukan [${sample._shape[0]}, ${sample._shape[1]}].`
        );
      }
      batch.setCol(j, sample._data);
    }
    return batch;
  }

  dispose() {
    this.batchInputBufferData = new Float32Array(0);
    this.batchTargetBufferData = new Float32Array(0);
    for (const layer of this.layers) {
      if (typeof (layer as any).dispose === 'function') {
        (layer as any).dispose();
      }
    }
  }

  // Memory helpers: iterate layers and call memory API when available
  resetMemory(): this {
    for (const layer of this.layers) {
      if (typeof (layer as any).resetMemory === "function") {
        (layer as any).resetMemory();
      }
    }
    return this;
  }

  saveMemory(path: string): this {
    const states: any[] = [];
    for (const layer of this.layers) {
      if (typeof (layer as any).getMemoryState === "function") {
        states.push((layer as any).getMemoryState());
      } else {
        states.push(null);
      }
    }
    writeFileSync(path, JSON.stringify(states), "utf-8");
    return this;
  }

  loadMemory(path: string): this {
    const raw = readFileSync(path, "utf-8");
    const states = JSON.parse(raw);
    for (let i = 0; i < states.length && i < this.layers.length; i++) {
      const st = states[i];
      if (!st) continue;
      if (typeof (this.layers[i] as any).setMemoryState === "function") {
        (this.layers[i] as any).setMemoryState(st);
      }
    }
    return this;
  }

  freezeMemoryWrites(): this {
    for (const layer of this.layers) {
      if (typeof (layer as any).freezeWrites === "function") (layer as any).freezeWrites();
    }
    return this;
  }

  enableMemoryWrites(): this {
    for (const layer of this.layers) {
      if (typeof (layer as any).enableWrites === "function") (layer as any).enableWrites();
    }
    return this;
  }
}
