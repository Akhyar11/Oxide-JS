import { Matrix, engine } from "@oxide-js/core";
import { BaseLayer, ForwardOptions } from "@oxide-js/layers";
import {
  Callback,
  CallbackLogs,
  CompileConfig,
  FitConfig,
  HistoryRecord,
  LossLike,
  MetricLike,
  ModelConfig,
  ModelSummaryRow,
  OptimizerLike,
  SerializedModel,
  WeightData
} from "./types.js";
import { resolveCompileConfig } from "./resolvers.js";
import { computeMetric, getMetricName } from "./metrics.js";
import { createBatches, trainValidationSplit } from "./data.js";

export abstract class BaseModel {
  public name: string;
  public trainable: boolean;
  public training: boolean = true;

  protected layers: BaseLayer[] = [];

  protected compiledLoss?: any;
  protected compiledOptimizer?: any;
  protected compiledMetrics: MetricLike[] = [];
  protected learningRate: number = 0.001;

  public isCompiled: boolean = false;
  public isBuilt: boolean = false;

  public inputShape: number[] = [];
  public outputShape: number[] = [];

  constructor(config?: ModelConfig) {
    this.name = config?.name || this.constructor.name;
    this.trainable = config?.trainable ?? true;
  }

  /**
   * Forward utama model.
   * Subclass seperti Sequential wajib implement.
   * Updated to support ForwardOptions for future compatibility.
   */
  public abstract forward(inputs: Matrix, optionsOrTraining?: ForwardOptions | boolean): Matrix;

  /**
   * Build model berdasarkan input shape.
   * Subclass boleh override jika punya graph non-linear.
   */
  public build(inputShape: number[]): void {
    this.inputShape = [...inputShape];

    let currentShape = [...inputShape];

    for (const layer of this.layers) {
      if (!layer.isBuilt) {
        layer.build(currentShape);
      }

      currentShape = [...layer.outputShape];
    }

    this.outputShape = currentShape;
    this.isBuilt = true;
  }

  /**
   * Tambah layer ke model.
   * Untuk BaseModel dibuat protected/public supaya Sequential bisa pakai langsung.
   */
  public add(layer: BaseLayer): this {
    this.layers.push(layer);
    this.isBuilt = false;
    return this;
  }

  /**
   * Ambil semua layer.
   */
  public getLayers(): BaseLayer[] {
    return [...this.layers];
  }

  /**
   * Ambil layer berdasarkan index.
   */
  public getLayer(index: number): BaseLayer {
    const layer = this.layers[index];

    if (!layer) {
      throw new Error(`[${this.name}] Layer index ${index} tidak ditemukan.`);
    }

    return layer;
  }

  /**
   * Cari layer berdasarkan nama.
   */
  public getLayerByName(name: string): BaseLayer | undefined {
    return this.layers.find(layer => layer.name === name);
  }

  /**
   * Jumlah layer.
   */
  public get layerCount(): number {
    return this.layers.length;
  }

  /**
   * Ubah model dan semua layer menjadi training mode.
   */
  public train(): void {
    this.training = true;

    for (const layer of this.layers) {
      layer.train();
    }
  }

  /**
   * Ubah model dan semua layer menjadi inference/eval mode.
   */
  public eval(): void {
    this.training = false;

    for (const layer of this.layers) {
      layer.eval();
    }
  }

  /**
   * Semua weights dari semua layer.
   */
  public get weights(): Matrix[] {
    return this.layers.flatMap(layer => layer.weights);
  }

  /**
   * Semua trainable weights.
   */
  public get trainableWeights(): Matrix[] {
    if (!this.trainable) return [];

    return this.layers.flatMap(layer => layer.trainableWeights);
  }

  /**
   * Semua non-trainable weights.
   */
  public get nonTrainableWeights(): Matrix[] {
    if (!this.trainable) {
      return this.layers.flatMap(layer => layer.weights);
    }

    return this.layers.flatMap(layer => layer.nonTrainableWeights);
  }

  /**
   * Alias untuk optimizer.
   */
  public getTrainableParameters(): Matrix[] {
    return this.trainableWeights;
  }

  /**
   * Semua parameter, baik trainable maupun non-trainable.
   */
  public getAllParameters(): Matrix[] {
    return this.weights;
  }

  /**
   * Clear gradient semua trainable parameter.
   * Ini tetap diperlukan walaupun gradient ditangani autodiff.
   */
  public zeroGrad(): void {
    for (const param of this.trainableWeights) {
      param.clearGrad();
    }
  }

  /**
   * Alias.
   */
  public clearGradients(): void {
    this.zeroGrad();
  }

  /**
   * Hitung total parameter semua layer.
   */
  public countParams(): number {
    return this.layers.reduce((total, layer) => {
      return total + layer.countParams();
    }, 0);
  }

  /**
   * Hitung total trainable parameter.
   */
  public countTrainableParams(): number {
    return this.trainableWeights.reduce((total, param) => {
      return total + param._shape.reduce((a, b) => a * b, 1);
    }, 0);
  }

  /**
   * Hitung total non-trainable parameter.
   */
  public countNonTrainableParams(): number {
    return this.nonTrainableWeights.reduce((total, param) => {
      return total + param._shape.reduce((a, b) => a * b, 1);
    }, 0);
  }

  /**
   * Compile model dengan loss, optimizer, dan metrics.
   * Supports both raw objects dan string-based resolution.
   */
  public compile(config: CompileConfig): void {
    const resolved = resolveCompileConfig(config);

    this.compiledLoss = resolved.loss;
    this.compiledOptimizer = resolved.optimizer;
    this.compiledMetrics = resolved.metrics;
    this.learningRate = resolved.learningRate;
    this.isCompiled = true;
  }

  /**
   * Compute metrics for predictions.
   */
  protected computeMetrics(yPred: Matrix, yTrue: Matrix): Record<string, number> {
    const metrics: Record<string, number> = {};

    for (const metric of this.compiledMetrics) {
      const name = getMetricName(metric);
      const value = computeMetric(metric, yPred, yTrue);
      metrics[name] = value;
    }

    return metrics;
  }

  /**
   * Pastikan model sudah compile sebelum training/evaluate.
   */
  protected assertCompiled(): void {
    if (!this.isCompiled) {
      throw new Error(
        `[${this.name}] Model belum di-compile. Panggil model.compile({ optimizer, loss }) terlebih dahulu.`
      );
    }

    if (!this.compiledLoss) {
      throw new Error(`[${this.name}] Loss function belum didefinisikan.`);
    }

    if (!this.compiledOptimizer) {
      throw new Error(`[${this.name}] Optimizer belum didefinisikan.`);
    }
  }

  /**
   * Hitung loss.
   * Dibuat fleksibel karena loss kamu bisa berupa function atau object.
   */
  protected computeLoss(yPred: Matrix, yTrue: Matrix): any {
    if (!this.compiledLoss) {
      throw new Error(`[${this.name}] Loss function belum tersedia.`);
    }

    let rawLoss: any;
    if (typeof this.compiledLoss === "function") {
      rawLoss = this.compiledLoss(yPred, yTrue);
    } else if (typeof this.compiledLoss.forward === "function") {
      rawLoss = this.compiledLoss.forward(yPred, yTrue);
    } else if (typeof this.compiledLoss.compute === "function") {
      rawLoss = this.compiledLoss.compute(yPred, yTrue);
    } else {
      throw new Error(`[${this.name}] Format loss function tidak valid.`);
    }

    // If it's already a Matrix (e.g. custom autodiff loss), return it directly!
    if (rawLoss instanceof Matrix) {
      return rawLoss;
    }

    // If it's [lossValue, gradientMatrix] (e.g. core cost function format)
    if (Array.isArray(rawLoss) && typeof rawLoss[0] === "number" && rawLoss[1] instanceof Matrix) {
      const [lossVal, dPred] = rawLoss;
      const lossMatrix = Matrix.fromFlat(new Float32Array([lossVal]), [1, 1]);

      engine.record([yPred], [lossMatrix], (grad) => {
        const scale = grad._data[0];
        const scaledGrad = Matrix.fromFlat(new Float32Array(dPred._data.length), dPred._shape);
        for (let i = 0; i < dPred._data.length; i++) {
          scaledGrad._data[i] = dPred._data[i] * scale;
        }
        return [scaledGrad];
      });

      return lossMatrix;
    }

    // If it's just a number, wrap it as a Matrix
    if (typeof rawLoss === "number") {
      return Matrix.fromFlat(new Float32Array([rawLoss]), [1, 1]);
    }

    return rawLoss;
  }

  /**
   * Optimizer step.
   * Dibuat fleksibel agar cocok dengan beberapa bentuk optimizer.
   */
  protected optimizerStep(): void {
    if (!this.compiledOptimizer) {
      throw new Error(`[${this.name}] Optimizer belum tersedia.`);
    }

    const params = this.trainableWeights;

    if (typeof this.compiledOptimizer.step === "function") {
      this.compiledOptimizer.step(params);
      return;
    }

    if (typeof this.compiledOptimizer.update === "function") {
      this.compiledOptimizer.update(params);
      return;
    }

    throw new Error(`[${this.name}] Optimizer tidak memiliki method step() atau update().`);
  }

  public trainStep(xBatch: Matrix, yBatch: Matrix): { loss: any; yPred: Matrix } {
    this.assertCompiled();
    this.train();

    this.zeroGrad();

    let yPred: Matrix | undefined;
    const tape = engine.grad(() => {
      yPred = this.forward(xBatch, true);
      return this.computeLoss(yPred, yBatch);
    });

    tape.backward(tape.result);

    this.optimizerStep();

    return { loss: tape.result, yPred: yPred! };
  }

  /**
   * Predict/inference.
   */
  public predict(inputs: Matrix): Matrix {
    this.eval();
    return this.forward(inputs, false);
  }

  /**
   * Evaluate model on data.
   * Returns loss and metrics.
   */
  public evaluate(x: Matrix, y: Matrix): {
    loss: number | undefined;
    yPred: Matrix;
    metrics: Record<string, number>;
  } {
    this.assertCompiled();
    this.eval();

    const yPred = this.forward(x, false);
    const loss = this.extractScalar(this.computeLoss(yPred, y));
    const metrics = this.computeMetrics(yPred, y);

    return { loss, yPred, metrics };
  }

  /**
   * Fit model on data with advanced training loop.
   * Supports mini-batching, validation, callbacks, and early stopping.
   */
  public fit(x: Matrix, y: Matrix, config: FitConfig = {}): HistoryRecord[] {
    this.assertCompiled();

    const epochs = config.epochs ?? 1;
    const batchSize = config.batchSize ?? x._shape[0];
    const shuffle = config.shuffle ?? false;
    const validationSplit = config.validationSplit;
    const verbose = config.verbose ?? 1;
    const callbacks = config.callbacks ?? [];

    // Add HistoryCallback if not already present
    const hasHistoryCallback = callbacks.some((cb) => cb.constructor.name === "HistoryCallback");
    const allCallbacks = hasHistoryCallback ? callbacks : callbacks;

    // Prepare training and validation data
    let xTrain = x;
    let yTrain = y;
    let xVal: Matrix | null = null;
    let yVal: Matrix | null = null;

    if (config.validationData) {
      [xVal, yVal] = config.validationData;
    } else if (validationSplit && validationSplit > 0) {
      const split = trainValidationSplit(x, y, validationSplit, shuffle);
      xTrain = split.xTrain;
      yTrain = split.yTrain;
      xVal = split.xVal;
      yVal = split.yVal;
    }

    const history: HistoryRecord[] = [];

    // Call onTrainBegin
    for (const callback of allCallbacks) {
      if (callback.onTrainBegin) {
        callback.onTrainBegin();
      }
    }

    // Training loop
    for (let epoch = 1; epoch <= epochs; epoch++) {
      // Call onEpochBegin
      const epochLogs: CallbackLogs = {};
      for (const callback of allCallbacks) {
        if (callback.onEpochBegin) {
          callback.onEpochBegin(epoch, epochLogs);
        }
      }

      // Create batches
      const batches = createBatches(xTrain, yTrain, batchSize, shuffle);
      let epochLoss = 0;
      let batchCount = 0;

      // Process each batch
      for (let batchIdx = 0; batchIdx < batches.length; batchIdx++) {
        const batch = batches[batchIdx];

        // Call onBatchBegin
        const batchLogs: CallbackLogs = { batch: batchIdx };
        for (const callback of allCallbacks) {
          if (callback.onBatchBegin) {
            callback.onBatchBegin(batchIdx, batchLogs);
          }
        }

        // Training step
        const result = this.trainStep(batch.x, batch.y);
        const batchLoss = this.extractScalar(result.loss) ?? 0;
        epochLoss += batchLoss;
        batchCount++;

        // Update batch logs and call onBatchEnd
        batchLogs.loss = batchLoss;
        for (const callback of allCallbacks) {
          if (callback.onBatchEnd) {
            callback.onBatchEnd(batchIdx, batchLogs);
          }
        }
      }

      // Compute average loss and metrics
      const avgLoss = batchCount > 0 ? epochLoss / batchCount : 0;
      const metrics = this.computeMetrics(this.forward(xTrain, false), yTrain);

      // Compute validation loss and metrics if available
      let valLoss: number | undefined;
      let valMetrics: Record<string, number> | undefined;

      if (xVal && yVal) {
        const evalResult = this.evaluate(xVal, yVal);
        valLoss = evalResult.loss;
        valMetrics = evalResult.metrics;
      }

      // Create history record
      const record: HistoryRecord = {
        epoch,
        loss: avgLoss,
        val_loss: valLoss,
        metrics,
        val_metrics: valMetrics
      };

      history.push(record);

      // Update epoch logs and call onEpochEnd
      epochLogs.loss = avgLoss;
      epochLogs.val_loss = valLoss;
      epochLogs.metrics = metrics;
      epochLogs.val_metrics = valMetrics;

      for (const callback of allCallbacks) {
        if (callback.onEpochEnd) {
          callback.onEpochEnd(epoch, epochLogs);
        }
      }

      // Check for early stopping
      let shouldStop = false;
      for (const callback of allCallbacks) {
        if (callback.shouldStop) {
          shouldStop = true;
          break;
        }
      }

      if (shouldStop) {
        if (verbose >= 1) {
          console.log(`Early stopping at epoch ${epoch}`);
        }
        break;
      }

      // Verbose output
      if (verbose >= 1) {
        let logStr = `Epoch ${epoch}/${epochs}`;
        logStr += ` - loss: ${avgLoss.toFixed(6)}`;

        if (valLoss !== undefined) {
          logStr += ` - val_loss: ${valLoss.toFixed(6)}`;
        }

        for (const [key, value] of Object.entries(metrics)) {
          logStr += ` - ${key}: ${(value as number).toFixed(4)}`;
        }

        if (valMetrics) {
          for (const [key, value] of Object.entries(valMetrics)) {
            logStr += ` - val_${key}: ${(value as number).toFixed(4)}`;
          }
        }

        console.log(logStr);
      }
    }

    // Call onTrainEnd
    const finalLogs: CallbackLogs = {};
    for (const callback of allCallbacks) {
      if (callback.onTrainEnd) {
        callback.onTrainEnd(finalLogs);
      }
    }

    return history;
  }

  /**
   * Ambil nilai scalar dari loss.
   * Disesuaikan dengan Matrix core kamu.
   */
  protected extractScalar(value: any): number | undefined {
    if (typeof value === "number") return value;

    if (value instanceof Matrix) {
      return value._data[0];
    }

    if (value?._data?.length > 0) {
      return value._data[0];
    }

    if (typeof value?.item === "function") {
      return value.item();
    }

    if (typeof value?.data === "number") {
      return value.data;
    }

    return undefined;
  }

  /**
   * Ambil semua weight data dari semua layer.
   */
  public getWeights(): WeightData[] {
    return this.layers.flatMap(layer => layer.getWeights());
  }

  /**
   * Set weights ke layer berdasarkan prefix nama layer.
   * Akan otomatis mendekompresi number[] kembali menjadi Float32Array jika diperlukan.
   */
  public setWeights(weightsData: WeightData[]): void {
    // Pastikan data berbentuk Float32Array saat diset
    const normalizedWeights = weightsData.map(w => ({
      ...w,
      data: w.data instanceof Float32Array ? w.data : new Float32Array(w.data)
    }));

    for (const layer of this.layers) {
      const layerWeights = normalizedWeights.filter(w => {
        return w.name.startsWith(`${layer.name}/`);
      });

      if (layerWeights.length > 0) {
        layer.setWeights(layerWeights);
      }
    }
  }

  /**
   * Config model.
   */
  public getConfig(): Record<string, any> {
    return {
      name: this.name,
      trainable: this.trainable,
      layers: this.layers.map(layer => layer.getKerasConfig())
    };
  }

  /**
   * Keras-style config.
   */
  public getKerasConfig(): Record<string, any> {
    return {
      class_name: this.constructor.name,
      config: this.getConfig()
    };
  }

  /**
   * Serialize model.
   * Otomatis mengompres Float32Array ke Array agar ukuran file JSON tidak membengkak.
   */
  public serialize(): SerializedModel {
    const compressedWeights = this.getWeights().map(w => ({
      ...w,
      data: Array.isArray(w.data) ? w.data : Array.from(w.data)
    }));

    return {
      class_name: this.constructor.name,
      name: this.name,
      trainable: this.trainable,
      config: this.getConfig(),
      layers: this.layers.map(layer => layer.getKerasConfig()),
      weights: compressedWeights
    };
  }

  /**
   * Info summary setiap layer.
   */
  public getSummaryInfo(): ModelSummaryRow[] {
    return this.layers.map(layer => layer.getSummaryInfo());
  }

  /**
   * Cetak summary model.
   */
  public summary(): void {
    const rows = this.getSummaryInfo();

    const divider = "=".repeat(80);

    console.log(divider);
    console.log(`Model: ${this.name}`);
    console.log(divider);
    console.log(
      `${"Layer".padEnd(24)} ${"Type".padEnd(22)} ${"Output Shape".padEnd(18)} ${"Param #".padStart(10)}`
    );
    console.log("-".repeat(80));

    for (const row of rows) {
      console.log(
        `${row.name.padEnd(24)} ${row.type.padEnd(22)} ${row.outputShape.padEnd(18)} ${String(row.paramCount).padStart(10)}`
      );
    }

    console.log("-".repeat(80));
    console.log(`Total params: ${this.countParams().toLocaleString()}`);
    console.log(`Trainable params: ${this.countTrainableParams().toLocaleString()}`);
    console.log(`Non-trainable params: ${this.countNonTrainableParams().toLocaleString()}`);
    console.log(divider);
  }

  /**
   * Validasi minimal input.
   */
  protected assertNotEmpty(): void {
    if (this.layers.length === 0) {
      throw new Error(`[${this.name}] Model tidak memiliki layer.`);
    }
  }

  /**
   * Membuat nama layer unik.
   * Sebaiknya dipanggil oleh Sequential.add().
   */
  protected makeUniqueLayerName(layer: BaseLayer): void {
    const existingNames = new Set(
      this.layers
        .filter(l => l !== layer)
        .map(l => l.name)
    );

    if (!existingNames.has(layer.name)) return;

    const baseName = layer.constructor.name;
    let index = 1;
    let candidate = `${baseName}_${index}`;

    while (existingNames.has(candidate)) {
      index++;
      candidate = `${baseName}_${index}`;
    }

    layer.name = candidate;
  }
}
