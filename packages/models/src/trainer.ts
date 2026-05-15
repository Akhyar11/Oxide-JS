import {
  Cost,
  Matrix,
  engine,
  FitConfig,
  FitResult,
  formatLoss,
  formatProgressBar,
  formatTime,
  mj,
  setLoss,
  shuffleInPlace,
  splitTrainValidation,
} from "@oxide-js/core";
import Module from "./module.js";

export type ModuleValue = Matrix | ModuleValue[] | { [key: string]: ModuleValue };

export interface ModuleLossResult {
  loss: number;
  grads: ModuleValue;
  weight?: number;
}

export type ModuleLoss =
  | Cost
  | ((yTrue: ModuleValue, yPred: ModuleValue) => ModuleLossResult | [number, Matrix]);

export interface ModuleFitConfig extends FitConfig {
  loss?: ModuleLoss;
  alpha?: number;
}

function isMatrix(value: unknown): value is Matrix {
  return value instanceof Matrix;
}

function isObjectLike(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function collectMatrices(value: ModuleValue, out: Matrix[] = []): Matrix[] {
  if (isMatrix(value)) {
    out.push(value);
    return out;
  }
  if (Array.isArray(value)) {
    for (const item of value) {
      collectMatrices(item, out);
    }
    return out;
  }
  for (const key of Object.keys(value)) {
    collectMatrices(value[key], out);
  }
  return out;
}

function forEachMatchingMatrix(
  a: ModuleValue,
  b: ModuleValue,
  visitor: (aMatrix: Matrix, bMatrix: Matrix) => void
): void {
  if (isMatrix(a) && isMatrix(b)) {
    visitor(a, b);
    return;
  }
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) {
      throw new Error(`Trainer: array structure mismatch (${a.length} !== ${b.length})`);
    }
    for (let i = 0; i < a.length; i++) {
      forEachMatchingMatrix(a[i], b[i], visitor);
    }
    return;
  }
  if (isObjectLike(a) && isObjectLike(b) && !Array.isArray(a) && !Array.isArray(b)) {
    const aKeys = Object.keys(a);
    const bKeys = Object.keys(b);
    if (aKeys.length !== bKeys.length || aKeys.some((key) => !(key in b))) {
      throw new Error("Trainer: object structure mismatch between target and prediction");
    }
    for (const key of aKeys) {
      forEachMatchingMatrix(a[key] as ModuleValue, b[key] as ModuleValue, visitor);
    }
    return;
  }
  throw new Error("Trainer: target/prediction structure mismatch");
}

function scaleModuleValue(value: ModuleValue, scale: number): ModuleValue {
  if (isMatrix(value)) {
    return scale === 1 ? value : mj.mul(value, scale);
  }
  if (Array.isArray(value)) {
    return value.map((item) => scaleModuleValue(item, scale));
  }
  const scaled: Record<string, ModuleValue> = {};
  for (const key of Object.keys(value)) {
    scaled[key] = scaleModuleValue(value[key], scale);
  }
  return scaled;
}

export default class Trainer<TModule extends Module = Module> {
  constructor(
    readonly module: TModule,
    private loss: ModuleLoss = "mse"
  ) {}

  setLoss(loss: ModuleLoss): this {
    this.loss = loss;
    return this;
  }

  trainBatch(x: ModuleValue, y: ModuleValue, alpha?: number): { loss: number; weight: number } {
    this.module.zeroGrad();
    const tape = engine.startTape();
    try {
      const pred = this.module.forward(x) as ModuleValue;
      const loss = this.buildAutodiffLoss(y, pred);
      tape.backward(loss);
      this.module.step(alpha);
      return {
        loss: loss._data[0],
        weight: this.computeWeight(y),
      };
    } finally {
      engine.endTape();
    }
  }

  fit(X: ModuleValue[], y: ModuleValue[], epochs: number, config: ModuleFitConfig = {}): FitResult {
    if (!Array.isArray(X) || !Array.isArray(y) || X.length === 0 || X.length !== y.length) {
      throw new Error("X dan y harus memiliki jumlah sample yang sama dan tidak kosong");
    }
    if (!Number.isFinite(epochs) || epochs < 1) {
      throw new Error("epochs harus >= 1");
    }

    const {
      batchSize = Math.max(1, Math.floor(X.length / 10)),
      validationSplit = 0,
      earlyStoppingPatience = Infinity,
      shuffle = true,
      verbose = false,
      onEpochEnd = () => {},
      monitorMetric = validationSplit > 0 ? "valLoss" : "loss",
      minDelta = 0,
      mode = "min",
      loss,
      alpha,
    } = config;

    if (loss !== undefined) {
      this.loss = loss;
    }
    if (!Number.isInteger(batchSize) || batchSize < 1) {
      throw new Error("batchSize harus >= 1");
    }
    if (validationSplit < 0 || validationSplit >= 1) {
      throw new Error("validationSplit harus antara 0 dan 1");
    }
    if (earlyStoppingPatience < 0) {
      throw new Error("earlyStoppingPatience harus >= 0");
    }

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
    this.module.train();

    for (let epoch = 0; epoch < epochs; epoch++) {
      const epochStartTime = Date.now();

      if (shuffle) {
        shuffleInPlace(trainIndices);
      }

      if (verbose) {
        const progress = formatProgressBar(0, trainX.length);
        const valStr = validationSplit > 0 ? ` | Val Loss: ${valLoss !== undefined ? formatLoss(valLoss) : "...."}` : "";
        process.stdout.write(
          `\rEpoch ${epoch + 1}/${epochs} ${progress} | Loss: ....${valStr} | 0.0 samples/s | ETA: --:--`
        );
      }

      let totalEpochLoss = 0;
      let totalEpochWeight = 0;

      for (let start = 0; start < trainX.length; start += batchSize) {
        const end = Math.min(start + batchSize, trainX.length);
        const currentBatchSize = end - start;
        const batchX = this.buildBatch(trainX, trainIndices, start, currentBatchSize, "input");
        const batchY = this.buildBatch(trainY, trainIndices, start, currentBatchSize, "target");
        const batchLossState = this.trainBatch(batchX, batchY, alpha);
        totalEpochLoss += batchLossState.loss * batchLossState.weight;
        totalEpochWeight += batchLossState.weight;

        if (verbose) {
          const elapsed = (Date.now() - epochStartTime) / 1000;
          const speed = end / Math.max(elapsed, 0.001);
          const eta = (trainX.length - end) / speed;
          const progress = formatProgressBar(end, trainX.length);
          const valStr = validationSplit > 0 ? ` | Val Loss: ${valLoss !== undefined ? formatLoss(valLoss) : "...."}` : "";
          process.stdout.write(
            `\rEpoch ${epoch + 1}/${epochs} ${progress} | Loss: ${formatLoss(batchLossState.loss)}${valStr} | ${speed.toFixed(1)} samples/s | ETA: ${formatTime(eta)}`
          );
        }
      }

      const epochLoss = totalEpochLoss / totalEpochWeight;
      history.loss.push(epochLoss);

      if (validationSplit > 0 && valX.length > 0) {
        valLoss = this.runValidation(valX, valY, batchSize, verbose);
        (history.valLoss as number[]).push(valLoss);
        this.module.train();
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
      const improved = mode === "min"
        ? metricValue < bestLoss - minDelta
        : metricValue > bestLoss + minDelta;

      if (improved) {
        bestLoss = metricValue;
        bestEpoch = epoch;
        noImprovementCount = 0;
      } else {
        noImprovementCount++;
      }

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

    this.module.eval();
    return {
      history,
      bestEpoch,
      bestLoss,
      stoppedEarly,
      stoppingEpoch,
    };
  }

  private resolveLossResult(yTrue: ModuleValue, yPred: ModuleValue): ModuleLossResult {
    if (typeof this.loss === "function") {
      const result = this.loss(yTrue, yPred);
      if (Array.isArray(result)) {
        return {
          loss: result[0],
          grads: result[1],
          weight: this.computeWeight(yTrue),
        };
      }
      return {
        loss: result.loss,
        grads: result.grads,
        weight: result.weight ?? this.computeWeight(yTrue),
      };
    }

    if (!isMatrix(yTrue) || !isMatrix(yPred)) {
      throw new Error(
        `Trainer: built-in loss '${this.loss}' only supports single-output Matrix targets. ` +
        "Use a custom loss function for multi-input or multi-output modules."
      );
    }
    const [lossValue, gradValue] = setLoss(this.loss)(yTrue, yPred);
    return {
      loss: lossValue,
      grads: gradValue,
      weight: this.computeWeight(yTrue),
    };
  }

  private buildAutodiffLoss(yTrue: ModuleValue, yPred: ModuleValue): Matrix {
    const result = this.resolveLossResult(yTrue, yPred);
    const loss = Matrix.fromFlat(new Float32Array([result.loss]), [1, 1]);
    const predictionLeaves = collectMatrices(yPred);
    const gradLeaves = collectMatrices(result.grads);
    if (predictionLeaves.length !== gradLeaves.length) {
      throw new Error(
        `Trainer: custom loss returned ${gradLeaves.length} gradient leaves for ${predictionLeaves.length} prediction leaves`
      );
    }

    const tape = engine.tape;
    if (tape) {
      tape.record(predictionLeaves, [loss], (grad: Matrix) => {
        const scaledGrads = collectMatrices(scaleModuleValue(result.grads, grad._data[0]));
        for (let i = 0; i < predictionLeaves.length; i++) {
          const predLeaf = predictionLeaves[i];
          const gradLeaf = scaledGrads[i];
          if (predLeaf.grad) predLeaf.grad.addInPlace(gradLeaf);
          else predLeaf.grad = gradLeaf;
        }
      }, { saveInput: false, saveOutput: false });
    }
    return loss;
  }

  private runValidation(valX: ModuleValue[], valY: ModuleValue[], batchSize: number, verbose: boolean): number {
    this.module.eval();
    let totalValLoss = 0;
    let totalValWeight = 0;
    const valIndices = Array.from({ length: valX.length }, (_, i) => i);
    const valStartTime = Date.now();

    for (let start = 0; start < valX.length; start += batchSize) {
      const end = Math.min(start + batchSize, valX.length);
      const currentBatchSize = end - start;
      const batchX = this.buildBatch(valX, valIndices, start, currentBatchSize, "input");
      const batchY = this.buildBatch(valY, valIndices, start, currentBatchSize, "target");
      const pred = this.module.forward(batchX) as ModuleValue;
      const result = this.resolveLossResult(batchY, pred);
      totalValLoss += result.loss * (result.weight ?? this.computeWeight(batchY));
      totalValWeight += result.weight ?? this.computeWeight(batchY);

      if (verbose) {
        const elapsed = (Date.now() - valStartTime) / 1000;
        const speed = end / Math.max(elapsed, 0.001);
        const eta = (valX.length - end) / speed;
        process.stdout.write(
          `\rValidating  ${formatProgressBar(end, valX.length)} | ${speed.toFixed(1)} samples/s | ETA: ${formatTime(eta)}`
        );
      }
    }

    if (verbose) process.stdout.write("\n");
    return totalValLoss / totalValWeight;
  }

  private computeWeight(value: ModuleValue): number {
    let weight = 0;
    for (const matrix of collectMatrices(value)) {
      weight += matrix._shape[1];
    }
    return Math.max(1, weight);
  }

  private buildBatch(
    samples: ModuleValue[],
    indices: number[],
    start: number,
    currentBatchSize: number,
    kind: "input" | "target"
  ): ModuleValue {
    if (currentBatchSize === 1) {
      return samples[indices[start]];
    }

    const first = samples[indices[start]];
    return this.batchFromStructure(first, samples, indices, start, currentBatchSize, kind);
  }

  private batchFromStructure(
    exemplar: ModuleValue,
    samples: ModuleValue[],
    indices: number[],
    start: number,
    currentBatchSize: number,
    kind: "input" | "target"
  ): ModuleValue {
    if (isMatrix(exemplar)) {
      return this.buildColumnBatchForLeaf(exemplar, samples, indices, start, currentBatchSize, kind);
    }

    if (Array.isArray(exemplar)) {
      return exemplar.map((item, index) => {
        const childSamples = samples.map((sample) => {
          if (!Array.isArray(sample)) {
            throw new Error(`Trainer.fit: expected array sample for ${kind}`);
          }
          return sample[index];
        });
        return this.batchFromStructure(item, childSamples, indices, start, currentBatchSize, kind);
      });
    }

    const batch: Record<string, ModuleValue> = {};
    for (const key of Object.keys(exemplar)) {
      const childSamples = samples.map((sample) => {
        if (!isObjectLike(sample) || Array.isArray(sample)) {
          throw new Error(`Trainer.fit: expected object sample for ${kind}`);
        }
        return sample[key] as ModuleValue;
      });
      batch[key] = this.batchFromStructure(exemplar[key], childSamples, indices, start, currentBatchSize, kind);
    }
    return batch;
  }

  private buildColumnBatchForLeaf(
    exemplar: Matrix,
    samples: ModuleValue[],
    indices: number[],
    start: number,
    currentBatchSize: number,
    kind: "input" | "target"
  ): Matrix {
    const [rows, cols] = exemplar._shape;
    if (cols !== 1) {
      throw new Error(
        `Trainer.fit: batchSize > 1 saat ini hanya mendukung leaf ${kind} berbentuk [rows, 1]. Ditemukan [${rows}, ${cols}].`
      );
    }

    const batch = mj.zeros([rows, currentBatchSize]);
    for (let j = 0; j < currentBatchSize; j++) {
      const sample = samples[indices[start + j]];
      if (!isMatrix(sample)) {
        throw new Error(`Trainer.fit: expected Matrix leaf for ${kind}`);
      }
      if (sample._shape[0] !== rows || sample._shape[1] !== 1) {
        throw new Error(
          `Trainer.fit: semua leaf ${kind} dalam satu batch harus memiliki shape [${rows}, 1]. Ditemukan [${sample._shape[0]}, ${sample._shape[1]}].`
        );
      }
      batch.setCol(j, sample._data);
    }
    return batch;
  }
}
