import { Optimizer, StatusLayer } from "@oxide-js/core";
import { mj } from "@oxide-js/core";
import { Matrix } from "@oxide-js/core";
import Dense from "./dense.js";

export interface AttentionPoolingConfig {
  units: number;
  maxTokens: number;
  alpha?: number;
  optimizer?: Optimizer;
  status?: StatusLayer;
  clipGradient?: number | boolean;
}

export default class AttentionPooling {
  name = "attention pooling layer";
  status: StatusLayer;
  params: number;
  inputShape: [number, number];
  outputShape: [number, number];
  units: number;
  maxTokens: number;
  alpha: number;
  optimizerName: Optimizer;
  clipGradient: number | boolean;

  private validLength = 1;
  private scorer: Dense;
  private lastInput: Matrix = mj.matrix([]);
  private lastWeights: Float32Array = new Float32Array(0);

  constructor({
    units,
    maxTokens,
    alpha = 0.01,
    optimizer = "adam",
    status = "train",
    clipGradient = 5.0,
  }: AttentionPoolingConfig) {
    this.units = units;
    this.maxTokens = maxTokens;
    this.alpha = alpha;
    this.optimizerName = optimizer;
    this.status = status;
    this.clipGradient = clipGradient;
    this.inputShape = [units, maxTokens];
    this.outputShape = [units, 1];
    this.scorer = new Dense({
      units,
      outputUnits: 1,
      activation: "linear",
      optimizer,
      alpha,
      clipGradient,
      status: "train",
    });
    this.params = this.scorer.params;
  }

  getParams(): Matrix[] {
    return this.scorer.getParams();
  }

  update(alpha: number): void {
    this.scorer.update(alpha);
  }

  setValidLength(validLength: number): void {
    if (!Number.isInteger(validLength) || validLength < 1) {
      throw new Error(`AttentionPooling.setValidLength: invalid validLength=${validLength}`);
    }
    this.validLength = Math.min(validLength, this.maxTokens);
  }

  forward(x: Matrix): Matrix {
    if (x._shape[0] !== this.units || x._shape[1] !== this.maxTokens) {
      throw new Error(
        `AttentionPooling.forward: expected [${this.units}, ${this.maxTokens}], got [${x._shape[0]}, ${x._shape[1]}]`
      );
    }

    this.lastInput = x;
    const scoreMatrix = this.scorer.forward(x);
    this.lastWeights = this.maskedSoftmax(scoreMatrix._data, this.validLength, this.maxTokens);

    const out = mj.zeros([this.units, 1]);
    for (let d = 0; d < this.units; d++) {
      let weightedSum = 0;
      const rowOffset = d * this.maxTokens;
      for (let t = 0; t < this.validLength; t++) {
        weightedSum += x._data[rowOffset + t] * this.lastWeights[t];
      }
      out._data[d] = weightedSum;
    }

    return out;
  }

  backward(_y: Matrix, err: Matrix, gradOnly = false): Matrix {
    if (err._shape[0] !== this.units || err._shape[1] !== 1) {
      throw new Error(`AttentionPooling.backward: expected err [${this.units}, 1], got [${err._shape[0]}, ${err._shape[1]}]`);
    }

    const dx = mj.zeros([this.units, this.maxTokens]);
    const weightGrad = new Float32Array(this.maxTokens);

    for (let d = 0; d < this.units; d++) {
      const rowOffset = d * this.maxTokens;
      const gradOut = err._data[d];
      for (let t = 0; t < this.validLength; t++) {
        dx._data[rowOffset + t] += gradOut * this.lastWeights[t];
        weightGrad[t] += gradOut * this.lastInput._data[rowOffset + t];
      }
    }

    let dot = 0;
    for (let t = 0; t < this.validLength; t++) {
      dot += weightGrad[t] * this.lastWeights[t];
    }

    const scoreGrad = mj.zeros([1, this.maxTokens]);
    for (let t = 0; t < this.validLength; t++) {
      scoreGrad._data[t] = this.lastWeights[t] * (weightGrad[t] - dot);
    }

    const scorerDx = this.scorer.backward(mj.matrix([]), scoreGrad, gradOnly);
    for (let i = 0; i < dx._data.length; i++) {
      dx._data[i] += scorerDx._data[i] ?? 0;
    }

    return dx;
  }

  compile(config: { alpha?: number; optimizer?: Optimizer; clipGradient?: number | boolean }): void {
    if (config.alpha !== undefined) this.alpha = config.alpha;
    if (config.optimizer !== undefined) this.optimizerName = config.optimizer;
    if (config.clipGradient !== undefined) this.clipGradient = config.clipGradient;
    this.scorer.compile({
      alpha: config.alpha,
      optimizer: config.optimizer,
      clipGradient: config.clipGradient,
    });
  }

  save() {
    return {
      name: this.name,
      status: this.status,
      units: this.units,
      maxTokens: this.maxTokens,
      alpha: this.alpha,
      optimizer: this.optimizerName,
      clipGradient: this.clipGradient,
      validLength: this.validLength,
      weight: this.scorer.weight._value,
      bias: this.scorer.bias._value,
    };
  }

  toKerasConfig() {
    return {
      class_name: "AttentionPooling",
      config: {
        units: this.units,
        maxTokens: this.maxTokens,
        name: `attention_pooling_${Math.floor(Math.random() * 1000)}`,
        trainable: true,
      }
    };
  }

  getWeightsManifest(): { name: string; shape: [number, number]; data: Float32Array }[] {
    return this.scorer.getWeightsManifest();
  }

  setWeightsFromBinary(weights: Record<string, Float32Array>): void {
    this.scorer.setWeightsFromBinary(weights);
  }

  load(data: {
    units?: number;
    maxTokens?: number;
    alpha?: number;
    optimizer?: Optimizer;
    status?: StatusLayer;
    clipGradient?: number | boolean;
    validLength?: number;
    weight?: number[][];
    bias?: number[][];
  }): void {
    this.units = data.units ?? this.units;
    this.maxTokens = data.maxTokens ?? this.maxTokens;
    this.alpha = data.alpha ?? this.alpha;
    this.optimizerName = data.optimizer ?? this.optimizerName;
    this.status = data.status ?? this.status;
    this.clipGradient = data.clipGradient ?? this.clipGradient;
    this.validLength = Math.max(1, Math.min(data.validLength ?? this.maxTokens, this.maxTokens));
    this.inputShape = [this.units, this.maxTokens];
    this.outputShape = [this.units, 1];
    this.scorer = new Dense({
      units: this.units,
      outputUnits: 1,
      activation: "linear",
      optimizer: this.optimizerName,
      alpha: this.alpha,
      clipGradient: this.clipGradient,
      status: "train",
    });
    if (data.weight && data.bias) {
      this.scorer.load(data.weight, data.bias, this.clipGradient);
    }
    this.scorer.compile({
      alpha: this.alpha,
      optimizer: this.optimizerName,
      clipGradient: this.clipGradient,
    });
    this.params = this.scorer.params;
  }

  private maskedSoftmax(scores: Float32Array, validLength: number, totalLength: number): Float32Array {
    const weights = new Float32Array(totalLength);
    let maxScore = Number.NEGATIVE_INFINITY;
    for (let t = 0; t < validLength; t++) {
      if (scores[t] > maxScore) maxScore = scores[t];
    }

    let sumExp = 0;
    for (let t = 0; t < validLength; t++) {
      const value = Math.exp(scores[t] - maxScore);
      weights[t] = value;
      sumExp += value;
    }

    if (sumExp <= 0 || !Number.isFinite(sumExp)) {
      const uniform = 1 / Math.max(1, validLength);
      for (let t = 0; t < validLength; t++) weights[t] = uniform;
      return weights;
    }

    for (let t = 0; t < validLength; t++) {
      weights[t] /= sumExp;
    }

    return weights;
  }
}
