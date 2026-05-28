import { Matrix } from "@oxide-js/core";

export type ModelMode = "training" | "inference";

export interface ModelConfig {
  name?: string;
  trainable?: boolean;
}

/**
 * Loss function type: can be a string name, raw function, or object with compute/forward methods.
 */
export type LossLike =
  | string
  | ((yPred: Matrix, yTrue: Matrix) => any)
  | {
      forward?: (yPred: Matrix, yTrue: Matrix) => any;
      compute?: (yPred: Matrix, yTrue: Matrix) => any;
    };

/**
 * Optimizer type: can be a string name or object with step/update methods.
 */
export type OptimizerLike =
  | string
  | {
      step?: (params: Matrix[]) => void;
      update?: (params: Matrix[]) => void;
    };

/**
 * Metric type: can be a string name, raw function, or stateful metric object.
 */
export type MetricLike =
  | string
  | ((yPred: Matrix, yTrue: Matrix) => number)
  | {
      name?: string;
      compute?: (yPred: Matrix, yTrue: Matrix) => number;
      updateState?: (yPred: Matrix, yTrue: Matrix) => void;
      result?: () => number;
      resetState?: () => void;
    };

/**
 * Compile configuration with loss, optimizer, and metrics.
 */
export interface CompileConfig {
  optimizer: OptimizerLike;
  loss: LossLike;
  metrics?: MetricLike[];
  learningRate?: number;
}

/**
 * Training configuration for fit().
 */
export interface FitConfig {
  epochs?: number;
  batchSize?: number;
  shuffle?: boolean;
  validationSplit?: number;
  validationData?: [Matrix, Matrix];
  verbose?: 0 | 1 | 2;
  callbacks?: Callback[];
}

/**
 * History record for each epoch.
 */
export interface HistoryRecord {
  epoch: number;
  loss?: number;
  val_loss?: number;
  metrics?: Record<string, number>;
  val_metrics?: Record<string, number>;
  [key: string]: unknown;
}

/**
 * Mini-batch of data.
 */
export interface Batch {
  x: Matrix;
  y: Matrix;
}

/**
 * Callback logs passed to callback hooks.
 */
export interface CallbackLogs {
  epoch?: number;
  batch?: number;
  loss?: number;
  val_loss?: number;
  metrics?: Record<string, number>;
  val_metrics?: Record<string, number>;
  [key: string]: unknown;
}

/**
 * Callback interface for training hooks.
 */
export interface Callback {
  onTrainBegin?(logs?: CallbackLogs): void | Promise<void>;
  onTrainEnd?(logs?: CallbackLogs): void | Promise<void>;
  onEpochBegin?(epoch: number, logs?: CallbackLogs): void | Promise<void>;
  onEpochEnd?(epoch: number, logs?: CallbackLogs): void | Promise<void>;
  onBatchBegin?(batch: number, logs?: CallbackLogs): void | Promise<void>;
  onBatchEnd?(batch: number, logs?: CallbackLogs): void | Promise<void>;
  shouldStop?: boolean;
}

export interface WeightData {
  name: string;
  shape: number[];
  physicalShape?: number[];
  dtype: "float32" | "float64";
  data: Float32Array | number[];
}

export interface ModelSummaryRow {
  name: string;
  type: string;
  outputShape: string;
  paramCount: number;
  trainable: boolean;
}

export interface SerializedModel {
  class_name: string;
  name: string;
  trainable: boolean;
  config: Record<string, any>;
  layers: any[];
  weights: WeightData[];
}
