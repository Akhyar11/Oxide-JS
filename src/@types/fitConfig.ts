export interface FitConfig {
  batchSize?: number;
  validationSplit?: number;
  earlyStoppingPatience?: number;
  shuffle?: boolean;
  verbose?: boolean;
  onEpochEnd?: (epoch: number, loss: number, valLoss?: number) => void;
  monitorMetric?: "loss" | "valLoss";
  minDelta?: number;
  mode?: "min" | "max";
}

export interface FitResult {
  history: {
    loss: number[];
    valLoss?: number[];
  };
  bestEpoch: number;
  bestLoss: number;
  stoppedEarly: boolean;
  stoppingEpoch?: number;
}
