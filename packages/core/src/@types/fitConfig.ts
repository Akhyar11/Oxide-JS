export type PaddingSide = "left" | "right";

export interface FitConfig {
  batchSize?: number;
  /** Run training batches through the Gradient Tape instead of the model's
   *  manual backward path. Default: false */
  autodiff?: boolean;
  validationSplit?: number;
  earlyStoppingPatience?: number;
  shuffle?: boolean;
  verbose?: boolean;
  onEpochEnd?: (epoch: number, loss: number, valLoss?: number) => void;
  monitorMetric?: "loss" | "valLoss";
  minDelta?: number;
  mode?: "min" | "max";
  /** Whether to dynamically trim PAD tokens from each batch before forward/backward.
   *  Only active for full-sequence targets (Y.shape[0] === X.shape[0]) and models
   *  that expose `getPadTokenId()` and `setPositionOffset()` (e.g. Transformers).
   *  Default: true */
  trimPadding?: boolean;
  /** Which side the PAD tokens are on.
   *  - "right" (default): trailing PADs are trimmed; positionOffset stays 0.
   *  - "left": leading PADs are trimmed; positionOffset is set so absolute positional
   *    encoding of real tokens remains unchanged.
   *  Default: "right" */
  paddingSide?: PaddingSide;
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
