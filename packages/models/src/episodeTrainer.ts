import {
  Matrix,
  engine,
  FitConfig,
  FitResult,
  formatLoss,
  formatProgressBar,
  formatTime,
  shuffleInPlace,
  splitTrainValidation,
} from "@oxide-js/core";
import Module from "./module.js";

export interface EpisodeRunnerContext<TModule extends Module, TInput, TTarget> {
  module: TModule;
  input: TInput;
  target: TTarget;
  episodeIndex: number;
  training: boolean;
}

export interface EpisodeRunResult {
  loss: Matrix;
  weight?: number;
}

export interface EpisodeFitConfig extends Omit<FitConfig, "batchSize"> {
  alpha?: number;
}

export type EpisodeRunner<TModule extends Module, TInput, TTarget> = (
  context: EpisodeRunnerContext<TModule, TInput, TTarget>
) => Matrix | EpisodeRunResult;

export default class EpisodeTrainer<
  TModule extends Module = Module,
  TInput = unknown,
  TTarget = unknown,
> {
  constructor(readonly module: TModule) {}

  trainEpisode(
    input: TInput,
    target: TTarget,
    runner: EpisodeRunner<TModule, TInput, TTarget>,
    alpha?: number,
    episodeIndex = 0
  ): { loss: number; weight: number } {
    this.module.zeroGrad();
    const tape = engine.startTape();
    try {
      const result = this.resolveRunResult(runner({
        module: this.module,
        input,
        target,
        episodeIndex,
        training: true,
      }));
      tape.backward(result.loss);
      this.module.step(alpha);
      return {
        loss: result.loss._data[0],
        weight: result.weight,
      };
    } finally {
      engine.endTape();
    }
  }

  fit(
    X: TInput[],
    y: TTarget[],
    epochs: number,
    runner: EpisodeRunner<TModule, TInput, TTarget>,
    config: EpisodeFitConfig = {}
  ): FitResult {
    if (!Array.isArray(X) || !Array.isArray(y) || X.length === 0 || X.length !== y.length) {
      throw new Error("EpisodeTrainer.fit: X dan y harus memiliki jumlah episode yang sama dan tidak kosong");
    }
    if (!Number.isFinite(epochs) || epochs < 1) {
      throw new Error("EpisodeTrainer.fit: epochs harus >= 1");
    }

    const {
      validationSplit = 0,
      earlyStoppingPatience = Infinity,
      shuffle = true,
      verbose = false,
      onEpochEnd = () => {},
      monitorMetric = validationSplit > 0 ? "valLoss" : "loss",
      minDelta = 0,
      mode = "min",
      alpha,
    } = config;

    if (validationSplit < 0 || validationSplit >= 1) {
      throw new Error("EpisodeTrainer.fit: validationSplit harus antara 0 dan 1");
    }
    if (earlyStoppingPatience < 0) {
      throw new Error("EpisodeTrainer.fit: earlyStoppingPatience harus >= 0");
    }

    const [trainX, valX] = splitTrainValidation(X, validationSplit);
    const [trainY, valY] = splitTrainValidation(y, validationSplit);
    if (trainX.length === 0) {
      throw new Error("EpisodeTrainer.fit: data train kosong setelah validationSplit");
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
          `\rEpoch ${epoch + 1}/${epochs} ${progress} | Loss: ....${valStr} | 0.0 episodes/s | ETA: --:--`
        );
      }

      let totalEpochLoss = 0;
      let totalEpochWeight = 0;

      for (let index = 0; index < trainIndices.length; index++) {
        const episodeIndex = trainIndices[index];
        const state = this.trainEpisode(trainX[episodeIndex], trainY[episodeIndex], runner, alpha, episodeIndex);
        totalEpochLoss += state.loss * state.weight;
        totalEpochWeight += state.weight;

        if (verbose) {
          const processed = index + 1;
          const elapsed = (Date.now() - epochStartTime) / 1000;
          const speed = processed / Math.max(elapsed, 0.001);
          const eta = (trainX.length - processed) / speed;
          const progress = formatProgressBar(processed, trainX.length);
          const valStr = validationSplit > 0 ? ` | Val Loss: ${valLoss !== undefined ? formatLoss(valLoss) : "...."}` : "";
          process.stdout.write(
            `\rEpoch ${epoch + 1}/${epochs} ${progress} | Loss: ${formatLoss(state.loss)}${valStr} | ${speed.toFixed(1)} episodes/s | ETA: ${formatTime(eta)}`
          );
        }
      }

      const epochLoss = totalEpochLoss / Math.max(totalEpochWeight, 1);
      history.loss.push(epochLoss);

      if (validationSplit > 0 && valX.length > 0) {
        valLoss = this.runValidation(valX, valY, runner, verbose);
        (history.valLoss as number[]).push(valLoss);
        this.module.train();
      }

      if (verbose) {
        const progress = formatProgressBar(trainX.length, trainX.length);
        const valStr = validationSplit > 0 ? ` | Val Loss: ${valLoss !== undefined ? formatLoss(valLoss) : "...."}` : "";
        const elapsed = (Date.now() - epochStartTime) / 1000;
        const speed = trainX.length / Math.max(elapsed, 0.001);
        process.stdout.write(
          `\rEpoch ${epoch + 1}/${epochs} ${progress} | Loss: ${formatLoss(epochLoss)}${valStr} | ${speed.toFixed(1)} episodes/s | ETA: 00:00\n`
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

  private runValidation(
    X: TInput[],
    y: TTarget[],
    runner: EpisodeRunner<TModule, TInput, TTarget>,
    verbose = false
  ): number {
    this.module.eval();
    let totalLoss = 0;
    let totalWeight = 0;

    for (let i = 0; i < X.length; i++) {
      const result = this.resolveRunResult(runner({
        module: this.module,
        input: X[i],
        target: y[i],
        episodeIndex: i,
        training: false,
      }));
      totalLoss += result.loss._data[0] * result.weight;
      totalWeight += result.weight;
    }

    const valLoss = totalLoss / Math.max(totalWeight, 1);
    if (verbose) {
      process.stdout.write(` | Val Loss: ${formatLoss(valLoss)}`);
    }
    return valLoss;
  }

  private resolveRunResult(result: Matrix | EpisodeRunResult): EpisodeRunResult & { weight: number } {
    if (result instanceof Matrix) {
      return {
        loss: result,
        weight: 1,
      };
    }
    if (!(result.loss instanceof Matrix)) {
      throw new Error("EpisodeTrainer: runner harus mengembalikan Matrix loss scalar atau { loss: Matrix, weight? }");
    }
    return {
      loss: result.loss,
      weight: result.weight ?? 1,
    };
  }
}
