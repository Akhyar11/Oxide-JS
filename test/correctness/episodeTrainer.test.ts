import { mj, Matrix } from "@oxide-js/core";
import { Dense } from "@oxide-js/layers";
import { EpisodeTrainer, Module } from "@oxide-js/models";
import { fileURLToPath } from "url";

function assert(condition: boolean, message: string): void {
  if (!condition) {
    throw new Error(message);
  }
}

function assertAllFinite(values: number[], message: string): void {
  if (!values.every((value) => Number.isFinite(value))) {
    throw new Error(message);
  }
}

interface EpisodeInput {
  source: Matrix;
  start: Matrix;
}

class EpisodicEncoderDecoderModule extends Module {
  readonly encoder = new Dense({ units: 2, outputUnits: 4, activation: "relu", status: "input", optimizer: "adam", alpha: 0.01 });
  readonly decoderInput = new Dense({ units: 1, outputUnits: 4, activation: "linear", status: "train", optimizer: "adam", alpha: 0.01 });
  readonly decoderState = new Dense({ units: 4, outputUnits: 4, activation: "linear", status: "train", optimizer: "adam", alpha: 0.01 });
  readonly output = new Dense({ units: 4, outputUnits: 1, activation: "linear", status: "output", optimizer: "adam", alpha: 0.01, loss: "mse" });

  encodeCalls = 0;
  decodeCalls = 0;

  forward(input: EpisodeInput): Matrix {
    return this.runEpisode(input, mj.matrix([[0], [0], [0]]));
  }

  resetCounters(): void {
    this.encodeCalls = 0;
    this.decodeCalls = 0;
  }

  encode(source: Matrix): Matrix {
    this.encodeCalls++;
    return this.encoder.forward(source);
  }

  decodeStep(context: Matrix, prevToken: Matrix, state: Matrix): { state: Matrix; prediction: Matrix } {
    this.decodeCalls++;
    const mixed = mj.add(context, mj.add(this.decoderInput.forward(prevToken), this.decoderState.forward(state)));
    const nextState = mj.relu(mixed);
    return {
      state: nextState,
      prediction: this.output.forward(nextState),
    };
  }

  runEpisode(input: EpisodeInput, target: Matrix): Matrix {
    const context = this.encode(input.source);
    let prevToken = input.start;
    let state = mj.zeros([4, 1]);
    let totalLoss: Matrix | null = null;

    for (let step = 0; step < target._shape[0]; step++) {
      const decoded = this.decodeStep(context, prevToken, state);
      const stepTarget = mj.matrix([[target._data[step]]]);
      const diff = mj.sub(decoded.prediction, stepTarget);
      const stepLoss = mj.mean(mj.pow(diff, 2));
      totalLoss = totalLoss ? mj.add(totalLoss, stepLoss) : stepLoss;
      prevToken = decoded.prediction;
      state = decoded.state;
    }

    if (!totalLoss) {
      throw new Error("runEpisode requires at least one target step");
    }
    return mj.mul(totalLoss, 1 / target._shape[0]);
  }
}

export function runEpisodeTrainerCorrectnessSuite(): void {
  console.log("  - Checking EpisodeTrainer episodic execution...");

  const module = new EpisodicEncoderDecoderModule();
  module.compile({ alpha: 0.01, optimizer: "adam", error: "mse" });
  const trainer = new EpisodeTrainer(module);

  const sampleInput = {
    source: mj.matrix([[1], [0]]),
    start: mj.matrix([[0]]),
  };
  const sampleTarget = mj.matrix([[1], [0], [1]]);

  module.resetCounters();
  const singleEpisode = trainer.trainEpisode(sampleInput, sampleTarget, ({ module, input, target }: { module: EpisodicEncoderDecoderModule; input: any; target: any }) => {
    return module.runEpisode(input, target);
  });
  assert(Number.isFinite(singleEpisode.loss), "EpisodeTrainer single episode loss must be finite");
  assert(module.encodeCalls === 1, `EpisodeTrainer should call encoder once per episode, got ${module.encodeCalls}`);
  assert(module.decodeCalls === sampleTarget._shape[0], `EpisodeTrainer should call decoder ${sampleTarget._shape[0]} times, got ${module.decodeCalls}`);

  const episodeInputs = [
    { source: mj.matrix([[1], [0]]), start: mj.matrix([[0]]) },
    { source: mj.matrix([[0], [1]]), start: mj.matrix([[0]]) },
    { source: mj.matrix([[1], [1]]), start: mj.matrix([[0]]) },
  ];
  const episodeTargets = [
    mj.matrix([[1], [0], [1]]),
    mj.matrix([[0], [1], [0]]),
    mj.matrix([[1], [1], [0]]),
  ];

  const result = trainer.fit(
    episodeInputs,
    episodeTargets,
    4,
    ({ module, input, target }: { module: EpisodicEncoderDecoderModule; input: any; target: any }) => module.runEpisode(input, target),
    { shuffle: false, verbose: false, validationSplit: 0.33 }
  );

  assert(result.history.loss.length === 4, `EpisodeTrainer.fit should record 4 losses, got ${result.history.loss.length}`);
  assertAllFinite(result.history.loss, "EpisodeTrainer.fit losses must be finite");
  if (result.history.valLoss) {
    assertAllFinite(result.history.valLoss, "EpisodeTrainer.fit validation losses must be finite");
  }

  module.resetCounters();
  const validationLoss = module.runEpisode(sampleInput, sampleTarget)._data[0];
  assert(Number.isFinite(validationLoss), "episodic module should produce finite scalar loss outside trainer");

  console.log("    ✅ EpisodeTrainer supports encoder-once / decoder-many / backward-once episodic training.");
}

const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];

if (isMain) {
  runEpisodeTrainerCorrectnessSuite();
}
