import { parentPort, workerData } from "worker_threads";
import { Transformers } from "../../src/models";
import Matrix from "../../src/matrix";
import { setForceDisableNative } from "../../src/math/rust_backend";

interface WorkerInput {
  samples: number[][];
  modelPath: string;
  vocabSize: number;
  padTokenId: number;
  contextLen: number;
  modelConfig: {
    units: number;
    heads: number;
    alpha: number;
  };
}

function run(): void {
  // Menghindari crash native addon ketika dipakai paralel di benchmark worker.
  setForceDisableNative(true);

  const cfg = workerData as WorkerInput;

  const model = new Transformers({
    units: cfg.modelConfig.units,
    seqLen: cfg.contextLen,
    vocabSize: cfg.vocabSize,
    heads: cfg.modelConfig.heads,
    alpha: cfg.modelConfig.alpha,
    padTokenId: cfg.padTokenId,
  });

  try {
    model.load(cfg.modelPath);
  } catch {
    // fallback ke bobot baru
  }

  model.compile({ alpha: cfg.modelConfig.alpha, optimizer: "adam", error: "softmaxCrossEntropy" });

  const inputs = cfg.samples.map((sample) => Matrix.fromFlat(Float64Array.from(sample), [cfg.contextLen, 1]));
  const start = Date.now();
  for (const input of inputs) {
    model.forward(input);
  }
  const elapsedMs = Date.now() - start;

  parentPort?.postMessage({ elapsedMs, processed: inputs.length });
}

run();
