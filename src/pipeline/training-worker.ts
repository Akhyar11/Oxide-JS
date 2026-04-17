/**
 * training-worker.ts
 *
 * Worker thread nyata untuk TransformerPipeline.
 * Menghindari crash native addon di worker thread (pakai JS-only path).
 *
 * Protocol:
 *   Terima: ForwardMessage | TrainMessage
 *   Kirim : ForwardResult  | TrainResult  | { type: 'ready' }
 */
import { parentPort, workerData } from "worker_threads";
import { Transformers } from "../models";
import Matrix from "../matrix";
import { setForceDisableNative } from "../math/rust_backend";

// ──────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────
export interface WorkerModelConfig {
  units: number;
  seqLen: number;
  vocabSize: number;
  heads: number;
  alpha: number;
  padTokenId: number;
}

interface WorkerInit {
  modelPath: string;
  modelConfig: WorkerModelConfig;
}

type MessageIn =
  | { type: "forward"; id: number; samples: number[][] }
  | { type: "train";   id: number; samples: Array<{ input: number[]; target: number }> };

interface ForwardResult {
  type: "forward_result";
  id: number;
  /** Flat logit arrays, satu per sample (vocabSize elemen masing-masing) */
  outputs: number[][];
  elapsedMs: number;
  processed: number;
}

interface TrainResult {
  type: "train_result";
  id: number;
  loss: number;
  processed: number;
}

// ──────────────────────────────────────────────
// Inisialisasi Model
// ──────────────────────────────────────────────

// Nonaktifkan native addon agar tidak crash di worker thread
setForceDisableNative(true);

const cfg = workerData as WorkerInit;
const { modelPath, modelConfig } = cfg;

const model = new Transformers({
  units:      modelConfig.units,
  seqLen:     modelConfig.seqLen,
  vocabSize:  modelConfig.vocabSize,
  heads:      modelConfig.heads,
  alpha:      modelConfig.alpha,
  padTokenId: modelConfig.padTokenId,
});

try {
  model.load(modelPath);
} catch {
  // Fallback ke bobot baru jika file belum ada
}

model.compile({
  alpha:     modelConfig.alpha,
  optimizer: "adam",
  error:     "softmaxCrossEntropy",
});

// Beri tahu parent bahwa worker sudah siap
parentPort!.postMessage({ type: "ready" });

// ──────────────────────────────────────────────
// Message Handler
// ──────────────────────────────────────────────
parentPort!.on("message", (msg: MessageIn) => {
  const { seqLen, vocabSize } = modelConfig;

  // ── FORWARD ──────────────────────────────────
  if (msg.type === "forward") {
    const inputs = msg.samples.map((s) =>
      Matrix.fromFlat(Float32Array.from(s), [seqLen, 1])
    );

    const start = Date.now();
    const outputs: number[][] = inputs.map((inp) => {
      const out = model.forward(inp);
      return Array.from(out._data.slice(0, vocabSize));
    });
    const elapsedMs = Date.now() - start;

    parentPort!.postMessage({
      type: "forward_result",
      id: msg.id,
      outputs,
      elapsedMs,
      processed: inputs.length,
    } as ForwardResult);
    return;
  }

  // ── TRAIN ────────────────────────────────────
  if (msg.type === "train") {
    let lossSum = 0;

    for (const s of msg.samples) {
      const input  = Matrix.fromFlat(Float32Array.from(s.input), [seqLen, 1]);
      const target = Matrix.fromFlat(new Float32Array([s.target]), [1, 1]);
      model.forward(input);
      model.backward(target);
      lossSum += model.loss;
    }

    const avgLoss = msg.samples.length > 0 ? lossSum / msg.samples.length : 0;

    parentPort!.postMessage({
      type: "train_result",
      id: msg.id,
      loss: avgLoss,
      processed: msg.samples.length,
    } as TrainResult);
    return;
  }
});
