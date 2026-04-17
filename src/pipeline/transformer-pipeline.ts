/**
 * transformer-pipeline.ts  — REAL multi-thread pipeline
 *
 * Sebelumnya: await Promise.resolve() saja (bohongan, semua di main thread).
 * Sekarang  : Worker Threads nyata via worker_threads API.
 *
 * Arsitektur — Data Parallelism:
 *   Main Thread menerima batch → split ke N workers → workers jalan PARALEL
 *   → kumpulkan hasil → return ke caller
 */
import * as path from "path";
import * as os from "os";
import { Worker } from "worker_threads";
import Transformers from "../models/transformers";
import Matrix from "../matrix";
import type { WorkerModelConfig } from "./training-worker";

// ──────────────────────────────────────────────
// Types (internal)
// ──────────────────────────────────────────────
interface WorkerSlot {
  worker: Worker;
  index: number;
}

// ──────────────────────────────────────────────
// TransformerPipeline
// ──────────────────────────────────────────────
export class TransformerPipeline {
  private readonly model: Transformers;
  private readonly numWorkers: number;
  private readonly microBatchSize: number;

  private slots: WorkerSlot[] = [];
  private resolvers = new Map<number, (v: any) => void>();
  private msgId = 0;
  private _initialized = false;

  /**
   * @param model           - Model utama di main thread (untuk fallback & backward)
   * @param numWorkers      - Jumlah worker thread (dibatasi oleh jumlah CPU)
   * @param microBatchSize  - Ukuran micro-batch yang dikirim ke setiap worker
   */
  constructor(model: Transformers, numWorkers: number, microBatchSize: number) {
    this.model = model;
    this.numWorkers = Math.max(1, Math.min(numWorkers, os.cpus().length));
    this.microBatchSize = Math.max(1, microBatchSize);
  }

  // ────────────────────────────────────────────
  // Init: spawn workers
  // ────────────────────────────────────────────
  /**
   * Spawn N worker threads dan tunggu semua READY.
   * WAJIB dipanggil sebelum forwardMicroBatches / trainBatch.
   */
  async init(modelPath: string, modelConfig: WorkerModelConfig): Promise<void> {
    if (this._initialized) return;

    // Path ke training-worker.ts (dijalankan via ts-node/register)
    const workerScript = path.join(__dirname, "training-worker.ts");
    const readyPromises: Promise<void>[] = [];

    for (let i = 0; i < this.numWorkers; i++) {
      const slot: WorkerSlot = { worker: null!, index: i };

      const readyP = new Promise<void>((onReady) => {
        const worker = new Worker(workerScript, {
          workerData: { modelPath, modelConfig } as { modelPath: string; modelConfig: WorkerModelConfig },
          execArgv: ["-r", "ts-node/register"],
          // Matikan native addon di worker agar tidak crash
          env: { ...process.env, ML_DISABLE_NATIVE: "1" },
        });

        let workerReady = false;

        worker.on("message", (msg: any) => {
          // Sinyal ready pertama kali
          if (msg.type === "ready" && !workerReady) {
            workerReady = true;
            onReady();
            return;
          }
          // Resolve pending promise untuk request lain
          const cb = this.resolvers.get(msg.id);
          if (cb !== undefined) {
            this.resolvers.delete(msg.id);
            cb(msg);
          }
        });

        worker.on("error", (err) => {
          console.error(`[Pipeline] Worker ${i} error: ${err.message}`);
        });

        slot.worker = worker;
      });

      this.slots.push(slot);
      readyPromises.push(readyP);
    }

    await Promise.all(readyPromises);
    this._initialized = true;
    console.log(`[Pipeline] ${this.numWorkers} worker(s) siap.`);
  }

  // ────────────────────────────────────────────
  // Private: send & wait
  // ────────────────────────────────────────────
  private sendAndWait(workerIdx: number, msg: object): Promise<any> {
    return new Promise((resolve) => {
      this.resolvers.set((msg as any).id, resolve);
      this.slots[workerIdx].worker.postMessage(msg);
    });
  }

  // ────────────────────────────────────────────
  // Forward single sample (untuk kompatibilitas API lama)
  // ────────────────────────────────────────────
  async forwardPipeline(input: Matrix): Promise<Matrix> {
    // Single sample → gunakan main thread agar backward tetap bisa dipakai
    return this.model.forward(input);
  }

  // ────────────────────────────────────────────
  // Forward micro-batches: BENAR-BENAR PARALEL
  // ────────────────────────────────────────────
  /**
   * Distribusi inputs ke semua workers secara paralel.
   * Workers menjalankan model.forward() pada chunk mereka BERSAMAAN.
   * Return: Matrix[] hasil logit (direkonstruksi dari worker output).
   */
  async forwardMicroBatches(inputs: Matrix[]): Promise<Matrix[]> {
    if (!this._initialized || this.slots.length === 0) {
      // Fallback: main thread
      return inputs.map((inp) => this.model.forward(inp));
    }

    // Bagi inputs ke N chunks
    const nWorkers = this.slots.length;
    const chunks: number[][][] = Array.from({ length: nWorkers }, () => []);
    const chunkOrder: number[] = []; // worker index untuk setiap input

    inputs.forEach((inp, i) => {
      const wIdx = i % nWorkers;
      chunks[wIdx].push(Array.from(inp._data));
      chunkOrder.push(wIdx);
    });

    // Kirim semua chunk ke workers BERSAMAAN (ini yang beneran paralel)
    const promises = this.slots.map((_, wIdx) => {
      if (chunks[wIdx].length === 0) {
        return Promise.resolve({ outputs: [] as number[][], processed: 0 });
      }
      const id = this.msgId++;
      return this.sendAndWait(wIdx, { type: "forward", id, samples: chunks[wIdx] });
    });

    // Tunggu semua workers selesai
    const results = await Promise.all(promises);

    // Rekonstruksi Matrix[] sesuai urutan input asli
    const vocabSize = this.model.vocabSize;
    const workerCounters = new Array(nWorkers).fill(0);
    const matrices: Matrix[] = inputs.map((_, i) => {
      const wIdx = chunkOrder[i];
      const localIdx = workerCounters[wIdx]++;
      const flat = results[wIdx]?.outputs?.[localIdx];
      if (!flat) {
        // Fallback kalau worker tidak mengembalikan output
        return this.model.forward(inputs[i]);
      }
      return Matrix.fromFlat(Float32Array.from(flat), [vocabSize, 1]);
    });

    return matrices;
  }

  // ────────────────────────────────────────────
  // Train batch: BENAR-BENAR PARALEL
  // ────────────────────────────────────────────
  /**
   * Split batch ke workers, tiap worker jalankan forward+backward pada chunk-nya.
   * Return: rata-rata loss dari semua workers.
   *
   * Catatan: setiap worker punya model-nya sendiri (copy bobot dari disk).
   * Ini adalah Asynchronous/Independent Data Parallelism.
   */
  async trainBatch(
    samples: Array<{ input: Matrix; target: Matrix }>
  ): Promise<number> {
    if (!this._initialized || this.slots.length === 0 || samples.length === 0) {
      // Fallback: main thread sequential
      let lossSum = 0;
      for (const s of samples) {
        this.model.forward(s.input);
        this.model.backward(s.target);
        lossSum += this.model.loss;
      }
      return samples.length > 0 ? lossSum / samples.length : 0;
    }

    const nWorkers = this.slots.length;
    const chunkSize = Math.ceil(samples.length / nWorkers);
    const promises: Promise<any>[] = [];

    for (let i = 0; i < nWorkers; i++) {
      const chunk = samples.slice(i * chunkSize, (i + 1) * chunkSize);
      if (chunk.length === 0) continue;

      const id = this.msgId++;
      const payload = chunk.map((s) => ({
        input:  Array.from(s.input._data),
        target: s.target._data[0],
      }));

      promises.push(
        this.sendAndWait(i, { type: "train", id, samples: payload })
      );
    }

    const results = await Promise.all(promises);

    // Rata-rata loss berbobot (weighted average by processed count)
    let totalLoss    = 0;
    let totalSamples = 0;
    for (const r of results) {
      const p = r?.processed ?? 0;
      totalLoss    += (r?.loss ?? 0) * p;
      totalSamples += p;
    }

    return totalSamples > 0 ? totalLoss / totalSamples : 0;
  }

  // ────────────────────────────────────────────
  // Shutdown
  // ────────────────────────────────────────────
  async shutdown(): Promise<void> {
    await Promise.all(this.slots.map((s) => s.worker.terminate()));
    this.slots = [];
    this.resolvers.clear();
    this._initialized = false;
  }

  get initialized(): boolean {
    return this._initialized;
  }
}

export default TransformerPipeline;
