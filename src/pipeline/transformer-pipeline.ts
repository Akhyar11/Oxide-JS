import Transformers from "../models/transformers";
import Matrix from "../matrix";

/**
 * Lightweight pipeline scheduler for transformer inference/training loops.
 *
 * Catatan:
 * - Ini bukan distributed pipeline antar-process.
 * - Fokusnya menjaga alur micro-batch rapi, bisa diintegrasikan ke training loop.
 */
export class TransformerPipeline {
  private readonly model: Transformers;
  private readonly numStages: number;
  private readonly maxInflightMicroBatches: number;

  constructor(model: Transformers, numStages: number, maxInflightMicroBatches: number) {
    this.model = model;
    this.numStages = Math.max(1, Math.floor(numStages));
    this.maxInflightMicroBatches = Math.max(1, Math.floor(maxInflightMicroBatches));
  }

  /**
   * Forward single sample melalui pipeline scheduler.
   * Saat ini eksekusi numeriknya tetap di model.forward; scheduler menjaga API pipeline tetap konsisten.
   */
  async forwardPipeline(input: Matrix): Promise<Matrix> {
    // Simulasi boundary stage agar alur async tetap stabil tanpa worker-thread overhead.
    for (let i = 0; i < this.numStages - 1; i++) {
      await Promise.resolve();
    }
    return this.model.forward(input);
  }

  /**
   * Forward beberapa micro-batch dengan batas inflight untuk stabilitas memori.
   */
  async forwardMicroBatches(inputs: Matrix[]): Promise<Matrix[]> {
    const outputs: Matrix[] = new Array(inputs.length);

    for (let i = 0; i < inputs.length; i += this.maxInflightMicroBatches) {
      const chunk = inputs.slice(i, i + this.maxInflightMicroBatches);
      const chunkOutputs = await Promise.all(chunk.map((input) => this.forwardPipeline(input)));
      for (let j = 0; j < chunkOutputs.length; j++) {
        outputs[i + j] = chunkOutputs[j];
      }
    }

    return outputs;
  }

  async shutdown(): Promise<void> {
    // no-op: tidak ada resource eksternal yang perlu ditutup.
    await Promise.resolve();
  }
}

export default TransformerPipeline;
