import Matrix from "../matrix/index.js";
import ones from "../math/ones.js";

type GradientFunc = (grad: Matrix, outputGrads?: Array<Matrix | null>) => void;

interface TapeNode {
  inputs: Matrix[];
  outputs: Matrix[];
  backward: GradientFunc;
  inputSnapshots: Float32Array[];
  outputSnapshots: Float32Array[];
  inputShapes: [number, number][];
  outputShapes: [number, number][];
}

export default class Tape {
  private nodes: TapeNode[] = [];
  private active: boolean = false;

  /**
   * Mulai merekam operasi
   */
  watch() {
    this.active = true;
    this.nodes = [];
  }

  /**
   * Berhenti merekam
   */
  stop() {
    this.active = false;
  }

  /**
   * Jalankan blok kode tanpa merekam gradien
   */
  noGrad<T>(fn: () => T): T {
    const prevActive = this.active;
    this.active = false;
    try {
      return fn();
    } finally {
      this.active = prevActive;
    }
  }

  isActive(): boolean {
    return this.active;
  }

  /**
   * Catat sebuah operasi ke dalam tape
   * @param inputs Matrix input
   * @param outputs Matrix output
   * @param backward Fungsi gradien
   * @param options Opsi snapshot untuk efisiensi
   */
  record(
    inputs: Matrix[], 
    outputs: Matrix[], 
    backward: GradientFunc,
    options: { saveInput?: boolean; saveOutput?: boolean } = { saveInput: true, saveOutput: true }
  ) {
    if (!this.active) return;

    for (const output of outputs) {
      output.grad = null;
    }
    
    // Simpan snapshot hanya jika benar-benar dibutuhkan oleh backward pass
    const inputSnapshots = options.saveInput ? inputs.map(m => new Float32Array(m._data)) : [];
    const outputSnapshots = options.saveOutput ? outputs.map(m => new Float32Array(m._data)) : [];
    const inputShapes = options.saveInput ? inputs.map(m => [m._shape[0], m._shape[1]] as [number, number]) : [];
    const outputShapes = options.saveOutput ? outputs.map(m => [m._shape[0], m._shape[1]] as [number, number]) : [];

    this.nodes.push({ 
      inputs, 
      outputs, 
      backward,
      inputSnapshots,
      outputSnapshots,
      inputShapes,
      outputShapes,
    });
  }

  /**
   * Jalankan backpropagation dari sebuah matrix output (loss)
   */
  backward(loss: Matrix) {
    const wasActive = this.active;
    this.active = false; // Nonaktifkan agar operasi gradien tidak terekam

    // Inisialisasi gradien loss dengan 1.0 (jika scalar) atau matrix ones
    if (!loss.grad) {
      loss.grad = ones(loss._shape);
    }

    // Jalankan nodes secara terbalik (LIFO)
    for (let i = this.nodes.length - 1; i >= 0; i--) {
      const node = this.nodes[i];
      const outputGrads = node.outputs.map((output) => output.grad);
      const outGrad = outputGrads.find((grad): grad is Matrix => grad !== null) ?? null;

      if (outGrad) {
        const hasInputSnapshots = node.inputSnapshots.length > 0;
        const hasOutputSnapshots = node.outputSnapshots.length > 0;

        // --- DATA SWAPPING (Time Travel) ---
        // Simpan referensi ke data asli (saat ini) jika ada snapshot
        const currentInputBuffers = hasInputSnapshots ? node.inputs.map(m => m._data) : [];
        const currentOutputBuffers = hasOutputSnapshots ? node.outputs.map(m => m._data) : [];
        const currentInputShapes = hasInputSnapshots ? node.inputs.map(m => [m._shape[0], m._shape[1]] as [number, number]) : [];
        const currentOutputShapes = hasOutputSnapshots ? node.outputs.map(m => [m._shape[0], m._shape[1]] as [number, number]) : [];

        // Pasang data snapshot (saat operasi direkam)
        if (hasInputSnapshots) {
          node.inputs.forEach((m, idx) => {
            m._data = node.inputSnapshots[idx];
            m._shape = [...node.inputShapes[idx]] as [number, number];
          });
        }
        if (hasOutputSnapshots) {
          node.outputs.forEach((m, idx) => {
            m._data = node.outputSnapshots[idx];
            m._shape = [...node.outputShapes[idx]] as [number, number];
          });
        }

        // Hitung gradien menggunakan data historis
        node.backward(outGrad, outputGrads);

        // Kembalikan data ke kondisi asli agar program utama tetap berjalan normal
        if (hasInputSnapshots) {
          node.inputs.forEach((m, idx) => {
            m._data = currentInputBuffers[idx];
            m._shape = currentInputShapes[idx];
          });
        }
        if (hasOutputSnapshots) {
          node.outputs.forEach((m, idx) => {
            m._data = currentOutputBuffers[idx];
            m._shape = currentOutputShapes[idx];
          });
        }
      }
    }
    
    this.active = false;
    this.nodes = [];
  }
}

// Tambahkan properti grad ke Matrix via deklarasi modul (jika perlu) atau langsung di class
declare module "../matrix/index.js" {
  interface Matrix {
    grad?: Matrix;
  }
}
