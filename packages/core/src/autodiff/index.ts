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

  isActive(): boolean {
    return this.active;
  }

  /**
   * Catat sebuah operasi ke dalam tape
   */
  record(inputs: Matrix[], outputs: Matrix[], backward: GradientFunc) {
    if (!this.active) return;
    
    // Simpan snapshot data saat ini agar aman dari modifikasi in-place nantinya
    const inputSnapshots = inputs.map(m => new Float32Array(m._data));
    const outputSnapshots = outputs.map(m => new Float32Array(m._data));
    const inputShapes = inputs.map(m => [m._shape[0], m._shape[1]] as [number, number]);
    const outputShapes = outputs.map(m => [m._shape[0], m._shape[1]] as [number, number]);

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
        // --- DATA SWAPPING (Time Travel) ---
        // Simpan referensi ke data asli (saat ini)
        const currentInputBuffers = node.inputs.map(m => m._data);
        const currentOutputBuffers = node.outputs.map(m => m._data);
        const currentInputShapes = node.inputs.map(m => [m._shape[0], m._shape[1]] as [number, number]);
        const currentOutputShapes = node.outputs.map(m => [m._shape[0], m._shape[1]] as [number, number]);

        // Pasang data snapshot (saat operasi direkam)
        node.inputs.forEach((m, idx) => {
          m._data = node.inputSnapshots[idx];
          m._shape = [...node.inputShapes[idx]] as [number, number];
        });
        node.outputs.forEach((m, idx) => {
          m._data = node.outputSnapshots[idx];
          m._shape = [...node.outputShapes[idx]] as [number, number];
        });

        // Hitung gradien menggunakan data historis
        node.backward(outGrad, outputGrads);

        // Kembalikan data ke kondisi asli agar program utama tetap berjalan normal
        node.inputs.forEach((m, idx) => {
          m._data = currentInputBuffers[idx];
          m._shape = currentInputShapes[idx];
        });
        node.outputs.forEach((m, idx) => {
          m._data = currentOutputBuffers[idx];
          m._shape = currentOutputShapes[idx];
        });
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
