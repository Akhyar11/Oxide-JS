import Matrix from "../matrix/index.js";
import ones from "../math/ones.js";

type GradientFunc = (grad: Matrix) => void;

interface TapeNode {
  inputs: Matrix[];
  outputs: Matrix[];
  backward: GradientFunc;
  inputSnapshots: Float32Array[];
  outputSnapshots: Float32Array[];
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
   * Catat sebuah operasi ke dalam tape
   */
  record(inputs: Matrix[], outputs: Matrix[], backward: GradientFunc) {
    if (!this.active) return;
    
    // Simpan snapshot data saat ini agar aman dari modifikasi in-place nantinya
    const inputSnapshots = inputs.map(m => new Float32Array(m._data));
    const outputSnapshots = outputs.map(m => new Float32Array(m._data));

    this.nodes.push({ 
      inputs, 
      outputs, 
      backward,
      inputSnapshots,
      outputSnapshots
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
      // Ambil gradien dari output pertama (asumsi mayoritas operasi punya 1 output)
      const outGrad = node.outputs[0].grad;
      
      if (outGrad) {
        // --- DATA SWAPPING (Time Travel) ---
        // Simpan referensi ke data asli (saat ini)
        const currentInputBuffers = node.inputs.map(m => m._data);
        const currentOutputBuffers = node.outputs.map(m => m._data);

        // Pasang data snapshot (saat operasi direkam)
        node.inputs.forEach((m, idx) => m._data = node.inputSnapshots[idx]);
        node.outputs.forEach((m, idx) => m._data = node.outputSnapshots[idx]);

        // Hitung gradien menggunakan data historis
        node.backward(outGrad);

        // Kembalikan data ke kondisi asli agar program utama tetap berjalan normal
        node.inputs.forEach((m, idx) => m._data = currentInputBuffers[idx]);
        node.outputs.forEach((m, idx) => m._data = currentOutputBuffers[idx]);
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
