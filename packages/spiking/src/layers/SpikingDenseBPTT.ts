import { BaseLayer, LayerConfig, ForwardOptions } from "@oxide-js/layers";
import { Matrix, mj } from "@oxide-js/core";
import { 
    isNativeAvailable, 
    lifStepNativeWrapper,
    maskSurrogateNativeWrapper,
    applyAddOnlyDeltaNativeWrapper
} from "../native_backend.js";
import dotProductAddOnly from "../math/dotProductAddOnly.js";

export interface SpikingDenseBPTTConfig extends LayerConfig {
  units: number;
  useBias?: boolean;
  kernelInitializer?: string;
  biasInitializer?: string;
  betaRange?: [number, number];
  thresholdRange?: [number, number];
}

export class SpikingDenseBPTT extends BaseLayer {
  public units: number;
  public useBias: boolean;
  public kernelInitializer: string;
  public biasInitializer: string;
  public betaRange: [number, number];
  public thresholdRange: [number, number];
  public beta!: Float32Array;
  public threshold!: Float32Array;

  public potentials!: Matrix;

  // History buffers for Backpropagation Through Time (BPTT)
  public historyInputs: Matrix[] = [];
  public historyPotentials: Matrix[] = [];
  public historySpikes: Matrix[] = [];
  public maxTimeSteps: number = 0;

  // Buffer untuk performa komputasi Forward
  private outSpikesDataBuffer?: Float32Array;

  public get kernel(): Matrix | undefined {
    return this.getParameter("kernel");
  }

  public get bias(): Matrix | undefined {
    return this.getParameter("bias");
  }

  constructor(config: SpikingDenseBPTTConfig) {
    super(config);
    this.units = config.units;
    this.useBias = config.useBias ?? true;
    this.kernelInitializer = config.kernelInitializer || "glorot_normal";
    this.biasInitializer = config.biasInitializer || "zeros";
    this.betaRange = config.betaRange || [0.8, 0.99];
    this.thresholdRange = config.thresholdRange || [0.5, 1.0];
  }

  public computeOutputShape(inputShape: number[]): number[] {
    const batch = inputShape[0] ?? -1;
    return [batch, this.units];
  }

  public build(inputShape: number[]): void {
    super.build(inputShape);

    const inFeatures = inputShape[inputShape.length - 1];

    const kernelVal = this.createInitializer(this.kernelInitializer, [inFeatures, this.units]);
    this.addParameter("kernel", kernelVal, true, [inFeatures, this.units]);

    if (this.useBias) {
      const biasVal = this.createInitializer(this.biasInitializer, [this.units, 1]);
      this.addParameter("bias", biasVal, true, [this.units, 1]);
    }
    
    // Inisialisasi beta (dengan pre-calculated bit-shift logic) dan threshold acak
    this.beta = new Float32Array(this.units);
    this.threshold = new Float32Array(this.units);
    for (let i = 0; i < this.units; i++) {
        // Pilih pangkat bit-shift secara acak (2 hingga 5)
        const shift = Math.floor(2 + Math.random() * 4); 
        // Pre-kalkulasi multiplier float agar loop sangat cepat tanpa perkalian ekstra
        this.beta[i] = 1.0 - (1.0 / Math.pow(2, shift)); 
        this.threshold[i] = this.thresholdRange[0] + Math.random() * (this.thresholdRange[1] - this.thresholdRange[0]);
    }
    
    // Inisialisasi state
    this.potentials = Matrix.fromFlat(new Float32Array(this.units), [1, this.units]); 
  }

  private ensurePotentialsShape(batch: number) {
    if (this.potentials._shape[0] !== batch || !this.outSpikesDataBuffer) {
       this.potentials = Matrix.fromFlat(new Float32Array(batch * this.units), [batch, this.units]);
       this.outSpikesDataBuffer = new Float32Array(batch * this.units);
    }
  }

  // Panggil fungsi ini SEBELUM satu kalimat/sequence baru mulai dimasukkan
  public resetSequence(timeSteps: number) {
    this.maxTimeSteps = timeSteps;
    this.historyInputs = new Array(timeSteps);
    this.historyPotentials = new Array(timeSteps);
    this.historySpikes = new Array(timeSteps);
    
    if (this.potentials) this.potentials._data.fill(0);
  }

  protected compute(inputs: Matrix, options?: ForwardOptions): Matrix {
      throw new Error("[SpikingDenseBPTT] Harap gunakan computeStep(inputs, t) dan resetSequence(t) untuk model BPTT, jangan gunakan compute().");
  }

  // Forward Pass untuk satu token di waktu ke-t
  public computeStep(inputs: Matrix, t: number): Matrix {
    if (t >= this.maxTimeSteps) {
        throw new Error(`[SpikingDenseBPTT] Time step ${t} melebihi batas maxTimeSteps ${this.maxTimeSteps}`);
    }

    const kernel = this.kernel!;
    const batch = inputs._shape[0];
    this.ensurePotentialsShape(batch);

    // 1. Simpan input ke dalam history buffer di index t
    this.historyInputs[t] = Matrix.fromFlat(new Float32Array(inputs._data), inputs._shape);

    // 2. Add-Only Spiking Dot Product
    let dot = dotProductAddOnly(inputs, kernel);

    if (this.useBias && this.bias) {
      mj.addBiasRow(dot, this.bias);
    }

    const outData = this.outSpikesDataBuffer!;
    outData.fill(0);
    const outSpikes = Matrix.fromFlat(outData, [batch, this.units]);

    // Buffer untuk menyimpan Potensial Membran di waktu t (SETELAH ditambahkan input, tapi SEBELUM ditembakkan/direset)
    // Ini krusial untuk evaluasi kedekatan threshold pada proses Surrogate Gradient BPTT
    const potAtT = new Float32Array(batch * this.units);

    if (isNativeAvailable()) {
        lifStepNativeWrapper(
            this.potentials._data,
            dot._data,
            outSpikes._data,
            potAtT, // argumen ke-4 di lifStepNative akan diisi oleh potensial pre-fire
            this.beta,
            this.threshold
        );
    } else {
        const potData = this.potentials._data;
        const dotData = dot._data;
        
        for (let b = 0; b < batch; b++) {
            const offset = b * this.units;
            // Tahap Leaky & Integrate
            for (let i = 0; i < this.units; i++) {
                const idx = offset + i;
                potData[idx] = Math.min((potData[idx] * this.beta[i]) + dotData[idx], 1.0);
                potAtT[idx] = potData[idx]; // Catat memori potensial di waktu t
            }
            // Tahap Fire & Reset
            for (let i = 0; i < this.units; i++) {
                const idx = offset + i;
                if (potData[idx] >= this.threshold[i]) {
                    outData[idx] = 1;
                    potData[idx] -= this.threshold[i]; // Soft Reset
                } else {
                    outData[idx] = 0;
                }
            }
        }
    }

    // 3. Simpan state (Potensial & Output Spikes) ke buffer di index t
    this.historyPotentials[t] = Matrix.fromFlat(potAtT, [batch, this.units]);
    this.historySpikes[t] = Matrix.fromFlat(new Float32Array(outSpikes._data), [batch, this.units]);

    return outSpikes;
  }

  // Backward Pass untuk belajar menggunakan BPTT
  // Dipanggil SATU KALI HANYA saat kalimat (sequence) selesai diproses
  public learnThroughTime(errorSequence: Matrix[], B: Matrix | undefined, learningRate: number = 0.01): void {
      if (this.maxTimeSteps === 0 || !this.historyInputs[0]) {
          throw new Error("[SpikingDenseBPTT] Belum ada data di memory. Panggil computeStep(inputs, t) terlebih dahulu.");
      }

      const batch = errorSequence[0]._shape[0];
      const inFeatures = this.historyInputs[0]._shape[1];
      const units = this.units;
      const kernel = this.kernel!._data;
      
      // Sinyal "Penyesalan" yang menjalar mundur dari masa depan ke masa lalu
      let temporalErrorData = new Float32Array(batch * units).fill(0);
      const windowSize = 1.0; // Lebar boxcar surrogate gradient

      // PRE-ALLOCATE buffers untuk menghilangkan BOTTLENECK Javascript (Zero Garbage Collection dalam loop)
      const totalErrorData = new Float32Array(batch * units);
      const maskedErrorData = new Float32Array(batch * units);
      const biasData = (this.useBias && this.bias) ? this.bias._data : new Float32Array(0);

      // Loop Mundur (Dari akhir kalimat ke awal kalimat)
      for (let t = this.maxTimeSteps - 1; t >= 0; t--) {
          let currentErrorData: Float32Array;
          
          if (B) {
              // Jika ini layer tersembunyi (Hidden): Evaluasi menggunakan matriks broadcast B
              let eHidden = mj.dotProduct(errorSequence[t], B, undefined, false, false);
              currentErrorData = eHidden._data;
          } else {
              // Jika ini layer Output: Error murni dari loss function
              currentErrorData = errorSequence[t]._data;
          }

          const pData = this.historyPotentials[t]._data;
          const inputData = this.historyInputs[t]._data;

          // Langkah A: Menyatukan Sinyal Spasial (Atas-Bawah) dan Temporal (Masa Depan-Masa Lalu)
          for (let i = 0; i < totalErrorData.length; i++) {
              totalErrorData[i] = currentErrorData[i] + temporalErrorData[i];
          }

          // Salin total error ke masked error, karena native wrapper akan menimpanya in-place
          maskedErrorData.set(totalErrorData);

          if (isNativeAvailable()) {
              // 1. Surrogate Mask Native (Zero Copy pointer passing ke Rust)
              maskSurrogateNativeWrapper(maskedErrorData, pData, this.threshold, windowSize);
              
              // 2. Add-Only Delta Rule Native (Zero Copy pointer passing ke Rust)
              applyAddOnlyDeltaNativeWrapper(
                  kernel,
                  biasData,
                  inputData,
                  maskedErrorData,
                  learningRate,
                  batch,
                  inFeatures,
                  units,
                  this.useBias
              );
              
              // 3. Hitung temporal error (leaky pathway)
              for (let b = 0; b < batch; b++) {
                  const offset = b * units;
                  for (let i = 0; i < units; i++) {
                      const idx = offset + i;
                      temporalErrorData[idx] = maskedErrorData[idx] * this.beta[i];
                  }
              }
          } else {
              // ============ FALLBACK JAVASCRIPT ============
              for (let b = 0; b < batch; b++) {
                  const offset = b * units;
                  for (let i = 0; i < units; i++) {
                      const idx = offset + i;
                      // Surrogate Boxcar Masking
                      if (Math.abs(pData[idx] - this.threshold[i]) <= windowSize) {
                          // maskedErrorData sudah berisi totalErrData karena di-.set() di atas
                      } else {
                          maskedErrorData[idx] = 0;
                      }
                      
                      // Menghitung Sinyal Temporal untuk dilanjutkan ke waktu t-1 (melewati jalur leaky/beta)
                      temporalErrorData[idx] = maskedErrorData[idx] * this.beta[i];
                  }
              }

              // Langkah B: Add-Only Delta Rule untuk update Bobot di waktu t
              for (let b = 0; b < batch; b++) {
                  const inOffset = b * inFeatures;
                  const errOffset = b * units;
                  for (let k = 0; k < inFeatures; k++) {
                      if (inputData[inOffset + k] > 0.5) {
                          const kOffset = k * units;
                          for (let j = 0; j < units; j++) {
                              kernel[kOffset + j] += learningRate * maskedErrorData[errOffset + j];
                              kernel[kOffset + j] = Math.max(-1.0, Math.min(1.0, kernel[kOffset + j]));
                          }
                      }
                  }
                  if (this.useBias && this.bias) {
                      for (let j = 0; j < units; j++) {
                          biasData[j] += (learningRate * maskedErrorData[errOffset + j]) / batch;
                          biasData[j] = Math.max(-1.0, Math.min(1.0, biasData[j]));
                      }
                  }
              }
          }
      }
  }

  public getConfig(): Record<string, any> {
    return {
      ...super.getConfig(),
      units: this.units,
      useBias: this.useBias,
      kernelInitializer: this.kernelInitializer,
      biasInitializer: this.biasInitializer
    };
  }
}
