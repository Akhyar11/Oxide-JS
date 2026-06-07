import { BaseLayer, LayerConfig, ForwardOptions } from "@oxide-js/layers";
import { Matrix, mj } from "@oxide-js/core";
import { isNativeAvailable, lifStepNativeWrapper } from "../native_backend.js";
import dotProductAddOnly from "../math/dotProductAddOnly.js";

export interface SpikingSelfAttentionConfig extends LayerConfig {
  d_model: number;
  sequenceLength: number;
  kernelInitializer?: string;
  betaRange?: [number, number];
  thresholdRange?: [number, number];
}

export class SpikingSelfAttention extends BaseLayer {
  public d_model: number;
  public sequenceLength: number;
  public kernelInitializer: string;
  public betaRange: [number, number];
  public thresholdRange: [number, number];

  // Q, K, V kernels
  public get kernelQ(): Matrix | undefined { return this.getParameter("kernelQ"); }
  public get kernelK(): Matrix | undefined { return this.getParameter("kernelK"); }
  public get kernelV(): Matrix | undefined { return this.getParameter("kernelV"); }

  // LIF state untuk Q, K, V (opsional, jika ingin akumulasi temporal)
  public betaQKV!: Float32Array;
  public thresholdQKV!: Float32Array;
  public potentialsQ!: Matrix;
  public potentialsK!: Matrix;
  public potentialsV!: Matrix;

  // LIF state untuk Attention Scores (Pengganti Softmax)
  public betaScores!: Float32Array;
  public thresholdScores!: Float32Array;
  public potentialsScores!: Matrix;

  // Cache input untuk Local Learning
  public lastInputs?: Matrix;

  constructor(config: SpikingSelfAttentionConfig) {
    super(config);
    this.d_model = config.d_model;
    this.sequenceLength = config.sequenceLength;
    this.kernelInitializer = config.kernelInitializer || "glorot_normal";
    this.betaRange = config.betaRange || [0.8, 0.99];
    this.thresholdRange = config.thresholdRange || [0.1, 0.3];
  }

  public computeOutputShape(inputShape: number[]): number[] {
    const batch = inputShape[0] ?? -1;
    // Asumsi input shape: [batch * seqLen, d_model]
    return [batch, this.d_model]; // Actually [batch * seqLen, d_model]
  }

  public build(inputShape: number[]): void {
    super.build(inputShape);

    const inFeatures = inputShape[inputShape.length - 1]; // Seharusnya sama dengan d_model

    // 1. Inisialisasi Bobot Q, K, V
    this.addParameter("kernelQ", this.createInitializer(this.kernelInitializer, [inFeatures, this.d_model]), true, [inFeatures, this.d_model]);
    this.addParameter("kernelK", this.createInitializer(this.kernelInitializer, [inFeatures, this.d_model]), true, [inFeatures, this.d_model]);
    this.addParameter("kernelV", this.createInitializer(this.kernelInitializer, [inFeatures, this.d_model]), true, [inFeatures, this.d_model]);

    // OPTIMIZATION: Scale up initial weights so neurons actually spike (prevent Layer 2 death)
    const scale = Math.sqrt(inFeatures);
    const kQ = this.kernelQ!._data;
    const kK = this.kernelK!._data;
    const kV = this.kernelV!._data;
    for(let i = 0; i < kQ.length; i++) {
        kQ[i] *= scale;
        kK[i] *= scale;
        kV[i] *= scale;
    }

    // 2. Inisialisasi parameter LIF untuk Q, K, V
    this.betaQKV = new Float32Array(this.d_model);
    this.thresholdQKV = new Float32Array(this.d_model);
    for (let i = 0; i < this.d_model; i++) {
        this.betaQKV[i] = this.betaRange[0] + Math.random() * (this.betaRange[1] - this.betaRange[0]);
        this.thresholdQKV[i] = this.thresholdRange[0] + Math.random() * (this.thresholdRange[1] - this.thresholdRange[0]); 
    }

    // 3. Inisialisasi parameter LIF untuk Attention Scores (Pengganti Softmax)
    this.betaScores = new Float32Array(this.sequenceLength);
    this.thresholdScores = new Float32Array(this.sequenceLength);
    for (let i = 0; i < this.sequenceLength; i++) {
        this.betaScores[i] = 0.9; 
        // Ambang batas diturunkan tajam untuk mencegah Dead Neurons
        this.thresholdScores[i] = 1.0; 
    }

    // Inisialisasi Potentials akan dilakukan secara dinamis pada saat forward
    this.potentialsQ = Matrix.fromFlat(new Float32Array(0), [0, 0]);
    this.potentialsK = Matrix.fromFlat(new Float32Array(0), [0, 0]);
    this.potentialsV = Matrix.fromFlat(new Float32Array(0), [0, 0]);
    this.potentialsScores = Matrix.fromFlat(new Float32Array(0), [0, 0]);
  }
  private sqDataBuffer?: Float32Array;
  private skDataBuffer?: Float32Array;
  private svDataBuffer?: Float32Array;
  private dummyLpBuffer?: Float32Array;
  private matchScoresBuffer?: Float32Array;
  private qGatedVBuffer?: Float32Array;
  private outSpikesBuffer?: Float32Array;
  private sScoresDataBuffer?: Float32Array;
  private dummyLpScoresBuffer?: Float32Array;
  private tempMatchesBuffer?: Float32Array;

  private ensurePotentialsShape(batchSeq: number, seqLen: number) {
    if (this.potentialsQ._shape[0] !== batchSeq || !this.sqDataBuffer) {
       this.potentialsQ = Matrix.fromFlat(new Float32Array(batchSeq * this.d_model), [batchSeq, this.d_model]);
       this.potentialsK = Matrix.fromFlat(new Float32Array(batchSeq * this.d_model), [batchSeq, this.d_model]);
       this.potentialsV = Matrix.fromFlat(new Float32Array(batchSeq * this.d_model), [batchSeq, this.d_model]);
       this.potentialsScores = Matrix.fromFlat(new Float32Array(batchSeq * seqLen), [batchSeq, seqLen]);
       
       this.sqDataBuffer = new Float32Array(batchSeq * this.d_model);
       this.skDataBuffer = new Float32Array(batchSeq * this.d_model);
       this.svDataBuffer = new Float32Array(batchSeq * this.d_model);
       this.dummyLpBuffer = new Float32Array(batchSeq * this.d_model);
       this.matchScoresBuffer = new Float32Array(batchSeq * seqLen);
       this.qGatedVBuffer = new Float32Array(batchSeq * this.d_model);
       this.outSpikesBuffer = new Float32Array(batchSeq * this.d_model);
       this.sScoresDataBuffer = new Float32Array(batchSeq * seqLen);
       this.dummyLpScoresBuffer = new Float32Array(batchSeq * seqLen);
       this.tempMatchesBuffer = new Float32Array(seqLen);
    }
  }

  public resetState() {
     if (this.potentialsQ) this.potentialsQ._data.fill(0);
     if (this.potentialsK) this.potentialsK._data.fill(0);
     if (this.potentialsV) this.potentialsV._data.fill(0);
     if (this.potentialsScores) this.potentialsScores._data.fill(0);
  }

  protected compute(inputs: Matrix, options?: ForwardOptions): Matrix {
    // Asumsi inputs adalah flat [batch * seqLen, d_model]
    const batchSeq = inputs._shape[0];
    const seqLen = this.sequenceLength;
    const batch = batchSeq / seqLen;
    const d_model = this.d_model;

    if (!Number.isInteger(batch)) {
        throw new Error(`[SpikingSelfAttention] Jumlah baris input (${batchSeq}) harus merupakan kelipatan dari sequenceLength (${seqLen}).`);
    }

    this.ensurePotentialsShape(batchSeq, seqLen);
    this.lastInputs = inputs; // Simpan untuk local learning

    // 1. Proyeksi Q, K, V (Hanya Addisi / Pergeseran Bit karena input spike biner)
    let dotQ = dotProductAddOnly(inputs, this.kernelQ!);
    let dotK = dotProductAddOnly(inputs, this.kernelK!);
    let dotV = dotProductAddOnly(inputs, this.kernelV!);

    // 2. LIF Step untuk menghasilkan S_Q, S_K, S_V (Matriks Biner)
    const sqData = this.sqDataBuffer!;
    sqData.fill(0);
    const skData = this.skDataBuffer!;
    skData.fill(0);
    const svData = this.svDataBuffer!;
    svData.fill(0);
    const dummyLp = this.dummyLpBuffer!;
    dummyLp.fill(0);

    // Q
    if (isNativeAvailable()) {
        lifStepNativeWrapper(this.potentialsQ._data, dotQ._data, sqData, dummyLp, this.betaQKV, this.thresholdQKV);
        lifStepNativeWrapper(this.potentialsK._data, dotK._data, skData, dummyLp, this.betaQKV, this.thresholdQKV);
        lifStepNativeWrapper(this.potentialsV._data, dotV._data, svData, dummyLp, this.betaQKV, this.thresholdQKV);
    } else {
        this.runLIF(this.potentialsQ._data, dotQ._data, sqData, batchSeq, d_model, this.betaQKV, this.thresholdQKV);
        this.runLIF(this.potentialsK._data, dotK._data, skData, batchSeq, d_model, this.betaQKV, this.thresholdQKV);
        this.runLIF(this.potentialsV._data, dotV._data, svData, batchSeq, d_model, this.betaQKV, this.thresholdQKV);
    }

    // 3. Menghitung Skor Kecocokan (SQ dot SK^T) menggunakan operasi AND / bit-wise addition
    // Hasilnya akan berukuran [batch * seqLen, seqLen]
    const matchScores = this.matchScoresBuffer!;
    matchScores.fill(0);
    
    for (let b = 0; b < batch; b++) {
        for (let i = 0; i < seqLen; i++) {
            const qBase = b * seqLen * d_model + i * d_model;
            // Pre-collect non-zero indices for Q to exploit sparsity
            const nonZeroQ: number[] = [];
            for (let d = 0; d < d_model; d++) {
                if (sqData[qBase + d] > 0) nonZeroQ.push(d);
            }
            if (nonZeroQ.length === 0) continue;

            let maxMatch = 0;
            const tempMatches = this.tempMatchesBuffer!;
            tempMatches.fill(0);
            
            for (let j = 0; j < seqLen; j++) {
                let matchCount = 0;
                const kBase = b * seqLen * d_model + j * d_model;
                for (let k = 0; k < nonZeroQ.length; k++) {
                    const d = nonZeroQ[k];
                    if (skData[kBase + d] > 0) matchCount++;
                }
                tempMatches[j] = matchCount;
                if (matchCount > maxMatch) {
                    maxMatch = matchCount;
                }
            }
            
            for (let j = 0; j < seqLen; j++) {
                if (maxMatch > 0) {
                    matchScores[b * seqLen * seqLen + i * seqLen + j] = tempMatches[j] / maxMatch;
                } else {
                    matchScores[b * seqLen * seqLen + i * seqLen + j] = 0;
                }
            }
        }
    }

    // 4. Pengganti Softmax: Lewatkan skor kecocokan ke lapisan LIF
    const sScoresData = this.sScoresDataBuffer!;
    sScoresData.fill(0);
    const dummyLpScores = this.dummyLpScoresBuffer!;
    dummyLpScores.fill(0);

    if (isNativeAvailable()) {
        lifStepNativeWrapper(this.potentialsScores._data, matchScores, sScoresData, dummyLpScores, this.betaScores, this.thresholdScores);
    } else {
        this.runLIF(this.potentialsScores._data, matchScores, sScoresData, batchSeq, seqLen, this.betaScores, this.thresholdScores);
    }

    const outData = this.outSpikesBuffer!;
    outData.fill(0);

    for (let b = 0; b < batch; b++) {
        for (let j = 0; j < seqLen; j++) {
            const vBase = b * seqLen * d_model + j * d_model;
            // Pre-collect non-zero indices for V to exploit sparsity
            const nonZeroV: number[] = [];
            for (let d = 0; d < d_model; d++) {
                if (svData[vBase + d] > 0) nonZeroV.push(d);
            }
            if (nonZeroV.length === 0) continue;

            for (let i = 0; i < seqLen; i++) {
                const gradedScore = matchScores[b * seqLen * seqLen + i * seqLen + j];
                if (gradedScore > 0) {
                    const outBase = b * seqLen * d_model + i * d_model;
                    for (let k = 0; k < nonZeroV.length; k++) {
                        const d = nonZeroV[k];
                        outData[outBase + d] += gradedScore * svData[vBase + d];
                    }
                }
            }
        }
    }

    // Opsional: Batasi output menjadi biner (spike) jika layer berikutnya menuntut binary matrix
    for (let i = 0; i < outData.length; i++) {
        if (outData[i] > 1.0) outData[i] = 1.0;
    }

    return Matrix.fromFlat(outData, [batchSeq, d_model]);
  }

  private runLIF(pot: Float32Array, input: Float32Array, output: Float32Array, batch: number, dim: number, beta: Float32Array, threshold: Float32Array) {
      for (let b = 0; b < batch; b++) {
          const offset = b * dim;
          for (let i = 0; i < dim; i++) {
              const idx = offset + i;
              pot[idx] = Math.min((pot[idx] * beta[i]) + input[idx], 1.0);
          }
          for (let i = 0; i < dim; i++) {
              const idx = offset + i;
              if (pot[idx] >= threshold[i]) {
                  output[idx] = 1.0;
                  pot[idx] -= threshold[i];
              } else {
                  output[idx] = 0.0;
              }
          }
      }
  }

  public learnAttention(errorSignal: Matrix, learningRate: number = 0.01) {
      if (!this.lastInputs) {
          throw new Error("[SpikingSelfAttention] Cannot run learning before forward() is executed.");
      }

      const err = errorSignal._data;
      const inputs = this.lastInputs._data;
      const batchSeq = this.lastInputs._shape[0];
      // Karena inputs masuk setelah layer 1, shape-nya [batchSeq, d_model]
      const inFeatures = this.lastInputs._shape[1] || this.d_model; 
      const d_model = this.d_model;

      // Update Local Learning: Karena fungsi non-differentiable rumit, 
      // kita mendistribusikan sinyal error secara merata ke kernel Q, K, dan V (Hebbian/Surrogate style)
      const kQ = this.kernelQ!._data;
      const kK = this.kernelK!._data;
      const kV = this.kernelV!._data;

      for (let b = 0; b < batchSeq; b++) {
          const errOffset = b * d_model;
          const inOffset = b * inFeatures;
          for (let i = 0; i < inFeatures; i++) {
              const inVal = inputs[inOffset + i];
              if (inVal > 0) { // Sparse update
                  const kOffset = i * d_model;
                  for (let d = 0; d < d_model; d++) {
                      // Dopamine drive sangat kecil untuk membangkitkan neuron mati tanpa over-saturate
                      const dopamine = 0.00005; 
                      
                      let deltaQ = (learningRate * err[errOffset + d] * inVal) + dopamine;
                      let deltaK = (learningRate * err[errOffset + d] * inVal) + dopamine;
                      let deltaV = (learningRate * err[errOffset + d] * inVal) + dopamine;
                      
                      kQ[kOffset + d] = Math.max(-1.0, Math.min(1.0, kQ[kOffset + d] + deltaQ));
                      kK[kOffset + d] = Math.max(-1.0, Math.min(1.0, kK[kOffset + d] + deltaK));
                      kV[kOffset + d] = Math.max(-1.0, Math.min(1.0, kV[kOffset + d] + deltaV));
                  }
              }
          }
      }
  }

  public getConfig(): Record<string, any> {
    return {
      ...super.getConfig(),
      d_model: this.d_model,
      sequenceLength: this.sequenceLength,
      kernelInitializer: this.kernelInitializer
    };
  }
}
