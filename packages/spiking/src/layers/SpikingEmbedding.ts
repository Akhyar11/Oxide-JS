import { BaseLayer, LayerConfig, ForwardOptions } from "@oxide-js/layers";
import { Matrix, mj } from "@oxide-js/core";
import {
  isNativeAvailable,
  lifStepNativeWrapper,
  maskSurrogateNativeWrapper,
  learnHebbianNativeWrapper
} from "../native_backend.js";

export interface SpikingEmbeddingConfig extends LayerConfig {
  inputDim: number; // Ukuran vocabulary
  outputDim: number; // Dimensi embedding (jumlah neuron)
  beta?: number; // Decay factor LIF
  threshold?: number; // Ambang batas Spike
  embeddingsInitializer?: string; // Tipe inisialisasi bobot
}

export class SpikingEmbedding extends BaseLayer {
  public inputDim: number;
  public outputDim: number;
  public beta: number;
  public threshold: number;

  public potentials!: Matrix;
  public lastPotentials?: Matrix;
  public lastInputs?: Matrix;
  public lastSpikes?: Matrix;

  public embeddingsInitializer: string;

  public get kernel(): Matrix | undefined {
    return this.getParameter("kernel");
  }

  constructor(config: SpikingEmbeddingConfig) {
    super(config);
    this.inputDim = config.inputDim;
    this.outputDim = config.outputDim;
    this.beta = config.beta ?? 0.9;
    this.threshold = config.threshold ?? 1.0;
    this.embeddingsInitializer = config.embeddingsInitializer || "glorot_normal";
  }

  public computeOutputShape(inputShape: number[]): number[] {
    const batch = inputShape[0] ?? -1;
    return [batch, this.outputDim];
  }

  public build(inputShape: number[]): void {
    super.build(inputShape);
    const kernelVal = this.createInitializer(this.embeddingsInitializer, [this.inputDim, this.outputDim]);
    this.addParameter("kernel", kernelVal, true, [this.inputDim, this.outputDim]);
  }

  public resetState() {
    if (this.potentials) this.potentials._data.fill(0);
    this.lastPotentials = undefined;
    this.lastInputs = undefined;
    this.lastSpikes = undefined;
  }

  private ensurePotentialsShape(batch: number) {
    if (!this.potentials || this.potentials._shape[0] !== batch) {
      this.potentials = Matrix.fromFlat(
        new Float32Array(batch * this.outputDim),
        [batch, this.outputDim]
      );
    }
  }

  protected compute(inputs: Matrix, options?: ForwardOptions): Matrix {
    const kernel = this.kernel!._data;
    const batch = inputs._shape[0];
    const inputData = inputs._data;

    this.ensurePotentialsShape(batch);

    // 1. Lookup Row (Pengganti dot-product)
    const dotData = new Float32Array(batch * this.outputDim);
    for (let b = 0; b < batch; b++) {
      const tokenId = Math.round(inputData[b]); // Asumsi input adalah ID token berukuran [batch, 1]

      // Jika token valid, ekstrak barisnya sebagai Arus (Current)
      if (tokenId >= 0 && tokenId < this.inputDim) {
        const kernelOffset = tokenId * this.outputDim;
        const dotOffset = b * this.outputDim;
        for (let j = 0; j < this.outputDim; j++) {
          dotData[dotOffset + j] = kernel[kernelOffset + j];
        }
      }
    }

    // 2 & 3. Leaky Integrate, Fire & Reset
    const outData = new Float32Array(batch * this.outputDim);
    const outSpikes = Matrix.fromFlat(outData, [batch, this.outputDim]);
    this.lastPotentials = Matrix.fromFlat(new Float32Array(batch * this.outputDim), [batch, this.outputDim]);

    if (isNativeAvailable()) {
      lifStepNativeWrapper(
        this.potentials._data,
        dotData,
        outSpikes._data,
        this.lastPotentials._data,
        this.beta,
        this.threshold
      );
    } else {
      const potData = this.potentials._data;
      const thresh = this.threshold;
      const lpData = this.lastPotentials._data;
      for (let i = 0; i < potData.length; i++) {
        potData[i] = (potData[i] * this.beta) + dotData[i];
        lpData[i] = potData[i];
      }
      for (let i = 0; i < potData.length; i++) {
        if (potData[i] >= thresh) {
          outData[i] = 1;
          potData[i] -= thresh;
        } else {
          outData[i] = 0;
        }
      }
    }

    // Simpan memori untuk update bobot
    this.lastInputs = inputs;
    this.lastSpikes = outSpikes;

    return outSpikes;
  }

  // Embedding hanya menerima instruksi belajar dari layer atasnya (eHidden yang sudah dikalikan matriks B)
  public learnEmbedding(errorFromNext: Matrix, B: Matrix, learningRate: number = 0.01): Matrix {
    if (!this.lastInputs) {
      throw new Error("[SpikingEmbedding] Cannot run learnEmbedding() before forward() is executed. 'lastInputs' is undefined.");
    }

    const kernel = this.kernel!._data;
    const inputData = this.lastInputs._data;
    const batch = this.lastInputs._shape[0];

    let eHiddenMatrix = mj.dotProduct(errorFromNext, B, undefined, false, false);

    // Surrogate Mask: Boxcar
    if (this.lastPotentials) {
      if (isNativeAvailable()) {
        maskSurrogateNativeWrapper(
          eHiddenMatrix._data,
          this.lastPotentials._data,
          this.threshold,
          1.0
        );
      } else {
        const eData = eHiddenMatrix._data;
        const pData = this.lastPotentials._data;
        const thresh = this.threshold;
        const windowSize = 1.0;

        for (let i = 0; i < eData.length; i++) {
          if (Math.abs(pData[i] - thresh) > windowSize) {
            eData[i] = 0;
          }
        }
      }
    }

    // Delta Rule Update pada baris Lookup (sangat efisien)
    const err = eHiddenMatrix._data;
    for (let b = 0; b < batch; b++) {
      const tokenId = Math.round(inputData[b]);
      if (tokenId >= 0 && tokenId < this.inputDim) {
        const kOffset = tokenId * this.outputDim;
        const errOffset = b * this.outputDim;
        for (let j = 0; j < this.outputDim; j++) {
          kernel[kOffset + j] += learningRate * err[errOffset + j];
        }
      }
    }

    return eHiddenMatrix;
  }

  /**
   * Word2Vec CBOW-style Hebbian Contrastive Learning
   * Memungkinkan pembelajaran embedding semantik secara topologis tanpa representation collapse.
   */
  public learnHebbian(
    tokens: number[] | Float32Array,
    positiveContext: Float32Array,
    negativeContexts: Float32Array[],
    learningRate: number = 0.01,
    marginPositive: number = 0.1,
    marginNegative: number = 0.05
  ): void {
    const kernel = this.kernel!._data;
    const dim = this.outputDim;

    if (isNativeAvailable()) {
      const negFlat = new Float32Array(negativeContexts.length * dim);
      for (let i = 0; i < negativeContexts.length; i++) {
        negFlat.set(negativeContexts[i], i * dim);
      }

      let tokensArray: Float32Array;
      if (tokens instanceof Float32Array) {
        tokensArray = tokens;
      } else {
        tokensArray = new Float32Array(tokens);
      }

      learnHebbianNativeWrapper(
        kernel,
        tokensArray,
        positiveContext,
        negFlat,
        negativeContexts.length,
        this.inputDim,
        dim,
        learningRate,
        marginPositive,
        marginNegative
      );
      return;
    }

    // Lacak token yang diperbarui agar bisa dinormalisasi
    const updatedTokens = new Set<number>();

    for (let i = 0; i < tokens.length; i++) {
      const tokenId = Math.round(tokens[i]);
      if (tokenId >= 0 && tokenId < this.inputDim) {
        updatedTokens.add(tokenId);
        const offset = tokenId * dim;
        
        for (let j = 0; j < dim; j++) {
          // 1. Update Positive Context (selalu dieksekusi)
          const posGradient = positiveContext[j] - kernel[offset + j];
          kernel[offset + j] += learningRate * posGradient * marginPositive;

          // 2. Update Negative Contexts (jika ada)
          for (let n = 0; n < negativeContexts.length; n++) {
            const negMean = negativeContexts[n];
            const negGradient = kernel[offset + j] - negMean[j];
            kernel[offset + j] += learningRate * negGradient * marginNegative;
          }
        }
      }
    }

    // L2 Normalization (Mencegah Dying Neuron atau Exploding Weights)
    for (const tokenId of updatedTokens) {
      const offset = tokenId * dim;
      let norm = 0;
      for (let j = 0; j < dim; j++) {
        norm += kernel[offset + j] * kernel[offset + j];
      }
      norm = Math.sqrt(norm + 1e-8);
      for (let j = 0; j < dim; j++) {
        kernel[offset + j] /= norm;
      }
    }
  }
}
