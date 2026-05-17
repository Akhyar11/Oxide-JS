import { BaseLayer, LayerConfig, type ForwardOptions } from "../base/BaseLayer.js";
import { Matrix, engine } from "@oxide-js/core";
import { isNativeAvailable, embeddingForwardNative, embeddingBackwardNative } from "../rust_backend.js";

export interface EmbeddingConfig extends LayerConfig {
  inputDim: number; // vocabulary size
  outputDim: number; // embedding dimension
  embeddingsInitializer?: string;
}

/**
 * Helper function to perform embedding lookup with custom backward pass recorded on tape
 */
function embeddingLookup(inputs: Matrix, embeddings: Matrix): Matrix {
  const numTokens = inputs._shape[0];
  const seqLen = inputs._shape[1] ?? 1;
  const totalTokens = numTokens * seqLen;
  const [vocabSize, embeddingDim] = embeddings._shape;

  const resultData = new Float32Array(totalTokens * embeddingDim);
  const inputsData = inputs._data;
  const embedData = embeddings._data;

  if (isNativeAvailable()) {
    embeddingForwardNative(inputsData, embedData, vocabSize, embeddingDim, resultData);
  } else {
    for (let i = 0; i < totalTokens; i++) {
      const idx = Math.floor(inputsData[i]); // token index
      if (idx < 0 || idx >= vocabSize) {
        throw new Error(`[Embedding] Token index ${idx} is out of vocabulary bounds [0, ${vocabSize - 1}].`);
      }
      const destOffset = i * embeddingDim;
      const srcOffset = idx * embeddingDim;
      for (let j = 0; j < embeddingDim; j++) {
        resultData[destOffset + j] = embedData[srcOffset + j];
      }
    }
  }

  const res = Matrix.fromFlat(resultData, [totalTokens, embeddingDim]);

  engine.record(
    [inputs, embeddings],
    [res],
    (grad: Matrix) => {
      // inputs does not require gradient (it's integer IDs)
      // embeddings requires gradient! We accumulate grad into a zero matrix of shape [vocabSize, embeddingDim]
      const gradEmbed = new Float32Array(vocabSize * embeddingDim);
      const gradOutData = grad._data;
      
      if (isNativeAvailable()) {
        embeddingBackwardNative(gradOutData, inputsData, vocabSize, embeddingDim, gradEmbed);
      } else {
        for (let i = 0; i < totalTokens; i++) {
          const idx = Math.floor(inputsData[i]);
          if (idx >= 0 && idx < vocabSize) {
            const srcOffset = i * embeddingDim;
            const destOffset = idx * embeddingDim;
            for (let j = 0; j < embeddingDim; j++) {
              gradEmbed[destOffset + j] += gradOutData[srcOffset + j];
            }
          }
        }
      }

      return [
        null, // inputs has no gradient
        Matrix.fromFlat(gradEmbed, [vocabSize, embeddingDim]) // embeddings gradient
      ];
    },
    { saveInput: true, saveOutput: false } // we need inputs to know the indices
  );

  return res;
}

export class Embedding extends BaseLayer {
  public inputDim: number;
  public outputDim: number;
  public embeddingsInitializer: string;

  constructor(config: EmbeddingConfig) {
    super(config);
    if (config.inputDim === undefined || config.inputDim <= 0) {
      throw new Error("[Embedding] 'inputDim' (vocabSize) wajib berupa angka positif.");
    }
    if (config.outputDim === undefined || config.outputDim <= 0) {
      throw new Error("[Embedding] 'outputDim' (embeddingDim) wajib berupa angka positif.");
    }
    this.inputDim = config.inputDim;
    this.outputDim = config.outputDim;
    this.embeddingsInitializer = config.embeddingsInitializer ?? "random";
  }

  /**
   * Menghitung output shape logis [batch * seqLen, embeddingDim]
   */
  public computeOutputShape(inputShape: number[]): number[] {
    const batch = inputShape[0] ?? -1;
    const seqLen = inputShape.length > 1 ? inputShape[1] : 1;
    if (batch === -1) {
      return [-1, this.outputDim];
    }
    return [batch * seqLen, this.outputDim];
  }

  /**
   * Menginisialisasi parameter 'embeddings'
   */
  public build(inputShape: number[]): void {
    if (this.isBuilt) return;

    this.inputShape = [...inputShape];
    this.outputShape = this.computeOutputShape(inputShape);

    // Initializer untuk embeddings table
    const embeddingsVal = this.createInitializer(this.embeddingsInitializer, [this.inputDim, this.outputDim]);
    this.addParameter("embeddings", embeddingsVal, true, [this.inputDim, this.outputDim]);

    this.isBuilt = true;
  }

  /**
   * Keras-style getter untuk mendapatkan matriks embeddings
   */
  public get embeddings(): Matrix | undefined {
    return this.getParameter("embeddings");
  }

  /**
   * Forward Pass matematika layer Embedding
   */
  protected compute(inputs: Matrix, options?: ForwardOptions): Matrix {
    const embeddings = this.embeddings;
    if (!embeddings) {
      throw new Error("[Embedding] Bobot 'embeddings' tidak terinisialisasi. Pastikan build() sudah dijalankan.");
    }
    return embeddingLookup(inputs, embeddings);
  }

  /**
   * Konfigurasi spesifik Keras
   */
  public getConfig(): Record<string, any> {
    return {
      ...super.getConfig(),
      inputDim: this.inputDim,
      outputDim: this.outputDim,
      embeddingsInitializer: this.embeddingsInitializer
    };
  }
}
