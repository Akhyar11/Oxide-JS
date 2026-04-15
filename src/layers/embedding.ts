import mj from "../math";
import Matrix from "../matrix";
import { Optimzier, OptimzierType, StatusLayer, matrix2d } from "../@types/type";
import setOptimizer from "../utils/setOptimizer";
import { isNativeAvailable, embeddingForwardNative, embeddingBackwardNative } from "../math/rust_backend";

export interface EmbeddingLayerParams {
  vocabSize: number;
  embeddingDim: number;
  alpha?: number;
  status?: StatusLayer;
  optimizer?: Optimzier;
  padTokenId?: number | null;
}

/**
 * Embedding Layer: Merubah index token (integer) menjadi vector dense
 */
export default class Embedding {
  name = "embedding layer";
  vocabSize: number;
  embeddingDim: number;
  weight: Matrix;
  status: StatusLayer;
  alpha: number;
  optimizerName: Optimzier;
  padTokenId: number | null;
  private optimizerWeight: OptimzierType;
  params: number;
  inputShape: [number, number] = [0, 0];
  outputShape: [number, number] = [0, 0];
  
  // State for backprop
  loss: number = 0;
  private inputIndices: number[] = [];
  
  constructor({
    vocabSize,
    embeddingDim,
    alpha = 0.01,
    status = "input",
    optimizer = "adam",
    padTokenId = null,
  }: EmbeddingLayerParams) {
    this.vocabSize = vocabSize;
    this.embeddingDim = embeddingDim;
    this.status = status;
    this.alpha = alpha;
    this.optimizerName = optimizer;
    this.padTokenId = padTokenId;
    
    // Weight shape: [embeddingDim, vocabSize]
    // Setiap kolom (vertikal) merepresentasikan satu kata/vektor
    this.weight = mj.random([embeddingDim, vocabSize]); 
    
    this.optimizerWeight = setOptimizer(optimizer, this.weight._shape, 1e-5);
    // Jumlah parameter total tabel embedding
    this.params = vocabSize * embeddingDim;
  }

  save() {
    return {
      name: this.name,
      status: this.status,
      vocabSize: this.vocabSize,
      embeddingDim: this.embeddingDim,
      alpha: this.alpha,
      optimizer: this.optimizerName,
      padTokenId: this.padTokenId,
      weight: this.weight._value,
    };
  }

  load(weight: matrix2d): void {
    this.weight._value = weight;
    this.weight._shape = [weight.length, weight[0]?.length ?? 0];
    this.vocabSize = this.weight._shape[1];
    this.params = this.vocabSize * this.embeddingDim;
  }

  compile({
    alpha = 0.1,
    optimizer = "adam",
  }: {
    alpha?: number;
    optimizer?: Optimzier;
  }): void {
    this.alpha = alpha;
    this.optimizerWeight = setOptimizer(optimizer, this.weight._shape, 1e-5);
    this.optimizerName = optimizer;
  }
  
  forward(x: Matrix): Matrix {
    // Memproses input berisi list index token, e.g. [1, 5, 2]
    // x bisa berupa matriks 1D [seqLen, 1] atau [1, seqLen]
    const flatX = mj.flatten(x);
    this.inputIndices = Array.from(flatX._data);
    
    const seqLen = this.inputIndices.length;
    this.inputShape = [seqLen, 1];
    
    // Karena ML_V2 kita memproses array vertikal (kolom per kolom),
    // output kita atur menjadi [embeddingDim, seqLen]
    this.outputShape = [this.embeddingDim, seqLen];

    if (isNativeAvailable()) {
        const out = embeddingForwardNative(this.inputIndices, this.weight._data, this.vocabSize, this.embeddingDim, this.padTokenId);
        return Matrix.fromFlat(out, [this.embeddingDim, seqLen]);
    }

    const outputData = new Float64Array(this.embeddingDim * seqLen);
    const weightData = this.weight._data;
    const weightCols = this.weight._shape[1];

    for (let j = 0; j < seqLen; j++) {
        const tokenIndex = Math.floor(this.inputIndices[j]);
        if (tokenIndex < 0 || tokenIndex >= this.vocabSize) {
            throw new Error(`Token index '${tokenIndex}' di luar kapasitas vocabulary (0 - ${this.vocabSize-1})`);
        }
        if (this.padTokenId !== null && tokenIndex === this.padTokenId) {
            continue;
        }
        for (let i = 0; i < this.embeddingDim; i++) {
            outputData[i * seqLen + j] = weightData[i * weightCols + tokenIndex];
        }
    }
    
    return Matrix.fromFlat(outputData, [this.embeddingDim, seqLen]);
  }
  
  backward(y: Matrix, err: Matrix): Matrix {
    // `err` adalah error/gradien dari layer setelahnya bertipe [embeddingDim, seqLen]
    const gradWeight = mj.zeros(this.weight._shape);
    const seqLen = this.inputIndices.length;
    if (isNativeAvailable()) {
        embeddingBackwardNative(this.inputIndices, err._data, gradWeight._data, this.vocabSize, this.embeddingDim, this.padTokenId);
    } else {
        const gradData = gradWeight._data;
        const errData = err._data;
        const vocabSize = this.weight._shape[1];
        
        // Kumpulkan dan akumulasi nilai gradien setiap token ke index yang relevan pada `weight`
        for (let i = 0; i < this.embeddingDim; i++) {
            for (let j = 0; j < seqLen; j++) {
                const tokenIndex = Math.floor(this.inputIndices[j]);
                if (this.padTokenId !== null && tokenIndex === this.padTokenId) {
                    continue;
                }
                gradData[i * vocabSize + tokenIndex] += errData[i * seqLen + j];
            }
        }
    }
    
    // Update bobot kamus embedding menggunakan optimizer
    const optimizerUpdate = this.optimizerWeight.calculate(gradWeight, this.alpha);
    this.weight = mj.sub(this.weight, optimizerUpdate);
    
    // Gradien dari inputnya index (x) tidak dapat diturunkan ulang ke depannya, 
    // Jadi dikembalikan dummy array zeros agar tidak crash.
    return mj.zeros([seqLen, 1]);
  }

  /**
   * Resize vocabulary size
   * @param newVocabSize - New vocabulary size
   */
  resize(newVocabSize: number): void {
    if (newVocabSize <= this.vocabSize) return;

    console.log(`[Embedding] Resizing vocab: ${this.vocabSize} -> ${newVocabSize}`);

    // 1. Create new weights matrix
    const newWeight = mj.random([this.embeddingDim, newVocabSize]);
    const oldWeightData = this.weight._data;
    const newWeightData = newWeight._data;

    // 2. Copy old weights
    for (let i = 0; i < this.embeddingDim; i++) {
        for (let j = 0; j < this.vocabSize; j++) {
            newWeightData[i * newVocabSize + j] = oldWeightData[i * this.vocabSize + j];
        }
    }

    // 3. Update state
    this.weight = newWeight;
    this.vocabSize = newVocabSize;
    this.params = newVocabSize * this.embeddingDim;

    // 4. Reset optimizer for new shape
    this.optimizerWeight = setOptimizer(this.optimizerName, this.weight._shape, 1e-5);
  }

  resetLoss(): void {
    this.loss = 0;
  }
}
