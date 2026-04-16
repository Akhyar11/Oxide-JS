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
  
  // Buffers
  private outputBuffer: Matrix | null = null;
  private gradWeightBuffer: Matrix | null = null;
  private errOutputBuffer: Matrix | null = null;
  
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
    this.weight = mj.xavier([embeddingDim, vocabSize]); 
    
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

    if (!this.outputBuffer || this.outputBuffer._shape[0] !== this.embeddingDim || this.outputBuffer._shape[1] !== seqLen) {
        this.outputBuffer = mj.zeros([this.embeddingDim, seqLen]);
    }
    const outputData = this.outputBuffer._data;

    if (isNativeAvailable()) {
        embeddingForwardNative(this.inputIndices, this.weight._data, this.vocabSize, this.embeddingDim, this.padTokenId, outputData);
        return this.outputBuffer;
    }

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
    
    return this.outputBuffer;
  }
  
  backward(y: Matrix, err: Matrix): Matrix {
    // `err` adalah error/gradien dari layer setelahnya bertipe [embeddingDim, seqLen]
    if (!this.gradWeightBuffer || this.gradWeightBuffer._shape[1] !== this.vocabSize || this.gradWeightBuffer._shape[0] !== this.embeddingDim) {
        this.gradWeightBuffer = mj.zeros([this.embeddingDim, this.vocabSize]);
    } else {
        this.gradWeightBuffer._data.fill(0);
    }
    const gradWeight = this.gradWeightBuffer;
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
    
    // Update bobot kamus embedding menggunakan optimizer In-Place
    const optimizerUpdate = this.optimizerWeight.calculate(gradWeight, this.alpha);
    this.weight.subInPlace(optimizerUpdate);
    
    // Gradien dari inputnya index (x) tidak dapat diturunkan ulang ke depannya, 
    // Jadi dikembalikan dummy array zeros agar tidak crash. (Menggunakan buffer)
    if (!this.errOutputBuffer || this.errOutputBuffer._shape[0] !== seqLen) {
        this.errOutputBuffer = mj.zeros([seqLen, 1]);
    } else {
        this.errOutputBuffer._data.fill(0);
    }
    return this.errOutputBuffer;
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
