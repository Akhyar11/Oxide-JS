import mj from "../math";
import Matrix from "../matrix";
import { Optimzier, OptimzierType, StatusLayer, matrix2d } from "../@types/type";
import setOptimizer from "../utils/setOptimizer";

export interface EmbeddingLayerParams {
  vocabSize: number;
  embeddingDim: number;
  alpha?: number;
  status?: StatusLayer;
  optimizer?: Optimzier;
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
    optimizer = "adam"
  }: EmbeddingLayerParams) {
    this.vocabSize = vocabSize;
    this.embeddingDim = embeddingDim;
    this.status = status;
    this.alpha = alpha;
    this.optimizerName = optimizer;
    
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
      weight: this.weight._value,
    };
  }

  load(weight: matrix2d): void {
    this.weight._value = weight;
    this.weight._shape = [weight.length, weight[0]?.length ?? 0];
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
    this.inputIndices = flatX._value[0];
    
    const seqLen = this.inputIndices.length;
    this.inputShape = [seqLen, 1];
    
    // Karena ML_V2 kita memproses array vertikal (kolom per kolom),
    // output kita atur menjadi [embeddingDim, seqLen]
    this.outputShape = [this.embeddingDim, seqLen];

    const outputArray: number[][] = [];
    for (let i = 0; i < this.embeddingDim; i++) {
        outputArray[i] = [];
        for (let j = 0; j < seqLen; j++) {
            const tokenIndex = Math.floor(this.inputIndices[j]);
            if (tokenIndex < 0 || tokenIndex >= this.vocabSize) {
                throw new Error(`Token index '${tokenIndex}' di luar kapasitas vocabulary (0 - ${this.vocabSize-1})`);
            }
            outputArray[i][j] = this.weight._value[i][tokenIndex];
        }
    }
    
    return new Matrix({ array: outputArray });
  }
  
  backward(y: Matrix, err: Matrix): Matrix {
    // `err` adalah error/gradien dari layer setelahnya bertipe [embeddingDim, seqLen]
    const gradWeight = mj.zeros(this.weight._shape);
    const seqLen = this.inputIndices.length;
    
    // Kumpulkan dan akumulasi nilai gradien setiap token ke index yang relevan pada `weight`
    for (let i = 0; i < this.embeddingDim; i++) {
        for (let j = 0; j < seqLen; j++) {
            const tokenIndex = Math.floor(this.inputIndices[j]);
            gradWeight._value[i][tokenIndex] += err._value[i][j];
        }
    }
    
    // Update bobot kamus embedding menggunakan optimizer
    const optimizerUpdate = this.optimizerWeight.calculate(gradWeight, this.alpha);
    this.weight = mj.sub(this.weight, optimizerUpdate);
    
    // Gradien dari inputnya index (x) tidak dapat diturunkan ulang ke depannya, 
    // Jadi dikembalikan dummy array zeros agar tidak crash.
    return mj.zeros([seqLen, 1]);
  }

  resetLoss(): void {
    this.loss = 0;
  }
}
