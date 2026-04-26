import mj from "../math";
import Matrix from "../matrix";
import { Optimzier, OptimzierType, StatusLayer, matrix2d } from "../@types/type";
import setOptimizer from "../utils/setOptimizer";
import { isNativeAvailable, embeddingForwardNative, embeddingBackwardSparseNative } from "../math/rust_backend";

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
  private inputIndices: Int32Array = new Int32Array(0);

  // Buffers
  private outputBuffer: Matrix | null = null;
  private gradWeightBuffer: Matrix | null = null;
  private errOutputBuffer: Matrix | null = null;
  private orderedInputBuffer: Int32Array = new Int32Array(0);

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
    this.embeddingDim = this.weight._shape[0];
    this.vocabSize = this.weight._shape[1];
    this.params = this.vocabSize * this.embeddingDim;
    this.optimizerWeight = setOptimizer(this.optimizerName, this.weight._shape, 1e-5);
  }

  compile({
    alpha,
    optimizer,
  }: {
    alpha?: number;
    optimizer?: Optimzier;
  }): void {
    if (alpha !== undefined) this.alpha = alpha;
    if (optimizer !== undefined) {
      this.optimizerWeight = setOptimizer(optimizer, this.weight._shape, this.alpha);
      this.optimizerName = optimizer;
    }
  }

  forward(x: Matrix): Matrix {
    return this.forwardWithLayout(x, "sample-major");
  }

  forwardTimeMajor(x: Matrix): Matrix {
    return this.forwardWithLayout(x, "time-major");
  }

  backward(y: Matrix, err: Matrix): Matrix {
    // Optimization: Use sparse updates instead of a full [embeddingDim, vocabSize] matrix
    const seqLen = this.inputIndices.length;
    const errData = err._data;

    let uniqueIndices: Int32Array;
    let smallGrad: Matrix;

    if (isNativeAvailable() && this.inputIndices instanceof Int32Array) {
      const res = embeddingBackwardSparseNative(this.inputIndices, errData, this.embeddingDim, this.padTokenId);
      uniqueIndices = res.uniqueIndices;
      smallGrad = Matrix.fromFlat(res.grad, [this.embeddingDim, uniqueIndices.length]);
    } else {
      // 1. Identify unique tokens and their first occurrence
      const uniqueIndicesArr: number[] = [];
      const indexMap = new Map<number, number>();
      for (let j = 0; j < seqLen; j++) {
        const tokenIndex = this.inputIndices[j];
        if (this.padTokenId !== null && tokenIndex === this.padTokenId) continue;
        if (!indexMap.has(tokenIndex)) {
          indexMap.set(tokenIndex, uniqueIndicesArr.length);
          uniqueIndicesArr.push(tokenIndex);
        }
      }

      uniqueIndices = new Int32Array(uniqueIndicesArr);
      const numUnique = uniqueIndices.length;

      // 2. Aggregate gradients into a small matrix [embeddingDim, numUnique]
      smallGrad = mj.zeros([this.embeddingDim, numUnique]);
      const smallGradData = smallGrad._data;

      for (let j = 0; j < seqLen; j++) {
        const tokenIndex = this.inputIndices[j];
        if (this.padTokenId !== null && tokenIndex === this.padTokenId) continue;
        const uIdx = indexMap.get(tokenIndex)!;
        for (let i = 0; i < this.embeddingDim; i++) {
          smallGradData[i * numUnique + uIdx] += errData[i * seqLen + j];
        }
      }
    }

    // 3. Update only the used embeddings using the sparse optimizer method
    this.optimizerWeight.updateSparse(this.weight, smallGrad, this.alpha, uniqueIndices);

    // Gradien dari inputnya index (x) tidak dapat diturunkan ulang ke depannya, 
    // Jadi dikembalikan dummy array zeros agar tidak crash. (Menggunakan buffer)
    // Shape matches the original input shape so downstream layers don't get a mismatch.
    const [inputRows, inputCols] = this.inputShape;
    if (!this.errOutputBuffer ||
      this.errOutputBuffer._shape[0] !== inputRows ||
      this.errOutputBuffer._shape[1] !== inputCols) {
      this.errOutputBuffer = mj.zeros([inputRows, inputCols]);
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

  private forwardWithLayout(x: Matrix, layout: "sample-major" | "time-major"): Matrix {
    const [rows, cols] = x._shape;
    const totalTokens = rows * cols;
    if (this.orderedInputBuffer.length !== totalTokens) {
      this.orderedInputBuffer = new Int32Array(totalTokens);
    }

    if (layout === "sample-major") {
      // Susun token secara sample-contiguous: semua token sample 0, lalu sample 1, dst.
      let writeIdx = 0;
      for (let col = 0; col < cols; col++) {
        for (let row = 0; row < rows; row++) {
          this.orderedInputBuffer[writeIdx++] = this.validateAndNormalizeTokenIndex(x._data[row * cols + col]);
        }
      }
    } else {
      for (let i = 0; i < totalTokens; i++) {
        this.orderedInputBuffer[i] = this.validateAndNormalizeTokenIndex(x._data[i]);
      }
    }
    this.inputIndices = this.orderedInputBuffer;

    const seqLen = totalTokens;
    this.inputShape = [rows, cols];
    this.outputShape = [this.embeddingDim, seqLen];

    if (!this.outputBuffer || this.outputBuffer._shape[0] !== this.embeddingDim || this.outputBuffer._shape[1] !== seqLen) {
      this.outputBuffer = mj.zeros([this.embeddingDim, seqLen]);
    }
    const outputData = this.outputBuffer._data;

    if (isNativeAvailable()) {
      embeddingForwardNative(this.inputIndices, this.weight._data, this.vocabSize, this.embeddingDim, this.padTokenId, outputData);
      return this.outputBuffer;
    }
    outputData.fill(0);

    const weightData = this.weight._data;
    const weightCols = this.weight._shape[1];

    for (let j = 0; j < seqLen; j++) {
      const tokenIndex = Math.floor(this.inputIndices[j]);
      if (this.padTokenId !== null && tokenIndex === this.padTokenId) {
        continue;
      }
      for (let i = 0; i < this.embeddingDim; i++) {
        outputData[i * seqLen + j] = weightData[i * weightCols + tokenIndex];
      }
    }

    return this.outputBuffer;
  }

  private validateAndNormalizeTokenIndex(rawTokenIndex: number): number {
    const tokenIndex = Math.floor(rawTokenIndex);
    if (!Number.isFinite(rawTokenIndex) || tokenIndex < 0 || tokenIndex >= this.vocabSize) {
      throw new Error(`Token index '${rawTokenIndex}' di luar kapasitas vocabulary (0 - ${this.vocabSize - 1})`);
    }
    return tokenIndex;
  }
}
