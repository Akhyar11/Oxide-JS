import mj from "../math";
import Matrix from "../matrix";
import { Optimzier, OptimzierType, StatusLayer, matrix2d } from "../@types/type";
import setOptimizer from "../utils/setOptimizer";
import { isNativeAvailable, embeddingForwardNative, embeddingBackwardSparseNative } from "../math/rust_backend";
import { readFileSync } from "fs";

export interface EmbeddingLayerParams {
  vocabSize: number;
  embeddingDim: number;
  alpha?: number;
  status?: StatusLayer;
  optimizer?: Optimzier;
  padTokenId?: number | null;
  trainable?: boolean;
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
  trainable: boolean;
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
    trainable = true,
  }: EmbeddingLayerParams) {
    this.vocabSize = vocabSize;
    this.embeddingDim = embeddingDim;
    this.status = status;
    this.alpha = alpha;
    this.optimizerName = optimizer;
    this.padTokenId = padTokenId;
    this.trainable = trainable;

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
      trainable: this.trainable,
      weight: this.weight._value,
    };
  }

  load(data: matrix2d | {
    vocabSize?: number;
    embeddingDim?: number;
    alpha?: number;
    optimizer?: Optimzier;
    status?: StatusLayer;
    padTokenId?: number | null;
    trainable?: boolean;
    weight: matrix2d;
  }): void {
    const resolved = this.resolveLoadPayload(data);
    this.weight = this.normalizeToMatrix(resolved.weight);
    this.weight._shape = [resolved.weight.length, resolved.weight[0]?.length ?? 0];
    this.embeddingDim = this.weight._shape[0];
    this.vocabSize = this.weight._shape[1];
    this.alpha = resolved.alpha ?? this.alpha;
    this.optimizerName = resolved.optimizer ?? this.optimizerName;
    this.status = resolved.status ?? this.status;
    this.padTokenId = resolved.padTokenId !== undefined ? resolved.padTokenId : this.padTokenId;
    this.trainable = resolved.trainable !== undefined ? resolved.trainable : this.trainable;
    this.params = this.vocabSize * this.embeddingDim;
    this.optimizerWeight = setOptimizer(this.optimizerName, this.weight._shape, this.alpha);
  }

  fillWeight(source: string | Matrix | number[][] | Float32Array | {
    weight?: number[][];
    layers?: any[];
    name?: string;
    vocabSize?: number;
    embeddingDim?: number;
    trainable?: boolean;
  }): void {
    const previousTrainable = this.trainable;
    const previousAlpha = this.alpha;
    const previousOptimizer = this.optimizerName;
    const previousStatus = this.status;
    const previousPadTokenId = this.padTokenId;
    const previousVocabSize = this.vocabSize;
    const previousEmbeddingDim = this.embeddingDim;

    const normalizedWeight = this.extractWeightFromFillSource(source);
    this.assignWeightPreservingConfig(normalizedWeight);

    this.trainable = previousTrainable;
    this.alpha = previousAlpha;
    this.optimizerName = previousOptimizer;
    this.status = previousStatus;
    this.padTokenId = previousPadTokenId;
    this.vocabSize = previousVocabSize;
    this.embeddingDim = previousEmbeddingDim;
    this.params = this.vocabSize * this.embeddingDim;
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
    return this.forwardWithLayout(x, "time-major");
  }

  forwardTimeMajor(x: Matrix): Matrix {
    return this.forwardWithLayout(x, "time-major");
  }

  backward(y: Matrix, err: Matrix): Matrix {
    if (!this.trainable) {
      return this.getOrCreateZeroInputGradient();
    }

    // ── Fast path: fused Rust backward+Adam update (zero JS allocations) ────
    const maybeAdam = this.optimizerWeight as any;
    if (
      isNativeAvailable() &&
      this.inputIndices instanceof Int32Array &&
      typeof maybeAdam.updateEmbeddingSparseNative === "function" &&
      maybeAdam.updateEmbeddingSparseNative(
        this.weight,
        this.inputIndices,
        err._data,
        this.alpha,
        this.embeddingDim,
        this.vocabSize,
        this.padTokenId
      )
    ) {
      return this.getOrCreateZeroInputGradient();
    }

    // ── Fallback: sparse JS/native split path ────────────────────────────────
    const seqLen = this.inputIndices.length;
    const errData = err._data;

    let uniqueIndices: Int32Array;
    let smallGrad: Matrix;

    // 1. Identify unique tokens and their first occurrence
    let uniqueIndicesBuffer = (this as any).uniqueIndicesBuffer as Int32Array | undefined;
    if (!uniqueIndicesBuffer || uniqueIndicesBuffer.length < seqLen) {
      uniqueIndicesBuffer = new Int32Array(Math.max(seqLen, Math.max(1, (uniqueIndicesBuffer?.length ?? 0) * 2)));
      (this as any).uniqueIndicesBuffer = uniqueIndicesBuffer;
    }
    
    let numUnique = 0;
    const indexMap = (this as any).indexMapCache || ((this as any).indexMapCache = new Map<number, number>());
    indexMap.clear();

    for (let j = 0; j < seqLen; j++) {
      const tokenIndex = this.inputIndices[j];
      if (this.padTokenId !== null && tokenIndex === this.padTokenId) continue;
      if (!indexMap.has(tokenIndex)) {
        indexMap.set(tokenIndex, numUnique);
        uniqueIndicesBuffer[numUnique] = tokenIndex;
        numUnique++;
      }
    }

    uniqueIndices = uniqueIndicesBuffer.subarray(0, numUnique);

    // 2. Aggregate gradients into a small matrix [embeddingDim, numUnique]
    const requiredGradLen = this.embeddingDim * numUnique;
    let gradWeightBufferData = (this as any).gradWeightBufferData as Float32Array | undefined;
    if (!gradWeightBufferData || gradWeightBufferData.length < requiredGradLen) {
      gradWeightBufferData = new Float32Array(Math.max(requiredGradLen, Math.max(1, (gradWeightBufferData?.length ?? 0) * 2)));
      (this as any).gradWeightBufferData = gradWeightBufferData;
    }
    
    smallGrad = Matrix.fromFlat(gradWeightBufferData.subarray(0, requiredGradLen), [this.embeddingDim, numUnique]);
    const smallGradData = smallGrad._data;
    smallGradData.fill(0);

    for (let j = 0; j < seqLen; j++) {
      const tokenIndex = this.inputIndices[j];
      if (this.padTokenId !== null && tokenIndex === this.padTokenId) continue;
      const uIdx = indexMap.get(tokenIndex)!;
      for (let i = 0; i < this.embeddingDim; i++) {
        smallGradData[i * numUnique + uIdx] += errData[i * seqLen + j];
      }
    }

    // 3. Update only the used embeddings using the sparse optimizer method
    this.optimizerWeight.updateSparse(this.weight, smallGrad, this.alpha, uniqueIndices);

    return this.getOrCreateZeroInputGradient();
  }

  /**
   * Returns a zero-filled Matrix shaped like the embedding layer's input.
   * Re-uses a pre-allocated buffer — no new allocation on the hot path.
   * Used both by the fused Adam fast path and the fallback sparse path.
   */
  private getOrCreateZeroInputGradient(): Matrix {
    const [inputRows, inputCols] = this.inputShape;
    const requiredErrLen = inputRows * inputCols;
    let errOutputBufferData = (this as any).errOutputBufferData as Float32Array | undefined;

    if (!errOutputBufferData || errOutputBufferData.length < requiredErrLen) {
      errOutputBufferData = new Float32Array(Math.max(requiredErrLen, Math.max(1, (errOutputBufferData?.length ?? 0) * 2)));
      (this as any).errOutputBufferData = errOutputBufferData;
    }

    this.errOutputBuffer = Matrix.fromFlat(errOutputBufferData.subarray(0, requiredErrLen), [inputRows, inputCols]);
    this.errOutputBuffer._data.fill(0);
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

  private resolveLoadPayload(data: matrix2d | {
    vocabSize?: number;
    embeddingDim?: number;
    alpha?: number;
    optimizer?: Optimzier;
    status?: StatusLayer;
    padTokenId?: number | null;
    trainable?: boolean;
    weight: matrix2d;
  }): {
    vocabSize?: number;
    embeddingDim?: number;
    alpha?: number;
    optimizer?: Optimzier;
    status?: StatusLayer;
    padTokenId?: number | null;
    trainable?: boolean;
    weight: matrix2d;
  } {
    if (Array.isArray(data)) {
      return { weight: data };
    }

    return {
      vocabSize: data.vocabSize,
      embeddingDim: data.embeddingDim,
      alpha: data.alpha,
      optimizer: data.optimizer,
      status: data.status,
      padTokenId: data.padTokenId,
      trainable: data.trainable ?? true,
      weight: data.weight,
    };
  }

  private extractWeightFromFillSource(source: string | Matrix | number[][] | Float32Array | {
    weight?: number[][];
    layers?: any[];
    name?: string;
    vocabSize?: number;
    embeddingDim?: number;
    trainable?: boolean;
  }): Matrix {
    if (typeof source === "string") {
      const parsed = JSON.parse(readFileSync(source, "utf-8"));
      return this.extractWeightFromJson(parsed);
    }

    if (source instanceof Matrix) {
      this.assertWeightShape(source._shape[0], source._shape[1]);
      return Matrix.fromFlat(new Float32Array(source._data), [...source._shape] as [number, number]);
    }

    if (source instanceof Float32Array) {
      return this.matrixFromFlatWeight(source);
    }

    if (Array.isArray(source)) {
      return this.matrixFromWeightArray(source);
    }

    return this.extractWeightFromJson(source);
  }

  private extractWeightFromJson(source: any): Matrix {
    if (Array.isArray(source)) {
      if (!source[0] || source[0].name !== this.name) {
        throw new Error("Embedding.fillWeight: JSON pretrained weight harus berasal dari Embedding layer atau model dengan layer pertama Embedding.");
      }
      return this.matrixFromWeightArray(source[0].weight);
    }

    if (source && Array.isArray(source.layers)) {
      if (!source.layers[0] || source.layers[0].name !== this.name) {
        throw new Error("Embedding.fillWeight: JSON pretrained weight harus berasal dari Embedding layer atau model dengan layer pertama Embedding.");
      }
      return this.matrixFromWeightArray(source.layers[0].weight);
    }

    if (source && typeof source === "object" && "weight" in source) {
      if (source.name !== undefined && source.name !== this.name) {
        throw new Error("Embedding.fillWeight: JSON pretrained weight harus berasal dari Embedding layer atau model dengan layer pertama Embedding.");
      }
      return this.matrixFromWeightArray(source.weight);
    }

    throw new Error("Embedding.fillWeight: JSON pretrained weight harus berasal dari Embedding layer atau model dengan layer pertama Embedding.");
  }

  private matrixFromFlatWeight(weight: Float32Array): Matrix {
    const expectedLength = this.embeddingDim * this.vocabSize;
    if (weight.length !== expectedLength) {
      throw new Error(`Embedding.fillWeight: dimensi weight tidak cocok. Expected [${this.embeddingDim}, ${this.vocabSize}], got [${this.embeddingDim}, ${weight.length / Math.max(1, this.embeddingDim)}].`);
    }
    for (let i = 0; i < weight.length; i++) {
      if (!Number.isFinite(weight[i])) {
        throw new Error(`Embedding.fillWeight: weight harus berisi finite number. Invalid value pada flat index ${i}.`);
      }
    }
    return Matrix.fromFlat(new Float32Array(weight), [this.embeddingDim, this.vocabSize]);
  }

  private matrixFromWeightArray(weight: number[][]): Matrix {
    const normalizedWeight = this.normalizeNumericMatrix(weight);
    this.assertWeightShape(normalizedWeight.length, normalizedWeight[0]?.length ?? 0);
    return new Matrix({ array: normalizedWeight });
  }

  private normalizeNumericMatrix(weight: unknown): number[][] {
    if (!Array.isArray(weight)) {
      throw new Error("Embedding.fillWeight: weight harus berupa 2D array numerik.");
    }

    const rows = weight.length;
    const normalized: number[][] = new Array(rows);
    let cols = -1;

    for (let i = 0; i < rows; i++) {
      const row = weight[i];
      if (!Array.isArray(row)) {
        throw new Error("Embedding.fillWeight: weight harus berupa 2D array numerik.");
      }
      if (cols === -1) cols = row.length;
      if (row.length !== cols) {
        throw new Error("Embedding.fillWeight: setiap row weight harus memiliki panjang yang sama.");
      }
      normalized[i] = new Array(row.length);
      for (let j = 0; j < row.length; j++) {
        const value = row[j];
        if (typeof value !== "number" || !Number.isFinite(value)) {
          throw new Error(`Embedding.fillWeight: weight harus berisi finite number. Invalid value pada [${i}, ${j}].`);
        }
        normalized[i][j] = value;
      }
    }

    return normalized;
  }

  private assertWeightShape(incomingRows: number, incomingCols: number): void {
    if (incomingRows !== this.embeddingDim || incomingCols !== this.vocabSize) {
      throw new Error(`Embedding.fillWeight: dimensi weight tidak cocok. Expected [${this.embeddingDim}, ${this.vocabSize}], got [${incomingRows}, ${incomingCols}].`);
    }
  }

  private assignWeightPreservingConfig(weight: Matrix): void {
    this.assertWeightShape(weight._shape[0], weight._shape[1]);
    this.weight = Matrix.fromFlat(new Float32Array(weight._data), [...weight._shape] as [number, number]);
    this.optimizerWeight = setOptimizer(this.optimizerName, this.weight._shape, this.alpha);
    this.params = this.vocabSize * this.embeddingDim;
  }

  private normalizeToMatrix(weight: matrix2d): Matrix {
    return new Matrix({ array: this.normalizeNumericMatrix(weight) });
  }

  private forwardWithLayout(x: Matrix, layout: "sample-major" | "time-major"): Matrix {
    const [rows, cols] = x._shape;
    const totalTokens = rows * cols;
    
    if (this.orderedInputBuffer.length < totalTokens) {
      this.orderedInputBuffer = new Int32Array(Math.max(totalTokens, Math.max(1, this.orderedInputBuffer.length * 2)));
    }

    if (layout === "time-major") {
      // Susun token secara time-major: Step 0 (Batch 0, Batch 1, ...), Step 1 (Batch 0, Batch 1, ...), dst.
      let writeIdx = 0;
      for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
          this.orderedInputBuffer[writeIdx++] = this.validateAndNormalizeTokenIndex(x._data[row * cols + col]);
        }
      }
    } else {
      // Susun token secara sample-contiguous: Sample 0 (Step 0, Step 1, ...), Sample 1 (Step 0, Step 1, ...), dst.
      let writeIdx = 0;
      for (let col = 0; col < cols; col++) {
        for (let row = 0; row < rows; row++) {
          this.orderedInputBuffer[writeIdx++] = this.validateAndNormalizeTokenIndex(x._data[row * cols + col]);
        }
      }
    }
    this.inputIndices = this.orderedInputBuffer.subarray(0, totalTokens);

    const seqLen = totalTokens;
    this.inputShape = [rows, cols];
    this.outputShape = [this.embeddingDim, seqLen];

    const requiredOutputLen = this.embeddingDim * seqLen;
    // We add outputBufferData to instance implicitly if not present yet
    let outputBufferData = (this as any).outputBufferData as Float32Array | undefined;
    if (!outputBufferData || outputBufferData.length < requiredOutputLen) {
      outputBufferData = new Float32Array(Math.max(requiredOutputLen, Math.max(1, (outputBufferData?.length ?? 0) * 2)));
      (this as any).outputBufferData = outputBufferData;
    }
    
    this.outputBuffer = Matrix.fromFlat(outputBufferData.subarray(0, requiredOutputLen), [this.embeddingDim, seqLen]);
    const outputData = this.outputBuffer._data;

    if (isNativeAvailable()) {
      embeddingForwardNative(this.inputIndices.subarray(0, totalTokens), this.weight._data, this.vocabSize, this.embeddingDim, this.padTokenId, outputData);
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

  dispose() {
    this.orderedInputBuffer = new Int32Array(0);
    this.outputBuffer = undefined as any;
    (this as any).outputBufferData = new Float32Array(0);
    (this as any).errOutputBufferData = new Float32Array(0);
    (this as any).gradWeightBufferData = new Float32Array(0);
    (this as any).uniqueIndicesBuffer = new Int32Array(0);
    this.gradWeightBuffer = undefined as any;
    this.errOutputBuffer = undefined as any;
  }
}
