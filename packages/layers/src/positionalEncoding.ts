import { mj } from "@oxide-js/core";
import { Matrix } from "@oxide-js/core";
import { StatusLayer } from "@oxide-js/core";

/**
 * Positional Encoding Layer
 * 
 * Menambahkan informasi posisi ke embedding vector.
 * Tanpa ini, model tidak tahu urutan kata: "saya makan nasi" = "nasi makan saya".
 * 
 * Menggunakan sinusoidal encoding (seperti paper "Attention Is All You Need"):
 *   PE(pos, 2i)   = sin(pos / 10000^(2i/dModel))
 *   PE(pos, 2i+1) = cos(pos / 10000^(2i/dModel))
 * 
 * Layer ini tidak punya parameter yang di-train (fixed encoding).
 */
export default class PositionalEncoding {
  name = "positional encoding";
  dModel: number;       // dimensi embedding
  maxSeqLen: number;    // panjang sequence maksimum
  inputShape: [number, number];
  outputShape: [number, number];
  params: number = 0;   // Tidak ada parameter trainable
  status: StatusLayer;
  loss: number = 0;

  // Tabel PE yang sudah diprecompute: [dModel, maxSeqLen]
  private peTable: Matrix;
  private resultBuffer: Matrix | null = null;
  private inferredPadMask: boolean[] = [];

  constructor({
    dModel,
    maxSeqLen = 512,
    status = "norm",
  }: {
    dModel: number;
    maxSeqLen?: number;
    status?: StatusLayer;
  }) {
    this.dModel = dModel;
    this.maxSeqLen = maxSeqLen;
    this.status = status;
    this.inputShape = [dModel, 0];
    this.outputShape = [dModel, 0];

    // Precompute tabel PE
    // Shape: [dModel, maxSeqLen]
    const pe: number[][] = [];
    for (let i = 0; i < dModel; i++) {
      pe[i] = [];
      for (let pos = 0; pos < maxSeqLen; pos++) {
        const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / dModel);
        if (i % 2 === 0) {
          pe[i][pos] = Math.sin(angle);
        } else {
          pe[i][pos] = Math.cos(angle);
        }
      }
    }
    this.peTable = new Matrix({ array: pe });
  }

  save() {
    return {
      name: this.name,
      status: this.status,
      dModel: this.dModel,
      maxSeqLen: this.maxSeqLen,
    };
  }

  /**
   * Forward: tambahkan positional encoding ke input embedding
   * Input shape:  [dModel, seqLen * batchSize]  (sample-major flattened)
   * Output shape: [dModel, seqLen * batchSize]
   *
   * @param x - embedding input [dModel, totalTokens]
   * @param positionOffset - absolute position of the first token in the sequence (default 0).
   *   Set to a non-zero value when the input has been left-trimmed so that real tokens
   *   retain their original absolute positions in the PE table.
   * @param seqLen - actual per-batch sequence length used for cycling column indices to
   *   positions within a sequence.  Defaults to maxSeqLen (original behavior).
   * @param padMask - optional sample-major mask where true means the whole token column is PAD.
   */
  forward(x: Matrix, positionOffset = 0, seqLen?: number, padMask?: boolean[]): Matrix {
    const actualTotalTokens = x._shape[1];
    const cycleLen = seqLen ?? this.maxSeqLen;
    this.inputShape = [this.dModel, actualTotalTokens];
    this.outputShape = [this.dModel, actualTotalTokens];

    const cols = actualTotalTokens;

    if (!this.resultBuffer || this.resultBuffer._shape[0] !== this.dModel || this.resultBuffer._shape[1] !== actualTotalTokens) {
      this.resultBuffer = mj.zeros([this.dModel, actualTotalTokens]);
    }
    const result = this.resultBuffer._data;
    result.fill(0);

    const xData = x._data;
    const peData = this.peTable._data;
    const peCols = this.peTable._shape[1];
    const effectivePadMask = padMask ?? this.inferPadColumns(x);

    for (let i = 0; i < this.dModel; i++) {
      const xOffset = i * cols;
      const peOffset = i * peCols;
      const outOffset = i * cols;
      for (let j = 0; j < cols; j++) {
        const localPos = j % cycleLen;
        const absolutePos = positionOffset + localPos;
        if (absolutePos >= this.maxSeqLen) {
          throw new Error(
            `PositionalEncoding: absolutePos ${absolutePos} (positionOffset=${positionOffset} + localPos=${localPos}) exceeds maxSeqLen=${this.maxSeqLen}. ` +
            `Increase maxSeqLen in the model configuration or reduce positionOffset.`
          );
        }
        const val = xData[xOffset + j];
        result[outOffset + j] = effectivePadMask[j] ? 0 : val + peData[peOffset + absolutePos];
      }
    }

    return this.resultBuffer;
  }

  /**
   * Backward: PE adalah konstanta, gradien langsung diteruskan tanpa modifikasi
   */
  backward(_y: Matrix, err: Matrix): Matrix {
    return err;
  }

  resetLoss(): void {
    this.loss = 0;
  }

  private inferPadColumns(x: Matrix): boolean[] {
    const [rows, cols] = x._shape;
    if (this.inferredPadMask.length !== cols) {
      this.inferredPadMask = new Array<boolean>(cols);
    }
    this.inferredPadMask.fill(true);

    const data = x._data;
    for (let j = 0; j < cols; j++) {
      for (let i = 0; i < rows; i++) {
        if (data[i * cols + j] !== 0) {
          this.inferredPadMask[j] = false;
          break;
        }
      }
    }

    return this.inferredPadMask;
  }
}
