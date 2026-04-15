import mj from "../math";
import Matrix from "../matrix";
import { StatusLayer } from "../@types/type";

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
   * Input shape:  [dModel, seqLen]
   * Output shape: [dModel, seqLen] (sama, hanya ditambah posisi)
   */
  forward(x: Matrix): Matrix {
    const seqLen = x._shape[1];
    this.inputShape = [this.dModel, seqLen];
    this.outputShape = [this.dModel, seqLen];

    // Ambil slice PE table sesuai panjang sequence
    const cols = x._shape[1];
    const result = new Float64Array(this.dModel * seqLen);
    const xData = x._data;
    const peData = this.peTable._data;
    const peCols = this.peTable._shape[1];

    for (let i = 0; i < this.dModel; i++) {
      const xOffset = i * cols;
      const peOffset = i * peCols;
      const outOffset = i * seqLen;
      for (let j = 0; j < seqLen; j++) {
        result[outOffset + j] = xData[xOffset + j] + peData[peOffset + j];
      }
    }

    return Matrix.fromFlat(result, [this.dModel, seqLen]);
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
}
