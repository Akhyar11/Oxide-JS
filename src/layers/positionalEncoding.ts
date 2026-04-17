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
  private resultBuffer: Matrix | null = null;

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
    const actualSeqLen = x._shape[1];
    const seqLen = this.maxSeqLen; // Use the configured maxSeqLen for cycling
    this.inputShape = [this.dModel, actualSeqLen];
    this.outputShape = [this.dModel, actualSeqLen];

    const cols = actualSeqLen;
    
    if (!this.resultBuffer || this.resultBuffer._shape[0] !== this.dModel || this.resultBuffer._shape[1] !== actualSeqLen) {
      this.resultBuffer = mj.zeros([this.dModel, actualSeqLen]);
    }
    const result = this.resultBuffer._data;
    result.fill(0);
    
    const xData = x._data;
    const peData = this.peTable._data;
    const peCols = this.peTable._shape[1];

    for (let i = 0; i < this.dModel; i++) {
      const xOffset = i * cols;
      const peOffset = i * peCols;
      const outOffset = i * cols;
      for (let j = 0; j < cols; j++) {
        const peIdx = j % seqLen; // Cycle PE for each batch item
        const val = xData[xOffset + j];
        result[outOffset + j] = val === 0 ? 0 : val + peData[peOffset + peIdx];
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
}
