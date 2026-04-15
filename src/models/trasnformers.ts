import mj from "../math";
import Matrix from "../matrix";
import { Matrix as MatrixType } from "../@types/type";
import Sequential, { SequentialLayers } from "./sequential";
import SelfAttantion from "../layers/selfAttantion";
import Dense from "../layers/dense";
import { CompileDenseLayers } from "../layers/dense";

interface TransformersConfig {
  units: number;          // ukuran input/embedding
  seqLen?: number;        // panjang sequence input
  denseUnits: number;     // ukuran output layer dense setelah attention
  alpha?: number;
  numHeads?: number;      // jumlah attention head (saat ini single head)
}

/**
 * Basic Transformer model:
 *   Input → [SelfAttantion] → flatten → [Dense (FFN)] → output
 *
 * Arsitektur sederhana tanpa positional encoding.
 * Untuk sequence tasks, input harus berbentuk [units, 1].
 */
export default class Transformers extends Sequential {
  private attentionLayer: SelfAttantion;
  private denseLayer: Dense;
  private attentionShape: [number, number];

  constructor({
    units,
    seqLen = 1,
    denseUnits,
    alpha = 0.01,
  }: TransformersConfig) {
    const attentionLayer = new SelfAttantion({
      units,
      seqLen,
      alpha,
      status: "input",
    });

    const attentionOutput = Math.floor(units / 2);

    const denseLayer = new Dense({
      units: attentionOutput * seqLen,
      outputUnits: denseUnits,
      activation: "linear",
      alpha,
      status: "output",
    });

    super({ layers: [attentionLayer, denseLayer] });
    this.attentionLayer = attentionLayer;
    this.denseLayer = denseLayer;
    this.attentionShape = [attentionOutput, seqLen];
  }

  forward(x: MatrixType): MatrixType {
    // 1. Self-Attention
    const attentionOut = this.attentionLayer.forward(x);
    this.attentionShape = [attentionOut._shape[0], attentionOut._shape[1]];

    // 2. Flatten output attention → [outputUnits * seqLen, 1]
    const n = attentionOut._shape[0] * attentionOut._shape[1];
    const flat = mj.reshape(attentionOut, [n, 1]);

    // 3. Dense feed-forward
    const output = this.denseLayer.forward(flat);
    return output;
  }

  backward(y: MatrixType) {
    // Backward dari Dense → dapatkan gradient ke attention output
    const errDense = this.denseLayer.backward(y, mj.matrix([[]]));
    this.loss = this.denseLayer.loss;

    // Un-flatten kembali ke shape attention output
    const errReshaped = mj.reshape(errDense, this.attentionShape);

    // Backward ke SelfAttantion
    this.attentionLayer.backward(y, errReshaped);
  }

  fit(
    X: MatrixType[],
    y: MatrixType[],
    epochs: number,
    cb: (loss: number) => any = (_) => {}
  ) {
    for (let i = 0; i < epochs; i++) {
      this.denseLayer.resetLoss();
      let epochLoss = 0;

      for (let j = 0; j < X.length; j++) {
        this.forward(X[j]);
        this.backward(y[j]);
        epochLoss = this.denseLayer.loss;
      }

      this.loss = epochLoss;
      cb(this.loss);

      if (this.loss < 0.01) return 0;
    }
  }
}
