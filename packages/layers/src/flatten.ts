import { mj, engine } from "@oxide-js/core";
import { Matrix } from "@oxide-js/core";
import { StatusLayer } from "@oxide-js/core";

/**
 * Flatten Layer: Melebur matriks multi-dimensi/2D menjadi vector 1 dimensi.
 * Biasanya digunakan sebelum layer Dense di ujung ekor CNN atau Self-Attention.
 */
export default class Flatten {
  name = "flatten layer";
  status: StatusLayer;
  loss = 0;
  params = 0; // Flatten tidak punya bobot
  inputShape: [number, number] = [0, 0];
  outputShape: [number, number] = [0, 0];

  constructor(status: StatusLayer = "input") {
    this.status = status;
  }
  
  forward(x: Matrix): Matrix {
    this.inputShape = [x._shape[0], x._shape[1]];
    const n = x._shape[0] * x._shape[1];
    this.outputShape = [n, 1];
    const out = mj.reshape(x, [n, 1]);

    const tape = engine.tape;
    if (tape) {
      tape.record([x], [out], (grad: Matrix) => {
        const dx = mj.reshape(grad, this.inputShape);
        if (x.grad) x.grad.addInPlace(dx);
        else x.grad = dx;
      });
    }

    return out;
  }

  getParams(): Matrix[] {
    return [];
  }

  update(_alpha: number): void {
    // No trainable parameters
  }
  
  backward(y: Matrix, err: Matrix, _gradOnly = false): Matrix {
    // Pada saat backward pass, error akan diproyeksikan (un-flatten) ke bentuk awal
    return mj.reshape(err, this.inputShape);
  }
  
  resetLoss(): void {
    this.loss = 0;
  }
  
  save() { 
    return { name: this.name, status: this.status }; 
  }

  toKerasConfig() {
    return {
      class_name: "Flatten",
      config: {
        data_format: "channels_last",
        name: `flatten_${Math.floor(Math.random() * 1000)}`,
        trainable: true,
      }
    };
  }
  
  load(): void { }
  
  compile(): void { } // Kosong karena tidak ada bobot (weights)
}
