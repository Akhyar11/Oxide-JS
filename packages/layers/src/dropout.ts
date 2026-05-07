import { mj, engine, StatusLayer } from "@oxide-js/core";
import { Matrix } from "@oxide-js/core";

export default class Dropout {
  name: string = "dropout layer";
  rate: number;
  mask: Matrix = mj.matrix([]);
  status: StatusLayer;
  private training: boolean = false;

  inputShape: [number, number] = [0, 0];
  outputShape: [number, number] = [0, 0];
  params: number = 0;
  private outputBuffer: Matrix = mj.matrix([]);

  constructor({ rate = 0.5, status = "input" }: { rate?: number; status?: StatusLayer }) {
    this.rate = rate;
    this.status = status;
    this.applyStatusTraining(status);
  }

  save() {
    return {
      name: this.name,
      status: this.status,
      rate: this.rate,
    };
  }

  toKerasConfig() {
    return {
      class_name: "Dropout",
      config: {
        rate: this.rate,
        noise_shape: null,
        seed: null,
        name: `dropout_${Math.floor(Math.random() * 1000)}`,
        trainable: true,
      }
    };
  }

  load({ rate, status }: { rate: number; status: StatusLayer }) {
    this.rate = rate;
    this.status = status;
    this.applyStatusTraining(status);
  }

  getParams(): Matrix[] {
    return [];
  }

  update(_alpha: number): void {
    // Dropout has no trainable parameters
  }

  forward(x: Matrix): Matrix {
    this.inputShape = [x._shape[0], x._shape[1]];
    this.outputShape = [x._shape[0], x._shape[1]];

    // Hanya lakukan dropout JIKA statusnya adalah 'train'
    // Jika 'test' atau status lain, kembalikan input tanpa modifikasi
    if (!this.training || this.rate === 0) {
      return x;
    }

    const scale = 1 / (1 - this.rate);
    if (this.outputBuffer._shape[0] !== x._shape[0] || this.outputBuffer._shape[1] !== x._shape[1]) {
      this.outputBuffer = Matrix.fromFlat(new Float32Array(x._data.length), x._shape);
      this.mask = Matrix.fromFlat(new Float32Array(x._data.length), x._shape);
    }

    const data = this.outputBuffer._data;
    const maskData = this.mask._data;

    for (let i = 0; i < x._data.length; i++) {
      if (Math.random() >= this.rate) {
        maskData[i] = scale;
        data[i] = x._data[i] * scale;
      } else {
        maskData[i] = 0;
        data[i] = 0;
      }
    }

    const tape = engine.tape;
    if (tape && this.training && this.rate > 0) {
      const currentMask = this.mask.clone();
      tape.record([x], [this.outputBuffer], (grad: Matrix) => {
        // dx = grad * mask
        const dx = mj.mul(grad, currentMask);
        if (x.grad) x.grad.addInPlace(dx);
        else x.grad = dx;
      });
    }

    return this.outputBuffer;
  }

  backward(y: Matrix, err: Matrix, _gradOnly = false): Matrix {
    if (!this.training || this.rate === 0) {
      return err;
    }
    return mj.mul(err, this.mask);
  }

  setTrainingMode(training: boolean): void {
    this.training = training;
    this.status = training ? "train" : "test";
  }

  isTraining(): boolean {
    return this.training;
  }

  private applyStatusTraining(status: StatusLayer): void {
    if (status === "train") this.training = true;
    else this.training = false;
  }
}
