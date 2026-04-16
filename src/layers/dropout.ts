import { StatusLayer } from "../@types/type";
import mj from "../math";
import Matrix from "../matrix";

export default class Dropout {
  name: string = "dropout layer";
  rate: number;
  mask: Matrix = mj.matrix([]);
  status: StatusLayer;
  
  inputShape: [number, number] = [0, 0];
  outputShape: [number, number] = [0, 0];
  params: number = 0;

  constructor({ rate = 0.5, status = "input" }: { rate?: number; status?: StatusLayer }) {
    this.rate = rate;
    this.status = status;
  }

  save() {
    return {
      name: this.name,
      status: this.status,
      rate: this.rate,
    };
  }

  load({ rate, status }: { rate: number; status: StatusLayer }) {
    this.rate = rate;
    this.status = status;
  }

  forward(x: Matrix): Matrix {
    this.inputShape = [x._shape[0], x._shape[1]];
    this.outputShape = [x._shape[0], x._shape[1]];
    
    // Only scale elements if rate is > 0
    if (this.rate === 0) {
      return x;
    }

    const scale = 1 / (1 - this.rate);
    const data = new Float64Array(x._data.length);
    const maskData = new Float64Array(x._data.length);

    for (let i = 0; i < x._data.length; i++) {
      if (Math.random() >= this.rate) {
        maskData[i] = scale;
        data[i] = x._data[i] * scale;
      } else {
        maskData[i] = 0;
        data[i] = 0;
      }
    }

    this.mask = Matrix.fromFlat(maskData, x._shape);
    return Matrix.fromFlat(data, x._shape);
  }

  backward(y: Matrix, err: Matrix): Matrix {
    if (this.rate === 0) {
      return err;
    }
    return mj.mul(err, this.mask);
  }
}
