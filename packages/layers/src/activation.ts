import { ActivationType, Cost, StatusLayer } from "@oxide-js/core";
import { mj } from "@oxide-js/core";
import { Matrix } from "@oxide-js/core";
import { setActivation } from "@oxide-js/core";
import { setLoss } from "@oxide-js/core";

export default class Activation {
  name: string = "activation layer";
  inputShape = [null, null];
  outputShape = [null, null];
  params = 0;
  loss = 0;
  activation: Function;
  lossFunc: Function;
  status: StatusLayer;
  activationName: ActivationType;
  lossName: Cost;
  private result: Matrix = mj.matrix([]);
  private dResult: Matrix = mj.matrix([]);
  private sumLoss: number = 0;
  private index: number = 0;
  constructor({
    activation,
    status = "input",
    loss = "mse",
  }: {
    activation: ActivationType;
    status?: StatusLayer;
    loss?: Cost;
  }) {
    this.activation = setActivation(activation);
    this.activationName = activation;
    this.status = status;
    this.lossFunc = setLoss(loss);
    this.lossName = loss;
  }

  save() {
    const data = {
      name: this.name,
      activation: this.activationName,
      status: this.status,
      loss: this.lossName,
    };
    return data;
  }

  load({
    activation,
    loss,
    status,
  }: {
    activation: ActivationType;
    loss: Cost;
    status: StatusLayer;
  }) {
    this.activation = setActivation(activation);
    this.lossFunc = setLoss(loss);
    this.activationName = activation;
    this.lossName = loss;
    this.status = status;
  }

  forward(x: Matrix) {
    [this.result, this.dResult] = this.activation(x);
    return this.result;
  }

  backward(y: Matrix, err: Matrix) {
    let e = err;
    let loss;
    if (this.status === "output") {
      [loss, e] = this.lossFunc(y, this.result);
      this.index++;
      this.sumLoss += loss;
      this.loss = this.sumLoss / this.index;
    }
    const errActivation = mj.mul(e, this.dResult);
    return errActivation;
  }
}
