import { ActivationType, Cost, StatusLayer, engine, mj } from "@oxide-js/core";
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

  toKerasConfig() {
    return {
      class_name: "Activation",
      config: {
        activation: this.activationName,
        name: `activation_${Math.floor(Math.random() * 1000)}`,
        trainable: false,
      }
    };
  }

  load(data: any) {
    if (data.activation !== undefined) {
      this.activation = setActivation(data.activation);
      this.activationName = data.activation;
    }
    if (data.loss !== undefined) {
      this.lossFunc = setLoss(data.loss);
      this.lossName = data.loss;
    }
    if (data.status !== undefined) {
      this.status = data.status;
    }
  }

  getParams(): Matrix[] {
    return [];
  }

  update(_alpha: number): void {
    // Activation layer has no trainable parameters
  }

  forward(x: Matrix) {
    [this.result, this.dResult] = this.activation(x);
    
    const tape = engine.tape;
    if (tape) {
      // Note: the individual activation functions (sigmoid, relu, etc.) 
      // in core already record themselves in the tape.
      // So we don't strictly need to record the Activation LAYER itself
      // unless we want to handle the 'output' status loss here.
    }
    
    return this.result;
  }

  backward(y: Matrix, err: Matrix, gradOnly = false) {
    let e = err;
    let loss;
    if (this.status === "output") {
      [loss, e] = this.lossFunc(y, this.result);
      this.index++;
      this.sumLoss += loss;
      this.loss = this.sumLoss / this.index;
    }
    const errActivation = mj.mul(e, this.dResult);
    if (!gradOnly) this.update(0);
    return errActivation;
  }
}
