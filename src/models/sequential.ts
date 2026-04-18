import { readFileSync, writeFileSync } from "fs";
import { Layers, Matrix } from "../@types/type";
import { setLayers } from "../utils";
import { CompileDenseLayers, Dense, Convolution } from "../layers";
import mj from "../math";

export type SequentialLayers = Layers[];

export default class Sequential {
  layers: SequentialLayers;
  loss = 0;
  protected isTrainingMode = true;
  constructor({ layers = [] }: { layers?: SequentialLayers } = {}) {
    this.layers = layers;
  }

  summary() {
    console.log("========== Model Info ==========");
    let totalParams = 0;
    for (let layer of this.layers) {
      console.log(`Layer name   : ${layer.name}`);
      console.log(`Layer input  : [${layer.inputShape}]`);
      console.log(`Layer output : [${layer.outputShape}]`);
      console.log(`Layer param  : ${layer.params}`);
      console.log("");
      totalParams += layer.params;
    }

    console.log("Total params =", totalParams);
    console.log("========== End Info ==========");
  }

  add(layer: Layers) {
    this.layers.push(layer);
  }

  save(path: string) {
    const data = [];
    for (let layer of this.layers) {
      data.push(layer.save());
    }
    const dataJson = JSON.stringify(data);
    writeFileSync(path, dataJson);
  }

  load(path: string) {
    const dataJson = readFileSync(path, "utf-8");
    const data = JSON.parse(dataJson);
    this.layers = [];
    this.layers = setLayers(data);
  }

  compile(config: CompileDenseLayers) {
    for (let layer of this.layers) {
      if (typeof (layer as any).compile === "function") {
        (layer as any).compile(config);
      }
    }
  }

  forward(x: Matrix): Matrix {
    let input = x;
    for (let layer of this.layers) {
      input = layer.forward(input);
    }
    return input;
  }

  backward(y: Matrix) {
    let err = mj.matrix([[]]);
    for (let i = this.layers.length - 1; i >= 0; i--) {
      err = this.layers[i].backward(y, err);
      if (this.layers[i].status === "output") this.loss = (this.layers[i] as any).loss;
    }
  }

  train(): this {
    this.isTrainingMode = true;
    for (const layer of this.layers) {
      if (typeof (layer as any).setTrainingMode === "function") {
        (layer as any).setTrainingMode(true);
      }
    }
    return this;
  }

  eval(): this {
    this.isTrainingMode = false;
    for (const layer of this.layers) {
      if (typeof (layer as any).setTrainingMode === "function") {
        (layer as any).setTrainingMode(false);
      }
    }
    return this;
  }

  predict(x: Matrix): Matrix {
    const wasTraining = this.isTrainingMode;
    this.eval();
    const out = this.forward(x);
    if (wasTraining) this.train();
    return out;
  }

  fit(
    X: Matrix[],
    y: Matrix[],
    epochs: number,
    cb: (err: number) => any = (_) => { },
  ) {
    this.train();
    for (let i = 0; i < epochs; i++) {
      // Reset akumulasi loss di setiap epoch
      for (let layer of this.layers) {
        if (typeof (layer as any).resetLoss === "function") {
          (layer as any).resetLoss();
        }
      }
      for (let j in X) {
        this.forward(X[j]);
        this.backward(y[j]);
      }
      cb(this.loss);
      if (this.loss < 0.01) {
        return 0;
      }
    }
  }
}
