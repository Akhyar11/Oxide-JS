import { readFileSync } from "fs";
import Sequential, { SequentialLayers } from "./sequential";
import { Matrix } from "../@types/type";
import { setLayers } from "../utils";
import { FitConfig, FitResult } from "../@types/fitConfig";

export default class DimentionalityReduction extends Sequential {
  layersEncode: SequentialLayers = [];
  layersDecode: SequentialLayers = [];
  constructor({ layers = [] }: { layers?: SequentialLayers } = {}) {
    super({ layers });
    let isLayerEncode = true;
    for (let layer of this.layers) {
      if (isLayerEncode) {
        this.layersEncode.push(layer);
        if (layer.status === "outputReduction") isLayerEncode = false;
      } else {
        this.layersDecode.push(layer);
      }
    }
    if (this.layers.length > 0 && this.layersDecode.length === 0) {
      console.warn(
        "[DimentionalityReduction] Tidak ada layer dengan status='outputReduction'. " +
        "Semua layer masuk ke encoder dan decoder akan kosong."
      );
    }
  }

  load(path: string): void {
    const dataJson = readFileSync(path, "utf-8");
    const data = JSON.parse(dataJson);
    this.layersEncode = [];
    this.layersDecode = [];
    this.layers = setLayers(data);
    let isLayerEncode = true;
    for (let layer of this.layers) {
      if (isLayerEncode) {
        this.layersEncode.push(layer);
        if (layer.status === "outputReduction") isLayerEncode = false;
      } else {
        this.layersDecode.push(layer);
      }
    }
  }

  encode(x: Matrix): Matrix {
    let input = x;
    for (let layer of this.layersEncode) {
      input = layer.forward(input);
    }
    return input;
  }

  decode(enc: Matrix): Matrix {
    let input = enc;
    for (let layer of this.layersDecode) {
      input = layer.forward(input);
    }
    return input;
  }

  fit(
    X: Matrix[],
    y: Matrix[],
    epochs: number,
    config?: FitConfig
  ): FitResult;
  fit(
    X: Matrix[],
    y: Matrix[],
    epochs: number,
    cb?: (loss: number) => any
  ): FitResult;
  fit(
    X: Matrix[],
    epochs: number,
    config?: FitConfig
  ): FitResult;
  fit(
    X: Matrix[],
    epochs: number,
    cb?: (loss: number) => any
  ): FitResult;
  fit(
    X: Matrix[],
    yOrEpoch: Matrix[] | number,
    epochsOrConfig?: number | FitConfig | ((loss: number) => any),
    configOrCb: FitConfig | ((loss: number) => any) = {}
  ): FitResult {
    if (typeof yOrEpoch === "number") {
      const epochs = yOrEpoch;
      const cfg = (epochsOrConfig ?? {}) as FitConfig | ((loss: number) => any);
      return super.fit(X, X, epochs, cfg as any);
    }

    const y = yOrEpoch;
    const epochs = epochsOrConfig as number;
    return super.fit(X, y, epochs, configOrCb as any);
  }
}
