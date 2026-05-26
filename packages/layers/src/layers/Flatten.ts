import { BaseLayer, LayerConfig, type ForwardOptions } from "../base/BaseLayer.js";
import { Matrix, mj } from "@oxide-js/core";

export interface FlattenConfig extends LayerConfig { }

export class Flatten extends BaseLayer {
  constructor(config?: FlattenConfig) {
    super(config || {});
  }

  public computeOutputShape(inputShape: number[]): number[] {
    if (inputShape.length === 0) {
      return [1, 1];
    }
    const total = inputShape.reduce((a, b) => a * b, 1);
    return [1, total];
  }

  protected compute(inputs: Matrix, options?: ForwardOptions): Matrix {
    return mj.reshape(inputs, [1, inputs._shape[0] * inputs._shape[1]]);
  }

  /**
   * Konfigurasi Keras
   */
  public getConfig(): Record<string, any> {
    return super.getConfig();
  }
}
