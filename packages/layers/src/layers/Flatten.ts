import { BaseLayer, LayerConfig, type ForwardOptions } from "../base/BaseLayer.js";
import { Matrix, mj } from "@oxide-js/core";

export interface FlattenConfig extends LayerConfig {}

export class Flatten extends BaseLayer {
  constructor(config?: FlattenConfig) {
    super(config || {});
  }

  /**
   * Mengubah logical shape multi-dimensi [batch, d1, d2, ...] menjadi 2D [batch, d1 * d2 * ...]
   */
  public computeOutputShape(inputShape: number[]): number[] {
    if (inputShape.length === 0) {
      return [1, 1];
    }
    const batch = inputShape[0] ?? -1;
    const features = inputShape.slice(1).reduce((a, b) => a * b, 1);
    return [batch, features];
  }

  /**
   * Forward Pass matematika layer Flatten.
   * Mereshape physical matrix menjadi 2D [batch, features]
   */
  protected compute(inputs: Matrix, options?: ForwardOptions): Matrix {
    const batch = inputs._shape[0];
    const totalElements = inputs._data.length;
    const features = totalElements / batch;

    if (inputs._shape[1] === features) {
      return inputs;
    }

    return mj.reshape(inputs, [batch, features]);
  }

  /**
   * Konfigurasi Keras
   */
  public getConfig(): Record<string, any> {
    return super.getConfig();
  }
}
