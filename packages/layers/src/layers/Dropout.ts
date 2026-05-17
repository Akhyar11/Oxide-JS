import { BaseLayer, LayerConfig } from "../base/BaseLayer.js";
import { Matrix, mj } from "@oxide-js/core";

export interface DropoutConfig extends LayerConfig {
  rate: number;
}

export class Dropout extends BaseLayer {
  public rate: number;

  constructor(config: DropoutConfig) {
    super(config);
    if (config.rate < 0 || config.rate >= 1) {
      throw new Error(`[Dropout] Nilai rate harus berada di rentang [0, 1). Mendapat: ${config.rate}`);
    }
    this.rate = config.rate;
  }

  /**
   * Output shape sama persis dengan input shape
   */
  public computeOutputShape(inputShape: number[]): number[] {
    return [...inputShape];
  }

  /**
   * Forward Pass matematika layer Dropout
   */
  protected compute(inputs: Matrix, isTraining?: boolean): Matrix {
    const training = isTraining ?? this.training;

    // Jika bukan dalam mode training atau rate adalah 0, behaves as an Identity layer
    if (!training || this.rate === 0) {
      return inputs;
    }

    const scale = 1 / (1 - this.rate);
    // Hasilkan mask binary acak berskala (scaled dropout)
    const mask = mj.map(inputs, () => (Math.random() >= this.rate ? scale : 0));

    return mj.mul(inputs, mask);
  }

  /**
   * Konfigurasi Keras
   */
  public getConfig(): Record<string, any> {
    return {
      ...super.getConfig(),
      rate: this.rate
    };
  }
}
