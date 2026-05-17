import { BaseLayer, LayerConfig, type ForwardOptions } from "../base/BaseLayer.js";
import { ActivationType, Matrix } from "@oxide-js/core";

export interface ActivationConfig extends LayerConfig {
  activation: ActivationType;
}

export class Activation extends BaseLayer {
  public activation: ActivationType;

  constructor(config: ActivationType | ActivationConfig) {
    if (typeof config === "string") {
      super();
      this.activation = config;
    } else {
      super(config);
      this.activation = config.activation;
    }
    // Validate activation immediately to throw error on invalid configuration
    try {
      this.resolveActivation(this.activation);
    } catch (err) {
      throw new Error(`[Activation] Invalid activation function '${this.activation}': ${(err as Error).message}`);
    }
  }

  /**
   * Output shape sama persis dengan input shape
   */
  public computeOutputShape(inputShape: number[]): number[] {
    return [...inputShape];
  }

  /**
   * Forward Pass menerapkan fungsi aktivasi ke inputs
   */
  protected compute(inputs: Matrix, options?: ForwardOptions): Matrix {
    try {
      const isSoftmax = this.activation === "softmax";
      const actFn = this.resolveActivation(this.activation, { row: isSoftmax });
      return actFn(inputs);
    } catch (err) {
      console.warn(
        `[Activation] Activation '${this.activation}' tidak ditemukan atau error: ${(err as Error).message}. Mengembalikan input asli.`
      );
      return inputs;
    }
  }

  /**
   * Konfigurasi spesifik Keras
   */
  public getConfig(): Record<string, any> {
    return {
      ...super.getConfig(),
      activation: this.activation
    };
  }
}
