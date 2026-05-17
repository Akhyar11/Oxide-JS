import { BaseLayer, LayerConfig } from "../base/BaseLayer.js";
import { Matrix, mj } from "@oxide-js/core";

export interface ResidualConfig extends LayerConfig {
  layer: BaseLayer;
  shortcut?: BaseLayer;
}

export class Residual extends BaseLayer {
  public layer: BaseLayer;
  public shortcut?: BaseLayer;

  constructor(config: ResidualConfig) {
    super(config);
    if (!config.layer) {
      throw new Error("[Residual] 'layer' wajib disertakan dalam konfigurasi.");
    }
    this.layer = config.layer;
    this.shortcut = config.shortcut;
  }

  public train(): void {
    super.train();
    this.layer.train();
    if (this.shortcut) {
      this.shortcut.train();
    }
  }

  public eval(): void {
    super.eval();
    this.layer.eval();
    if (this.shortcut) {
      this.shortcut.eval();
    }
  }

  public get weights(): Matrix[] {
    const w = [...super.weights];
    w.push(...this.layer.weights);
    if (this.shortcut) {
      w.push(...this.shortcut.weights);
    }
    return w;
  }

  public get trainableWeights(): Matrix[] {
    if (!this.trainable) return [];
    const w = [...super.trainableWeights];
    w.push(...this.layer.trainableWeights);
    if (this.shortcut) {
      w.push(...this.shortcut.trainableWeights);
    }
    return w;
  }

  public get nonTrainableWeights(): Matrix[] {
    const w = [...super.nonTrainableWeights];
    w.push(...this.layer.nonTrainableWeights);
    if (this.shortcut) {
      w.push(...this.shortcut.nonTrainableWeights);
    }
    return w;
  }

  public getTrainableParameters(): Matrix[] {
    if (!this.trainable) return [];
    const params = [...super.getTrainableParameters()];
    params.push(...this.layer.getTrainableParameters());
    if (this.shortcut) {
      params.push(...this.shortcut.getTrainableParameters());
    }
    return params;
  }

  public countParams(): number {
    let count = super.countParams();
    count += this.layer.countParams();
    if (this.shortcut) {
      count += this.shortcut.countParams();
    }
    return count;
  }

  public clearGradients(): void {
    super.clearGradients();
    this.layer.clearGradients();
    if (this.shortcut) {
      this.shortcut.clearGradients();
    }
  }

  public computeOutputShape(inputShape: number[]): number[] {
    return this.layer.computeOutputShape(inputShape);
  }

  public build(inputShape: number[]): void {
    if (this.isBuilt) return;

    this.inputShape = [...inputShape];

    // Build main layer
    this.layer.build(inputShape);
    this.outputShape = this.layer.computeOutputShape(inputShape);

    // Build shortcut layer if provided
    if (this.shortcut) {
      this.shortcut.build(inputShape);
      const shortcutOutputShape = this.shortcut.computeOutputShape(inputShape);
      if (shortcutOutputShape.join(",") !== this.outputShape.join(",")) {
        throw new Error(
          `[Residual] Output shape mismatch. Main layer output shape is [${this.outputShape.join(", ")}], but shortcut output shape is [${shortcutOutputShape.join(", ")}].`
        );
      }
    } else {
      // If no shortcut is provided, inputShape must match main layer output shape
      if (this.inputShape.join(",") !== this.outputShape.join(",")) {
        throw new Error(
          `[Residual] Input shape [${this.inputShape.join(", ")}] must match Output shape [${this.outputShape.join(", ")}] when no shortcut projection layer is provided.`
        );
      }
    }

    this.isBuilt = true;
  }

  protected compute(inputs: Matrix, isTraining?: boolean): Matrix {
    const fx = this.layer.forward(inputs, isTraining);
    const shortcutX = this.shortcut ? this.shortcut.forward(inputs, isTraining) : inputs;
    return mj.add(fx, shortcutX);
  }

  public getConfig(): Record<string, any> {
    return {
      ...super.getConfig(),
      layer: this.layer.getKerasConfig(),
      shortcut: this.shortcut ? this.shortcut.getKerasConfig() : undefined,
    };
  }
}
