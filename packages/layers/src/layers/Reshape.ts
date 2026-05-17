import { BaseLayer, LayerConfig } from "../base/BaseLayer.js";
import { Matrix, mj } from "@oxide-js/core";

export interface ReshapeConfig extends LayerConfig {
  targetShape: number[];
}

export class Reshape extends BaseLayer {
  public targetShape: number[];

  constructor(config: ReshapeConfig) {
    super(config);
    if (!config.targetShape || config.targetShape.length === 0) {
      throw new Error("[Reshape] 'targetShape' wajib diberikan dan tidak boleh kosong.");
    }
    this.targetShape = config.targetShape;
  }

  /**
   * Menghitung output shape logis [batch, ...targetShape] dengan dukungan inferensi -1
   */
  public computeOutputShape(inputShape: number[]): number[] {
    const batch = inputShape[0] ?? -1;
    const totalInputFeatures = inputShape.slice(1).reduce((a, b) => a * b, 1);

    // Cari tahu apakah ada dimensi -1
    let minusOneIndex = -1;
    let otherProduct = 1;
    for (let i = 0; i < this.targetShape.length; i++) {
      if (this.targetShape[i] === -1) {
        if (minusOneIndex !== -1) {
          throw new Error("[Reshape] Hanya boleh ada satu dimensi bernilai -1 di targetShape.");
        }
        minusOneIndex = i;
      } else {
        otherProduct *= this.targetShape[i];
      }
    }

    const resolvedShape = [...this.targetShape];
    if (minusOneIndex !== -1) {
      if (totalInputFeatures % otherProduct !== 0) {
        throw new Error(
          `[Reshape] Total fitur input ${totalInputFeatures} tidak dapat dibagi rata dengan dimensi target ${otherProduct}.`
        );
      }
      resolvedShape[minusOneIndex] = totalInputFeatures / otherProduct;
    } else {
      const targetProduct = this.targetShape.reduce((a, b) => a * b, 1);
      if (totalInputFeatures !== targetProduct) {
        throw new Error(
          `[Reshape] Total fitur input (${totalInputFeatures}) tidak sama dengan dimensi target (${targetProduct}).`
        );
      }
    }

    return [batch, ...resolvedShape];
  }

  /**
   * Forward Pass matematika layer Reshape.
   * Mereshape physical matrix menjadi [batch, physicalFeatures]
   */
  protected compute(inputs: Matrix, isTraining?: boolean): Matrix {
    const batch = inputs._shape[0];
    
    // Dapatkan output shape logis yang sesungguhnya (tanpa -1)
    const logicalOutputShape = this.computeOutputShape(inputs._shape);
    const physicalFeatures = logicalOutputShape.slice(1).reduce((a, b) => a * b, 1);

    if (inputs._shape[1] === physicalFeatures) {
      return inputs;
    }

    return mj.reshape(inputs, [batch, physicalFeatures]);
  }

  /**
   * Konfigurasi Keras
   */
  public getConfig(): Record<string, any> {
    return {
      ...super.getConfig(),
      targetShape: this.targetShape
    };
  }
}
