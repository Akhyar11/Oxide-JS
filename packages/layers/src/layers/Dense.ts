import { BaseLayer, LayerConfig, type ForwardOptions } from "../base/BaseLayer.js";
import { ActivationType, Matrix, mj } from "@oxide-js/core";

export interface DenseConfig extends LayerConfig {
  units: number;
  activation?: ActivationType;
  useBias?: boolean;
  kernelInitializer?: string;
  biasInitializer?: string;
}

export class Dense extends BaseLayer {
  public units: number;
  public activation: string;
  public useBias: boolean;
  public kernelInitializer: string;
  public biasInitializer: string;

  // Getters cepat untuk akses kernel dan bias
  public get kernel(): Matrix | undefined {
    return this.getParameter("kernel");
  }

  public get bias(): Matrix | undefined {
    return this.getParameter("bias");
  }

  constructor(config: DenseConfig) {
    super(config);
    this.units = config.units;
    this.activation = config.activation || "linear";
    this.useBias = config.useBias ?? true;
    this.kernelInitializer = config.kernelInitializer || "glorot_normal";
    this.biasInitializer = config.biasInitializer || "zeros";
  }

  /**
   * Menghitung output shape secara deterministik [batch, units]
   */
  public computeOutputShape(inputShape: number[]): number[] {
    const batch = inputShape[0] ?? -1;
    return [batch, this.units];
  }

  /**
   * Menginisialisasi parameter kernel [inFeatures, units] dan bias [units, 1]
   */
  public build(inputShape: number[]): void {
    super.build(inputShape);

    const inFeatures = inputShape[inputShape.length - 1];

    // Inisialisasi kernel
    const kernelVal = this.createInitializer(this.kernelInitializer, [inFeatures, this.units]);
    this.addParameter("kernel", kernelVal, true, [inFeatures, this.units]);

    // Inisialisasi bias (jika aktif)
    if (this.useBias) {
      const biasVal = this.createInitializer(this.biasInitializer, [this.units, 1]);
      this.addParameter("bias", biasVal, true, [this.units, 1]);
    }
  }

  /**
   * Forward Pass matematika layer Dense
   */
  protected compute(inputs: Matrix, options?: ForwardOptions): Matrix {
    const kernel = this.kernel!;

    // 1. dot = inputs * kernel
    let dot = mj.dotProduct(inputs, kernel);

    // 2. Tambahkan bias jika digunakan
    if (this.useBias && this.bias) {
      // dot shape: [batch, units], bias shape: [units, 1]
      // Transpose dot ke [units, batch] agar in-place addBias bekerja
      const dotT = mj.transpose(dot);
      mj.addBias(dotT, this.bias);
      dot = mj.transpose(dotT);
    }

    // 3. Aplikasikan fungsi aktivasi
    let output = dot;
    if (this.activation !== "linear") {
      try {
        const isSoftmax = this.activation === "softmax";
        const actFn = this.resolveActivation(this.activation, { row: isSoftmax });
        output = actFn(dot);
      } catch (err) {
        console.warn(
          `[Dense] Activation '${this.activation}' tidak ditemukan atau error: ${(err as Error).message}. Menggunakan 'linear'.`
        );
      }
    }

    return output;
  }

  /**
   * Konfigurasi spesifik Keras
   */
  public getConfig(): Record<string, any> {
    return {
      ...super.getConfig(),
      units: this.units,
      activation: this.activation,
      useBias: this.useBias,
      kernelInitializer: this.kernelInitializer,
      biasInitializer: this.biasInitializer
    };
  }
}
