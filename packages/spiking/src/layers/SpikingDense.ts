import { BaseLayer, LayerConfig, ForwardOptions } from "@oxide-js/layers";
import { Matrix, mj } from "@oxide-js/core";
import { 
    isNativeAvailable, 
    lifStepNativeWrapper, 
    maskSurrogateNativeWrapper, 
    applyAddOnlyDeltaNativeWrapper 
} from "../native_backend.js";
import dotProductAddOnly from "../math/dotProductAddOnly.js";
export interface SpikingDenseConfig extends LayerConfig {
  units: number;
  useBias?: boolean;
  kernelInitializer?: string;
  biasInitializer?: string;
}

export class SpikingDense extends BaseLayer {
  public units: number;
  public useBias: boolean;
  public kernelInitializer: string;
  public biasInitializer: string;
  public beta!: Float32Array;
  public threshold!: Float32Array;

  public potentials!: Matrix;
  public lastPotentials?: Matrix;
  public lastInputs?: Matrix;
  public lastSpikes?: Matrix;

  public get kernel(): Matrix | undefined {
    return this.getParameter("kernel");
  }

  public get bias(): Matrix | undefined {
    return this.getParameter("bias");
  }

  constructor(config: SpikingDenseConfig) {
    super(config);
    this.units = config.units;
    this.useBias = config.useBias ?? true;
    this.kernelInitializer = config.kernelInitializer || "glorot_normal";
    this.biasInitializer = config.biasInitializer || "zeros";
  }

  public computeOutputShape(inputShape: number[]): number[] {
    const batch = inputShape[0] ?? -1;
    return [batch, this.units];
  }

  public build(inputShape: number[]): void {
    super.build(inputShape);

    const inFeatures = inputShape[inputShape.length - 1];

    const kernelVal = this.createInitializer(this.kernelInitializer, [inFeatures, this.units]);
    this.addParameter("kernel", kernelVal, true, [inFeatures, this.units]);

    if (this.useBias) {
      const biasVal = this.createInitializer(this.biasInitializer, [this.units, 1]);
      this.addParameter("bias", biasVal, true, [this.units, 1]);
    }
    
    // Inisialisasi beta dan threshold secara acak untuk setiap neuron
    this.beta = new Float32Array(this.units);
    this.threshold = new Float32Array(this.units);
    for (let i = 0; i < this.units; i++) {
        this.beta[i] = 0.8 + Math.random() * 0.19; 
        this.threshold[i] = 0.5 + Math.random() * 0.5; // Max 1.0
    }
    
    // Inisialisasi state
    this.potentials = Matrix.fromFlat(new Float32Array(this.units), [1, this.units]); 
  }

  private ensurePotentialsShape(batch: number) {
    if (this.potentials._shape[0] !== batch) {
       this.potentials = Matrix.fromFlat(new Float32Array(batch * this.units), [batch, this.units]);
    }
  }

  public resetState() {
     if (this.potentials) this.potentials._data.fill(0);
     this.lastPotentials = undefined;
     this.lastInputs = undefined;
     this.lastSpikes = undefined;
  }

  protected compute(inputs: Matrix, options?: ForwardOptions): Matrix {
    const kernel = this.kernel!;
    const batch = inputs._shape[0];
    this.ensurePotentialsShape(batch);

    // 1. Spiking-optimized matrix multiplication (Add-Only)
    let dot = dotProductAddOnly(inputs, kernel);

    // 2. Add bias
    if (this.useBias && this.bias) {
      mj.addBiasRow(dot, this.bias);
    }

    // 3 & 4. Leaky Integrate, Fire & Reset
    const outData = new Float32Array(batch * this.units);
    const outSpikes = Matrix.fromFlat(outData, [batch, this.units]);
    this.lastPotentials = Matrix.fromFlat(new Float32Array(batch * this.units), [batch, this.units]);

    if (isNativeAvailable()) {
        lifStepNativeWrapper(
            this.potentials._data,
            dot._data,
            outSpikes._data,
            this.lastPotentials._data,
            this.beta,
            this.threshold
        );
    } else {
        const potData = this.potentials._data;
        const dotData = dot._data;
        const lpData = this.lastPotentials._data;
        
        for (let b = 0; b < batch; b++) {
            const offset = b * this.units;
            for (let i = 0; i < this.units; i++) {
                const idx = offset + i;
                potData[idx] = Math.min((potData[idx] * this.beta[i]) + dotData[idx], 1.0); // Clamp potential max 1.0
                lpData[idx] = potData[idx];
            }
            for (let i = 0; i < this.units; i++) {
                const idx = offset + i;
                if (potData[idx] >= this.threshold[i]) {
                    outData[idx] = 1;
                    potData[idx] -= this.threshold[i];
                } else {
                    outData[idx] = 0;
                }
            }
        }
    }

    // Simpan memori untuk belajar
    this.lastInputs = inputs;
    this.lastSpikes = outSpikes;

    return outSpikes;
  }

  public learnOutput(errorSignal: Matrix, learningRate: number = 0.01): Matrix {
      this.applyAddOnlyDelta(errorSignal, learningRate);
      return errorSignal; 
  }

  public learnHidden(errorFromNext: Matrix, B: Matrix, learningRate: number = 0.01): Matrix {
      // Broadcast error mundur
      let eHidden = mj.dotProduct(errorFromNext, B, undefined, false, false); // E * B
      
      // Surrogate Mask: Boxcar (Murni Add-Only mask, tanpa perkalian float!)
      if (this.lastPotentials) {
          const eData = eHidden._data;
          const pData = this.lastPotentials._data;
          const windowSize = 1.0; 
          
          if (isNativeAvailable()) {
              maskSurrogateNativeWrapper(
                  eData,
                  pData,
                  this.threshold,
                  windowSize
              );
          } else {
              const batch = eHidden._shape[0];
              for (let b = 0; b < batch; b++) {
                  const offset = b * this.units;
                  for (let i = 0; i < this.units; i++) {
                      const idx = offset + i;
                      if (Math.abs(pData[idx] - this.threshold[i]) > windowSize) {
                          eData[idx] = 0; 
                      }
                  }
              }
          }
      }

      this.applyAddOnlyDelta(eHidden, learningRate);
      return eHidden;
  }

  private applyAddOnlyDelta(errorSignal: Matrix, learningRate: number) {
      if (!this.lastInputs || !this.lastSpikes) {
          throw new Error("[SpikingDense] Cannot run learning before forward() is executed. 'lastInputs' or 'lastSpikes' is undefined.");
      }
      
      const kernel = this.kernel!._data;
      const inputs = this.lastInputs._data;
      const err = errorSignal._data;
      
      const batch = this.lastInputs._shape[0];
      const inFeatures = this.lastInputs._shape[1];
      const units = this.units;

      if (isNativeAvailable()) {
          const dummyBias = this.useBias && this.bias ? this.bias._data : new Float32Array(0);
          applyAddOnlyDeltaNativeWrapper(
              kernel,
              dummyBias,
              inputs,
              err,
              learningRate,
              batch,
              inFeatures,
              units,
              this.useBias
          );
          
          for(let i = 0; i < kernel.length; i++) kernel[i] = Math.max(-1.0, Math.min(1.0, kernel[i]));
          if (this.useBias && this.bias) {
              const biasData = this.bias._data;
              for(let i = 0; i < biasData.length; i++) biasData[i] = Math.max(-1.0, Math.min(1.0, biasData[i]));
          }
      } else {
          // Delta rule add-only
          for (let b = 0; b < batch; b++) {
              const inOffset = b * inFeatures;
              const errOffset = b * units;
              
              for (let k = 0; k < inFeatures; k++) {
                  // HANYA update jika input menyala (Spike > 0.5) -> Add Only Update!
                  if (inputs[inOffset + k] > 0.5) { 
                      const kOffset = k * units;
                      for (let j = 0; j < units; j++) {
                          kernel[kOffset + j] += learningRate * err[errOffset + j];
                          kernel[kOffset + j] = Math.max(-1.0, Math.min(1.0, kernel[kOffset + j]));
                      }
                  }
              }

              if (this.useBias && this.bias) {
                  const biasData = this.bias._data;
                  for (let j = 0; j < units; j++) {
                      biasData[j] += (learningRate * err[errOffset + j]) / batch;
                      biasData[j] = Math.max(-1.0, Math.min(1.0, biasData[j]));
                  }
              }
          }
      }
  }

  public getConfig(): Record<string, any> {
    return {
      ...super.getConfig(),
      units: this.units,
      useBias: this.useBias,
      kernelInitializer: this.kernelInitializer,
      biasInitializer: this.biasInitializer
    };
  }
}
