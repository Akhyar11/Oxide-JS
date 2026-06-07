import { BaseLayer, LayerConfig, ForwardOptions } from "@oxide-js/layers";
import { Matrix, mj } from "@oxide-js/core";
import { 
    isNativeAvailable, 
    lifStepNativeWrapper, 
    maskSurrogateNativeWrapper,
    applyEmbeddingDeltaNativeWrapper
} from "../native_backend.js";

export interface SpikingEmbeddingConfig extends LayerConfig {
  inputDim: number;
  outputDim: number;
  embeddingsInitializer?: string;
  betaRange?: [number, number];
  thresholdRange?: [number, number];
}

export class SpikingEmbedding extends BaseLayer {
  public inputDim: number;
  public outputDim: number;
  public embeddingsInitializer: string;
  
  public betaRange: [number, number];
  public thresholdRange: [number, number];
  public beta!: Float32Array;
  public threshold!: Float32Array;

  public potentials!: Matrix;
  public lastPotentials?: Matrix;
  public lastInputs?: Matrix;
  public lastSpikes?: Matrix;

  public get embeddings(): Matrix | undefined {
    return this.getParameter("embeddings");
  }

  constructor(config: SpikingEmbeddingConfig) {
    super(config);
    this.inputDim = config.inputDim;
    this.outputDim = config.outputDim;
    this.embeddingsInitializer = config.embeddingsInitializer || "glorot_normal";
    this.betaRange = config.betaRange || [0.8, 0.99];
    this.thresholdRange = config.thresholdRange || [0.01, 0.1];
  }

  public computeOutputShape(inputShape: number[]): number[] {
    const batch = inputShape[0] ?? -1;
    return [batch, this.outputDim];
  }

  public build(inputShape: number[]): void {
    super.build(inputShape);

    const embVal = this.createInitializer(this.embeddingsInitializer, [this.inputDim, this.outputDim]);
    
    // Scale up the embedding values because Glorot Normal makes them too small for large vocabularies (e.g. 32000)
    // which prevents the LIF neurons from ever reaching the threshold.
    const scaleFactor = Math.sqrt(this.inputDim);
    for (let i = 0; i < embVal._data.length; i++) {
        embVal._data[i] *= scaleFactor;
    }

    this.addParameter("embeddings", embVal, true, [this.inputDim, this.outputDim]);
    
    // Inisialisasi beta dan threshold secara acak untuk setiap neuron
    this.beta = new Float32Array(this.outputDim);
    this.threshold = new Float32Array(this.outputDim);
    for (let i = 0; i < this.outputDim; i++) {
        this.beta[i] = this.betaRange[0] + Math.random() * (this.betaRange[1] - this.betaRange[0]);
        this.threshold[i] = this.thresholdRange[0] + Math.random() * (this.thresholdRange[1] - this.thresholdRange[0]); 
    }
    
    // Potentials start at 0, shape [batch, outputDim]. 
    this.potentials = Matrix.fromFlat(new Float32Array(this.outputDim), [1, this.outputDim]); 
  }

  private dotDataBuffer?: Float32Array;
  private outDataBuffer?: Float32Array;

  private ensurePotentialsShape(batch: number) {
    if (this.potentials._shape[0] !== batch || !this.dotDataBuffer) {
       this.potentials = Matrix.fromFlat(new Float32Array(batch * this.outputDim), [batch, this.outputDim]);
       this.dotDataBuffer = new Float32Array(batch * this.outputDim);
       this.outDataBuffer = new Float32Array(batch * this.outputDim);
       this.lastPotentials = Matrix.fromFlat(new Float32Array(batch * this.outputDim), [batch, this.outputDim]);
    }
  }

  public resetState() {
     if (this.potentials) this.potentials._data.fill(0);
     if (this.lastPotentials) this.lastPotentials._data.fill(0);
     this.lastInputs = undefined;
     this.lastSpikes = undefined;
  }

  protected compute(inputs: Matrix, options?: ForwardOptions): Matrix {
    const batch = inputs._shape[0];
    this.ensurePotentialsShape(batch);
    
    // 1. Embedding lookup
    const emb = this.embeddings!;
    const dotData = this.dotDataBuffer!;
    dotData.fill(0);
    
    for (let b = 0; b < batch; b++) {
        const tokenIdx = inputs._data[b]; 
        if (tokenIdx >= 0 && tokenIdx < this.inputDim) {
            const embOffset = tokenIdx * this.outputDim;
            const dotOffset = b * this.outputDim;
            for (let i = 0; i < this.outputDim; i++) {
                dotData[dotOffset + i] = emb._data[embOffset + i];
            }
        }
    }
    
    // 2. Leaky Integrate and Fire (LIF Restore untuk Spiking Murni)
    const outData = this.outDataBuffer!;
    outData.fill(0);
    const outSpikes = Matrix.fromFlat(outData, [batch, this.outputDim]);
    // lastPotentials is already ensured in shape

    if (isNativeAvailable()) {
        lifStepNativeWrapper(
            this.potentials._data,
            dotData,
            outData,
            this.lastPotentials!._data,
            this.beta,
            this.threshold
        );
    } else {
        const potData = this.potentials._data;
        const lpData = this.lastPotentials!._data;
        
        for (let b = 0; b < batch; b++) {
            const offset = b * this.outputDim;
            for (let i = 0; i < this.outputDim; i++) {
                const idx = offset + i;
                potData[idx] = Math.min((potData[idx] * this.beta[i]) + dotData[idx], 1.0); // Clamp potential max 1.0
                lpData[idx] = potData[idx];
            }
            for (let i = 0; i < this.outputDim; i++) {
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
    
    this.lastInputs = inputs;
    this.lastSpikes = outSpikes;

    return outSpikes;
  }

  public learnEmbedding(errorSignal: Matrix, B: Matrix, learningRate: number = 0.01): Matrix {
      // Broadcast error mundur (Feedback Alignment)
      let eHidden = mj.dotProduct(errorSignal, B, undefined, false, false); // E * B
      
      // Surrogate Mask: Boxcar
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
                  const offset = b * this.outputDim;
                  for (let i = 0; i < this.outputDim; i++) {
                      const idx = offset + i;
                      if (Math.abs(pData[idx] - this.threshold[i]) > windowSize) {
                          eData[idx] = 0; 
                      }
                  }
              }
          }
      }

      this.applyEmbeddingDelta(eHidden, learningRate);
      return eHidden;
  }
  
  private applyEmbeddingDelta(errorSignal: Matrix, learningRate: number) {
      if (!this.lastInputs || !this.lastSpikes) {
          throw new Error("[SpikingEmbedding] Cannot run learning before forward() is executed.");
      }
      
      const embeddings = this.embeddings!._data;
      const inputs = this.lastInputs._data;
      const err = errorSignal._data;
      
      const batch = this.lastInputs._shape[0];
      const outputDim = this.outputDim;

      if (isNativeAvailable()) {
          applyEmbeddingDeltaNativeWrapper(
              embeddings,
              inputs,
              err,
              learningRate,
              this.inputDim,
              outputDim
          );
      } else {
          for (let b = 0; b < batch; b++) {
              const tokenIdx = inputs[b];
              if (tokenIdx >= 0 && tokenIdx < this.inputDim) {
                  const embOffset = tokenIdx * outputDim;
                  const errOffset = b * outputDim;
                  for (let j = 0; j < outputDim; j++) {
                      embeddings[embOffset + j] += learningRate * err[errOffset + j];
                      embeddings[embOffset + j] = Math.max(-1.0, Math.min(1.0, embeddings[embOffset + j])); // Clamp weight [-1, 1]
                  }
              }
          }
      }
  }

  public getConfig(): Record<string, any> {
    return {
      ...super.getConfig(),
      inputDim: this.inputDim,
      outputDim: this.outputDim,
      embeddingsInitializer: this.embeddingsInitializer
    };
  }
}
