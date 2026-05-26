import { SpikingNetwork } from "../core/SpikingNetwork.js";

/**
 * Konfigurasi untuk Spike-Timing-Dependent Plasticity (STDP)
 */
export interface STDPConfig {
  learningRate: number;
  tauPlus: number;     // Decay rate for pre-synaptic trace (e.g., 0.8)
  tauMinus: number;    // Decay rate for post-synaptic trace (e.g., 0.8)
  aPlus: number;       // Max weight change for LTP (Long-Term Potentiation)
  aMinus: number;      // Max weight change for LTD (Long-Term Depression)
  wMax: number;        // Maximum weight limit
  wMin: number;        // Minimum weight limit
}

export class STDP {
  public network: SpikingNetwork;
  public config: STDPConfig;

  // Traces to remember "recent" spikes without needing a history buffer
  public preTraces: Float32Array;
  public postTraces: Float32Array;

  constructor(network: SpikingNetwork, config?: Partial<STDPConfig>) {
    this.network = network;
    this.config = {
      learningRate: config?.learningRate ?? 0.01,
      tauPlus: config?.tauPlus ?? 0.8,
      tauMinus: config?.tauMinus ?? 0.8,
      aPlus: config?.aPlus ?? 1.0,
      aMinus: config?.aMinus ?? 1.0,
      wMax: config?.wMax ?? 5.0,
      wMin: config?.wMin ?? -5.0,
    };

    const N = network.numNeurons;
    this.preTraces = new Float32Array(N);
    this.postTraces = new Float32Array(N);
  }

  /**
   * Called ONCE per time step AFTER network.step()
   * Updates traces and applies STDP weight changes based on spikes.
   */
  public updateWeights(): void {
    const N = this.network.numNeurons;
    const spikes = this.network.spikes;

    // 1. Update trace decay and trace spikes
    for (let i = 0; i < N; i++) {
      this.preTraces[i] *= this.config.tauPlus;
      this.postTraces[i] *= this.config.tauMinus;

      if (spikes[i] === 1) {
        this.preTraces[i] = 1.0;
        this.postTraces[i] = 1.0;
      }
    }

    // 2. Event-driven weight updates (Only loop through actual connections)
    // STDP rule:
    // If Pre spikes: LTD -> W -= trace_post * aMinus
    // If Post spikes: LTP -> W += trace_pre * aPlus
    
    for (let i = 0; i < N; i++) {
      const targets = this.network.postSynapticIndices[i];
      const w = this.network.weights[i];

      const preSpiked = spikes[i] === 1;

      for (let k = 0; k < targets.length; k++) {
        const j = targets[k];
        const postSpiked = spikes[j] === 1;

        let dw = 0;

        // Long-Term Potentiation (Pre fired, then Post fired)
        if (postSpiked) {
          dw += this.config.learningRate * this.config.aPlus * this.preTraces[i];
        }

        // Long-Term Depression (Post fired, then Pre fired)
        if (preSpiked) {
          dw -= this.config.learningRate * this.config.aMinus * this.postTraces[j];
        }

        if (dw !== 0) {
          w[k] += dw;

          // Clip weights to prevent them from exploding
          if (w[k] > this.config.wMax) w[k] = this.config.wMax;
          if (w[k] < this.config.wMin) w[k] = this.config.wMin;
        }
      }
    }
  }
}
