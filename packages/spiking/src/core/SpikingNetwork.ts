export class SpikingNetwork {
  public numNeurons: number;
  public potentials: Float32Array;
  public thresholds: Float32Array;
  public spikes: Uint8Array;
  
  // Adjacency List for sparse, add-only processing
  // postSynapticIndices[i] contains an array of post-synaptic neuron indices 'j' that receive signals from neuron 'i'
  public postSynapticIndices: number[][];
  // weights[i] contains the corresponding weights for the connections in postSynapticIndices[i]
  public weights: Float32Array[];

  // Membrane decay factor
  public beta: number;

  constructor(numNeurons: number, beta: number = 0.9, defaultThreshold: number = 1.0) {
    this.numNeurons = numNeurons;
    this.potentials = new Float32Array(numNeurons);
    this.thresholds = new Float32Array(numNeurons);
    this.spikes = new Uint8Array(numNeurons);
    this.postSynapticIndices = Array.from({ length: numNeurons }, () => []);
    this.weights = Array.from({ length: numNeurons }, () => new Float32Array(0));
    this.beta = beta;

    for (let i = 0; i < numNeurons; i++) {
      this.thresholds[i] = defaultThreshold;
    }
  }

  /**
   * Defines a connection from neuron 'pre' to neuron 'post' with a given weight.
   */
  public connect(pre: number, post: number, weight: number): void {
    if (pre < 0 || pre >= this.numNeurons || post < 0 || post >= this.numNeurons) {
      throw new Error(`Neuron index out of bounds. Must be between 0 and ${this.numNeurons - 1}`);
    }
    
    const targets = this.postSynapticIndices[pre];
    const w = this.weights[pre];
    
    // Check if connection already exists
    const idx = targets.indexOf(post);
    if (idx !== -1) {
      w[idx] = weight; // Update weight
      return;
    }

    // Add new connection
    this.postSynapticIndices[pre].push(post);
    const newWeights = new Float32Array(w.length + 1);
    newWeights.set(w);
    newWeights[w.length] = weight;
    this.weights[pre] = newWeights;
  }

  /**
   * Injects an external current or spike directly into a neuron's potential.
   */
  public injectCurrent(neuronIdx: number, current: number): void {
    if (neuronIdx >= 0 && neuronIdx < this.numNeurons) {
      this.potentials[neuronIdx] += current;
    }
  }

  /**
   * Advances the network by one time step using Add-Only Sparse Processing.
   */
  public step(): void {
    // 1. Accumulate pre-synaptic spikes (Add-Only, Multiplication-Free matrix operation)
    for (let i = 0; i < this.numNeurons; i++) {
      if (this.spikes[i] === 1) {
        const targets = this.postSynapticIndices[i];
        const w = this.weights[i];
        // Only addition operations here!
        for (let k = 0; k < targets.length; k++) {
          const j = targets[k];
          this.potentials[j] += w[k];
        }
      }
    }

    // Reset spikes for the current time step
    this.spikes.fill(0);

    // 2. Membrane potential decay, threshold check, and fire
    for (let j = 0; j < this.numNeurons; j++) {
      // Leaky integration
      this.potentials[j] *= this.beta;

      // Fire spike if threshold is crossed
      if (this.potentials[j] >= this.thresholds[j]) {
        this.spikes[j] = 1;
        // Soft reset: subtract threshold
        this.potentials[j] -= this.thresholds[j];
      }
    }
  }

  /**
   * Resets the internal state of the network.
   */
  public resetState(): void {
    this.potentials.fill(0);
    this.spikes.fill(0);
  }

  /**
   * Serializes the network topology and parameters to a plain JSON object.
   * State (potentials and spikes) are NOT saved, only the trained weights.
   */
  public toJSON(): object {
    return {
      numNeurons: this.numNeurons,
      beta: this.beta,
      thresholds: Array.from(this.thresholds),
      postSynapticIndices: this.postSynapticIndices,
      weights: this.weights.map(w => Array.from(w))
    };
  }

  /**
   * Creates a SpikingNetwork instance from a serialized JSON object.
   */
  public static fromJSON(data: any): SpikingNetwork {
    const net = new SpikingNetwork(data.numNeurons, data.beta);
    
    if (data.thresholds) {
      net.thresholds.set(data.thresholds);
    }

    if (data.postSynapticIndices && data.weights) {
      net.postSynapticIndices = data.postSynapticIndices;
      net.weights = data.weights.map((wArray: number[]) => new Float32Array(wArray));
    }

    return net;
  }
}
