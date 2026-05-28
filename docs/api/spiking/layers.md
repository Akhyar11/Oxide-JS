# 🧠 Spiking Layers API Reference

The `@oxide-js/spiking` module introduces specialized **Biomimetic Neural Network** layers. These layers model the behavior of biological neurons using **Leaky Integrate-and-Fire (LIF)** dynamics and output discrete binary spikes (`1` or `0`) instead of continuous floating-point activations. 

Because the outputs are discrete, mathematical operations such as vector multiplications can be reduced to pure **Add-Only** updates (`dotProductAddOnly`), bypassing computationally expensive floating-point multiplications entirely on the forward pass.

---

## 1. `SpikingDense`

The `SpikingDense` layer is the fundamental building block of a Spiking Neural Network (SNN). It implements a fully connected feed-forward architecture, but with biological LIF dynamics. 

### Configuration (`SpikingDenseConfig`)

Extends standard `LayerConfig`.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `units` | `number` | **Required** | The dimensionality of the output space (number of neurons). |
| `useBias` | `boolean` | `true` | Whether the layer uses a bias vector. |
| `beta` | `number` | `0.9` | The decay factor for the membrane potential (Leakage rate). |
| `threshold` | `number` | `1.0` | The voltage threshold at which the neuron fires a spike. |
| `kernelInitializer` | `string` | `"glorot_normal"` | Initializer for the synaptic weight matrix. |
| `biasInitializer` | `string` | `"zeros"` | Initializer for the bias vector. |

### LIF Dynamics (Forward Pass)

The `SpikingDense` layer processes incoming binary spikes through the following equation:

1. **Add-Only Integration**: $I(t) = W \cdot S(t) + b$ (implemented via fast lookup additions).
2. **Leaky Integration**: $V(t) = V(t-1) \times \beta + I(t)$.
3. **Fire & Reset**: If $V(t) \ge \theta$, emit spike $1$ and subtract $\theta$ from $V(t)$. Else emit spike $0$.

### Surrogate Gradient Learning

Because the spike function (Heaviside step function) is non-differentiable (derivative is 0 everywhere except at threshold where it is infinite), standard backpropagation fails. `SpikingDense` automatically uses a **Boxcar Surrogate Gradient** during backpropagation:
- Gradients pass through the neuron *only* if the membrane potential $V(t)$ was close to the threshold $\theta$.

```ts
import { SpikingDense } from "@oxide-js/spiking";

const spikingLayer = new SpikingDense({
    units: 256,
    beta: 0.9,
    threshold: 1.0,
    useBias: true
});

// Assume inputs are binary spikes Matrix (e.g., from another spiking layer)
const outputSpikes = spikingLayer.forward(inputs); 
```

---

## 2. `SpikingEmbedding`

The `SpikingEmbedding` layer acts as the entry point for NLP tasks in an SNN. It behaves like a standard Embedding lookup table, translating discrete token IDs into dense vectors. However, instead of passing these dense vectors directly, it treats them as **input currents** to internal LIF neurons, which then emit spikes.

### Configuration (`SpikingEmbeddingConfig`)

Extends standard `LayerConfig`.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `inputDim` | `number` | **Required** | Size of the vocabulary (max integer index + 1). |
| `outputDim` | `number` | **Required** | Dimension of the continuous embedding space. |
| `beta` | `number` | `0.9` | Decay factor for the internal LIF neurons. |
| `threshold` | `number` | `1.0` | Firing threshold for the internal LIF neurons. |
| `embeddingsInitializer` | `string` | `"glorot_normal"` | Weight initializer. |

### Word2Vec CBOW-style Hebbian Contrastive Learning

A major issue in SNNs trained with backpropagation is **Representation Collapse**, where semantic spaces converge into a single dense point due to vanishing surrogate gradients across deep time steps.

`SpikingEmbedding` offers a native **Hebbian Contrastive Learning** update method (`learnHebbian`) that applies Continuous Bag of Words (CBOW) logic directly to the embedding weights, skipping the LIF surrogate gradient bottleneck entirely.

```ts
/**
 * @param tokens Array of token IDs in the current sentence
 * @param positiveContext Mean vector of the current sentence
 * @param negativeContexts Array of mean vectors from random past sentences
 * @param learningRate Step size for the update
 * @param marginPositive Pull force towards the positive context
 * @param marginNegative Push force away from negative contexts
 */
spikingEmbedding.learnHebbian(
    tokens,
    positiveContext,
    negativeContexts,
    0.01,
    0.1,
    0.05
);
```

By manually orchestrating `learnHebbian`, you can train SNN embeddings for downstream Semantic Search tasks (like ChromaDB indexing) without encountering vector collapse.

---

> **Note on Native Acceleration**: Both `SpikingDense` and `SpikingEmbedding` automatically hook into `@oxide-js/core`'s Rust Native Backend. The LIF computations (`lifStepNativeWrapper`), Surrogate Masking, and Add-Only Delta Updates run completely in heavily optimized Rust routines (via Rayon parallel processing) when native dependencies are present.
