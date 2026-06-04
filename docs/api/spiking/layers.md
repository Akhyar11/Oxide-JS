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
| `kernelInitializer` | `string` | `"glorot_normal"` | Initializer for the synaptic weight matrix. |
| `biasInitializer` | `string` | `"zeros"` | Initializer for the bias vector. |

### Heterogeneous LIF Dynamics (Forward Pass)

Unlike standard SNNs that use a global leak factor (`beta`) and `threshold`, Oxide-JS utilizes **Heterogeneous Neuron Dynamics**. Every neuron is initialized with its own random `beta` ($0.8 - 0.99$) and `threshold` ($0.5 - 1.0$). This stochasticity promotes model diversity and prevents premature synchronization or representation collapse.

The `SpikingDense` layer processes incoming binary spikes through the following equation:

1. **Add-Only Integration**: $I(t) = W \cdot S(t) + b$ (implemented via fast lookup additions).
2. **Leaky Integration**: $V(t) = \min(V(t-1) \times \beta_i + I(t), 1.0)$. The potential is **clamped at 1.0** to prevent overflow.
3. **Fire & Reset**: If $V(t) \ge \theta_i$, emit spike $1$ and subtract $\theta_i$ from $V(t)$. Else emit spike $0$.

### Surrogate Gradient Learning

Because the spike function (Heaviside step function) is non-differentiable (derivative is 0 everywhere except at threshold where it is infinite), standard backpropagation fails. `SpikingDense` automatically uses a **Boxcar Surrogate Gradient** during backpropagation:
- Gradients pass through the neuron *only* if the membrane potential $V(t)$ was close to the threshold $\theta_i$.

> **Stability Normalization**: To prevent gradient explosions in temporal learning, all synaptic weights (kernels and biases) are strictly clipped to the `[-1.0, 1.0]` range after every weight update.

```ts
import { SpikingDense } from "@oxide-js/spiking";

const spikingLayer = new SpikingDense({
    units: 256,
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
| `embeddingsInitializer` | `string` | `"glorot_normal"` | Weight initializer. |

*Note: Similar to `SpikingDense`, `SpikingEmbedding` automatically initializes and utilizes heterogeneous neuron dynamics (per-neuron `beta` and `threshold`) and clamps embedding weights to `[-1.0, 1.0]` during updates.*

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

> **Note on Native Acceleration**: Both `SpikingDense` and `SpikingEmbedding` automatically hook into `@oxide-js/core`'s Rust Native Backend. The LIF computations (`lifStepNativeWrapper`), Surrogate Masking, and Add-Only Delta Updates run completely in heavily optimized Rust routines (via **Rayon Data Parallelism**) when native dependencies are present. The Rust implementations handle slicing correctly to ensure `Send + Sync` constraints are satisfied across thread pools.
