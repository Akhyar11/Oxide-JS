# 🗣️ Spiking Sentence Embedder API Reference

The `SpikingSentenceEmbedder` is a high-level model container, similar to `Sequential`, but specifically tailored for creating topological semantic representations of entire sentences using **Biomimetic Spiking Neural Networks**.

It combines a `SpikingEmbedding` layer and a `SpikingDense` layer into a single sequence processing engine. The ultimate goal is to convert an array of token IDs into a single, highly compressed **Semantic Vector (Dense Float32Array)** of shape `[1, embeddingDim]` that correctly identifies cosine similarity relationships (e.g. for Vector Databases like ChromaDB).

---

## 1. `SpikingSentenceEmbedder`

### Constructor

```ts
import { SpikingSentenceEmbedder } from "@oxide-js/spiking";

const model = new SpikingSentenceEmbedder(
    vocabSize,     // Total number of tokens in the vocabulary
    embeddingDim   // Dimension of the output semantic vector (e.g. 256)
);
```

### Architecture Pipeline

When `forward(tokens)` is called, the embedder performs the following sequence:

1. **Temporal Unrolling**: The input `tokens` array represents time steps.
2. **Current Generation (`SpikingEmbedding`)**: For each time step $t$, the token $x_t$ triggers the Embedding lookup to produce an input current.
3. **LIF Dynamics (`SpikingEmbedding`)**: The embedding neurons integrate this current and emit binary spikes.
4. **Synaptic Transmission (`SpikingDense`)**: The binary spikes are transmitted to the `SpikingDense` layer using *Add-Only* operations (no floating point multiplication).
5. **Output LIF Dynamics (`SpikingDense`)**: The dense layer integrates the incoming spikes.
6. **Time Averaging**: After all time steps are processed, the accumulated internal membrane potentials ($V$) of the final `SpikingDense` layer are averaged over time. This continuous mean vector represents the final **Semantic Vector**.

---

## 2. High-Level Native Contrastive Learning

Training an SNN using Backpropagation Through Time (BPTT) with Surrogate Gradients for semantic clustering often leads to poor performance or vector collapse.

To solve this, `SpikingSentenceEmbedder` exposes a native Continuous Bag of Words (CBOW) Hebbian learning method called `learnContrastive`.

### `learnContrastive()`

This method automatically:
1. Performs the forward pass to get the positive context mean vector.
2. Calls the internal `SpikingEmbedding` layer's `learnHebbian()` method to apply Hebbian contrastive updates.
3. Automatically avoids Representation Collapse by pushing the token representations away from a buffer of negative contexts.

```ts
/**
 * Executes a single Contrastive Learning step.
 * 
 * @param tokens Array of token IDs in the current sentence
 * @param negativeContexts Array of mean vectors from previous sentences
 * @param learningRate Step size for the update
 * @param marginPositive Pull force towards the positive context (default: 0.1)
 * @param marginNegative Push force away from negative contexts (default: 0.05)
 * @returns The generated mean semantic vector (useful for pushing to the negative buffer)
 */
const meanVector = model.learnContrastive(
    tokens, 
    negativeContexts, 
    0.01, 
    0.1, 
    0.05
);
```

### E2E Training Example

Here is how you can train a semantic embedding topology efficiently using a sliding history buffer of negative contexts:

```ts
import { SpikingSentenceEmbedder } from "@oxide-js/spiking";
import { Matrix } from "@oxide-js/core";

const vocabSize = 30000;
const embeddingDim = 256;
const model = new SpikingSentenceEmbedder(vocabSize, embeddingDim);

// History buffer for negative sampling (Contrastive Learning)
const historyBuffer: Matrix[] = [];
const historyMax = 1000;
const learningRate = 0.01;

const sentencesTokens = [
    [12, 45, 87, 10], // Token IDs for "The cat sat on the mat"
    [32, 11, 99, 14], // Token IDs for "A dog barked loudly"
    // ...
];

for (const tokens of sentencesTokens) {
    // Pick 5 random negative contexts from the buffer
    const negatives = [];
    for (let i = 0; i < 5; i++) {
        if (historyBuffer.length > 0) {
            const randIdx = Math.floor(Math.random() * historyBuffer.length);
            negatives.push(historyBuffer[randIdx]);
        }
    }

    // 1. Train natively using Contrastive Hebbian Learning
    const meanVec = model.learnContrastive(
        tokens, 
        negatives, 
        learningRate,
        0.1,  // Positive Margin
        0.05  // Negative Margin
    );

    // 2. Add the resulting mean vector to the negative history buffer
    historyBuffer.push(meanVec);
    if (historyBuffer.length > historyMax) {
        historyBuffer.shift();
    }
}
```

---

## 3. Serialization (Saving/Loading)

`SpikingSentenceEmbedder` extends the `BaseModel` abstract class, which means it inherits all native Keras-style serialization hooks and automatic JSON JSON array compression for weights.

```ts
import * as fs from "fs";

// Save
const serialized = model.serialize();
fs.writeFileSync("spiking_model.json", JSON.stringify(serialized));

// Load
const loadedData = JSON.parse(fs.readFileSync("spiking_model.json", "utf-8"));
model.setWeights(loadedData.weights);
```
*Note: Thanks to the automated Float32Array-to-Array compression implemented in Oxide-JS v2.5.0, these JSON exports are up to 4x smaller than raw JSON Float32Arrays.*
