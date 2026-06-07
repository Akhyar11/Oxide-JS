export { default as dotProductAddOnly } from "./math/dotProductAddOnly.js";
export { SpikingDense, type SpikingDenseConfig } from "./layers/SpikingDense.js";
export { SpikingDenseBPTT, type SpikingDenseBPTTConfig } from "./layers/SpikingDenseBPTT.js";
export { SpikingEmbedding, type SpikingEmbeddingConfig } from "./layers/SpikingEmbedding.js";
export { SpikingSelfAttention, type SpikingSelfAttentionConfig } from "./layers/SpikingSelfAttention.js";
// export { SpikingSentenceEmbedder, type SpikingSentenceConfig } from "./models/SpikingSentenceEmbedder.js";
export { contrastiveHebbianNativeWrapper, isNativeAvailable } from "./native_backend.js";
