import { Cost, StatusLayer, Optimizer } from "@oxide-js/core";
import {
  Activation,
  AdaptiveMemoryRNN,
  Convolution,
  Dense,
  Dropout,
  Embedding,
  Flatten,
  GRU,
  LayerNormalization,
  LSTM,
  MultiHeadAttention,
  PositionalEncoding,
  RNN,
  SelfAttention,
  MemoryBank,
  AttentionPooling,
} from "../index.js";
// We'll define a local type or import from a shared types file in layers
import { Layers } from "../types.js";

export type LayerFactory = (data: any) => Layers;

export const layerRegistry: Record<string, LayerFactory> = Object.create(null);

export function registerLayer(name: string, factory: LayerFactory): void {
  layerRegistry[name] = factory;
}

registerLayer("dense layer", (data) => {
  const dense = new Dense({
    units: data.units_input ?? data.units_in ?? 0, // Fallback for input units if not provided
    outputUnits: data.outputUnits ?? data.units,
    activation: data.activation,
    optimizer: data.optimizer,
    status: data.status,
    loss: data.loss,
    clipGradient: data.clipGradient,
  });
  dense.load(data.weight ?? data.kernel, data.bias, data.clipGradient);
  return dense;
});

registerLayer("activation layer", (data) => {
  const activation = new Activation({
    activation: data.activation,
    status: data.status,
    loss: data.loss,
  });
  activation.load(data);
  return activation;
});

registerLayer("convolution layer", (data) => {
  const kernelSize = data.kernel_size ?? data.kernelSize ?? [3, 3]; // Default fallback if missing
  const inputShape = data.inputShape ?? data.batch_input_shape?.slice(1) ?? [0, 0];
  const convolution = new Convolution({
    kernelSize,
    inputShape,
    activation: data.activation,
    loss: data.loss,
    optimizer: data.optimizer,
    status: data.status,
    clipGradient: data.clipGradient,
  });
  convolution.load(data.kernel, data.bias, data.clipGradient);
  return convolution;
});

registerLayer("embedding layer", (data) => {
  const embedding = new Embedding({
    vocabSize: data.vocabSize ?? data.input_dim,
    embeddingDim: data.embeddingDim ?? data.output_dim,
    alpha: data.alpha,
    optimizer: data.optimizer,
    status: data.status,
    padTokenId: (data.padTokenId !== undefined ? data.padTokenId : (data.mask_zero ? 0 : null)),
    trainable: data.trainable ?? true,
  });
  embedding.load(data);
  return embedding;
});

registerLayer("positional encoding", (data) => {
  return new PositionalEncoding({
    dModel: data.dModel,
    maxSeqLen: data.maxSeqLen,
    status: data.status,
  });
});

registerLayer("layer normalization", (data) => {
  const layerNormalization = new LayerNormalization({
    units: data.units,
    status: data.status,
    clipGradient: data.clipGradient,
  });
  layerNormalization.load(data.gamma, data.beta, data.clipGradient);
  return layerNormalization;
});

registerLayer("self attention layer", (data) => {
  const selfAttention = new SelfAttention({
    units: data.units,
    alpha: data.alpha,
    status: data.status,
    clipGradient: data.clipGradient,
  });
  selfAttention.load(data.q, data.k, data.v, data.clipGradient);
  return selfAttention;
});

registerLayer("multi head attention layer", (data) => {
  const multiHeadAttention = new MultiHeadAttention({
    units: data.units,
    heads: data.heads,
    seqLen: data.seqLen,
    alpha: data.alpha,
    status: data.status,
    clipGradient: data.clipGradient,
  });
  multiHeadAttention.load(data);
  return multiHeadAttention;
});

registerLayer("dropout layer", (data) => {
  const dropout = new Dropout({
    rate: data.rate,
    status: data.status,
  });
  dropout.load(data);
  return dropout;
});

registerLayer("rnn layer", (data) => {
  const rnn = new RNN({
    units: data.units_input ?? data.units_in ?? 0,
    hiddenUnits: data.hiddenUnits ?? data.units,
    activation: data.activation,
    returnSequences: data.returnSequences,
    returnState: data.returnState,
    stateful: data.stateful,
    alpha: data.alpha,
    optimizer: data.optimizer,
    status: data.status,
    clipGradient: data.clipGradient,
    loss: data.loss,
  });
  rnn.load(data);
  return rnn;
});

registerLayer("lstm layer", (data) => {
  const lstm = new LSTM({
    units: data.units_input ?? data.units_in ?? 0,
    hiddenUnits: data.hiddenUnits ?? data.units,
    forgetBias: data.forgetBias,
    returnSequences: data.returnSequences,
    returnState: data.returnState,
    stateful: data.stateful,
    alpha: data.alpha,
    optimizer: data.optimizer,
    status: data.status,
    clipGradient: data.clipGradient,
    loss: data.loss,
  });
  lstm.load(data);
  return lstm;
});

registerLayer("gru layer", (data) => {
  const gru = new GRU({
    units: data.units_input ?? data.units_in ?? 0,
    hiddenUnits: data.hiddenUnits ?? data.units,
    returnSequences: data.returnSequences,
    returnState: data.returnState,
    stateful: data.stateful,
    bidirectional: data.bidirectional,
    alpha: data.alpha,
    optimizer: data.optimizer,
    status: data.status,
    clipGradient: data.clipGradient,
    loss: data.loss,
  });
  gru.load(data);
  return gru;
});

registerLayer("adaptive memory rnn layer", (data) => {
  const layer = new AdaptiveMemoryRNN({
    units: data.units,
    hiddenUnits: data.hiddenUnits,
    activation: data.activation,
    memorySlots: data.memorySlots,
    memoryDim: data.memoryDim,
    returnSequences: data.returnSequences,
    returnState: data.returnState,
    stateful: data.stateful,
    alpha: data.alpha,
    optimizer: data.optimizer,
    status: data.status,
    clipGradient: data.clipGradient,
    loss: data.loss,
  });
  layer.load(data);
  return layer;
});

registerLayer("memory bank layer", (data) => {
  const layer = new MemoryBank({
    units: data.units,
    memorySlots: data.memorySlots,
    outputUnits: data.outputUnits,
    mode: data.mode,
    similarity: data.similarity,
    readTopK: data.readTopK,
    persistence: data.persistence,
    resetOnInit: data.resetOnInit,
    writeEnabled: data.writeEnabled,
    overwriteThreshold: data.overwriteThreshold,
    alpha: data.alpha,
    optimizer: data.optimizer,
    clipGradient: data.clipGradient,
    status: data.status,
  });
  layer.load(data);
  return layer;
});

registerLayer("attention pooling layer", (data) => {
  const layer = new AttentionPooling({
    units: data.units,
    maxTokens: data.maxTokens,
    alpha: data.alpha,
    optimizer: data.optimizer,
    status: data.status,
    clipGradient: data.clipGradient,
  });
  layer.load(data);
  return layer;
});

registerLayer("flatten layer", (data) => new Flatten(data.status));

// Keras/TFJS Compatibility Mapping
registerLayer("Dense", layerRegistry["dense layer"]);
registerLayer("Embedding", layerRegistry["embedding layer"]);
registerLayer("Conv2D", layerRegistry["convolution layer"]);
registerLayer("Dropout", layerRegistry["dropout layer"]);
registerLayer("Flatten", layerRegistry["flatten layer"]);
registerLayer("SimpleRNN", layerRegistry["rnn layer"]);
registerLayer("LSTM", layerRegistry["lstm layer"]);
registerLayer("GRU", layerRegistry["gru layer"]);
registerLayer("LayerNormalization", layerRegistry["layer normalization"]);
registerLayer("SelfAttention", layerRegistry["self attention layer"]);
registerLayer("SelfAttention", layerRegistry["self attention layer"]);
registerLayer("MultiHeadAttention", layerRegistry["multi head attention layer"]);
registerLayer("AdaptiveMemoryRNN", layerRegistry["adaptive memory rnn layer"]);
registerLayer("MemoryBank", layerRegistry["memory bank layer"]);
registerLayer("Activation", layerRegistry["activation layer"]);
registerLayer("PositionalEncoding", layerRegistry["positional encoding"]);


registerLayer("AttentionPooling", (data) => {
  const pooling = new AttentionPooling({
    units: data.units ?? data.params,
    maxTokens: data.maxTokens ?? 1024,
    status: data.status,
    alpha: data.alpha,
    optimizer: data.optimizer,
    clipGradient: data.clipGradient,
  });
  if (data.weight && data.bias) {
    pooling.load(data);
  }
  return pooling;
});

export default function setLayers(data: any): Layers[] {
  const layers: Layers[] = [];
  for (const layer of data) {
    const factory = layerRegistry[layer.name];
    if (!factory) {
      console.warn(`[setLayers] Layer tidak dikenal dan dilewati: '${layer.name}'`);
      continue;
    }
    layers.push(factory(layer));
  }

  return layers;
}
