import { Layers } from "../@types/type";
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
} from "../layers";
import { SequentialLayers } from "../models/sequential";

export type LayerFactory = (data: any) => Layers;

export const layerRegistry: Record<string, LayerFactory> = Object.create(null);

export function registerLayer(name: string, factory: LayerFactory): void {
  layerRegistry[name] = factory;
}

registerLayer("dense layer", (data) => {
  const dense = new Dense({
    units: data.units,
    outputUnits: data.outputUnits,
    activation: data.activation,
    optimizer: data.optimizer,
    status: data.status,
    loss: data.loss,
    clipGradient: data.clipGradient,
  });
  dense.load(data.weight, data.bias, data.clipGradient);
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
  const convolution = new Convolution({
    kernelSize: data.kernelSize,
    inputShape: data.inputShape,
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
    vocabSize: data.vocabSize,
    embeddingDim: data.embeddingDim,
    alpha: data.alpha,
    optimizer: data.optimizer,
    status: data.status,
    padTokenId: data.padTokenId ?? null,
  });
  embedding.load(data.weight);
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
    units: data.units,
    hiddenUnits: data.hiddenUnits,
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
    units: data.units,
    hiddenUnits: data.hiddenUnits,
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
    units: data.units,
    hiddenUnits: data.hiddenUnits,
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

registerLayer("flatten", (data) => new Flatten(data.status));
registerLayer("flatten layer", (data) => new Flatten(data.status));

export default function setLayers(data: any): SequentialLayers {
  const layers: SequentialLayers = [];
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
