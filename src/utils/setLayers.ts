import { Activation, Convolution, Dense } from "../layers";
import { SequentialLayers } from "../models/sequential";

export default function setLayers(data: any) {
  const layers: SequentialLayers = [];
  for (let layer of data) {
    if (layer.name === "dense layer") {
      const dense = new Dense({
        units: layer.units,
        outputUnits: layer.outputUnits,
        activation: layer.activation,
        optimizer: layer.optimizer,
        status: layer.status,
        loss: layer.loss,
        clipGradient: layer.clipGradient,
      });
      dense.load(layer.weight, layer.bias, layer.clipGradient);
      layers.push(dense);
    } else if (layer.name === "activation layer") {
      const activation = new Activation({
        activation: layer.activation,
        status: layer.status,
        loss: layer.loss,
      });
      layers.push(activation);
    } else if (layer.name === "convolution layer") {
      const convolution = new Convolution({
        kernelSize: layer.kernelSize,
        inputShape: layer.inputShape,
        activation: layer.activation,
        loss: layer.loss,
        optimizer: layer.optimizer,
        status: layer.status,
        clipGradient: layer.clipGradient,
      });
      convolution.load(layer.kernel, layer.bias, layer.clipGradient);
      layers.push(convolution);
    } else if (layer.name === "embedding layer") {
      const { Embedding } = require("../layers");
      const embedding = new Embedding({
        vocabSize: layer.vocabSize,
        embeddingDim: layer.embeddingDim,
        alpha: layer.alpha,
        optimizer: layer.optimizer,
        status: layer.status,
        padTokenId: layer.padTokenId ?? null,
      });
      embedding.load(layer.weight);
      layers.push(embedding);
    } else if (layer.name === "positional encoding") {
      const { PositionalEncoding } = require("../layers");
      const pe = new PositionalEncoding({
        dModel: layer.dModel,
        maxSeqLen: layer.maxSeqLen,
        status: layer.status,
      });
      layers.push(pe);
    } else if (layer.name === "layer normalization") {
      const { LayerNormalization } = require("../layers");
      const ln = new LayerNormalization({
        units: layer.units,
        status: layer.status,
        clipGradient: layer.clipGradient,
      });
      ln.load(layer.gamma, layer.beta, layer.clipGradient);
      layers.push(ln);
    } else if (layer.name === "self attention layer") {
      const { SelfAttention } = require("../layers");
      const attn = new SelfAttention({
        units: layer.units,
        alpha: layer.alpha,
        status: layer.status,
        clipGradient: layer.clipGradient,
      });
      attn.load(layer.q, layer.k, layer.v, layer.clipGradient);
      layers.push(attn);
    } else if (layer.name === "flatten") {
      const { Flatten } = require("../layers");
      layers.push(new Flatten());
    } else {
      console.warn(`[setLayers] Layer tidak dikenal dan dilewati: '${layer.name}'`);
    }
  }

  return layers;
}
