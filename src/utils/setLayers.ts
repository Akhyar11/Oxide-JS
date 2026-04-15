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
      });
      dense.load(layer.weight, layer.bias);
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
      });
      convolution.load(layer.kernel, layer.bias);
      layers.push(convolution);
    } else if (layer.name === "embedding layer") {
      // Lazy import untuk menghindari circular dep 
      const { Embedding } = require("../layers");
      const embedding = new Embedding({
        vocabSize: layer.vocabSize,
        embeddingDim: layer.embeddingDim,
        alpha: layer.alpha,
        optimizer: layer.optimizer,
        status: layer.status
      });
      embedding.load(layer.weight);
      layers.push(embedding);
    } else {
      console.warn(`[setLayers] Layer tidak dikenal dan dilewati: '${layer.name}'`);
    }
  }

  return layers;
}
