import { Cost } from "../@types/type";
import MeanSquerError from "../cost/mse";
import CategoricalCrossEntropy, { BinaryCrossEntropy } from "../cost/crossEntropy";
import SoftmaxCrossEntropy from "../cost/softmaxCrossEntropy";

export default function setLoss(cost: Cost) {
  switch (cost) {
    case "mse":
      return MeanSquerError;
    case "crossEntropy":
      return CategoricalCrossEntropy;
    case "binaryCrossEntropy":
      return BinaryCrossEntropy;
    case "softmaxCrossEntropy":
      return SoftmaxCrossEntropy;
    default:
      throw new Error(`Loss function '${cost}' tidak dikenal. Pilih: mse, crossEntropy, binaryCrossEntropy, softmaxCrossEntropy`);
  }
}

