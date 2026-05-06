import { FitConfig, FitResult } from "@oxide-js/core";
import { Matrix } from "@oxide-js/core";

export interface TrainableModel {
  forward(x: Matrix, batchSize?: number): Matrix;
  backward(y: Matrix, batchSize?: number): void;
  fit(X: Matrix[], y: Matrix[], epochs: number, config?: FitConfig): FitResult;
  predict(x: Matrix): Matrix;
  train(): this;
  eval(): this;
}
