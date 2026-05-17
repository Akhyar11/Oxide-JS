# 🏷️ TypeScript Types API Reference

The **[types](file:///home/akhyar/Dokumen/Code/NODE_JS/Oxide-JS/packages/core/src/@types/type.ts)** module provides standardized types, signatures, and interfaces for the entire **Oxide-JS** monorepo ecosystem. 

---

## 🧮 Numeric & Tensor Structures

### 1. `vector`
An alias representing a flat list of numbers.
- **Definition**: `type vector = number[]`
- **Example**:
  ```ts
  import { vector } from "@oxide-js/core";
  const line: vector = [1.0, 2.0, 3.0];
  ```

### 2. `matrix2d`
An alias representing nested 2-dimensional numbers.
- **Definition**: `type matrix2d = number[][]`
- **Example**:
  ```ts
  import { matrix2d } from "@oxide-js/core";
  const grid: matrix2d = [
    [1, 2],
    [3, 4]
  ];
  ```

### 3. `matrix3d`
An alias representing nested 3-dimensional numbers.
- **Definition**: `type matrix3d = number[][][]`
- **Example**:
  ```ts
  import { matrix3d } from "@oxide-js/core";
  const cube: matrix3d = [
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
  ];
  ```

### 4. `MatrixShape`
Tuple signature defining matrix dimensions `[rows, cols]`.
- **Definition**: `type MatrixShape = [number, number]`
- **Example**:
  ```ts
  import { MatrixShape } from "@oxide-js/core";
  const shape: MatrixShape = [128, 64];
  ```

### 5. `MatrixFlatData`
Specifies typed array structures allowed as backing data inside matrices.
- **Definition**: `type MatrixFlatData = Float32Array | Float64Array`
- **Example**:
  ```ts
  import { MatrixFlatData } from "@oxide-js/core";
  const flatBuffer: MatrixFlatData = new Float32Array([1.0, 2.0, 3.0]);
  ```

### 6. `MatrixCollection`
Accepts a matrix object or a raw number for scalar operations.
- **Definition**: `type MatrixCollection = Matrix | number`
- **Example**:
  ```ts
  import { MatrixCollection, Matrix } from "@oxide-js/core";
  
  function applyOffset(input: Matrix, offset: MatrixCollection) {
    if (typeof offset === "number") {
      return input.add(offset);
    }
    return input.add(offset);
  }
  ```

---

## ⚙️ Model Tuning & Callback Configuration

### 1. `FitConfig`
Configuration structure controlling parameters for high-level model fitting loops.
- **Definition**:
  ```ts
  interface FitConfig {
    batchSize?: number;
    autodiff?: boolean;             // Tracks dynamic tape recordings instead of manual backpropagation paths
    validationSplit?: number;       // Ratio of dataset row segments split for validation [0.0 - 1.0]
    earlyStoppingPatience?: number; // Epoch plateau tolerance before stopping training early
    shuffle?: boolean;              // Re-shuffle training batches every epoch
    verbose?: boolean;              // Print progress logs to console
    monitorMetric?: "loss" | "valLoss";
    minDelta?: number;              // Minimum improvement to qualify as progress
    mode?: "min" | "max";
    trimPadding?: boolean;          // Dynamically trim PAD tokens (Transformer sequence optimization)
    paddingSide?: "left" | "right";
    onEpochEnd?: (epoch: number, loss: number, valLoss?: number) => void;
  }
  ```
- **Example**:
  ```ts
  import { FitConfig } from "@oxide-js/core";

  const config: FitConfig = {
    batchSize: 32,
    autodiff: true,
    validationSplit: 0.1,
    earlyStoppingPatience: 5,
    verbose: true,
    onEpochEnd: (epoch, loss, valLoss) => {
      console.log(`Epoch ${epoch} finished. Train Loss: ${loss}, Val Loss: ${valLoss}`);
    }
  };
  ```

### 2. `FitResult`
Object returned after completing fitting cycles, containing history logs.
- **Definition**:
  ```ts
  interface FitResult {
    history: {
      loss: number[];
      valLoss?: number[];
    };
    bestEpoch: number;
    bestLoss: number;
    stoppedEarly: boolean;
    stoppingEpoch?: number;
  }
  ```
- **Example**:
  ```ts
  import { FitResult } from "@oxide-js/core";

  function printResults(result: FitResult) {
    console.log("Best training epoch completed:", result.bestEpoch);
    console.log("Lowest recorded loss value:", result.bestLoss);
    if (result.stoppedEarly) {
      console.log(`Training stopped early at epoch: ${result.stoppingEpoch}`);
    }
  }
  ```

---

## 🏷️ Tag Enumerations

### 1. `ActivationType`
Supported string tags identifying layer activation routines.
- **Definition**:
  ```ts
  type ActivationType = 
    | "sigmoid" | "tanh" | "relu" | "lRelu" | "linear" | "softmax"
    | "elu" | "gelu" | "hardsigmoid" | "hardswish" | "mish" | "selu"
    | "softplus" | "softsign" | "swish";
  ```
- **Example**:
  ```ts
  import { ActivationType } from "@oxide-js/core";
  const activeType: ActivationType = "relu";
  ```

### 2. `Optimizer`
Supported string tags identifying training optimization algorithms.
- **Definition**:
  ```ts
  type Optimizer = "sgd" | "adaGrad" | "momentum" | "nag" | "adam";
  ```
- **Example**:
  ```ts
  import { Optimizer } from "@oxide-js/core";
  const optName: Optimizer = "adam";
  ```

### 3. `Cost`
Supported string tags identifying error cost functions.
- **Definition**:
  ```ts
  type Cost = 
    | "mse" | "mae" | "huber" | "logCosh" | "hinge" | "squaredHinge"
    | "klDivergence" | "poisson" | "crossEntropy" 
    | "binaryCrossEntropy" | "softmaxCrossEntropy";
  ```
- **Example**:
  ```ts
  import { Cost } from "@oxide-js/core";
  const costType: Cost = "softmaxCrossEntropy";
  ```

### 4. `StatusLayer`
Tags detailing computational states or target roles for layers inside deep networks.
- **Definition**:
  ```ts
  type StatusLayer =
    | "input"            // Layer acts as initial network input gateway
    | "output"           // Final output predictor
    | "norm"             // Normalization step
    | "outputReduction"  // Decoupled bottleneck layer (e.g. autoencoders)
    | "convOutput"       // Intermediate post-convolutional layer
    | "train"            // Standard trainable mid-network layer
    | "test";            // Verification layer
  ```
- **Example**:
  ```ts
  import { StatusLayer } from "@oxide-js/core";
  const layerState: StatusLayer = "train";
  ```
