# 🏗️ BaseModel API Reference

The **[BaseModel](file:///home/akhyar/Dokumen/Code/NODE_JS/Oxide-JS/packages/models/src/BaseModel.ts)** class is the primary abstract base class for neural network models in **@oxide-js/models**. It manages compiling, state propagation, early-stopping loops, weight serialization/deserialization, and high-level training pipelines.

---

## 📐 Architecture & Properties

Every model extending `BaseModel` inherits the following core properties and state attributes:

* **`name: string`** — The name of the model instance (defaults to the class constructor name).
* **`trainable: boolean`** — Global lock. If `false`, prevents parameters of all layers in the model from updating during training.
* **`training: boolean`** — Indicates whether the model is in active training mode (`true`) or inference/evaluation mode (`false`).
* **`isCompiled: boolean`** — Set to `true` after calling `compile()`.
* **`isBuilt: boolean`** — Set to `true` once layer parameter grids are built based on input shapes.
* **`inputShape: number[]`** — The expected dimensions of raw inputs.
* **`outputShape: number[]`** — The final dimensions of calculated predictions.

---

## ⚙️ Core Lifecycle Methods

### 🧪 `compile(config: CompileConfig): void`
Prepares the model for training by resolving loss functions, optimizers, and monitoring metrics.
```ts
model.compile({
  optimizer: "adam", // or sgd, momentum, nag, adagrad
  loss: "softmaxCrossEntropy", // or mse, mae, huber, logCosh, hinge, crossEntropy, binaryCrossEntropy
  learningRate: 0.001,
  metrics: ["accuracy", "mae"]
});
```

### 🔮 `predict(inputs: Matrix): Matrix`
Performs an inference forward pass. It automatically switches the model and all nested layers into **evaluation/inference mode** (`training = false`) to disable regularizers like `Dropout`.
```ts
const predictions = model.predict(testInputs);
```

### 📊 `evaluate(x: Matrix, y: Matrix)`
Evaluates predictions against target outputs. Returns calculated loss and metrics.
```ts
const { loss, metrics } = model.evaluate(inputs, targets);
console.log(`Evaluation loss: ${loss}, accuracy: ${metrics.accuracy}`);
```

### 🔄 `trainStep(xBatch: Matrix, yBatch: Matrix): { loss: Matrix; yPred: Matrix }`
Executes a single mini-batch forward-backward step:
1. Puts the model in training mode and clears historical parameter gradients.
2. Tracks operations using the Autodiff Engine (`engine.grad()`).
3. Computes the forward pass and loss value.
4. Backpropagates gradients through the calculated computational tape (`tape.backward(loss)`).
5. Invokes the resolved optimizer step to update model parameters.

### 💾 `serialize(): SerializedModel`
Serializes the entire model topology, hyperparameters, and layer structures into a standard Keras-compatible JSON schema. 
* **Compression Optimization**: As of v2.5.0, physical matrix values (Float32Array) are automatically converted into standard `number[]` arrays during serialization. This optimization prevents JSON from saving arrays as verbose `{ "0": 0.1, "1": 0.2 }` objects, reducing the resulting JSON model file size by up to **75%**.

### 📥 `setWeights(weightsData: WeightData[]): void`
Loads physical weights back into the model parameters. The method intelligently detects if the incoming weight data is a compressed `number[]` array and will automatically decompress it back into a zero-copy `Float32Array` before loading it into the `Matrix` tensors.

---

## 🔍 Parameter Tracking & Registry

`BaseModel` delegates weight tracking downstream to its registered layers, exposing clean registries:

* **`model.weights`** — Returns a flat array of all matrices inside the model.
* **`model.trainableWeights`** — Returns only parameter matrices that are trainable (e.g. weights/biases of trainable layers).
* **`model.nonTrainableWeights`** — Returns parameter matrices marked as non-trainable (e.g. Batch Normalization moving moments).
* **`model.countParams()`** — Sums the parameter count across all layers.
* **`model.countTrainableParams()`** — Sums only active trainable parameters.
* **`model.countNonTrainableParams()`** — Sums only non-trainable parameter matrices.

---

## 🛠️ Custom Model Subclassing Example

This example demonstrates how to subclass `BaseModel` directly to create a custom neural network, bypassing standard `Sequential` layers when complex non-linear routing (such as multi-branch connections) is needed.

```ts
import { BaseModel } from "@oxide-js/models";
import { Dense } from "@oxide-js/layers";
import { Matrix, engine, mj } from "@oxide-js/core";

// 1. Define the Custom Multi-Branch model extending BaseModel
export class DualBranchRegressor extends BaseModel {
  private branchA: Dense;
  private branchB: Dense;
  private merger: Dense;

  constructor() {
    super({ name: "dual_branch_regressor" });

    // Instantiating layers
    this.branchA = new Dense({ name: "branch_a", units: 4, activation: "relu" });
    this.branchB = new Dense({ name: "branch_b", units: 4, activation: "tanh" });
    this.merger = new Dense({ name: "merger", units: 8, outputUnits: 1 });

    // Register layers to the BaseModel catalog
    this.add(this.branchA);
    this.add(this.branchB);
    this.add(this.merger);
  }

  // 2. Implement the abstract forward pass
  public forward(inputs: Matrix, optionsOrTraining?: any): Matrix {
    // Run forward pass through parallel branches
    const featA = this.branchA.forward(inputs, optionsOrTraining);
    const featB = this.branchB.forward(inputs, optionsOrTraining);

    // Merge features (add element-wise)
    const merged = mj.add(featA, featB);

    // Run final projection
    return this.merger.forward(merged, optionsOrTraining);
  }
}

// 3. E2E execution of the Custom Model
const model = new DualBranchRegressor();

model.compile({
  optimizer: "adam",
  loss: "mse",
  learningRate: 0.01
});

// Train inputs [batch=2, features=4] -> Target output [batch=2, target=1]
const inputs = Matrix.fromFlat(new Float32Array([1.0, 2.0, -1.0, 0.5, 0.2, 0.4, 0.8, -0.2]), [2, 4]);
const targets = Matrix.fromFlat(new Float32Array([1.5, 0.8]), [2, 1]);

// Fit for 5 epochs
console.log("Fitting custom multi-branch regressor...");
model.fit(inputs, targets, { epochs: 5, verbose: 1 });

// Output model architecture summary
model.summary();
```
