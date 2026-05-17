# 🛠️ Utilities API Reference

The **[utils](file:///home/akhyar/Dokumen/Code/NODE_JS/Oxide-JS/packages/core/src/utils/index.ts)** workspace package exports highly optimized helper functions to manage training pipelines, format indicators, calculate mathematical similarities, and perform sequence batch trimming.

---

## ⚡ Dynamic Padding Trim (For Transformers)

### `trimPaddingBatch(x: Matrix, y: Matrix, padId: number, paddingSide: "left" | "right"): TrimPaddingBatchResult`
Optimizes causal multi-block self-attention computations dynamically. By detecting and slicing leading (left) or trailing (right) padding tokens out of active batch tensors on the fly, it reduces the sequence dimension length from $O(\text{seqLen}^2)$ to $O(\text{effectiveSeqLen}^2)$.
- **Arguments**:
  - `x` - Input batch matrix of shape `[seqLen, batchSize]`.
  - `y` - Shifted targets batch matrix of shape `[seqLen, batchSize]`.
  - `padId` - The token index used for padding (e.g. `0`).
  - `paddingSide` - The alignment direction where padding is located (`"left"` or `"right"`).
- **Returns** a `TrimPaddingBatchResult` object:
  - `x: Matrix` - Sliced input matrix with shape `[effectiveSeqLen, batchSize]`.
  - `y: Matrix` - Sliced target matrix with shape `[effectiveSeqLen, batchSize]`.
  - `positionOffset: number` - Offset index (useful for shifting positional encodings during left-padding).
  - `effectiveSeqLen: number` - Re-mapped sequence length.
  - `trimmed: boolean` - Tells whether any slicing operations were executed.
- **Example**:
  ```ts
  import { trimPaddingBatch, mj } from "@oxide-js/core";

  // Batch of 2 samples, max sequence length 4. (0 is PAD token)
  const x = mj.matrix([
    [10, 11],
    [12, 13],
    [0,  0],
    [0,  0]
  ]); // [4, 2] shape
  
  const y = mj.matrix([
    [11, 12],
    [13, 14],
    [0,  0],
    [0,  0]
  ]); // [4, 2] shape

  const result = trimPaddingBatch(x, y, 0, "right");
  console.log("Trimmed:", result.trimmed);                 // true
  console.log("Effective Sequence Length:", result.effectiveSeqLen); // 2
  result.x.print(); // Shape [2, 2]
  ```

---

## 🎛️ Resolution Helper Modules

### 1. `setActivation(config: string | object): Activation`
Resolves user configurations into standard, autodiff-compliant activation function containers.
- **Example**:
  ```ts
  import { setActivation } from "@oxide-js/core";
  const reluFunc = setActivation("relu");
  ```

### 2. `setLoss(config: string | object): Cost`
Resolves target config settings or string keys into cost function closures.
- **Example**:
  ```ts
  import { setLoss } from "@oxide-js/core";
  const mseLoss = setLoss("mse");
  ```

### 3. `setOptimizer(config: string | object, shape: [number, number]): OptimizerType`
Initializes optimizer state instances, resolving string tags (e.g. `"adam"`) and configuring weights shapes.
- **Example**:
  ```ts
  import { setOptimizer } from "@oxide-js/core";
  const optimizer = setOptimizer("adam", [32, 64]);
  ```

---

## 📊 Similarity Primitives

### `cosineSimilarity(a: Matrix, b: Matrix): number`
Calculates the spatial cosine similarity metric between two dense multi-dimensional matrices.
- **Math**: $\text{Similarity} = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\|_2 \|\mathbf{B}\|_2}$
- **Example**:
  ```ts
  import { cosineSimilarity, mj } from "@oxide-js/core";

  const a = mj.matrix([[1.0, 2.0]]);
  const b = mj.matrix([[2.0, 4.0]]);
  console.log(cosineSimilarity(a, b)); // 1.0 (perfectly parallel)
  ```

---

## 🏗️ Training & UI Formatting Utilities

### 1. `shuffleInPlace(array: any[]): void`
Performs in-place array shuffling using the highly uniform **Fisher-Yates** randomization algorithm.
- **Example**:
  ```ts
  import { shuffleInPlace } from "@oxide-js/core";

  const arr = [1, 2, 3, 4, 5];
  shuffleInPlace(arr);
  console.log(arr); // Randomly ordered array
  ```

### 2. `splitTrainValidation(x: Matrix, y: Matrix, validationSplit: number, shuffle?: boolean)`
Partitions raw input and target tensor rows into training and validation subsets based on validation ratios.
- **Example**:
  ```ts
  import { splitTrainValidation, mj } from "@oxide-js/core";

  const x = mj.random([100, 10]);
  const y = mj.random([100, 1]);

  const splitted = splitTrainValidation(x, y, 0.2, true);
  console.log(splitted.trainX._shape); // [80, 10]
  console.log(splitted.valX._shape);   // [20, 10]
  ```

### 3. `formatLoss(loss: number | Matrix): string`
Extracts internal scalars from numerical cost outputs and formats float numbers into clean strings.
- **Example**:
  ```ts
  import { formatLoss, mj } from "@oxide-js/core";

  const lossMat = mj.matrix([[0.045678]]);
  console.log(formatLoss(lossMat)); // "0.0457"
  console.log(formatLoss(0.123));   // "0.1230"
  ```

### 4. `formatProgressBar(current: number, total: number, elapsedMs: number): string`
Renders standard progress indicators displaying percentage bars, progress ratios, and processing speed per step.
- **Example**:
  ```ts
  import { formatProgressBar } from "@oxide-js/core";

  // Progress 50 out of 100, spent 1000ms
  const bar = formatProgressBar(50, 100, 1000);
  console.log(bar); // [========--------] 50% - 20ms/step
  ```

### 5. `formatTime(elapsedMs: number): string`
Formats raw millisecond timers into highly readable human representations.
- **Example**:
  ```ts
  import { formatTime } from "@oxide-js/core";
  console.log(formatTime(125000)); // "2m 5s"
  ```
