# Auto-Diff Module

The Auto-Diff (Automatic Differentiation) module provides the engine for recording mathematical operations and calculating gradients automatically using the Reverse-Mode Differentiation (Backpropagation) technique.

## Import

```ts
import { engine, Tape } from "@oxide-js/core"
```

## Overview

Oxide-JS uses a **Gradient Tape** system. When a tape is active, it "watches" operations performed on `Matrix` objects (like `dotProduct`, `add`, etc.) and records them into a computational graph. When `backward()` is called, the engine traverses this graph in reverse to compute gradients.

Gradients are stored directly in the `grad` property of each `Matrix` object involved in the computation.

---

## Engine API

The `engine` is a singleton instance that manages the active tape lifecycle.

### `engine.startTape(): Tape`

Starts a new recording session. Returns a `Tape` instance.

### `engine.endTape(): void`

Stops the current recording session and cleans up the active tape reference.

### `engine.grad(fn: () => void): Tape`

A convenience helper that starts a tape, executes the provided function, stops the tape, and returns it.

```ts
const tape = engine.grad(() => {
  const z = mj.dotProduct(x, w);
  // ... more operations
});
```

---

## Tape API

### `tape.backward(loss: Matrix): void`

Calculates gradients starting from the `loss` matrix. It automatically initializes the gradient of the loss to `1.0` if not already set.

- **Note:** During the backward pass, the engine temporarily restores the input/output data as it was when the operation was recorded. This ensures mathematical correctness even if the matrices were modified in-place later.

```ts
const tape = engine.startTape();
const z = mj.dotProduct(x, w);
tape.backward(z);
engine.endTape();

console.log(w.grad); // Matrix containing gradients for w
```

---

## Matrix Integration

Every `Matrix` object has a nullable `grad` property of type `Matrix`.

### `matrix.grad: Matrix | null`

Stores the accumulated gradient for this matrix. After a `backward()` call, you can access this property to perform parameter updates (e.g., in an Optimizer).

### `matrix.clearGrad(): void`

Resets the gradient to `null`. (Recommended between training iterations).

---

## Advanced: Manual Recording

If you are implementing a custom operation or layer and want it to support Auto-Diff, you can manually record it:

### `tape.record(inputs, outputs, backwardFn)`

| Parameter | Type | Description |
|---|---|---|
| `inputs` | `Matrix[]` | List of input matrices |
| `outputs` | `Matrix[]` | List of output matrices |
| `backwardFn` | `(grad: Matrix) => void` | Function to compute gradients for inputs given the output gradient |

**Example:**

```ts
const tape = engine.tape;
if (tape) {
  tape.record([a, b], [res], (grad) => {
    // grad is the gradient of the output (res)
    // We compute gradients for a and b and add them to their .grad property
    a.grad = mj.add(a.grad || mj.zeros(a._shape), someGradA);
    b.grad = mj.add(b.grad || mj.zeros(b._shape), someGradB);
  });
}
```

---

## Layers & Models Integration

The Auto-Diff system is integrated into all Oxide-JS layers. This allows you to combine high-level layers with low-level matrix operations in a single training loop.

### Standard Training Loop (Manual Auto-Diff)

The following pattern is used to train models using the Gradient Tape manually, providing full control over the gradient flow.

```ts
import { mj, engine, Adam } from "@oxide-js/core";
import { Dense } from "@oxide-js/layers";

const layer = new Dense({ units: 10, outputUnits: 1 });
const params = layer.getParams();
const optimizers = new Map(params.map((p) => [p, new Adam(p._shape)]));

// Training iteration
for (let epoch = 0; epoch < 100; epoch++) {
  for (const p of params) p.clearGrad();

  // 1. Start recording
  const tape = engine.startTape();

  // 2. Forward Pass
  // Every layer automatically records its operations on the active tape
  const prediction = layer.forward(input);
  const diff = mj.sub(prediction, target);
  const loss = mj.mean(mj.pow(diff, 2)); // MSE

  // 3. Backward Pass (Auto-Diff)
  // This populates .grad for all involved parameters
  tape.backward(loss);
  engine.endTape();

  // 4. Update Parameters
  for (const p of params) {
    if (p.grad) {
      optimizers.get(p)!.apply(p, 0.001);
    }
  }
}
```

### Advantages of Tape-based Training

- **Dynamic Graphs**: You can use standard JavaScript control flow (`if`, `for`) inside the `engine.grad` block.
- **Interoperability**: High-level layers like `Dense` or `RNN` can be mixed with low-level matrix math.
- **Gradient Accumulation**: Simply don't call `clearGrad()` to accumulate gradients over multiple forward passes.

---

## The `gradOnly` Pattern

For complex layers (like `Transformers` or `RNN`) that have their own optimized internal backward implementations, Oxide-JS uses the `gradOnly` pattern to synchronize with the Tape engine.

- **`backward(grad, gradOnly = true)`**:
  - When `gradOnly` is `true`, the layer computes gradients and **adds** them to the `.grad` property of its parameters.
  - It **does not** perform weight updates using its internal learning rate/optimizer.
  - This is essential when you want to use a top-level Auto-Diff optimizer or accumulate gradients.

**Example: Mixed Layer and Manual Math**

```ts
const tape = engine.startTape();

const h = dense.forward(input);
const activated = mj.relu(h);
const loss = mj.mean(mj.pow(mj.sub(activated, target), 2));

tape.backward(loss);
engine.endTape();
```

Inside `dense.forward`, the layer detects the active tape and records its operation. When `tape.backward` is called, it eventually calls `dense.backward(grad, true)` to populate the weights' gradients.
### Performance Considerations

- **Tape Overhead**: The recording overhead is typically **< 3%** for deep models.
- **Memory**: Each recorded operation stores snapshots of its inputs/outputs. For extremely large models, ensure you call `engine.endTape()` as soon as possible to release memory.
- **In-Place Operations**: The engine handles in-place modifications (like `addInPlace`) correctly by snapshotting buffers before they are modified.

---

## Implementation Workflow (Internal)

If you are developing a new layer, follow this standardized flow:

1.  **Forward**: Perform the operation and call `tape.record(...)` if `engine.tape` is active.
2.  **Backward**: Implement a `backward(y, err, gradOnly)` method.
3.  **Accumulation**: Inside `backward`, if `gradOnly` is true, accumulate gradients into the `.grad` property of your trainable matrices:
    ```ts
    if (this.weight.grad) {
      this.weight.grad.addInPlace(newGrad);
    } else {
      this.weight.grad = newGrad;
    }
    ```
4.  **Registration**: Implement `getParams()` and `update(alpha)` to allow external optimizers to manage your layer.
