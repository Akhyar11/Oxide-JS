# 🔄 Automatic Differentiation (Autodiff) API Reference

Oxide-JS features a powerful, lightweight **Dynamic Gradient Tape** automatic differentiation engine. It records mathematical operations on tensors during the forward pass and performs reverse-mode automatic differentiation (backpropagation) to calculate gradients for all trainable parameters.

---

## 🏗️ The `Tape` Class

The **[Tape](file:///home/akhyar/Dokumen/Code/NODE_JS/Oxide-JS/packages/core/src/autodiff/index.ts)** monitors mathematical steps by storing execution nodes. Each node represents a single operation, keeping track of input/output matrices, versions, shapes, and custom analytical derivative callbacks.

### 📌 Core Instance Methods

#### 1. `watch(): void`
Starts recording operations. Clears any historical nodes.
- **Example**:
  ```ts
  import { Tape } from "@oxide-js/core";
  const tape = new Tape();
  tape.watch();
  ```

#### 2. `stop(): void`
Stops recording operations.
- **Example**:
  ```ts
  tape.stop();
  ```

#### 3. `noGrad<T>(fn: () => T): T`
Temporarily halts the tape recording to execute a block of code without calculating or saving gradient graphs. Highly useful during evaluation, prediction, and parameter updates to save execution memory.
- **Example**:
  ```ts
  // Evaluate loss without recording any operations on the tape
  const valLoss = tape.noGrad(() => {
    return evaluateModel(valData);
  });
  ```

#### 4. `record(inputs: Matrix[], outputs: Matrix[], backward: GradientFunc, options?: TapeRecordOptions): void`
Registers a mathematical operation on the active tape.
- **`TapeRecordOptions`**:
  - `saveInput?: boolean` - If `true`, snapshots input matrix elements to prevent post-mutation corruption during backpropagation. Default `true`.
  - `saveOutput?: boolean` - If `true`, snapshots output matrix elements. Default `true`.
  - `requireInputStability?: boolean` - If `true`, throws an error if an input matrix is mutated without a saved snapshot. Default `false`.
  - `requireOutputStability?: boolean` - If `true`, throws an error if an output matrix is mutated without a saved snapshot. Default `false`.
- **Example**:
  ```ts
  import { Matrix } from "@oxide-js/core";

  const x = Matrix.fromFlat([2], [1, 1]);
  x.requiresGrad = true;
  const y = Matrix.fromFlat([4], [1, 1]); // y = x * 2

  // Register custom forward multiplication
  tape.record([x], [y], (grad) => {
    // dy/dx = 2 -> multiply by upstream gradient
    const dx = Matrix.fromFlat([grad.get(0, 0) * 2], [1, 1]);
    return [dx];
  }, { saveInput: true, saveOutput: true });
  ```

#### 5. `backward(loss: Matrix, upstreamGrad?: Matrix): void`
Triggers backpropagation. Traverses recorded execution nodes in reverse order (LIFO), calculating gradients via analytic derivative callbacks and updating parameter `grad` properties in-place.
- **Data Swapping (Snapshot Time Travel)**: To support operations where inputs/outputs are mutated in-place, `backward` swaps historical snapshots back into the tensor objects during derivative evaluations, restoring current runtime buffers immediately afterward.
- **Example**:
  ```ts
  // y is the final computed loss matrix
  tape.backward(y);
  console.log("x gradient:", x.grad?.get(0, 0));
  ```

---

## ⚙️ The Global `engine` Singleton

The **[engine](file:///home/akhyar/Dokumen/Code/NODE_JS/Oxide-JS/packages/core/src/autodiff/engine.ts)** singleton manages nested tape allocations and offers high-level closures to wrap forward passes.

### 📌 Core Instance Methods

#### 1. `grad<T>(fn: () => T): Tape & { result: T }`
Starts a tape, runs the forward function `fn()`, stops the tape, and returns the tape container coupled with the function's returned value (`result`).
- **Best Use**: Custom training steps.
- **Example**:
  ```ts
  import { engine, mj } from "@oxide-js/core";

  const x = mj.matrix([[2.0]]);
  x.requiresGrad = true;

  const tape = engine.grad(() => {
    // Operations inside this closure will be recorded
    return mj.add(mj.mul(x, x), 10); // x^2 + 10
  });

  tape.backward(tape.result);
  console.log("x gradient:", x.grad?.get(0, 0)); // 4.0 (2 * x)
  ```

#### 2. `noGrad<T>(fn: () => T): T`
Global wrapper to execute code without recording gradients on any active tape.
- **Example**:
  ```ts
  import { engine } from "@oxide-js/core";

  const outputs = engine.noGrad(() => {
    return model.forward(inputs);
  });
  ```

---

## 🛠️ Custom Autograd Operation Example

This complete example demonstrates how to build and record a custom mathematical layer (e.g. $y = x^2 + 2x$), record it on the global `engine`, and compute the derivative ($\frac{dy}{dx} = 2x + 2$):

```ts
import { Matrix, engine, mj } from "@oxide-js/core";

// 1. Initialize input requiring gradient
const x = Matrix.fromFlat(new Float32Array([3.0]), [1, 1]);
x.requiresGrad = true;

// 2. Wrap calculations inside engine.grad
const tape = engine.grad(() => {
  // Let's implement: y = x^2 + 2x
  const xSquared = Matrix.fromFlat(new Float32Array([x._data[0] ** 2]), [1, 1]);
  const twoX = Matrix.fromFlat(new Float32Array([2 * x._data[0]]), [1, 1]);

  // Target loss prediction matrix (y = xSquared + twoX)
  const y = mj.add(xSquared, twoX);

  // Register custom analytical gradients on the Tape
  engine.record([x], [xSquared], (grad) => {
    // d(x^2)/dx = 2x
    const dx = Matrix.fromFlat(new Float32Array([2 * x._data[0] * grad._data[0]]), [1, 1]);
    return [dx];
  });

  engine.record([x], [twoX], (grad) => {
    // d(2x)/dx = 2
    const dx = Matrix.fromFlat(new Float32Array([2 * grad._data[0]]), [1, 1]);
    return [dx];
  });

  return y;
});

// 3. Trigger backpropagation
tape.backward(tape.result);

console.log("Output y value:", tape.result._data[0]); // Prints 15.0 (3^2 + 2*3)
console.log("Calculated dy/dx gradient:", x.grad?._data[0]); // Prints 8.0 (2*3 + 2)
```
