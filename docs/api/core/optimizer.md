# 📈 Optimizers API Reference

Optimizers in **Oxide-JS** update the parameters (weights and biases) of the neural network during training to minimize the calculated loss. Each optimizer exposes standard methods for full parameter updates and sparse parameters (Embedding) indexing updates, fully accelerated by optimized **Rust Native Kernels** on execution hot paths.

---

## 🏗️ Core API Interface

All optimizers implement the following standard lifecycle API:

### 1. `calculate(grad: Matrix, alpha: number): Matrix`
Calculates the update delta to subtract from the parameter weights.
- **Arguments**:
  - `grad` - The parameter's gradient matrix (`param.grad`).
  - `alpha` - The active scalar learning rate.
- **Returns**: A `Matrix` containing computed parameter update steps.

### 2. `updateSparse(target: Matrix, grad: Matrix, alpha: number, indices: Int32Array): void`
Performs sparse, targeted in-place updates. Extremely useful for embedding lookup tables where only indices matching active vocabulary tokens in a sequence are calculated and updated.
- **Arguments**:
  - `target` - The weights matrix to update (e.g. `Embedding.weight`).
  - `grad` - The sparse gradient matrix.
  - `alpha` - The scalar learning rate.
  - `indices` - Unique active index identifiers.

### 3. `apply(target: Matrix, alpha: number): void`
Finds `target.grad`, executes `calculate()`, subtracts weight changes in-place via `subInPlace()`, and clears the gradient.

---

## 📈 Optimizer Catalog

### 1. `SGD` (Stochastic Gradient Descent)
Standard learning rate scaling of gradient adjustments.
- **Math**: $\theta_t = \theta_{t-1} - \alpha \cdot g_t$
- **Example**:
  ```ts
  import { SGD, mj } from "@oxide-js/core";

  const weights = mj.matrix([[1.0, 2.0]]);
  weights.grad = mj.matrix([[0.1, -0.2]]);

  const opt = new SGD();
  opt.apply(weights, 0.01);
  weights.print(); // [[0.99, 2.002]]
  ```

### 2. `Adam` (Adaptive Moment Estimation)
Calculates adaptive learning rates for each parameter by tracking exponentially decaying moving averages of both past gradients (mean) and squared past gradients (uncentered variance).
- **Math**: 
  - $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$ (First moment)
  - $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$ (Second moment)
  - $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$ (Bias correction 1)
  - $\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$ (Bias correction 2)
  - $\theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t$
- **Constructor Options**: `new Adam(shape: [number, number], beta1?: number, beta2?: number, epsilon?: number)`
- **Example**:
  ```ts
  import { Adam, mj } from "@oxide-js/core";

  const shape: [number, number] = [2, 2];
  const weights = mj.zeros(shape);
  weights.grad = mj.ones(shape);

  const opt = new Adam(shape, 0.9, 0.999, 1e-8);
  opt.apply(weights, 0.001);
  ```

### 3. `Momentum` (Classical Momentum SGD)
Accelerates SGD by tracking velocity vectors along the descent slope to navigate ravines.
- **Math**:
  - $v_t = \gamma v_{t-1} + \alpha g_t$ (Velocity)
  - $\theta_t = \theta_{t-1} - v_t$
- **Constructor Options**: `new Momentum(shape: [number, number], momentum?: number)`
- **Example**:
  ```ts
  import { Momentum, mj } from "@oxide-js/core";

  const shape: [number, number] = [1, 2];
  const weights = mj.matrix([[1.0, 2.0]], shape);
  weights.grad = mj.matrix([[0.1, 0.1]], shape);

  const opt = new Momentum(shape, 0.9);
  opt.apply(weights, 0.01);
  ```

### 4. `NAG` (Nesterov Accelerated Gradient)
An advanced momentum formulation that calculates gradients ahead of the current position, acting as a braking mechanism to stabilize convergence.
- **Math**:
  - $v_t = \gamma v_{t-1} + \alpha \nabla f(\theta_{t-1} - \gamma v_{t-1})$
  - $\theta_t = \theta_{t-1} - v_t$
- **Constructor Options**: `new NAG(shape: [number, number], momentum?: number)`
- **Example**:
  ```ts
  import { NAG, mj } from "@oxide-js/core";

  const shape: [number, number] = [1, 2];
  const weights = mj.matrix([[1.0, 2.0]], shape);
  weights.grad = mj.matrix([[0.1, 0.1]], shape);

  const opt = new NAG(shape, 0.9);
  opt.apply(weights, 0.01);
  ```

### 5. `AdaGrad` (Adaptive Gradient Algorithm)
Adapts learning rates globally per feature by dividing by the sum of historical squared gradients.
- **Math**:
  - $G_t = G_{t-1} + g_t^2$
  - $\theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{G_t} + \epsilon} \cdot g_t$
- **Constructor Options**: `new AdaGrad(shape: [number, number], epsilon?: number)`
- **Example**:
  ```ts
  import { AdaGrad, mj } from "@oxide-js/core";

  const shape: [number, number] = [1, 2];
  const weights = mj.matrix([[1.0, 2.0]], shape);
  weights.grad = mj.matrix([[0.1, 0.1]], shape);

  const opt = new AdaGrad(shape, 1e-8);
  opt.apply(weights, 0.01);
  ```

---

## ⚡ Sparse Updates (Embeddings Example)

Optimizers support highly efficient sparse updates using `updateSparse`, updating only the rows of parameters that were modified during embedding lookup:

```ts
import { Adam, mj } from "@oxide-js/core";

// 1. Embedding weight matrix (vocabulary size = 5, embed dim = 3)
const embedWeight = mj.random([5, 3]);

// 2. We only looked up indices [1, 3] in the forward pass.
// Gradients will be computed for index 1 and index 3.
const sparseGrad = mj.matrix([
  [0.1, 0.2, 0.3], // gradient for index 1
  [-0.1, 0.0, 0.2] // gradient for index 3
]);
const activeIndices = new Int32Array([1, 3]);

// 3. Instantiate Adam with full embedding shape
const optimizer = new Adam([5, 3]);

// 4. Update sparse parameter rows in-place
optimizer.updateSparse(embedWeight, sparseGrad, 0.01, activeIndices);
```
