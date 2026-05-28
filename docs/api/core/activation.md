# ⚡ Activation Functions API Reference

Activation functions in **Oxide-JS** are mathematical mappings that introduce non-linearity into neural network layers. All activation functions are fully integrated with the **[Gradient Tape](file:///home/akhyar/Dokumen/Code/NODE_JS/Oxide-JS/packages/core/src/autodiff/index.ts)** auto-diff engine (recording backward derivative steps automatically) and leverage accelerated **Rust Native Kernels** on execution hot paths.

---

## 🚀 API Signature

Each activation function has a consistent signature:
```ts
activationFunction(x: Matrix): Matrix
```
- **Inputs**: A `Matrix` containing pre-activation elements.
- **Outputs**: A fresh `Matrix` of the same shape with the activation function applied.
- **Auto-Diff**: Automatically tracks output tensors and records backward derivative callbacks.

---

## 📈 Standard Activation Catalog

### 1. `relu(x: Matrix): Matrix`
**Rectified Linear Unit**. Maps negative values to 0, leaving positive values unchanged.
- **Math**: $f(x) = \max(0, x)$
- **Example**:
  ```ts
  import { mj } from "@oxide-js/core";
  const inputs = mj.matrix([[-1.5, 2.0]]);
  const out = mj.relu(inputs);
  out.print(); // [[0.0, 2.0]]
  ```

### 2. `lRelu(x: Matrix, alpha?: number): Matrix`
**Leaky Rectified Linear Unit**. Introduces a small slope ($\alpha$) for negative inputs to prevent the "dead ReLU" problem.
- **Math**: $f(x) = \max(\alpha \cdot x, x)$ (Default $\alpha = 0.01$).
- **Example**:
  ```ts
  import { mj } from "@oxide-js/core";
  const inputs = mj.matrix([[-1.5, 2.0]]);
  const out = mj.lRelu(inputs, 0.1);
  out.print(); // [[-0.15, 2.0]]
  ```

### 3. `sigmoid(x: Matrix): Matrix`
Maps inputs to a logistic range between $(0, 1)$.
- **Math**: $f(x) = \frac{1}{1 + e^{-x}}$
- **Example**:
  ```ts
  import { mj } from "@oxide-js/core";
  const inputs = mj.matrix([[0.0]]);
  const out = mj.sigmoid(inputs);
  out.print(); // [[0.5]]
  ```

### 4. `tanh(x: Matrix): Matrix`
**Hyperbolic Tangent**. Maps inputs to a symmetric range between $(-1, 1)$.
- **Math**: $f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$
- **Example**:
  ```ts
  import { mj } from "@oxide-js/core";
  const inputs = mj.matrix([[0.0]]);
  const out = mj.tanh(inputs);
  out.print(); // [[0.0]]
  ```

### 5. `softmax(x: Matrix): Matrix`
Computes normal probability distributions over columns or batches.
- **Math**: $f(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$
- **Example**:
  ```ts
  import { mj } from "@oxide-js/core";
  const inputs = mj.matrix([[1.0, 2.0, 3.0]]);
  const out = mj.softmax(inputs);
  out.print(); // Prints normalized probabilities sum = 1.0
  ```

### 6. `linear(x: Matrix): Matrix`
A pass-through activation function that returns the input values unchanged.
- **Math**: $f(x) = x$
- **Example**:
  ```ts
  import { mj } from "@oxide-js/core";
  const inputs = mj.matrix([[5.0]]);
  const out = mj.linear(inputs);
  out.print(); // [[5.0]]
  ```

---

## 🔬 Advanced Activation Catalog

### 1. `elu(x: Matrix, alpha?: number): Matrix`
Exponential Linear Unit. Smoother than ReLU around 0.
- **Math**: $f(x) = \begin{cases} x & x > 0 \\ \alpha(e^x - 1) & x \le 0 \end{cases}$ (Default $\alpha = 1.0$).
- **Example**:
  ```ts
  import { mj } from "@oxide-js/core";
  const inputs = mj.matrix([[-1.0, 1.0]]);
  mj.elu(inputs, 1.0).print();
  ```

### 2. `selu(x: Matrix): Matrix`
Scaled Exponential Linear Unit for self-normalizing networks ($\lambda \approx 1.0507, \alpha \approx 1.67326$).
- **Math**: $f(x) = \lambda \begin{cases} x & x > 0 \\ \alpha(e^x - 1) & x \le 0 \end{cases}$
- **Example**:
  ```ts
  import { mj } from "@oxide-js/core";
  const inputs = mj.matrix([[-1.0, 1.0]]);
  mj.selu(inputs).print();
  ```

### 3. `gelu(x: Matrix): Matrix`
Gaussian Error Linear Unit. The standard in Transformer (GPT/BERT) architectures.
- **Math**: $f(x) = x \cdot \Phi(x) \approx 0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$
- **Example**:
  ```ts
  import { mj } from "@oxide-js/core";
  const inputs = mj.matrix([[0.0, 2.0]]);
  mj.gelu(inputs).print();
  ```

### 4. `swish(x: Matrix, beta?: number): Matrix`
Self-gated activation function originally discovered by Google.
- **Math**: $f(x) = x \cdot \text{sigmoid}(\beta \cdot x)$ (Default $\beta = 1.0$).
- **Example**:
  ```ts
  import { mj } from "@oxide-js/core";
  const inputs = mj.matrix([[2.0]]);
  mj.swish(inputs).print();
  ```

### 5. `mish(x: Matrix): Matrix`
A highly smooth self-gated activation function.
- **Math**: $f(x) = x \cdot \tanh(\text{softplus}(x))$
- **Example**:
  ```ts
  import { mj } from "@oxide-js/core";
  const inputs = mj.matrix([[1.0]]);
  mj.mish(inputs).print();
  ```

### 6. `softplus(x: Matrix): Matrix`
A smooth approximation of the ReLU function.
- **Math**: $f(x) = \ln(1 + e^x)$
- **Example**:
  ```ts
  import { mj } from "@oxide-js/core";
  const inputs = mj.matrix([[0.0]]);
  mj.softplus(inputs).print(); // [[0.693147]]
  ```

### 7. `softsign(x: Matrix): Matrix`
A smooth mathematical curve mapping symmetric scales.
- **Math**: $f(x) = \frac{x}{1 + |x|}$
- **Example**:
  ```ts
  import { mj } from "@oxide-js/core";
  const inputs = mj.matrix([[2.0, -2.0]]);
  mj.softsign(inputs).print(); // [[0.6666, -0.6666]]
  ```

### 8. `hardSigmoid(x: Matrix): Matrix`
Piecewise linear approximation of sigmoid. Fast computation.
- **Math**: $f(x) = \max(0, \min(1, 0.2x + 0.5))$
- **Example**:
  ```ts
  import { mj } from "@oxide-js/core";
  const inputs = mj.matrix([[0.0]]);
  mj.hardSigmoid(inputs).print(); // [[0.5]]
  ```

### 9. `hardSwish(x: Matrix): Matrix`
Piecewise approximation of Swish. Efficient for mobile/embedded deployment.
- **Math**: $f(x) = x \cdot \text{hardSigmoid}(x) = x \cdot \frac{\text{clip}(x+3, 0, 6)}{6}$
- **Example**:
  ```ts
  import { mj } from "@oxide-js/core";
  const inputs = mj.matrix([[2.0]]);
  mj.hardSwish(inputs).print();
  ```

### 10. `threshold(x: Matrix, threshold?: number, leak?: number): Matrix`
Step function designed for **Spiking Neural Networks** (SNN) Leaky Integrate-and-Fire output potentials.
- **Math**: $f(x) = \begin{cases} 1 & x \ge threshold \\ leak & x < threshold \end{cases}$ (Default $threshold = 1.0, leak = 0.0$).
- **Example**:
  ```ts
  import { mj } from "@oxide-js/core";
  const inputs = mj.matrix([[0.5, 1.2]]);
  mj.threshold(inputs, 1.0, 0.0).print(); // [[0.0, 1.0]]
  ```
