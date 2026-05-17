# 📉 Cost Functions API Reference

Cost (Loss) functions in **Oxide-JS** measure the difference between the model's predictions and actual targets. They compute a scalar error value and its analytical gradient with respect to the predictions, enabling backpropagation.

---

## 🚀 API Signature

All core loss functions share a uniform signature:
```ts
lossFunction(yTrue: Matrix, yPred: Matrix): [number, Matrix]
```
- **Arguments**:
  - `yTrue` - The target label matrix.
  - `yPred` - The model's prediction matrix.
- **Returns**: A tuple `[lossValue, dPred]` where:
  - `lossValue: number` - The scalar average error.
  - `dPred: Matrix` - The computed analytical gradient with respect to predictions, of the same shape as `yPred`.
- **Autodiff Integration**: The high-level `BaseModel` handles this returned tuple automatically, registering the custom derivative onto the `Gradient Tape` dynamically during backward loops.

---

## 📉 Loss Function Catalog

### 1. `MeanSquaredError`
**Mean Squared Error (MSE)**. Measures the average squared difference between predictions and targets.
- **Math**: $L = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$
- **Gradient**: $\frac{\partial L}{\partial \hat{y}_i} = \frac{2}{N} (\hat{y}_i - y_i)$
- **Example**:
  ```ts
  import { MeanSquaredError, mj } from "@oxide-js/core";

  const yTrue = mj.matrix([[1.0, 0.0]]);
  const yPred = mj.matrix([[0.8, 0.2]]);
  const [loss, grad] = MeanSquaredError(yTrue, yPred);

  console.log("Loss:", loss); // 0.04
  grad.print();               // [[-0.2, 0.2]]
  ```

### 2. `CategoricalCrossEntropy`
**Categorical Cross-Entropy (CCE)**. Measures performance of classification models outputting probability values between 0 and 1.
- **Math**: $L = -\frac{1}{B} \sum_{b=1}^{B} \sum_{c=1}^{C} y_{b,c} \ln(\hat{y}_{b,c})$
- **Gradient**: $\frac{\partial L}{\partial \hat{y}_{b,c}} = -\frac{y_{b,c}}{B \cdot \hat{y}_{b,c}}$
- **Requirements**: Active predictions `yPred` must be pre-normalized using `softmax` or equivalent.
- **Example**:
  ```ts
  import { CategoricalCrossEntropy, mj } from "@oxide-js/core";

  const yTrue = mj.matrix([[0.0, 1.0, 0.0]]); // One-hot labels
  const yPred = mj.matrix([[0.1, 0.8, 0.1]]); // Class probabilities (sum = 1.0)
  const [loss, grad] = CategoricalCrossEntropy(yTrue, yPred);

  console.log("Loss:", loss); // -ln(0.8) ~ 0.223
  grad.print();
  ```

### 3. `BinaryCrossEntropy`
**Binary Cross-Entropy (BCE)**. Used for single-label, binary classification tasks.
- **Math**: $L = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \ln(\hat{y}_i) + (1 - y_i) \ln(1 - \hat{y}_i) \right]$
- **Gradient**: $\frac{\partial L}{\partial \hat{y}_i} = -\frac{1}{N} \left[ \frac{y_i}{\hat{y}_i} - \frac{1 - y_i}{1 - \hat{y}_i} \right]$
- **Example**:
  ```ts
  import { BinaryCrossEntropy, mj } from "@oxide-js/core";

  const yTrue = mj.matrix([[1.0]]);
  const yPred = mj.matrix([[0.9]]);
  const [loss, grad] = BinaryCrossEntropy(yTrue, yPred);

  console.log("Loss:", loss); // -ln(0.9) ~ 0.105
  grad.print();
  ```

### 4. `SoftmaxCrossEntropy`
A unified categorical loss function that computes Softmax internally before calculating Categorical Cross-Entropy. This formulation is highly stable, preventing numerical overflows or division-by-zero during backpropagation.
- **Math**: combines Softmax and Cross Entropy.
- **Gradient**: $\frac{\partial L}{\partial x_i} = \frac{1}{B} (P_i - y_i)$ where $P$ is the softmax probability vector.
- **Best Use**: Categorical output classification layers in feed-forward and recurrent models.
- **Example**:
  ```ts
  import { SoftmaxCrossEntropy, mj } from "@oxide-js/core";

  const yTrue = mj.matrix([[0.0, 1.0, 0.0]]); // One-hot targets
  const logits = mj.matrix([[1.0, 5.0, 1.0]]); // Raw, unnormalized outputs
  const [loss, grad] = SoftmaxCrossEntropy(yTrue, logits);

  console.log("Loss:", loss);
  grad.print();
  ```
