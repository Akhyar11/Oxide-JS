import { describe, expect, it } from "vitest";
import { setForceDisableNative } from "../src/math/rust_backend.js";
import mj from "../src/math/index.js";
import convolution from "../src/math/convolution.js";
import addBias from "../src/math/addBias.js";
import sumAxis from "../src/math/sumAxis.js";
import clipGradients from "../src/math/clipGradients.js";
import Adam from "../src/optimizer/adam.js";
import SGD from "../src/optimizer/sgd.js";
import Adagrad from "../src/optimizer/adaGrad.js";
import Momentum from "../src/optimizer/momentum.js";
import NAG from "../src/optimizer/nag.js";
import Matrix from "../src/matrix/index.js";

function expectMatricesToBeClose(a: Matrix, b: Matrix, maxDiff = 1e-5) {
  expect(a._shape).toEqual(b._shape);
  let maxActualDiff = 0;
  for (let i = 0; i < a._data.length; i++) {
    const diff = Math.abs(a._data[i] - b._data[i]);
    if (diff > maxActualDiff) {
      maxActualDiff = diff;
    }
  }
  expect(maxActualDiff).toBeLessThan(maxDiff);
}

function expectArraysToBeClose(a: number[], b: number[], maxDiff = 1e-5) {
  expect(a.length).toEqual(b.length);
  for (let i = 0; i < a.length; i++) {
    expect(Math.abs(a[i] - b[i])).toBeLessThan(maxDiff);
  }
}

describe("JS vs Rust Math Parity - Complete Suite", () => {
  it("should have exact parity for primitive math operations", () => {
    const size = 32;
    const aData = mj.random([size, size]);
    const bData = mj.random([size, size]);
    
    const getClones = () => [mj.add(aData, 0), mj.add(bData, 0)];

    // add
    setForceDisableNative(true);
    let [a1, b1] = getClones();
    const jsAdd = mj.add(a1, b1);
    setForceDisableNative(false);
    let [a2, b2] = getClones();
    const rustAdd = mj.add(a2, b2);
    expectMatricesToBeClose(jsAdd, rustAdd);

    // sub
    setForceDisableNative(true);
    const jsSub = mj.sub(a1, b1);
    setForceDisableNative(false);
    const rustSub = mj.sub(a2, b2);
    expectMatricesToBeClose(jsSub, rustSub);

    // mul
    setForceDisableNative(true);
    const jsMul = mj.mul(a1, b1);
    setForceDisableNative(false);
    const rustMul = mj.mul(a2, b2);
    expectMatricesToBeClose(jsMul, rustMul);

    // div (avoid div by zero)
    const b1Safe = mj.add(b1, 1);
    const b2Safe = mj.add(b2, 1);
    setForceDisableNative(true);
    const jsDiv = mj.div(a1, b1Safe);
    setForceDisableNative(false);
    const rustDiv = mj.div(a2, b2Safe);
    expectMatricesToBeClose(jsDiv, rustDiv);

    // dotProduct
    setForceDisableNative(true);
    const jsDot = mj.dotProduct(a1, b1);
    setForceDisableNative(false);
    const rustDot = mj.dotProduct(a2, b2);
    expectMatricesToBeClose(jsDot, rustDot, 1e-4);
  });

  it("should have exact parity for in-place math operations", () => {
    const size = 32;
    const aData = mj.random([size, size]);
    const bData = mj.random([size, size]);
    const getClones = () => [mj.add(aData, 0), mj.add(bData, 0)];

    // addInPlace
    let [ja1, jb1] = getClones();
    let [ra1, rb1] = getClones();
    setForceDisableNative(true);
    ja1.addInPlace(jb1);
    setForceDisableNative(false);
    ra1.addInPlace(rb1);
    expectMatricesToBeClose(ja1, ra1);

    // subInPlace
    let [ja2, jb2] = getClones();
    let [ra2, rb2] = getClones();
    setForceDisableNative(true);
    ja2.subInPlace(jb2);
    setForceDisableNative(false);
    ra2.subInPlace(rb2);
    expectMatricesToBeClose(ja2, ra2);

    // mulInPlace
    let [ja3, jb3] = getClones();
    let [ra3, rb3] = getClones();
    setForceDisableNative(true);
    ja3.mulInPlace(jb3);
    setForceDisableNative(false);
    ra3.mulInPlace(rb3);
    expectMatricesToBeClose(ja3, ra3);
  });

  it("should have exact parity for complex math and utility operations", () => {
    const aData = mj.random([16, 16]);

    // convolution
    const kData = mj.random([3, 3]);
    setForceDisableNative(true);
    const jsConv = convolution(mj.add(aData, 0), mj.add(kData, 0));
    setForceDisableNative(false);
    const rustConv = convolution(mj.add(aData, 0), mj.add(kData, 0));
    expectMatricesToBeClose(jsConv, rustConv, 1e-4);

    // addBias
    const biasData = mj.random([16, 1]);
    const jsBiasTarget = mj.add(aData, 0);
    const rustBiasTarget = mj.add(aData, 0);
    setForceDisableNative(true);
    addBias(jsBiasTarget, biasData);
    setForceDisableNative(false);
    addBias(rustBiasTarget, biasData);
    expectMatricesToBeClose(jsBiasTarget, rustBiasTarget);

    // sumAxis
    setForceDisableNative(true);
    const jsSum0 = sumAxis(aData, 0);
    const jsSum1 = sumAxis(aData, 1);
    setForceDisableNative(false);
    const rustSum0 = sumAxis(aData, 0);
    const rustSum1 = sumAxis(aData, 1);
    expectMatricesToBeClose(jsSum0, rustSum0, 1e-4);
    expectMatricesToBeClose(jsSum1, rustSum1, 1e-4);

    // clipGradients
    const jsClipTarget = mj.mul(mj.add(aData, 0), 10);
    const rustClipTarget = mj.mul(mj.add(aData, 0), 10);
    setForceDisableNative(true);
    clipGradients(jsClipTarget, 2);
    setForceDisableNative(false);
    clipGradients(rustClipTarget, 2);
    expectMatricesToBeClose(jsClipTarget, rustClipTarget);
  });

  it("should have exact parity for activations and loss", () => {
    const aData = mj.random([32, 32]);
    const getClones = () => [mj.add(aData, 0), mj.add(aData, 0)];

    // relu
    setForceDisableNative(true);
    const jsRelu = mj.relu(aData);
    setForceDisableNative(false);
    const rustRelu = mj.relu(aData);
    expectMatricesToBeClose(jsRelu, rustRelu);

    // sigmoid
    setForceDisableNative(true);
    const jsSigmoid = mj.sigmoid(aData);
    setForceDisableNative(false);
    const rustSigmoid = mj.sigmoid(aData);
    expectMatricesToBeClose(jsSigmoid, rustSigmoid, 1e-5);

    // tanh
    setForceDisableNative(true);
    const jsTanh = mj.tanh(aData);
    setForceDisableNative(false);
    const rustTanh = mj.tanh(aData);
    expectMatricesToBeClose(jsTanh, rustTanh, 1e-5);

    // softmax
    setForceDisableNative(true);
    const jsSoftmax = mj.softmax(aData);
    setForceDisableNative(false);
    const rustSoftmax = mj.softmax(aData);
    expectMatricesToBeClose(jsSoftmax, rustSoftmax, 1e-5);

    // lRelu
    const lReluData = mj.sub(aData, 0.5); // contains negatives
    setForceDisableNative(true);
    const jsLRelu = mj.lRelu(lReluData);
    setForceDisableNative(false);
    const rustLRelu = mj.lRelu(lReluData);
    expectMatricesToBeClose(jsLRelu, rustLRelu, 1e-5);

    // pow
    setForceDisableNative(true);
    const jsPow = mj.pow(aData, 3);
    setForceDisableNative(false);
    const rustPow = mj.pow(aData, 3);
    expectMatricesToBeClose(jsPow, rustPow, 1e-4);

    // absm
    setForceDisableNative(true);
    const jsAbsm = mj.absm(lReluData);
    setForceDisableNative(false);
    const rustAbsm = mj.absm(lReluData);
    expectMatricesToBeClose(jsAbsm, rustAbsm, 1e-5);

    // expm
    setForceDisableNative(true);
    const jsExpm = mj.expm(aData);
    setForceDisableNative(false);
    const rustExpm = mj.expm(aData);
    expectMatricesToBeClose(jsExpm, rustExpm, 1e-4);

    // logm
    setForceDisableNative(true);
    const jsLogm = mj.logm(aData);
    setForceDisableNative(false);
    const rustLogm = mj.logm(aData);
    expectMatricesToBeClose(jsLogm, rustLogm, 1e-4);

    // transpose
    const rectData = mj.random([16, 32]);
    setForceDisableNative(true);
    const jsTranspose = mj.transpose(rectData);
    setForceDisableNative(false);
    const rustTranspose = mj.transpose(rectData);
    expectMatricesToBeClose(jsTranspose, rustTranspose, 1e-5);

    // threshold
    setForceDisableNative(true);
    const jsThreshold = mj.threshold(aData, 0.5, 0.1);
    setForceDisableNative(false);
    const rustThreshold = mj.threshold(aData, 0.5, 0.1);
    expectMatricesToBeClose(jsThreshold, rustThreshold, 1e-5);


    // mse
    const bData = mj.random([32, 32]);
    setForceDisableNative(true);
    const jsMse = mj.mse(aData, bData);
    setForceDisableNative(false);
    const rustMse = mj.mse(aData, bData);
    expect(Math.abs(jsMse - rustMse)).toBeLessThan(1e-5);
  });

  it("should have exact parity for optimizers", () => {
    const size = 16;
    const grad = mj.random([size, size]);
    
    // Adam
    const optAdamJs = new Adam([size, size]);
    const optAdamRust = new Adam([size, size]);
    setForceDisableNative(true);
    const jsAdam = optAdamJs.calculate(mj.add(grad, 0), 0.01);
    setForceDisableNative(false);
    const rustAdam = optAdamRust.calculate(mj.add(grad, 0), 0.01);
    expectMatricesToBeClose(jsAdam, rustAdam, 1e-4);

    // SGD
    const optSgdJs = new SGD([size, size]);
    const optSgdRust = new SGD([size, size]);
    setForceDisableNative(true);
    const jsSgd = optSgdJs.calculate(mj.add(grad, 0), 0.01);
    setForceDisableNative(false);
    const rustSgd = optSgdRust.calculate(mj.add(grad, 0), 0.01);
    expectMatricesToBeClose(jsSgd, rustSgd, 1e-5);

    // Adagrad
    const optAdagradJs = new Adagrad([size, size], 0.1);
    const optAdagradRust = new Adagrad([size, size], 0.1);
    setForceDisableNative(true);
    const jsAdagrad = optAdagradJs.calculate(mj.add(grad, 0), 0.01);
    setForceDisableNative(false);
    const rustAdagrad = optAdagradRust.calculate(mj.add(grad, 0), 0.01);
    expectMatricesToBeClose(jsAdagrad, rustAdagrad, 1e-4);

    // Momentum
    const optMomJs = new Momentum([size, size]);
    const optMomRust = new Momentum([size, size]);
    setForceDisableNative(true);
    const jsMom = optMomJs.calculate(mj.add(grad, 0), 0.01);
    setForceDisableNative(false);
    const rustMom = optMomRust.calculate(mj.add(grad, 0), 0.01);
    expectMatricesToBeClose(jsMom, rustMom, 1e-5);

    // NAG
    const optNagJs = new NAG([size, size]);
    const optNagRust = new NAG([size, size]);
    setForceDisableNative(true);
    const jsNag = optNagJs.calculate(mj.add(grad, 0), 0.01);
    setForceDisableNative(false);
    const rustNag = optNagRust.calculate(mj.add(grad, 0), 0.01);
    expectMatricesToBeClose(jsNag, rustNag, 1e-5);
  });
});
