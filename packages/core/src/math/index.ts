import absm from "./absm.js";
import add, { addInto } from "./add.js";
import concat from "./concat.js";
import div from "./div.js";
import dotDiv from "./dotDiv.js";
import dotMul from "./dotMul.js";
import dotProduct from "./dotProduct.js";
import dotSub from "./dotSub.js";
import dotSum from "./dotSum.js";
import expm from "./expm.js";
import flatten from "./flatten.js";
import logm from "./logm.js";
import map from "./map.js";
import matrix from "./matrix.js";
import mean from "./mean.js";
import mul from "./mul.js";
import ones from "./ones.js";
import random from "./random.js";
import reshape from "./reshape.js";
import sub, { subInto } from "./sub.js";
import transpose from "./transpose.js";
import zeros from "./zeros.js";
import convolution from "./convolution.js";
import norm from "./norm.js";
import xavier from "./xavier.js";
import he from "./he.js";
import addBias from "./addBias.js";
import sumAxis from "./sumAxis.js";
import clipGradients from "./clipGradients.js";

import pow from "./pow.js";

const mj = {
  absm,
  add,
  addInto,
  concat,
  div,
  dotDiv,
  dotMul,
  dotProduct,
  dotSub,
  dotSum,
  expm,
  flatten,
  logm,
  map,
  matrix,
  mean,
  mul,
  ones,
  pow,
  random,
  reshape,
  sub,
  subInto,
  transpose,
  zeros,
  convolution,
  norm,
  xavier,
  he,
  addBias,
  sumAxis,
  clipGradients,
};

export default mj;
