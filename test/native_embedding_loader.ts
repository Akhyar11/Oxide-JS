import { embeddingForwardNative, isNativeAvailable } from "../src/math/rust_backend";

function assert(condition: boolean, message: string) {
  if (!condition) {
    throw new Error(message);
  }
}

assert(isNativeAvailable(), "native backend should be available for this regression test");

const out = new Float32Array(4);

embeddingForwardNative(
  [1, 2],
  new Float32Array([
    0, 1, 2,
    10, 11, 12,
  ]),
  3,
  2,
  null,
  out,
);

assert(out[0] === 1, `expected out[0] to be 1, got ${out[0]}`);
assert(out[1] === 2, `expected out[1] to be 2, got ${out[1]}`);
assert(out[2] === 11, `expected out[2] to be 11, got ${out[2]}`);
assert(out[3] === 12, `expected out[3] to be 12, got ${out[3]}`);

console.log("native_embedding_loader passed");
