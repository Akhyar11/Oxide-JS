import mj from "../src/math";
import { performance } from "perf_hooks";

const size = 512;
const a = mj.random([size, size]);
const b = mj.random([size, size]);

console.log(`=== Benchmarking Matrix Operations (${size}x${size}) ===\n`);

// 1. ADDITION
let startTime = performance.now();
for (let i = 0; i < 1000; i++) {
    const r = mj.add(a, b);
}
let endTime = performance.now();
console.log(`Standard Add (Allocation): ${(endTime - startTime).toFixed(2)} ms`);

startTime = performance.now();
for (let i = 0; i < 1000; i++) {
    a.addInPlace(b);
}
endTime = performance.now();
console.log(`In-Place Add (No Allocation): ${(endTime - startTime).toFixed(2)} ms`);

// 2. DOT PRODUCT
console.log(`\n=== Benchmarking Dot Product (128x128) ===\n`);
const d1 = mj.random([128, 128]);
const d2 = mj.random([128, 128]);
const out = mj.zeros([128, 128]);

startTime = performance.now();
for (let i = 0; i < 500; i++) {
    const r = mj.dotProduct(d1, d2);
}
endTime = performance.now();
console.log(`Standard DotProduct: ${(endTime - startTime).toFixed(2)} ms`);

startTime = performance.now();
for (let i = 0; i < 500; i++) {
    mj.dotProduct(d1, d2, out);
}
endTime = performance.now();
console.log(`Optimized DotProduct (with Out buffer): ${(endTime - startTime).toFixed(2)} ms`);
