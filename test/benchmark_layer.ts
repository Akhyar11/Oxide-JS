import mj from "../src/math";
import Matrix from "../src/matrix";
import Dense from "../src/layers/dense";

/**
 * Benchmark untuk Dense Layer spesifik:
 * Input Shape: [64, 1]
 * Output Shape: [20183, 1]
 * Params: 1,311,895
 */

const ITERATIONS = 1000;
const WARMUP = 100;

async function runBenchmark() {
    console.log("=== Dense Layer Performance Benchmark ===");
    console.log(`Layer: Dense (64 -> 20183)`);
    console.log(`Input: [64, 1]`);
    console.log(`Output: [20183, 1]`);
    console.log(`Total Params: 1,311,895\n`);

    // Initialize Layer
    const layer = new Dense({
        units: 64,
        outputUnits: 20183,
        activation: "relu",
        optimizer: "adam",
        status: "input" 
    });

    // Create dummy input and target/error
    const input = mj.random([64, 1]);
    const error = mj.random([20183, 1]);

    console.log("Warming up...");
    for (let i = 0; i < WARMUP; i++) {
        layer.forward(input);
        layer.backward(input, error);
    }

    // Benchmark Forward
    console.log(`Running ${ITERATIONS} forward passes...`);
    const startForward = process.hrtime.bigint();
    for (let i = 0; i < ITERATIONS; i++) {
        layer.forward(input);
    }
    const endForward = process.hrtime.bigint();
    const forwardTimeMs = Number(endForward - startForward) / 1e6;
    const avgForward = forwardTimeMs / ITERATIONS;

    // Benchmark Backward
    console.log(`Running ${ITERATIONS} backward passes...`);
    const startBackward = process.hrtime.bigint();
    for (let i = 0; i < ITERATIONS; i++) {
        layer.backward(input, error);
    }
    const endBackward = process.hrtime.bigint();
    const backwardTimeMs = Number(endBackward - startBackward) / 1e6;
    const avgBackward = backwardTimeMs / ITERATIONS;

    console.log("\n--- Results ---");
    console.log(`Forward Pass  : ${avgForward.toFixed(4)} ms/op (${(1000 / avgForward).toFixed(2)} ops/sec)`);
    console.log(`Backward Pass : ${avgBackward.toFixed(4)} ms/op (${(1000 / avgBackward).toFixed(2)} ops/sec)`);
    console.log(`Total Step    : ${(avgForward + avgBackward).toFixed(4)} ms/step`);
}

runBenchmark().catch(console.error);
