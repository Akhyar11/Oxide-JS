import { describe, it, expect } from "vitest";
import { PoissonEncoder } from "../src/encoding/PoissonEncoder.js";

describe("PoissonEncoder", () => {
  it("should initialize with default firing rate", () => {
    const encoder = new PoissonEncoder();
    expect(encoder.maxFiringRate).toBe(1.0);
  });

  it("should throw error if firing rate is out of bounds", () => {
    expect(() => new PoissonEncoder(0.0)).toThrow();
    expect(() => new PoissonEncoder(-0.5)).toThrow();
    expect(() => new PoissonEncoder(1.5)).toThrow();
  });

  it("should encode 0.0 strictly as 0 spikes", () => {
    const encoder = new PoissonEncoder();
    const train = encoder.generateSpikeTrain(0.0, 100);
    const totalSpikes = train.reduce((a, b) => a + b, 0);
    expect(totalSpikes).toBe(0);
  });

  it("should encode 1.0 strictly as 100% spikes if maxFiringRate is 1.0", () => {
    const encoder = new PoissonEncoder(1.0);
    const train = encoder.generateSpikeTrain(1.0, 100);
    const totalSpikes = train.reduce((a, b) => a + b, 0);
    expect(totalSpikes).toBe(100);
  });

  it("should encode 0.5 as roughly 50% spikes over large samples", () => {
    const encoder = new PoissonEncoder(1.0);
    const steps = 1000;
    const train = encoder.generateSpikeTrain(0.5, steps);
    const totalSpikes = train.reduce((a, b) => a + b, 0);
    
    // In 1000 steps, a 50% probability should yield ~500 spikes
    // We allow a large variance (e.g., 400 to 600) to prevent flaky tests
    expect(totalSpikes).toBeGreaterThan(400);
    expect(totalSpikes).toBeLessThan(600);
  });

  it("should encode array of values", () => {
    const encoder = new PoissonEncoder(1.0);
    // [0.0, 1.0, 0.0] -> should definitely yield [0, 1, 0]
    const spikes = encoder.encodeArray([0.0, 1.0, 0.0]);
    expect(spikes[0]).toBe(0);
    expect(spikes[1]).toBe(1);
    expect(spikes[2]).toBe(0);
  });
});
