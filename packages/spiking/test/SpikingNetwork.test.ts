import { describe, it, expect, beforeEach } from "vitest";
import { SpikingNetwork } from "../src/core/SpikingNetwork.js";

describe("SpikingNetwork (Add-Only Architecture)", () => {
  let net: SpikingNetwork;

  beforeEach(() => {
    // 3 neurons, decay beta = 0.5 (fast decay for testing), threshold = 10
    net = new SpikingNetwork(3, 0.5, 10.0);
  });

  it("should initialize correctly", () => {
    expect(net.numNeurons).toBe(3);
    expect(net.potentials).toHaveLength(3);
    expect(net.spikes).toHaveLength(3);
    expect(net.thresholds[0]).toBe(10.0);
  });

  it("should accumulate potential when current is injected", () => {
    net.injectCurrent(0, 5.0);
    expect(net.potentials[0]).toBe(5.0);

    // After 1 step, potential should decay (5.0 * 0.5 = 2.5)
    net.step();
    expect(net.potentials[0]).toBe(2.5);
    expect(net.spikes[0]).toBe(0); // No spike yet
  });

  it("should fire a spike and reset potential when threshold is reached", () => {
    // Inject enough current to cross threshold (10.0)
    // Considering decay happens before threshold check in our step logic:
    // V_t = V_{t-1} * beta -> 20.0 * 0.5 = 10.0
    net.injectCurrent(1, 20.0);
    net.step();

    expect(net.spikes[1]).toBe(1); // Neuron 1 fired!
    expect(net.potentials[1]).toBe(0); // Soft reset: 10.0 - 10.0 = 0.0
  });

  it("should perform Add-Only propagation correctly", () => {
    // Connect Neuron 0 -> Neuron 2 with weight 8.0
    net.connect(0, 2, 8.0);
    
    // Force Neuron 0 to spike in the current state
    net.spikes[0] = 1;

    // Run step
    net.step();

    // Neuron 0 spike should propagate its weight to Neuron 2
    // Since Neuron 2 started at 0.0, it receives +8.0, then decays (8.0 * 0.5 = 4.0)
    // Wait, let's trace the step() logic carefully:
    // 1. Add-Only: if spikes[0] === 1 -> potentials[2] += 8.0 (now 8.0)
    // 2. Decay: potentials[2] *= 0.5 -> 4.0
    expect(net.potentials[2]).toBe(4.0);
    
    // Check that spikes were reset
    expect(net.spikes[0]).toBe(0);
  });

  it("should handle multiple connections and spikes", () => {
    // N0 -> N2 (w=6.0)
    // N1 -> N2 (w=4.0)
    net.connect(0, 2, 6.0);
    net.connect(1, 2, 4.0);

    // Force N0 and N1 to spike
    net.spikes[0] = 1;
    net.spikes[1] = 1;

    net.step();

    // Potential of N2 should be (6.0 + 4.0) = 10.0
    // Then decay: 10.0 * 0.5 = 5.0
    expect(net.potentials[2]).toBe(5.0);
  });

  it("should trigger cascading spikes over time steps", () => {
    // N0 -> N1 (w=20.0)
    // N1 -> N2 (w=20.0)
    net.connect(0, 1, 20.0);
    net.connect(1, 2, 20.0);

    // Inject 20 to N0 to guarantee a spike after decay (20 * 0.5 = 10)
    net.injectCurrent(0, 20.0);

    // Step 1: N0 decays to 10, spikes, resets to 0.
    net.step();
    expect(net.spikes[0]).toBe(1);
    expect(net.potentials[1]).toBe(0); // N0's spike will affect N1 in the NEXT step.

    // Step 2: N0 spike affects N1. N1 gets 20.0. Decays to 10.0. N1 spikes.
    net.step();
    expect(net.spikes[0]).toBe(0);
    expect(net.spikes[1]).toBe(1);
    expect(net.potentials[1]).toBe(0); // 10.0 - threshold(10) = 0
    
    // Step 3: N1 spike affects N2. N2 gets 20.0. Decays to 10.0. N2 spikes.
    net.step();
    expect(net.spikes[1]).toBe(0);
    expect(net.spikes[2]).toBe(1);
  });
});
