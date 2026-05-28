import { describe, it, expect, beforeEach } from "vitest";
import { SpikingNetwork } from "../src/core/SpikingNetwork.js";
import { STDP } from "../src/learning/STDP.js";

describe("STDP Learning Rule", () => {
  let net: SpikingNetwork;
  let stdp: STDP;

  beforeEach(() => {
    // 2 Neurons, standard decay
    net = new SpikingNetwork(2, 0.5, 10.0);
    // Connect N0 to N1 with weight 2.0
    net.connect(0, 1, 2.0);

    // Initialize STDP
    stdp = new STDP(net, {
      learningRate: 0.1,
      tauPlus: 0.8,
      tauMinus: 0.8,
      aPlus: 1.0,
      aMinus: 1.0,
      wMax: 5.0,
      wMin: -5.0
    });
  });

  it("should increase weight (LTP) if Pre spikes before Post", () => {
    const initialWeight = net.weights[0][0];

    // Step 1: Force Pre (N0) to spike
    net.injectCurrent(0, 20.0);
    net.step(); 
    stdp.updateWeights();

    // Step 2: Force Post (N1) to spike
    net.injectCurrent(1, 20.0);
    net.step();
    stdp.updateWeights();

    const finalWeight = net.weights[0][0];
    
    // Weight should increase
    expect(finalWeight).toBeGreaterThan(initialWeight);
  });

  it("should decrease weight (LTD) if Post spikes before Pre", () => {
    const initialWeight = net.weights[0][0];

    // Step 1: Force Post (N1) to spike first
    net.injectCurrent(1, 20.0);
    net.step(); 
    stdp.updateWeights();

    // Step 2: Force Pre (N0) to spike
    net.injectCurrent(0, 20.0);
    net.step();
    stdp.updateWeights();

    const finalWeight = net.weights[0][0];
    
    // Weight should decrease
    expect(finalWeight).toBeLessThan(initialWeight);
  });

  it("should respect weight limits (wMax and wMin)", () => {
    // Set weight near max
    net.weights[0][0] = 4.9;
    stdp.config.learningRate = 1.0; // Force large jump
    
    // Repeatedly trigger LTP
    for(let i=0; i<5; i++) {
      net.injectCurrent(0, 20.0);
      net.step(); stdp.updateWeights();
      
      net.injectCurrent(1, 20.0);
      net.step(); stdp.updateWeights();
    }

    // Weight should be clamped at wMax = 5.0
    expect(net.weights[0][0]).toBeCloseTo(5.0);
  });
});
