import { describe, it, expect, beforeEach } from 'vitest';
import { Matrix } from '@oxide-js/core';
import { SpikingDenseBPTT } from '../src/layers/SpikingDenseBPTT.js';

describe('SpikingDenseBPTT Layer', () => {
  const units = 4;
  const inFeatures = 3;
  const batch = 2;
  const maxTimeSteps = 3;

  let layer: SpikingDenseBPTT;

  beforeEach(() => {
    layer = new SpikingDenseBPTT({
      units,
      kernelInitializer: 'ones',
      useBias: false
    });
    layer.build([batch, inFeatures]);
  });

  it('should initialize parameters correctly', () => {
    expect(layer.units).toBe(units);
    expect(layer.kernel).toBeDefined();
    
    // Mengecek apakah inisialisasi dynamic beta (Bit-Shift Decay Float) bekerja
    expect(layer.beta).toBeDefined();
    expect(layer.beta.length).toBe(units);
    
    for (let i = 0; i < units; i++) {
        // Karena kita melakukan pre-kalkulasi multiplier: 1.0 - (1.0 / Math.pow(2, shift))
        // dimana shift = 2 hingga 5 (1/4 hingga 1/32)
        // Maka rentang beta yang valid adalah 0.75 hingga 0.96875
        expect(layer.beta[i]).toBeGreaterThanOrEqual(0.75);
        expect(layer.beta[i]).toBeLessThanOrEqual(0.96875);
    }
  });

  it('should throw error when calling compute() directly', () => {
    const inputs = Matrix.fromFlat(new Float32Array(batch * inFeatures), [batch, inFeatures]);
    expect(() => {
        // @ts-ignore
        layer.compute(inputs);
    }).toThrowError(/Harap gunakan computeStep/);
  });

  it('should process sequence, enforce BPTT limits, and store history correctly', () => {
    layer.resetSequence(maxTimeSteps);
    
    expect(layer.maxTimeSteps).toBe(maxTimeSteps);
    expect(layer.historyInputs.length).toBe(maxTimeSteps);

    // Dummy binary spike input
    const inputData = new Float32Array(batch * inFeatures).fill(1);
    const inputs = Matrix.fromFlat(inputData, [batch, inFeatures]);

    // Time Step 0
    const out0 = layer.computeStep(inputs, 0);
    expect(out0._shape).toEqual([batch, units]);
    expect(layer.historyInputs[0]).toBeDefined();
    expect(layer.historyPotentials[0]).toBeDefined();
    expect(layer.historySpikes[0]).toBeDefined();

    // Time Step 1
    const out1 = layer.computeStep(inputs, 1);
    expect(out1._shape).toEqual([batch, units]);
    expect(layer.historyInputs[1]).toBeDefined();

    // Time Step 2
    const out2 = layer.computeStep(inputs, 2);
    expect(out2._shape).toEqual([batch, units]);
    
    // Time Step 3 (Exceeds maxTimeSteps -> Harus Error)
    expect(() => {
        layer.computeStep(inputs, 3);
    }).toThrowError(/melebihi batas maxTimeSteps/);
  });

  it('should run learnThroughTime properly without crashing', () => {
    layer.resetSequence(maxTimeSteps);
    
    const inputData = new Float32Array(batch * inFeatures).fill(1);
    const inputs = Matrix.fromFlat(inputData, [batch, inFeatures]);

    // Jalankan seluruh sekuens
    for (let t = 0; t < maxTimeSteps; t++) {
        layer.computeStep(inputs, t);
    }

    // Siapkan urutan error palsu untuk pengujian (error di t=0, t=1, t=2)
    const errors = [];
    for (let t = 0; t < maxTimeSteps; t++) {
        errors.push(Matrix.fromFlat(new Float32Array(batch * units).fill(0.1), [batch, units]));
    }

    // Uji BPTT untuk Output Layer (parameter B = undefined)
    expect(() => {
        layer.learnThroughTime(errors, undefined, 0.01);
    }).not.toThrow();

    // Uji BPTT untuk Hidden Layer (parameter B = Identity Matrix Broadcast)
    const B = Matrix.fromFlat(new Float32Array(units * units).fill(1), [units, units]);
    expect(() => {
        layer.learnThroughTime(errors, B, 0.01);
    }).not.toThrow();
  });
  it('should correctly accumulate potentials and trigger spikes deterministically over time', () => {
    // Buat layer deterministik
    const testLayer = new SpikingDenseBPTT({
        units: 2,
        useBias: false,
        kernelInitializer: 'zeros'
    });
    testLayer.build([1, 2]); // batch=1, inFeatures=2

    // Override kernel manual: [ [0.6, 0.0], [0.0, 0.8] ]
    testLayer.kernel!._data.set([0.6, 0.0, 0.0, 0.8]);

    // Override konstan beta (0.5 agar mudah dihitung) dan threshold (1.0)
    testLayer.beta.fill(0.5);
    testLayer.threshold.fill(1.0);

    // Siapkan sequence 3 time steps
    testLayer.resetSequence(3);

    // Input selalu menyala setiap timestep: [1, 1]
    const inputs = Matrix.fromFlat(new Float32Array([1, 1]), [1, 2]);

    // --- TIME STEP 0 ---
    const out0 = testLayer.computeStep(inputs, 0);
    expect(out0._data).toEqual(new Float32Array([0, 0]));
    
    // History potentials sebelum spike harus mencatat nilai [0.6, 0.8]
    expect(testLayer.historyPotentials[0]._data[0]).toBeCloseTo(0.6, 5);
    expect(testLayer.historyPotentials[0]._data[1]).toBeCloseTo(0.8, 5);

    // --- TIME STEP 1 ---
    const out1 = testLayer.computeStep(inputs, 1);
    expect(out1._data).toEqual(new Float32Array([0, 1]));
    
    expect(testLayer.historyPotentials[1]._data[0]).toBeCloseTo(0.9, 5);
    expect(testLayer.historyPotentials[1]._data[1]).toBeCloseTo(1.0, 5);

    // --- TIME STEP 2 ---
    const out2 = testLayer.computeStep(inputs, 2);
    expect(out2._data).toEqual(new Float32Array([1, 0]));

    expect(testLayer.historyPotentials[2]._data[0]).toBeCloseTo(1.0, 5);
    expect(testLayer.historyPotentials[2]._data[1]).toBeCloseTo(0.8, 5);
  });
});
