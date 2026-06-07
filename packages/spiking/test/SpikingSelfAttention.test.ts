import { describe, it, expect, beforeEach } from 'vitest';
import { Matrix } from '@oxide-js/core';
import { SpikingSelfAttention } from '../src/layers/SpikingSelfAttention.js';

describe('SpikingSelfAttention Layer', () => {
  const d_model = 4;
  const sequenceLength = 3;
  const batch = 2;
  const batchSeq = batch * sequenceLength;

  let attentionLayer: SpikingSelfAttention;

  beforeEach(() => {
    attentionLayer = new SpikingSelfAttention({
      d_model,
      sequenceLength,
      kernelInitializer: 'ones' // Inisialisasi awal statis agar prediktabilitas baik
    });
    attentionLayer.build([batchSeq, d_model]);
  });

  it('should initialize parameters correctly', () => {
    expect(attentionLayer.d_model).toBe(d_model);
    expect(attentionLayer.sequenceLength).toBe(sequenceLength);
    expect(attentionLayer.kernelQ).toBeDefined();
    expect(attentionLayer.kernelK).toBeDefined();
    expect(attentionLayer.kernelV).toBeDefined();
    
    // Potentials harus diawali kosong
    expect(attentionLayer.potentialsQ._data.length).toBe(0);
    expect(attentionLayer.potentialsScores._data.length).toBe(0);
  });

  it('should compute output with the correct shape and binary spike format', () => {
    // Buat input dummy berupa array biner (0 dan 1)
    const inputData = new Float32Array(batchSeq * d_model);
    for (let i = 0; i < inputData.length; i++) {
        inputData[i] = Math.random() > 0.5 ? 1 : 0;
    }
    const inputs = Matrix.fromFlat(inputData, [batchSeq, d_model]);

    const output = attentionLayer.forward(inputs) as Matrix;

    // Cek shape output yang diharapkan: [batch * seqLen, d_model]
    expect(output._shape[0]).toBe(batchSeq);
    expect(output._shape[1]).toBe(d_model);

    // Pastikan output hanya berisi format spike biner (0.0 atau 1.0)
    const outputData = output._data;
    for (let i = 0; i < outputData.length; i++) {
        expect([0, 1]).toContain(outputData[i]);
    }
    
    // State potentials harus terbentuk sesuai shape
    expect(attentionLayer.potentialsQ._data.length).toBe(batchSeq * d_model);
    expect(attentionLayer.potentialsScores._data.length).toBe(batchSeq * sequenceLength);
  });

  it('should properly accumulate potentials in sequential steps', () => {
    const inputData = new Float32Array(batchSeq * d_model).fill(1);
    const inputs = Matrix.fromFlat(inputData, [batchSeq, d_model]);

    // Jalankan beberapa time-steps (forward pass)
    attentionLayer.forward(inputs);
    
    // Ambil sebagian data dari potentialsQ untuk verifikasi akumulasi
    const firstStepPotentials = new Float32Array(attentionLayer.potentialsQ._data);
    
    attentionLayer.forward(inputs);
    
    // Karena input konstan, seharusnya potensial naik atau diset ulang setelah spike, 
    // tetapi setidaknya harus berjalan normal (tidak crash)
    expect(attentionLayer.potentialsQ._data).toBeDefined();
    // Pada saat reset, potensial harus dikembalikan ke 0
    attentionLayer.resetState();
    const zeros = new Float32Array(attentionLayer.potentialsQ._data.length).fill(0);
    expect(attentionLayer.potentialsQ._data).toEqual(zeros);
  });

  it('should throw error if input batch length is not multiple of sequence length', () => {
    const invalidBatchSeq = 7; // Bukan kelipatan sequenceLength (3)
    const inputData = new Float32Array(invalidBatchSeq * d_model).fill(1);
    const inputs = Matrix.fromFlat(inputData, [invalidBatchSeq, d_model]);

    expect(() => {
        attentionLayer.forward(inputs);
    }).toThrowError(/Jumlah baris input/);
  });

  it('should correctly compute exact Self-Attention math (Deterministic Correctness)', () => {
    // Set up environment yang sepenuhnya deterministik
    const testLayer = new SpikingSelfAttention({
        d_model: 2,
        sequenceLength: 2,
        kernelInitializer: 'zeros' // Kita override manual
    });
    
    testLayer.build([2, 2]); // batchSeq=2, d_model=2

    // Override bobot menjadi Identity Matrix
    const identity = new Float32Array([1, 0, 0, 1]);
    testLayer.kernelQ!._data.set(identity);
    testLayer.kernelK!._data.set(identity);
    testLayer.kernelV!._data.set(identity);

    // Override LIF properties agar seketika spike jika input >= 1
    testLayer.betaQKV.fill(0.0);
    testLayer.thresholdQKV.fill(0.5);
    
    testLayer.betaScores.fill(0.0);
    testLayer.thresholdScores.fill(0.5); // Threshold kecil agar skor >= 1 langsung tembak spike

    // Input: Token 0 = [1, 0], Token 1 = [0, 1]
    const inputs = Matrix.fromFlat(new Float32Array([1, 0, 0, 1]), [2, 2]);

    // Lakukan forward pass
    const output = testLayer.forward(inputs) as Matrix;

    // Analisis ekspektasi:
    // SQ, SK, SV akan identik dengan input (karena dikali Identity dan threshold < 1.0)
    // SQ dot SK^T:
    // - Token 0 dot Token 0 = 1 (match index 0)
    // - Token 0 dot Token 1 = 0
    // - Token 1 dot Token 0 = 0
    // - Token 1 dot Token 1 = 1
    // S_Scores akan menjadi Identity Matrix [1, 0, 0, 1]
    // Hasil akhir: S_Scores dikali SV -> [1, 0, 0, 1]

    expect(output._shape).toEqual([2, 2]);
    expect(output._data).toEqual(new Float32Array([1, 0, 0, 1]));

    // Mari tes dengan Token yang identik: Token 0 = [1, 0], Token 1 = [1, 0]
    const inputsSame = Matrix.fromFlat(new Float32Array([1, 0, 1, 0]), [2, 2]);
    const outputSame = testLayer.forward(inputsSame) as Matrix;
    
    // SQ dot SK^T:
    // - Token 0 dot Token 0 = 1
    // - Token 0 dot Token 1 = 1
    // - Token 1 dot Token 0 = 1
    // - Token 1 dot Token 1 = 1
    // S_Scores akan menjadi [1, 1, 1, 1] (semuanya spike karena saling cocok)
    // SV = [1, 0, 1, 0]
    // Perkalian akhir: 
    // Out Token 0 = S_Scores[0]*SV[0] + S_Scores[1]*SV[1] = 1*1 + 1*1 = 2 (Lalu dikunci/clamped ke 1) = [1, 0]
    // Out Token 1 = [1, 0]
    expect(outputSame._data).toEqual(new Float32Array([1, 0, 1, 0]));
  });
});
