import { Matrix } from "@oxide-js/core";
import { BaseModel } from "@oxide-js/models";
import { SpikingEmbedding } from "../layers/SpikingEmbedding.js";
import { SpikingDense } from "../layers/SpikingDense.js";

export interface SpikingSentenceConfig {
    vocabSize: number;
    embedDim: number;  
    beta?: number;     
    threshold?: number;
}

export class SpikingSentenceEmbedder extends BaseModel {
    public vocabSize: number;
    public embedDim: number;
    
    public embedding: SpikingEmbedding;
    public contextLayer: SpikingDense;

    constructor(config: SpikingSentenceConfig) {
        super();
        this.vocabSize = config.vocabSize;
        this.embedDim = config.embedDim;
        const beta = config.beta ?? 0.9; 
        const threshold = config.threshold ?? 1.0;

        this.embedding = new SpikingEmbedding({
            inputDim: this.vocabSize,
            outputDim: this.embedDim,
            beta: beta,
            threshold: threshold,
            embeddingsInitializer: "glorot_normal"
        });
        
        this.contextLayer = new SpikingDense({
            units: this.embedDim,
            beta: beta,
            threshold: threshold,
            useBias: true,
            kernelInitializer: "glorot_normal"
        });

        this.add(this.embedding);
        this.add(this.contextLayer);
    }

    public resetState() {
        this.embedding.resetState();
        this.contextLayer.resetState();
    }

    /**
     * Membaca sebuah kalimat utuh dan mengubahnya menjadi Vektor Semantik tunggal (Spike Count)
     * @param inputs Matrix Token ID dari kalimat (shape: [batch=1, seq_len])
     * @param optionsOrTraining Opsi forward (tidak dipakai di SNN ini, tapi dibutuhkan oleh abstract method)
     * @returns Matrix Vektor berukuran `[1, embedDim]` yang berisi total Spike (Representasi Makna Kalimat)
     */
    public forward(inputs: Matrix, optionsOrTraining?: any): Matrix {
        if (!this.isBuilt) {
            this.build([1, 1]); // SNN layer selalu memproses kata per kata
        }
        
        this.resetState();
        const semanticVector = new Float32Array(this.embedDim);
        
        const seqLen = inputs._shape.length > 1 ? inputs._shape[1] : inputs._shape[0];
        const inputData = inputs._data;

        for (let i = 0; i < seqLen; i++) {
            const x = Matrix.fromFlat(new Float32Array([inputData[i]]), [1, 1]);
            
            // Default readingTime = 3 timestep
            for (let t = 0; t < 3; t++) {
                const wordSpikes = this.embedding.forward(x) as Matrix;
                const contextSpikes = this.contextLayer.forward(wordSpikes) as Matrix;

                const outData = contextSpikes._data;
                for(let j=0; j<this.embedDim; j++) {
                    semanticVector[j] += outData[j];
                }
            }
        }
        
        return Matrix.fromFlat(semanticVector, [1, this.embedDim]);
    }

    /**
     * Melatih model sentence embedder menggunakan prinsip Word2Vec CBOW-style Hebbian Contrastive Learning.
     * Secara otomatis mengambil rata-rata vektor konteks kalimat saat ini (Positive), 
     * dan menolaknya dari list mean vector negatif (NegativeContexts).
     * Metode ini bypass context layer dan langsung melatih embedding untuk mencegah representation collapse.
     * 
     * @returns Float32Array rata-rata vektor (meanVec) dari kalimat saat ini yang dapat disimpan ke historyBuffer.
     */
    public learnContrastive(
        tokens: number[] | Float32Array,
        negativeContexts: Float32Array[],
        learningRate: number = 0.01,
        marginPositive: number = 0.1,
        marginNegative: number = 0.05
    ): Float32Array {
        const wordEmbeddings: { tokenId: number, vec: Float32Array }[] = [];
        const kernel = this.embedding.getParameter('kernel')!._data;
        const dim = this.embedDim;
        
        for (let i = 0; i < tokens.length; i++) {
            const tokenId = Math.round(tokens[i]);
            if (tokenId >= 0 && tokenId < this.vocabSize) {
                const offset = tokenId * dim;
                const vec = new Float32Array(dim);
                for (let j = 0; j < dim; j++) vec[j] = kernel[offset + j];
                wordEmbeddings.push({ tokenId, vec });
            }
        }
        
        const meanVec = new Float32Array(dim);
        if (wordEmbeddings.length > 0) {
            for (const w of wordEmbeddings) {
                for (let j = 0; j < dim; j++) meanVec[j] += w.vec[j];
            }
            for (let j = 0; j < dim; j++) meanVec[j] /= wordEmbeddings.length;

            this.embedding.learnHebbian(
                tokens, 
                meanVec, 
                negativeContexts, 
                learningRate, 
                marginPositive, 
                marginNegative
            );
        }
        
        return meanVec;
    }
}
