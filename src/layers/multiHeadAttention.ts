import { StatusLayer } from "../@types/type";
import mj from "../math";
import Matrix from "../matrix";
import SelfAttention from "./selfAttention";
import Dense from "./dense";

interface MultiHeadAttentionLayer {
  units: number;
  heads: number;
  seqLen: number;
  alpha?: number;
  status?: StatusLayer;
}

export default class MultiHeadAttention {
  name = "multi head attention layer";
  units: number;
  heads: number;
  headUnits: number;
  seqLen: number;
  alpha: number;
  status: StatusLayer;
  
  attentionHeads: SelfAttention[];
  wo: Dense;
  
  inputShape: [number, number];
  outputShape: [number, number];
  params: number;
  loss: number = 0;

  private input: Matrix = mj.matrix([]);

  constructor({ units, heads, seqLen, alpha = 0.1, status = "input" }: MultiHeadAttentionLayer) {
    this.units = units;
    this.heads = heads;
    this.seqLen = seqLen;
    this.alpha = alpha;
    this.status = status;
    
    this.inputShape = [units, seqLen];
    this.outputShape = [units, seqLen];
    
    if (this.units % this.heads !== 0) {
      throw new Error(`units (${units}) must be divisible by heads (${heads})`);
    }
    this.headUnits = this.units / this.heads;

    this.attentionHeads = [];
    for (let i = 0; i < heads; i++) {
       this.attentionHeads.push(new SelfAttention({
         units: this.units,
         outputUnits: this.headUnits,
         seqLen,
         alpha,
         status: "input"
       }));
    }

    this.wo = new Dense({
      units: this.units,
      outputUnits: this.units,
      activation: "linear",
      alpha,
    });

    this.params = this.heads * this.attentionHeads[0].params + this.wo.params;
  }

  compile({ alpha, optimizer }: { alpha?: number; optimizer?: any }) {
    if (alpha !== undefined) this.alpha = alpha;
    for (const head of this.attentionHeads) {
      head.compile({ alpha, optimizer });
    }
    this.wo.compile({ alpha, optimizer });
  }

  forward(x: Matrix): Matrix {
    this.input = x;
    const seqLen = x._shape[1];
    
    // Concatenate outputs from all heads
    const catData = new Float64Array(this.units * seqLen);
    
    for (let i = 0; i < this.heads; i++) {
        const headOut = this.attentionHeads[i].forward(x);
        const headData = headOut._data;
        for (let r = 0; r < this.headUnits; r++) {
           const rowIdx = i * this.headUnits + r;
           const targetOffset = rowIdx * seqLen;
           const srcOffset = r * seqLen;
           for (let c = 0; c < seqLen; c++) {
              catData[targetOffset + c] = headData[srcOffset + c];
           }
        }
    }
    
    const concatenated = Matrix.fromFlat(catData, [this.units, seqLen]);
    return this.wo.forward(concatenated);
  }

  backward(y: Matrix, err: Matrix): Matrix {
    const dCat = this.wo.backward(y, err);
    const seqLen = dCat._shape[1];
    
    let gradInput = mj.zeros(this.input._shape);
    
    for (let i = 0; i < this.heads; i++) {
        const headErrData = new Float64Array(this.headUnits * seqLen);
        const dCatData = dCat._data;
        for (let r = 0; r < this.headUnits; r++) {
           const rowIdx = i * this.headUnits + r;
           const srcOffset = rowIdx * seqLen;
           const targetOffset = r * seqLen;
           for (let c = 0; c < seqLen; c++) {
              headErrData[targetOffset + c] = dCatData[srcOffset + c];
           }
        }
        const headErr = Matrix.fromFlat(headErrData, [this.headUnits, seqLen]);
        const headGrad = this.attentionHeads[i].backward(y, headErr);
        
        gradInput.addInPlace(headGrad);
    }
    return gradInput;
  }

  save() {
    return {
       name: this.name,
       units: this.units,
       heads: this.heads,
       alpha: this.alpha,
       attentionHeads: this.attentionHeads.map(h => h.save()),
       wo: this.wo.save()
    };
  }

  load(data: any) {
    if (data.attentionHeads) {
       for (let i = 0; i < this.heads; i++) {
          this.attentionHeads[i].load(data.attentionHeads[i].q, data.attentionHeads[i].k, data.attentionHeads[i].v);
       }
    }
    if (data.wo) {
       this.wo.load(data.wo.weight, data.wo.bias);
    }
  }
}
