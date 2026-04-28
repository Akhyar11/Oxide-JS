# Reproducibility Artifacts: Empirical Analysis of Recurrent Neural Network Stabilities for Indonesian Sentiment Classification

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Framework](https://img.shields.io/badge/Framework-%40akhyar11%2Fml--v1-blueviolet)](https://www.npmjs.com/package/@akhyar11/ml-v1)

This repository serves as the official reproducibility artifact and experimental companion for the upcoming journal publication on the stability and convergence dynamics of recurrent neural networks (RNNs) in natural language processing (NLP). Specifically, it details the methodology, topology, and granular statistical findings corresponding to our empirical study on Indonesian document-level sentiment classification.

## 📖 1. Overview & Objectives

Neural network training entails inherent stochasticity—driven by random weight initializations, permutation of mini-batches, and numerical variances during gradient propagation. Consequently, singular training iterations yield statistically insufficient evidence to establish the superiority of one architectural paradigm over another. 

The primary objective of this empirical study is to rigorously evaluate the **training stability, convergence velocity, and generalization capability** of three foundational recurrent architectures:
1. **Vanilla RNN** (Recurrent Neural Network)
2. **LSTM** (Long Short-Term Memory)
3. **GRU** (Gated Recurrent Unit)

All models in this study were strictly synthesized and evaluated using the `@akhyar11/ml-v1` native TypeScript library to entirely eliminate cross-framework variability, thereby isolating the architectural characteristics as the sole independent variables.

## 🔬 2. Methodology & Experimental Design

To conduct a robust comparative analysis, this experiment executes a rigorous **Repeated-Measures Design**. Each architectural variant was subjected to **10 independent training repetitions** (epochs = 30). Before each stochastic manifestation, the embedding and recurrent weight matrices were completely reset to ensure isolated initialization states.

### 2.1. Network Topology Pipeline

To ensure algorithmic parity during comparison, all models utilize a controlled homologous pipeline. The solitary independent variable across the three variants is the internal calculation schema of the recurrent processing unit itself. 

```mermaid
graph LR
    A[BPE Tokenizer] -->|128-token Array| B(Embedding Layer)
    B -->|Dense Vector Transformation| C{RNN / LSTM / GRU Unit}
    C -->|Aggregated Hidden State| D[Dense Linear Output]

    classDef input fill:#1e293b,stroke:#475569,color:#f8fafc;
    classDef embed fill:#312e81,stroke:#4f46e5,color:#f8fafc;
    classDef recur fill:#064e3b,stroke:#10b981,color:#f8fafc;
    classDef out fill:#881337,stroke:#f43f5e,color:#f8fafc;

    class A input;
    class B embed;
    class C recur;
    class D out;
```

### 2.2. Architectural Hyperparameters

- **Vocabulary Space**: 15,182 discrete tokens
- **Input Dimension / Context Window**: 128 (Left-padded normalization)
- **Embedding Subspace Dimension**: 128
- **Recurrent Hidden State Units**: 32
- **Latent Stateful Sequence Matrix**: Disabled (`stateful: false`)
- **Return Temporal Sequences**: Disabled (`returnSequences: false`)
- **Output Classes**: 3 (Negative, Positive, Neutral)
- **Loss Objective Function**: Categorical Cross-Entropy (Softmax Output)
- **Stochastic Optimizer**: Adam (Adaptive Moment Estimation)
- **Initial Learning Rate ($\eta$)**: 0.001

## 📊 3. Dataset Characteristics

The empirical observations are derived from the **SMSA Toolkit (Document-level Sentiment Analysis)**, an integral subset of the recognized **IndoNLU** benchmark suite. The dataset provides thousands of highly-contextualized text strings compiled from Indonesian social media vernacular.

| Dataset Parameter | Specification |
| :--- | :--- |
| **Origin Task** | Indonesian Document-Level Sentiment Analysis |
| **Classification Categories** | Negative, Positive, Neutral |
| **Original Training Corpus Size**| 11,000 document samples |
| **Training Allocation (80%)** | 8,800 sample instances |
| **Internal Validation Allocation (20%)** | 2,200 sample instances |
| **External Hold-out Validation** | 1,260 sample instances |

## 🏆 4. Aggregate Empirical Results & Indicator Matrices

The empirical data recorded below encapsulates the statistical variables explicitly extracted across all 10 independent training permutations. The matrix includes detailed telemetry regarding their standard deviations ($\sigma$) and stability indexes, effectively quantifying architectural volatility.

A specialized interactive telemetry visualization dashboard (`stability_report.html`) is also provided within the repository source code to explore these multi-dimensional matrices in real-time.

### 4.1. Primary Convergence Matrices (Internal Validation)
| Architecture Variant | Trainable Parameters | Mean Convergence Loss | Internal Peak Accuracy | 
| :--- | :---: | :---: | :---: | 
| **LSTM** | `1,964,003` | `0.0093` | **89.24%** | 
| **GRU** | `1,958,851` | `0.1321` | 84.55% | 
| **RNN** | `1,948,547` | `0.0141` | 83.62% | 

### 4.2. Granular External Hold-out Evaluation Matrices
To verify the global utility against overfitting, an independent hold-out test was conducted against a static block of 1,260 validation sequences. The following matrix presents the **statistical mean and standard deviation ($\sigma$)** accumulated exactly across all 10 stochastic training repetitions.

| Architecture Variant | Mean Accuracy ($\pm\sigma$) | Mean Macro F1 | Mean Weighted F1 | Empirical Stability Score |
| :--- | :---: | :---: | :---: | :---: |
| **LSTM** | **85.30%** ($\pm 1.11\%$) | **80.74%** | **85.16%** | **99% (Highly Stable)** |
| **RNN** | 77.64% ($\pm 1.88\%$) | 67.37% | 76.72% | 98% (Stable) |
| **GRU** | 72.98% ($\pm 12.72\%$) | 66.46% | 72.66% | 87% (High Variance)$^\dagger$ |

*$^\dagger$ Note: The GRU model exhibited severe mathematical divergence during repetition Run #2 (External Accuracy dropping catastrophically to 44.60%), manifesting extreme volatility which heavily penalized its empirical stability score and mean accuracy.*

### 4.3. Peak State Confusion Matrices
To further diagnose inter-class differentiation logic across the algorithms, the following confusion matrices from the highest-performing iteration of each architecture are provided. 

**1. LSTM Optimal Checkpoint (Run #5)**
- Peak Validation Accuracy: **86.98%** | Weighted F1-Score: **86.94%**

| Actual `\` Predicted | Negative | Positive | Neutral |
| :--- | :---: | :---: | :---: |
| **Negative** | **343** | 46 | 5 |
| **Positive** | 58 | **664** | 13 |
| **Neutral** | 29 | 13 | **89** |

**2. GRU Optimal Checkpoint (Run #5)**
- Peak Validation Accuracy: **82.62%** | Weighted F1-Score: **82.16%**

| Actual `\` Predicted | Negative | Positive | Neutral |
| :--- | :---: | :---: | :---: |
| **Negative** | **279** | 101 | 14 |
| **Positive** | 36 | **683** | 16 |
| **Neutral** | 21 | 31 | **79** |

**3. Vanilla RNN Optimal Checkpoint (Run #2)**
- Peak Validation Accuracy: **79.68%** | Weighted F1-Score: **78.99%**

| Actual `\` Predicted | Negative | Positive | Neutral |
| :--- | :---: | :---: | :---: |
| **Negative** | **279** | 102 | 13 |
| **Positive** | 54 | **666** | 15 |
| **Neutral** | 38 | 34 | **59** |

## 📈 5. Analytical Discoveries & Conclusions

Based on the controlled multi-run trials facilitated via the `@akhyar11/ml-v1` framework, our findings objectively quantify the behavioral tendencies of sequence modeling frameworks. The LSTM architecture achieved the best overall performance under the evaluated library version, dataset split, and experimental configuration. Below are the theoretical and mathematical justifications supported by the compiled data:

1. **Superiority in Minority Class Differentiation**: The **Long Short-Term Memory (LSTM)** architecture consistently performed better than both GRU and Vanilla RNN, particularly in differentiating subtle classes. While GRU achieved high overall accuracy (84.55%) by aggressively predicting the majority "Positive" sentiment class, LSTM successfully modeled the minority "Neutral" sentiment (identifying 89 Neutral samples compared to GRU's 79 and RNN's 59). This allowed LSTM to achieve a much higher mean F1-Score of **85.16%** compared to GRU's **72.66%**.

2. **Decoupled Memory and Activation Mechanisms**: The core structural advantage of LSTM over GRU lies in its independent memory unit (Cell State $C_t$) decoupled from the activation output (Hidden State $h_t$). GRU merges these pathways into a singular vector, causing immediate overwriting of historical context by recent tokens. Because Indonesian document-level sentiment often places crucial contextual modifiers far apart, the LSTM's dedicated Cell State acts as an isolated highway effectively carrying latent semantic meaning across the 128-token span without dilution.

3. **Output Gate Gating Logic**: Unlike GRU, LSTM incorporates an explicit *Output Gate*. This gate dynamically controls what portion of the internal memory gets exposed to the external hidden layer. In highly varied semantic spaces, this mechanism empowers the LSTM to suppress irrelevant historical noise (like sarcasm markers that get neutralized) from polluting the final classification layer—a capability absent in the streamlined GRU constraint.

4. **Stochastic Training Stability**: Supported by its structural complexity, the LSTM demonstrated a near-perfect stability score of **99%** with an extremely low variance ($\sigma = 1.11\%$). In contrast, GRU suffered from unpredictable catastrophic unlearning on Run #2 (dropping to 44.60%). The explicit `forget_gate` implementation in LSTM constructs a more bounded and robust gradient manifold backward through time (BPTT), effectively immunizing the model against the stochastic volatility triggered by random weight initializations.

## ⚙️ 6. Reproducibility Hardware & Environment

To support deterministic replication, all 30 sub-experiments were compiled sequentially under uniform thermal and computational conditions.
- **Compute Framework**: `@akhyar11/ml-v1` version 2.3.0 Ecosystem
- **Processor**: 11th Gen Intel® Core™ i5-1135G7 (Microarchitecture: Tiger Lake)
- **System Memory**: 15.41 GB Available
- **Host OS**: Linux CachyOS (Kernel optimized for computational workloads)
- **Cumulative Execution Duration**: ~4 hours, 20 minutes, 34 seconds

## 📖 7. Citations & References

If you extract statistical configurations or telemetry artifacts from this open-source repository for secondary research, please apply proper citation referencing the underlying IndoNLU dataset distributions:

```bibtex
@inproceedings{wilie2020indonlu,
  title={IndoNLU: Evaluating and Winning Indonesian Natural Language Understanding},
  author={Wilie, Bryan and Vincentio, Karissa and Winata, Genta Indra and Cahyawijaya, Samuel and Li, Xiaohong and Zy, Zhi Yuan and Khusainov, Ruslan and Lovenia, Holy and Chandawati, Ika Alfina and Kurniawan, Derry Tanti and others},
  booktitle={Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing},
  pages={501--516},
  year={2020}
}

@misc{hanafie_kaggle_sentiment,
  author       = {Alvin Hanafie},
  title        = {Dataset for Indonesian Sentiment Analysis},
  howpublished = {Kaggle Repository: \url{https://www.kaggle.com/datasets/alvinhanafie/dataset-for-indonesian-sentiment-analysis}},
}
```

---
*This repository and its integrated telemetry reporting mechanisms were systematically architected specifically to support reproducibility and empirical validation in machine learning studies.*
