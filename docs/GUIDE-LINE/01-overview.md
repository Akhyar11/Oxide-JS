# System Documentation: Overview

Welcome to the official documentation of **Oxide-JS**, a custom machine learning framework designed to provide full control, flexibility, and high performance for AI researchers and developers.

## What is Oxide-JS?

**Oxide-JS** is a low-to-mid-level machine learning library built with **TypeScript** and accelerated by a **Rust (N-API)** backend. This project was born from the need for a transparent ML ecosystem, where every mathematical operation and training loop logic can be manually inspected and modified without depending on complex commercial frameworks.

## Vision and Goals

The project is designed with several main goals:
- **Full Control**: Providing the ability to manage every technical detail, from matrix shapes to parameter update mechanisms.
- **Hybrid Efficiency**: Combining the convenience of coding in TypeScript with the execution speed of critical numerical operations using Rust.
- **Architecture Research**: Serving as a playground to experiment with custom model architectures such as Transformers, Dimensionality Reduction, and more.
- **Modular Monorepo**: Decoupling core math, layers, and models into independent packages for better maintainability and scalability.

---

## Core Architecture

The system is organized as a monorepo divided into several specialized packages:

### 1. Core Logic (`packages/core`)
The foundation of the framework containing:
- **Matrix & Math**: The `Matrix` class based on `Float32Array` with flat memory layout for cache efficiency.
- **Math Primitives**: Basic operations like `dotProduct`, `add`, `sumAxis`, and `clipGradients`.
- **Hybrid Backend**: Modular Rust kernels for hot-path operations (Multi-Head Attention, LayerNorm, etc.) with automatic JS fallback.
- **Tokenizer**: Implementation of the Byte Pair Encoding (BPE) Tokenizer with multilingual pre-tokenization.

### 2. Neural Network Components (`packages/layers`)
This package provides building blocks for composing models:
- **Linear/Dense**: Fully-connected layers with optimizer support.
- **Recurrent**: `RNN`, `LSTM`, and `GRU` for hidden-state-based sequence modeling.
- **Attention**: Implementation of *Self-Attention* and *Multi-Head Attention* with causal masking schemes.
- **Specialized**: *Embedding*, *Dropout*, *Layer Normalization*, *Positional Encoding*, and *Flatten*.

### 3. Model Composition (`packages/models`)
High-level abstractions for managing data flow:
- **Sequential**: Linear layer stacking.
- **Transformers**: Complete NLP architecture with full-sequence causal language modeling.
- **Dimensionality Reduction**: Specialized models for data dimension reduction.

---

## Main Features

- **Matrix-Driven**: All operations center on efficient matrix manipulation.
- **BPE-Native**: Built-in support for advanced tokenization.
- **Optimizer & Loss Functions**: Various choices like Adam, SGD, MSE, and Softmax Cross-Entropy.
- **Training Workflow**: Intuitive API with `.forward()`, `.backward()`, and `.fit()` methods.

---

## Performance Philosophy

Oxide-JS prioritizes performance through:
1. **Pre-allocated Buffers**: Reducing Garbage Collection (GC) frequency during intense training loops.
2. **Native Dispatching**: Using `napi-rs` to minimize overhead between JavaScript and Rust layers.
3. **Modular Kernels**: High-performance Rust kernels separated by domain (math, activations, optimizers) for optimal execution.

---

> [!NOTE]
> This project is under active development (v2.3.1). Always ensure compatibility between the library version and the native kernels used.

**Next Steps:**
Continue to the [Installation](02-installation.md) section to start setting up your development environment.
