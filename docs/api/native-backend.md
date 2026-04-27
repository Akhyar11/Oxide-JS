# Native Backend

The Rust native backend provides hardware-accelerated implementations of the most computationally intensive operations in ML-V1.

## Import

```ts
import { isNativeAvailable } from "@akhyar11/ml-v1"
```

## Overview

ML-V1 includes a Rust extension (via [napi-rs](https://napi.rs/)) that accelerates critical hot paths:

- Matrix dot-product (GEMM)
- Activation functions (sigmoid, relu, softmax, etc.)
- Layer normalization
- Embedding lookup
- Attention computation
- Adam and other optimizer weight updates
- Full fused recurrent loops (RNN, LSTM, GRU) for high-speed BPTT
- Fused Sparse Optimizer updates for NLP embedding training

When the native addon is not available, the library falls back transparently to pure JavaScript implementations. There is no API difference; the same functions work in both cases.

---

## API Reference

### `isNativeAvailable(): boolean`

Returns `true` if the compiled Rust native addon is loaded and active.

```ts
import { isNativeAvailable } from "@akhyar11/ml-v1"

console.log("Native active:", isNativeAvailable());
```

---

## Disabling the Native Backend

Set the `ML_DISABLE_NATIVE` environment variable to force JavaScript-only execution. Useful for debugging, regression comparisons, and CI environments without a compiled native addon.

```bash
ML_DISABLE_NATIVE=1 node your-script.js
```

---

## Building the Native Addon

The native addon is compiled automatically during `npm install` if the Rust toolchain is present.

### Prerequisites

1. **Rust Toolchain** — install via [rustup.rs](https://rustup.rs/).
2. **C/C++ Build Tools** — `build-essential` on Linux, Xcode CLI on macOS, MSVC on Windows.

### Manual Build

```bash
# Release build (optimized)
npm run build:rust

# Debug build
npm run build:rust:debug
```

If Rust is not installed, `npm install` prints a warning and the library continues with the JavaScript fallback. Performance will be noticeably slower for large models.

---

## Output Buffer Safety

Native `add` and `sub` kernels assume the output buffer does not alias either input buffer. Always ensure `out._data` does not share storage with `a._data` or `b._data` when calling `mj.add(a, b, out)` or `mj.sub(a, b, out)`.

---

## Notes

- Adaptive dispatch thresholds determine whether each operation is routed to Rust or JavaScript based on workload size. Small matrices may still use the JavaScript path even when the native backend is available.
- The native addon is platform-specific. The `.node` binary must match the current OS and Node.js version. Run `npm run build:rust` after upgrading Node.js.
- Start debugging unexpected behavior with `ML_DISABLE_NATIVE=1` to isolate whether an issue is in the native or JavaScript path.
