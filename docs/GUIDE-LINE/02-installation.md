# Installation and Setup Guide

Follow the steps below to set up the **Oxide-JS** development environment on your local machine.

## Main Prerequisites

Before starting, ensure you have the following software installed:
1. **Node.js**: LTS version (v20.x or newer recommended).
2. **Rust Toolchain**: Required to build the native backend (`cargo`, `rustc`). Install via [rustup.rs](https://rustup.rs/).
3. **C/C++ Build Tools**: Required by the native binding compiler (e.g. `build-essential` on Linux).

---

## Installation Steps

### 1. Clone the Repository
Clone the monorepo to your local machine:
```bash
git clone <repository-url>
cd Oxide-JS
```

### 2. Install Node.js Dependencies
Use npm to install all required packages for the entire monorepo:
```bash
npm install
```

### 3. Build Native Backend
To achieve maximum performance, you need to build the modular Rust kernels using `napi-rs`.

**For Production Build (Release):**
```bash
npm run build:rust
```

The build process will produce a `.node` binary file inside `packages/core/` (e.g., `packages/core/oxide-native.linux-x64-gnu.node`).

---

## Installation Verification

After installation is complete, you can verify if the native backend is active by running a simple test:

```ts
import { isNativeAvailable } from "@oxide-js/core";

console.log("Native Backend Status:", isNativeAvailable());
```

Alternatively, run the full test suite:
```bash
npm test
```

Notes:
- `npm test` runs the root suite `test/index.ts`.
- The suite validates both **Correctness** and **Benchmarks** across all packages.

---

## Environment Configuration

### Forcibly Disabling the Native Backend
If you encounter issues with the native binary or want to perform debugging on the pure JavaScript implementation, you can use the `ML_DISABLE_NATIVE` environment variable:

```bash
ML_DISABLE_NATIVE=1 node your-script.js
```

### TypeScript Usage
Since this project is a monorepo, you should build all packages before using them in external projects:
```bash
npm run build
```
This will compile `@oxide-js/core`, `@oxide-js/layers`, and `@oxide-js/models`.

---

> [!WARNING]
> If you change your operating system or CPU architecture (e.g., from Linux to macOS), you **must** rerun `npm run build:rust` so that the binary matches the new platform.

**Next Steps:**
Learn how to use this library in the [Practical Tutorial](03-tutorial.md).
