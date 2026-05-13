# @oxide-js/layers

## 0.2.0

### Minor Changes

- 7f5e0a4: feat: add external attention inputs, memory bank access hooks, and autodiff tape fixes

  This release adds external query/key/value injection for multi-head attention,
  external read/write access hooks for memory bank operations, and autodiff fixes
  for scalar math recording, tape snapshot restore, multi-output gradients, and
  loss/result access through `engine.grad()`.

### Patch Changes

- Updated dependencies [7f5e0a4]
  - @oxide-js/core@0.2.0

## 0.1.1

### Patch Changes

- 143482d: fix: point package exports to compiled JS files to avoid type-stripping errors in Node.js v25+
- Updated dependencies [143482d]
  - @oxide-js/core@0.1.1

## 0.1.0

### Minor Changes

- ef8dd3d: feat: add Keras-style model serialization and Gradient Tape auto-diff support

### Patch Changes

- Updated dependencies [ef8dd3d]
  - @oxide-js/core@0.1.0

## 1.0.0

### Major Changes

- aac3b3a: Initial release of Oxide-JS modular monorepo. Transitioned from ML-V1 to Oxide-JS with modular architecture and Rust native acceleration.

### Patch Changes

- Updated dependencies [aac3b3a]
  - @oxide-js/core@1.0.0
