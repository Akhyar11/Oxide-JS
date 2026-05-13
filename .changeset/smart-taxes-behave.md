---
"@oxide-js/core": minor
"@oxide-js/layers": minor
---

feat: add external attention inputs, memory bank access hooks, and autodiff tape fixes

This release adds external query/key/value injection for multi-head attention,
external read/write access hooks for memory bank operations, and autodiff fixes
for scalar math recording, tape snapshot restore, multi-output gradients, and
loss/result access through `engine.grad()`.
