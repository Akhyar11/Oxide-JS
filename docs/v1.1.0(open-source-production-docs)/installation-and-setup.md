# Installation and Setup

## Environment requirements
- Node.js 18+
- npm
- Rust toolchain (`cargo`, `rustc`) untuk backend native
- Platform yang didukung oleh artefak napi-rs

## Install dependency
```bash
npm install
```

## Build TypeScript (manual)
```bash
npx tsc
```

## Build Rust backend
```bash
npm run build:rust
```
Debug build:
```bash
npm run build:rust:debug
```

## Verifikasi native aktif
```ts
import { isNativeAvailable } from "../src/math/rust_backend";
console.log(isNativeAvailable());
```

## Menonaktifkan native (debug/fallback test)
```bash
ML_DISABLE_NATIVE=1 npx ts-node test/test.ts
```

## Setup troubleshooting
- Jika build rust gagal: cek toolchain rust dan compiler C/C++ di OS Anda.
- Jika `Native backend not available`: pastikan file `.node` hasil build tersedia dan cocok platform.
- Jika test TypeScript gagal karena dependency: pastikan `npm install` sukses.

## Catatan audit saat dokumen ini dibuat
- `npm test` jalan tetapi ada 1 assert presisi floating.
- `npx tsc --noEmit` gagal pada import test ke `project/math-bot/main` (folder tidak ada di snapshot ini).
