# Native Backend (Rust)

## Cara kerja
- `src/math/rust_backend.ts` mencoba `require("../../index.js")`.
- `index.js` adalah loader N-API auto-generated yang memilih binary sesuai OS/arch.
- Jika load gagal, path JS fallback dipakai.

## Native vs fallback
- Native aktif jika module berhasil dimuat dan tidak di-force disable.
- Fallback aktif jika native unavailable atau `ML_DISABLE_NATIVE=1`.

## Fungsi penting yang dipercepat
- dot product
- element-wise add/sub/mul/div
- softmax + backward
- layer norm + backward
- relu/sigmoid/tanh
- embedding forward/backward
- MHA forward/backward
- adam update
- addBias, sumAxis, clipGradients

## Build
```bash
npm run build:rust
```

## Verifikasi
```ts
import { isNativeAvailable } from "../src/math/rust_backend";
console.log("native:", isNativeAvailable());
```

## Troubleshooting
- Build gagal: pastikan rust toolchain + linker tersedia.
- Runtime gagal load binary: cocokkan platform/arch binary.
- Worker thread crash native: pipeline worker sudah memaksa disable native (`setForceDisableNative(true)`).

## Performance notes
- Native memberi dampak paling jelas pada operasi matriks besar dan MHA.
- Untuk debugging numerik, bandingkan hasil native vs fallback dengan test hotpath yang ada.
