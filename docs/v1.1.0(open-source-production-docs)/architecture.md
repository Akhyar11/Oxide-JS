# Architecture

## High-level flow
`Matrix -> Math Ops -> Layer -> Model -> Training/Inference`

## Komponen utama
1. **Matrix (`src/matrix`)**
   - Penyimpanan data flat (`Float32Array`) + shape `[rows, cols]`.
2. **Math (`src/math`)**
   - Primitive numerik (`dotProduct`, `add`, `sumAxis`, `reshape`, dst).
   - Memilih native path saat tersedia.
3. **Layer (`src/layers`)**
   - Transformasi fitur (dense, attention, embedding, dll).
4. **Model (`src/models`)**
   - Orkestrasi forward/backward antar layer.
5. **Tokenizer (`src/tokenizer`)**
   - Persiapan teks untuk task NLP.
6. **Native backend (`src-rust`)**
   - Akselerasi fungsi kritikal.

## Relasi matrix -> math -> layer -> model -> training
- Layer menerima `Matrix` dan memanggil operasi `mj.*`.
- Model memanggil `layer.forward()` berurutan, lalu `layer.backward()` terbalik.
- Optimizer mengubah bobot layer selama backward pass.

## Peran backend Rust
- Fungsi native dipanggil lewat `src/math/rust_backend.ts`.
- Jika native gagal dimuat, fallback JS tetap berjalan.

## Data flow (Transformer)
1. Input token index `[seqLen, batch]`
2. `Embedding` -> `[units, seqLen*batch]`
3. `PositionalEncoding`
4. `LayerNorm + MHA + residual`
5. `LayerNorm + FFN + residual`
6. Ambil last token state per sample
7. `Dense output` -> logits `[vocabSize, batch]`

## Shape convention penting
- Matrix disimpan row-major (`index = i*cols + j`).
- Dense menerima `[units, seqLen]` dan mengeluarkan `[outputUnits, seqLen]`.
- Target sparse klasifikasi token biasanya `[1, batch]`.
