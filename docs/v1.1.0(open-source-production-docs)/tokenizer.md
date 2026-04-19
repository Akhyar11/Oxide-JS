# Tokenizer (BPE)

## Kelas
`BPETokenizer` (`src/tokenizer/bpe.ts`)

## Training awal
```ts
const tokenizer = new BPETokenizer({ vocabSize: 200, minFrequency: 2 });
tokenizer.train(["aku suka matematika", "aku suka fisika"]);
```

## Update vocabulary (incremental)
```ts
tokenizer.update(["aku belajar statistika"], 260);
```

## Encode / decode
```ts
const ids = tokenizer.encode("aku suka matematika");
const idsSpecial = tokenizer.encodeWithSpecial("aku suka matematika");
const text = tokenizer.decode(idsSpecial);
```

## Special token
Default special token internal:
- `<PAD>`
- `<UNK>`
- `<BOS>`
- `<EOS>`

Dapat akses pad id:
```ts
const padId = tokenizer.getPadId();
```

## Pad sequence
```ts
const padded = tokenizer.padSequence(idsSpecial, 32);
```

## Save / load
```ts
tokenizer.save("./dataset/bpe_vocab.json");
const loaded = BPETokenizer.load("./dataset/bpe_vocab.json");
```

## Catatan implementasi
- Ada proses `sanitize()` untuk membersihkan token/merge yang dianggap polluted.
- `update()` dapat memanfaatkan placeholder `<UNUSED_*>` atau `<RESERVED_*>`.
