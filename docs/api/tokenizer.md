# Tokenizer

BPE (Byte Pair Encoding) tokenizer and pre-tokenizer utilities for text preprocessing.

## Import

```ts
import {
  BPETokenizer,
  charPreTokenizer,
  unicodeGraphemePreTokenizer,
  unicodeWordPreTokenizer,
  whitespacePreTokenizer,
  scriptAwarePreTokenizer
} from "@oxide-js/core"

import type {
  BPETrainingEncodeOptions,
  BPETokenizerOptions,
  BuiltInPreTokenizer,
  PreTokenizer
} from "@oxide-js/core"
```

## Overview

`BPETokenizer` implements the Byte Pair Encoding algorithm, splitting rare words into subwords while keeping frequent words as single tokens. It supports Unicode-aware pre-tokenization for multilingual corpora.

---

## API Reference

### `BPETokenizer`

#### `constructor(config: BPETokenizerOptions)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `vocabSize` | `number` | — | Target vocabulary size |
| `minFrequency` | `number` | `2` | Minimum merge frequency |
| `preTokenizer` | `BuiltInPreTokenizer \| PreTokenizer` | `"char"` | Pre-tokenization strategy |
| `specialTokens` | `string[]` | — | Additional special tokens to preserve |

```ts
import { BPETokenizer } from "@oxide-js/core"

const tokenizer = new BPETokenizer({
  vocabSize: 1000,
  minFrequency: 2
});
```

---

#### `train(texts: string[]): void`

Trains the tokenizer on a corpus. Builds the initial vocabulary and applies BPE merges up to `vocabSize`.

```ts
tokenizer.train(["saya makan nasi", "kamu makan roti"]);
```

---

#### `update(texts: string[], newVocabSize?: number): void`

Incrementally continues training on a new corpus without resetting existing token IDs. Useful for gradually expanding the vocabulary.

```ts
tokenizer.update(["matematika diskrit", "logika proposisional"], 1200);
```

---

#### `encode(text: string): number[]`

Encodes text to token IDs without adding special tokens.

```ts
const ids = tokenizer.encode("saya makan");
// [12, 45, 67]
```

#### `encodeForTraining(text: string, options?: BPETrainingEncodeOptions): number[]`

Encode khusus untuk training. Method ini melakukan random branch per pemanggilan:

- jika branch aktif, tokenizer tetap mengecek apakah kata utuh ada di vocab/corpus
- jika branch nonaktif, tokenizer mengabaikan kecocokan kata-utuh dan memaksa hasil subword/kombinasi token

Tujuannya agar model kadang melihat kata yang sama dalam bentuk pecahan subword, sehingga lebih tahan terhadap kata yang kurang lengkap atau OOV parsial.

```ts
const ids = tokenizer.encodeForTraining("belajar", {
  fullWordLookupProbability: 0.5
});
```

#### `decode(ids: number[]): string`

Converts token IDs back to text.

```ts
const text = tokenizer.decode([12, 45, 67]);
// "saya makan"
```

---

#### `encodeWithSpecial(text: string): number[]`

Encodes text and wraps it with **BOS** (beginning-of-sequence) and **EOS** (end-of-sequence) tokens. Recommended for generative and Transformer models.

---

#### `padSequence(ids: number[], maxLength: number): number[]`

Pads or truncates a token ID sequence to a fixed length using the PAD token.

```ts
const padded = tokenizer.padSequence([1, 2], 5);
// [1, 2, 0, 0, 0]  (PAD_ID = 0)
```

---

#### `save(path: string): void`

Saves the tokenizer vocabulary and configuration to a JSON file.

```ts
tokenizer.save("./model/vocab.json");
```

#### `static load(path: string, options?: { preTokenizer?: PreTokenizer }): BPETokenizer`

Loads a tokenizer from a saved JSON file.

```ts
const loadedTokenizer = BPETokenizer.load("./model/vocab.json");
```

> **Serialization note:** Built-in pre-tokenizer names (`"char"`, `"unicode-grapheme"`, etc.) are saved in the JSON file. Custom pre-tokenizer functions are **not** serialized — the saved metadata records `"custom"`, and the same function must be passed again on load:
>
> ```ts
> const loaded = BPETokenizer.load("./model/vocab.json", { preTokenizer: myPreTokenizer });
> ```

---

#### Helper Methods

| Method | Description |
|---|---|
| `getVocabSize()` | Number of tokens stored in the vocabulary map |
| `getVocabularyCapacity()` | Effective ID capacity (`maxTokenId + 1`). Use this to sync `Embedding` / `Transformers.vocabSize` |
| `getTokenId(token)` | ID of a specific token |
| `getToken(id)` | Token string for a specific ID |
| `getPadId()` | PAD token ID |

> [!CAUTION]
> For `Embedding` layers or `Transformers`, use `tokenizer.getVocabularyCapacity()` — not `getVocabSize()` — to determine `vocabSize`. Token IDs may not be dense after incremental updates, and using a size that is too small causes index-out-of-bounds errors in the model.

---

### Pre-tokenizers

Pre-tokenizers split raw text into initial segments before BPE merges. Oxide-JS provides five built-in strategies:

| Name | Export | Description |
|---|---|---|
| `"char"` | `charPreTokenizer` | Split into Unicode code points (default; backward-compatible) |
| `"unicode-grapheme"` | `unicodeGraphemePreTokenizer` | Split on grapheme clusters (safer character boundaries) |
| `"unicode-word"` | `unicodeWordPreTokenizer` | Split on Unicode word boundaries |
| `"whitespace"` | `whitespacePreTokenizer` | Split on whitespace |
| `"script-aware"` | `scriptAwarePreTokenizer` | Script-aware segmentation for mixed multilingual text |

Pre-tokenizers can be used directly as functions `(text: string) => string[]`:

```ts
import { scriptAwarePreTokenizer } from "@oxide-js/core"

const tokens = scriptAwarePreTokenizer("ꦱꦺꦴꦥꦺꦴ");
// ["ꦱꦺꦴ", "ꦥꦺꦴ"]
```

#### Choosing a Pre-tokenizer

- **`"char"`** — default; works for most Latin-script tasks and is backward-compatible.
- **`"unicode-grapheme"`** — use for scripts with combining marks (e.g. Thai, Devanagari) where a "character" visually spans multiple code points.
- **`"script-aware"`** — recommended for mixed-script corpora (Latin, Arabic, Japanese, Chinese, Thai, Korean, Javanese, emoji, math symbols).
- **Custom function** — pass `(text: string) => string[]` for language-specific tokenization needs.

`Intl.Segmenter` is used when the runtime supports it; fallback behavior is deterministic but may be less linguistically accurate.

#### Multilingual Example

```ts
import { BPETokenizer } from "@oxide-js/core"

const tokenizer = new BPETokenizer({
  vocabSize: 1000,
  preTokenizer: "script-aware"
});

tokenizer.train([
  "hello world",
  "مرحبا بالعالم",
  "こんにちは世界",
  "你好世界",
  "ภาษาไทย",
  "한국어테스트",
  "ꦱꦺꦴꦥꦺꦴ",
  "x² + y² = z²",
  "hello ꦱꦺꦴꦥꦺꦴ 😊 你好"
]);
```

---

## Types

```ts
import type {
  BPETrainingEncodeOptions,
  BPETokenizerOptions,
  BuiltInPreTokenizer,
  PreTokenizer
} from "@oxide-js/core"
```

### `BPETokenizerOptions`

```ts
type BPETokenizerOptions = {
  vocabSize?: number;
  minFrequency?: number;
  preTokenizer?: BuiltInPreTokenizer | PreTokenizer;
};
```

### `BPETrainingEncodeOptions`

```ts
type BPETrainingEncodeOptions = {
  fullWordLookupProbability?: number;
  random?: () => number;
};
```

### `BuiltInPreTokenizer`

```ts
type BuiltInPreTokenizer =
  | "char"
  | "unicode-grapheme"
  | "unicode-word"
  | "whitespace"
  | "script-aware";
```

### `PreTokenizer`

```ts
type PreTokenizer = (text: string) => string[];
```

---

## Notes

- BPE alone is insufficient for many writing systems. Pre-tokenization is important for scripts without spaces, combining marks, emoji sequences, and mixed multilingual text.
- `"script-aware"` is a general-purpose built-in; for language-specific behavior, pass a custom pre-tokenizer.
- Token IDs assigned during `train()` / `update()` are stable — existing IDs are never reassigned.
