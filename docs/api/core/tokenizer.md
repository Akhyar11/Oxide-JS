# 🔠 BPE Tokenizer API Reference

The **[BPETokenizer](file:///home/akhyar/Dokumen/Code/NODE_JS/Oxide-JS/packages/core/src/tokenizer/index.ts)** is a fast, flexible, and robust Byte-Pair Encoding (BPE) subword tokenizer implemented in TypeScript. It supports BPE vocabulary building, incremental updates, sequence padding, custom grapheme/word segmentation pipelines, and saving/loading configurations.

---

## 🏗️ Initialization & Config

### `new BPETokenizer(options?: BPETokenizerOptions)`
Creates a new BPE Tokenizer instance.

- **`BPETokenizerOptions`**:
  - `vocabSize?: number` - The maximum vocabulary limit. Default is standard `1000`.
  - `minFrequency?: number` - Minimum occurrences of a character pair required to trigger subword merges. Default is `2`.
  - `preTokenizer?: BuiltInPreTokenizer | PreTokenizer` - Pre-tokenization behavior. Can be a string identifier or a custom callback function: `(text: string) => string[]`.

- **Example (Standard Init)**:
  ```ts
  import { BPETokenizer } from "@oxide-js/core";
  const tokenizer = new BPETokenizer({ vocabSize: 5000, preTokenizer: "whitespace" });
  ```

- **Example (Custom Callback Init)**:
  ```ts
  import { BPETokenizer } from "@oxide-js/core";
  
  // Custom pre-tokenizer that splits text by dash boundaries
  const myCustomPre = (text: string) => text.split("-");
  const tokenizer = new BPETokenizer({ vocabSize: 1000, preTokenizer: myCustomPre });
  ```

---

## 🗺️ Built-in Pre-Tokenizers

Before BPE performs subword merges, text is split into basic token parts. You can import and use these pre-tokenizers directly:

### 1. `charPreTokenizer`
Splits text into single characters. Best for character-level models.
- **Example**:
  ```ts
  import { charPreTokenizer } from "@oxide-js/core";
  console.log(charPreTokenizer("Hi!")); // ["H", "i", "!"]
  ```

### 2. `whitespacePreTokenizer`
Splits text strictly by space boundaries.
- **Example**:
  ```ts
  import { whitespacePreTokenizer } from "@oxide-js/core";
  console.log(whitespacePreTokenizer("Hello World")); // ["Hello", "World"]
  ```

### 3. `unicodeGraphemePreTokenizer`
Splits text respecting full Unicode grapheme clusters (emojis, diacritics).
- **Example**:
  ```ts
  import { unicodeGraphemePreTokenizer } from "@oxide-js/core";
  console.log(unicodeGraphemePreTokenizer("A🏽")); // ["A", "🏽"]
  ```

### 4. `unicodeWordPreTokenizer`
Splits text using standard linguistic word boundary characters.
- **Example**:
  ```ts
  import { unicodeWordPreTokenizer } from "@oxide-js/core";
  console.log(unicodeWordPreTokenizer("Hello, world!")); // ["Hello", ",", "world", "!"]
  ```

### 5. `scriptAwarePreTokenizer`
Advanced multilingual mode that separates distinct language script regions (Hiragana/Katakana, Cyrillic, Latin, Arabic, etc.) into unique lexical segments automatically.
- **Example**:
  ```ts
  import { scriptAwarePreTokenizer } from "@oxide-js/core";
  console.log(scriptAwarePreTokenizer("Hello世界")); // ["Hello", "世界"]
  ```

---

## 🌀 Instance Methods

### 1. Training & Incremental Updates

#### `train(corpus: string[]): void`
Trains BPE vocabulary from scratch using a training text array.
- **Example**:
  ```ts
  const tokenizer = new BPETokenizer({ vocabSize: 1000 });
  tokenizer.train(["hello world", "welcome to oxide-js"]);
  ```

#### `update(corpus: string[]): void`
Performs **incremental updates** to the vocabulary, merging new character pairs from a fresh corpus without wiping or resetting existing trained tokens.
- **Example**:
  ```ts
  tokenizer.update(["deep learning", "neural networks"]);
  ```

---

### 2. Encoding (Text ➡️ IDs)

#### `encode(text: string): number[]`
Encodes input text into vocabulary subword IDs. Does **not** include special tokens.
- **Example**:
  ```ts
  const ids = tokenizer.encode("hello"); // [12, 45]
  ```

#### `encodeWithSpecial(text: string): number[]`
Encodes text and appends sequence indicators (e.g. `<s>` at start, `</s>` at end).
- **Example**:
  ```ts
  const ids = tokenizer.encodeWithSpecial("hello"); // [1, 12, 45, 2]
  ```

#### `padSequence(ids: number[], length: number): number[]`
Pads or truncates a list of token IDs to match a fixed target context length. Uses standard `<pad>` token ID (0) for padding.
- **Example**:
  ```ts
  const ids = tokenizer.encode("hello"); // [12, 45]
  const padded = tokenizer.padSequence(ids, 5); // [12, 45, 0, 0, 0]
  ```

---

### 3. Decoding (IDs ➡️ Text)

#### `decode(ids: number[]): string`
Converts token IDs back to a reconstructed string, resolving subword merges and stripping `<pad>` elements.
- **Example**:
  ```ts
  const text = tokenizer.decode([12, 45]);
  console.log(text); // "hello"
  ```

---

### 4. Serialization (Save/Load)

#### `save(path: string): void`
Saves BPE vocab tables, merge rules, and pre-tokenizer configurations to a local JSON file.
- **Example**:
  ```ts
  tokenizer.save("./my_tokenizer.json");
  ```

#### `load(path: string, options?: { preTokenizer?: PreTokenizer }): void`
Loads tokenizer state from a JSON file.
- **Example (Standard Load)**:
  ```ts
  const tokenizer = new BPETokenizer();
  tokenizer.load("./my_tokenizer.json");
  ```
- **Example (Load with Custom preTokenizer)**:
  ```ts
  const tokenizer = new BPETokenizer();
  tokenizer.load("./my_tokenizer.json", {
    preTokenizer: (text) => text.split("-")
  });
  ```
