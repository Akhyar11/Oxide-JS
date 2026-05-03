# MemoryBank BPE Dataset

Dataset ini adalah versi **raw text** untuk test `MemoryBank` dengan `BPETokenizer` di repo `Akhyar11/ML-V1`.

Tidak ada token id input yang digenerate di dataset ini. Semua input berupa teks seperti:

```text
store key_03 value_11
update key_03 value_07
query key_03
note key_20 value_02
```

Token id harus dibuat oleh tokenizer repo:

```ts
const tokenizer = new BPETokenizer({ vocabSize: 256, minFrequency: 2, preTokenizer: "char" });
tokenizer.train(corpus);
const ids = tokenizer.encode(turn.text);
```

## Files

- `train.jsonl`: 12,000 episodes
- `val.jsonl`: 2,400 episodes
- `test.jsonl`: 2,400 episodes
- `smoke.jsonl`: 64 episodes
- `bpe_corpus.txt`: corpus untuk melatih BPE tokenizer
- `metadata.json`: daftar key/value/operator
- `bpe_memory_dataset_loader.ts`: helper TypeScript untuk repo kamu

## Task

Dalam satu episode, model membaca beberapa turn:

- `store key_x value_y`
- `update key_x value_z`
- `query key_x`
- `note ...` noise

Target hanya dihitung pada turn `query`.

Untuk setiap query:

```json
{
  "turn_index": 8,
  "query_text": "query key_03",
  "key_text": "key_03",
  "target_value_text": "value_07",
  "target_class": 7
}
```

`target_class` adalah class output `0..23`, sesuai `value_00..value_23`.

## Suggested training flow

```ts
const episodes = loadBpeMemoryEpisodes("train.jsonl");
const tokenizer = trainMemoryBpeTokenizer("bpe_corpus.txt", 256);

for (const episode of episodes) {
  model.resetMemory();

  for (let t = 0; t < episode.turns.length; t++) {
    const x = encodeTurnToMatrix(tokenizer, episode.turns[t].text, maxTurnTokens);
    const pred = model.forward(x);

    const q = getQueryForTurn(episode, t);
    if (q) {
      const y = makeTargetMatrix(q.target_class);
      model.backward(y);
    }
  }
}
```

Memory di-reset di awal episode, bukan antar turn.
