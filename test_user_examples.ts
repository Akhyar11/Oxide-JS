import BPETokenizer from "./src/tokenizer/bpe";
import * as fs from "fs";

async function test() {
    console.log("=== Verifying Sanitization of User Examples ===");
    
    const vocabPath = "./tmp_user_dirty_vocab.json";
    const WORD_BOUNDARY = "▁";

    // "Dirty" tokens from user's list
    const dirtyTokens = [
        "▁(2+4+",
        "▁σ(x−x̄)(y−ȳ)",
        "▁`append(",
        "▁`append()`",
        "▁`extend(",
        "▁`extend()`"
    ];
    
    const dirtyData = {
        config: { vocabSize: 5000, specialTokens: ["<PAD>", "<UNK>", "<BOS>", "<EOS>"] },
        merges: dirtyTokens.map(t => [WORD_BOUNDARY, t.substring(1)]) as [string, string][],
        vocab: {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
            [WORD_BOUNDARY]: 4,
            "makan": 5
        } as Record<string, number>
    };
    
    // Add dirty tokens to vocab
    dirtyTokens.forEach((t, i) => {
        dirtyData.vocab[t] = 1000 + i;
    });

    fs.writeFileSync(vocabPath, JSON.stringify(dirtyData, null, 2));

    console.log(`\nLoading vocabulary with ${dirtyTokens.length} dirty tokens...`);
    const tokenizer = BPETokenizer.load(vocabPath);
    
    console.log("\nChecking tokens after sanitization:");
    for (const t of dirtyTokens) {
        const id = tokenizer.getTokenId(t);
        console.log(`  "${t}": ${id !== undefined ? "STILL EXISTS (FAIL)" : "REMOVED (SUCCESS)"}`);
    }
    
    const idMakan = tokenizer.getTokenId("makan");
    console.log(`  "makan": ${idMakan !== undefined ? "EXISTS (SUCCESS)" : "REMOVED (FAIL)"}`);

    fs.unlinkSync(vocabPath);
}

test().catch(console.error);
