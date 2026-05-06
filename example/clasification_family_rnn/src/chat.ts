import readline from "readline";
import { mj } from "@oxide-js/core";
import { Sequential } from "@oxide-js/models";;
import { BPETokenizer } from "@oxide-js/core";;
import path from "path";
import { fileURLToPath } from "url";
import { dirname } from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Load model & tokenizer
const model = new Sequential();
model.load(path.join(__dirname, "../model.json"));
const tokenizer = BPETokenizer.load(
    path.join(__dirname, "../tokenizer.json")
);

function argmax(arr: number[] | Float32Array) {
    let maxIndex = 0;
    let maxValue = arr[0]!;

    for (let i = 1; i < arr.length; i++) {
        if (arr[i]! > maxValue) {
            maxValue = arr[i]!;
            maxIndex = i;
        }
    }

    return maxIndex;
}

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

function predict(text: string) {
    let tokens = tokenizer.encode(text);

    if (tokens.length === 0) {
        console.log("Teks kosong / tidak dikenali");
        return;
    }

    // IMPORTANT: sama seperti training (TANPA padding panjang)
    const x = mj.matrix(tokens.map((id: number) => [id]));

    const logits = model.predict(x);
    const pred = argmax(logits._data);

    const label = pred === 1 ? "positive" : "negative";

    console.log(`\nTeks: ${text}`);
    console.log("Prediksi:", label);
    console.log("Confidence:", logits._data);
}

function ask() {
    rl.question("\nMasukkan teks: ", (input) => {
        if (input === "exit") {
            rl.close();
            return;
        }

        predict(input);
        ask();
    });
}

ask();