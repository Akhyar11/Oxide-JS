import { BPETokenizer, mj, Matrix } from "@oxide-js/core";
import { Sequential } from "@oxide-js/models";;

function argmax(matrix: Matrix): number {
    const data = matrix._data;
    if (data.length === 0) return 0;

    let maxIndex = 0;
    let maxValue = data[0]!;

    for (let i = 1; i < data.length; i++) {
        const val = data[i]!;
        if (val > maxValue) {
            maxValue = val;
            maxIndex = i;
        }
    }

    return maxIndex;
}

export interface EvaluationResult {
    accuracy: number;
    macroF1: number;
    weightedF1: number;
    processed: number;
    skipped: number;
}

export function runEvaluation(
    model: Sequential,
    tokenizer: BPETokenizer,
    rawDataset: any[]
): EvaluationResult {
    model.eval();

    // 3 Classes: 0: negative, 1: positive, 2: neutral
    const classNames = ["negative", "positive", "neutral"];
    const confusion: [number[], number[], number[]] = [
        [0, 0, 0], // actual 0 (negative)
        [0, 0, 0], // actual 1 (positive)
        [0, 0, 0]  // actual 2 (neutral)
    ];

    let processed = 0;
    let skipped = 0;

    for (let i = 0; i < rawDataset.length; i++) {
        const item = rawDataset[i];
        const text = String(item?.text ?? "").trim();
        const labelStr = String(item?.label ?? "").trim();

        if (!text || (labelStr !== "positive" && labelStr !== "negative" && labelStr !== "neutral")) {
            skipped++;
            continue;
        }

        const tokenIds = tokenizer.encode(text);
        if (tokenIds.length === 0) {
            skipped++;
            continue;
        }

        const x = mj.matrix(tokenIds.map((id: number) => [id]));
        const logits = model.predict(x);
        const predIndex = argmax(logits);

        let actualIndex: 0 | 1 | 2;
        if (labelStr === "positive") actualIndex = 1;
        else if (labelStr === "negative") actualIndex = 0;
        else actualIndex = 2; // neutral

        const row = confusion[actualIndex];
        if (row && row[predIndex] !== undefined) {
            row[predIndex]++;
        }

        processed++;
        if (processed % 500 === 0) {
            process.stdout.write(`Processed ${processed}/${rawDataset.length} samples...\n`);
        }
    }

    console.log("\n--- Evaluation Results ---");
    console.log(`Processed: ${processed}, Skipped: ${skipped}`);

    console.log("\nConfusion Matrix (Actual \\ Predicted):");
    console.log("          Neg   Pos   Neu");
    for (let i = 0; i < 3; i++) {
        const row = confusion[i]!;
        console.log(`${classNames[i]!.padEnd(8)}: ${String(row[0]).padStart(5)} ${String(row[1]).padStart(5)} ${String(row[2]).padStart(5)}`);
    }

    let totalF1 = 0;
    let totalPrecision = 0;
    let totalRecall = 0;
    let weightedF1 = 0;

    console.log("\nPer-class Metrics:");
    for (let i = 0; i < 3; i++) {
        const tp = confusion[i]![i]!;
        const fp = confusion[0]![i]! + confusion[1]![i]! + confusion[2]![i]! - tp;
        const fn = confusion[i]![0]! + confusion[i]![1]! + confusion[i]![2]! - tp;

        const precision = tp / (tp + fp) || 0;
        const recall = tp / (tp + fn) || 0;
        const f1 = (2 * precision * recall) / (precision + recall) || 0;

        totalPrecision += precision;
        totalRecall += recall;
        totalF1 += f1;

        const classCount = confusion[i]![0]! + confusion[i]![1]! + confusion[i]![2]!;
        weightedF1 += f1 * (classCount / processed);

        console.log(`${classNames[i]!.padEnd(8)} -> Precision: ${(precision * 100).toFixed(2)}%, Recall: ${(recall * 100).toFixed(2)}%, F1: ${(f1 * 100).toFixed(2)}%`);
    }

    const macroPrecision = totalPrecision / 3;
    const macroRecall = totalRecall / 3;
    const macroF1 = totalF1 / 3;

    let totalCorrect = confusion[0]![0]! + confusion[1]![1]! + confusion[2]![2]!;
    const accuracy = totalCorrect / processed;

    console.log("--------------------------");
    console.log(`Accuracy:         ${(accuracy * 100).toFixed(2)}%`);
    console.log(`Macro F1 Score:   ${(macroF1 * 100).toFixed(2)}%`);
    console.log(`Weighted F1 Score:${(weightedF1 * 100).toFixed(2)}%`);
    console.log("--------------------------");

    return {
        accuracy,
        macroF1,
        weightedF1,
        processed,
        skipped
    };
}
