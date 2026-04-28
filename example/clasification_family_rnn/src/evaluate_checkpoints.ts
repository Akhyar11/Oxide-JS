import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { dirname } from "path";
import { BPETokenizer, Sequential, mj, Matrix } from "@akhyar11/ml-v1";
import { getDatasetValid } from "./get_dataset.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

function argmax(matrix: Matrix): number {
    const data = matrix._data;
    if (data.length === 0) return 0;
    let maxIndex = 0;
    let maxValue = data[0]!;
    for (let i = 1; i < data.length; i++) {
        if (data[i]! > maxValue) {
            maxValue = data[i]!;
            maxIndex = i;
        }
    }
    return maxIndex;
}

async function run() {
    console.log("Loading tokenizer...");
    const tokenizer = BPETokenizer.load(path.join(__dirname, "../tokenizer.json"));
    
    console.log("Loading dataset...");
    const rawDatasetValid = getDatasetValid();
    
    const outputJSON: any = {
      sourceExperiment: "stability_experiment_20260427_230251.json",
      evaluationDataset: "dataset/valid_preprocess.tsv",
      models: {
        RNN: {},
        LSTM: {},
        GRU: {}
      }
    };
    
    const logDir = path.join(__dirname, "../log");
    const files = fs.readdirSync(logDir);
    
    const modelTypes = ["RNN", "LSTM", "GRU"];
    const suffix = "_20260427_230251.json";
    
    for (const modelType of modelTypes) {
        const prefix = `model_${modelType.toLowerCase()}_run`;
        
        for (const file of files) {
            if (file.startsWith(prefix) && file.endsWith(suffix)) {
                const runStr = file.replace(prefix, "").replace(suffix, "");
                const runKey = `run_${runStr}`;
                
                console.log(`Evaluating ${modelType} ${runKey}...`);
                const modelPath = path.join(logDir, file);
                
                const model = new Sequential({ layers: [] });
                // @ts-ignore
                model.load(modelPath);
                model.eval();
                
                let processed = 0;
                let skipped = 0;
                const confusion: [number[], number[], number[]] = [
                    [0, 0, 0], 
                    [0, 0, 0], 
                    [0, 0, 0]  
                ];
                
                for (let i = 0; i < rawDatasetValid.length; i++) {
                    const item = rawDatasetValid[i];
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
            
                    confusion[actualIndex]![predIndex]!++;
                    processed++;
                }
                
                let totalF1 = 0;
                let totalPrecision = 0;
                let totalRecall = 0;
                let weightedF1 = 0;
                
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
                }
                
                const macroF1 = totalF1 / 3;
                let totalCorrect = confusion[0]![0]! + confusion[1]![1]! + confusion[2]![2]!;
                const accuracy = totalCorrect / processed;
                
                outputJSON.models[modelType][runKey] = {
                    accuracy,
                    macroF1,
                    weightedF1,
                    confusionMatrix: confusion
                };
                
                // Clear model
                // @ts-ignore
                if (model.dispose) model.dispose();
            }
        }
    }
    
    const outPath = path.join(__dirname, "../log/checkpoint_evaluation_20260428.json");
    fs.writeFileSync(outPath, JSON.stringify(outputJSON, null, 2));
    console.log(`Evaluation done. Result saved to ${outPath}`);
}

run().catch(console.error);
