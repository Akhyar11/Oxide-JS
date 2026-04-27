import fs from "fs"
import path from "path"
import { fileURLToPath } from "url";
import { dirname } from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export const dataset_path1 = path.join(__dirname, '../dataset/train_preprocess.tsv');
export const dataset_path2 = path.join(__dirname, '../dataset/valid_preprocess.tsv');


function readDataset(filePath: string) {
    if (!fs.existsSync(filePath)) {
        console.error(`Dataset not found at: ${filePath}`);
        return [];
    }

    const content = fs.readFileSync(filePath, 'utf-8');
    const lines = content.split('\n').filter(line => line.trim() !== '');

    if (lines.length <= 1) return [];

    const header = lines[0]!;
    const separator = header.includes('\t') ? '\t' : ',';

    return lines.slice(1).map((line) => {
        const parts = line.split(separator);

        let text: string;
        let label: string;

        if (separator === '\t') {
            // Format TSV: text \t sentiment
            text = parts[0] || "";
            label = parts[1] || "";
        } else {
            // Legacy CSV fallback or standard 2-column CSV
            // If parts[1] exists and parts[2] exists, it might be the old id,label,text format
            if (parts.length > 2) {
                label = parts[1]!;
                text = parts.slice(2).join(',');
            } else {
                text = parts[0] || "";
                label = parts[1] || "";
            }
        }

        return {
            text: text.trim(),
            label: label.trim(),
        }
    })
}

export function getDatasetTrain() {
    return readDataset(dataset_path1);
}

export function getDatasetValid() {
    return readDataset(dataset_path2);
}