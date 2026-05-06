import path from "path"
import { fileURLToPath } from "url";
import { dirname } from "path";
import { BPETokenizer } from "@oxide-js/core";;
import { getDatasetTrain } from "./get_dataset.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const dataset = getDatasetTrain()
const texts = [...dataset.map((item) => item.text as string), 'negative', 'positive', 'neutral']

const tokenizer = new BPETokenizer({ minFrequency: 2 })

tokenizer.train(texts)
tokenizer.save(path.join(__dirname, '../tokenizer.json'))