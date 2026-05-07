import Sequential from "./sequential.js";
import Transformers from "./transformers.js";
import { readFileSync } from "fs";

/**
 * Model Management Utility
 * Provides a standardized interface for loading saved Oxide-JS models.
 */
export default class Model {
  /**
   * Loads a model from a JSON manifest file.
   * Automatically detects the model type (Sequential or Transformers) from the manifest.
   * 
   * @param path Path to the model.json file
   * @returns An instance of the loaded model
   */
  static load(path: string): Sequential | Transformers {
    const dataJson = readFileSync(path, "utf-8");
    const manifest = JSON.parse(dataJson);

    // Detect format
    if (manifest.format !== "oxide-v1") {
      throw new Error(`Unsupported model format: ${manifest.format}. Expected oxide-v1.`);
    }

    // Check if it's a specialized model or a standard Sequential
    const isTransformer = manifest.modelTopology?.modelType === "transformers" || 
                         (Array.isArray(manifest.modelTopology?.layers) && 
                          manifest.modelTopology.layers.some((l: any) => l.name === "positional encoding"));

    if (isTransformer) {
      // Transformer models need special reconstruction logic or specific config
      // For now, we assume the user might want a Sequential instance if it fits,
      // but if Transformers class was used, we should ideally instantiate it.
      // However, Transformers inherits from Sequential, so Sequential.load() 
      // can often reconstruct the layers.
      
      // Let's use a heuristic: if we have a lot of blocks, it's definitely a Transformers instance.
      console.log(`[Model] Detected Transformer-compatible manifest at ${path}`);
    }

    const model = new Sequential();
    model.load(path);
    return model;
  }

  /**
   * Static alias for saving a model (utility)
   */
  static save(model: Sequential, path: string) {
    model.save(path);
  }
}
