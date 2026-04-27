
export type WorkspacePolicy = "grow" | "bounded" | "exact";

export interface WorkspaceConfig {
  policy?: WorkspacePolicy;
  maxCapacityMultiplier?: number;
  shrinkThreshold?: number;
}

export class MemoryManager {
  private static defaultPolicy: WorkspacePolicy = "bounded";
  private static defaultShrinkThreshold = 4;

  static ensureCapacity(
    buffer: Float32Array,
    required: number,
    config: WorkspaceConfig = {}
  ): Float32Array {
    const policy = config.policy || this.defaultPolicy;
    const shrinkThreshold = config.shrinkThreshold || this.defaultShrinkThreshold;

    if (policy === "exact") {
      if (buffer.length === required) return buffer;
      return new Float32Array(required);
    }

    if (buffer.length < required) {
      // Always grow if too small
      const nextCapacity = Math.max(required, buffer.length * 2 || 1024);
      return new Float32Array(nextCapacity);
    }

    if (policy === "bounded" && buffer.length > required * shrinkThreshold) {
      // Shrink if way too big
      return new Float32Array(required);
    }

    return buffer;
  }
}
