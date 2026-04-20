export function shuffleInPlace<T>(arr: T[]): void {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

export function splitTrainValidation<T>(data: T[], validationSplit: number): [T[], T[]] {
  const numTrain = Math.ceil(data.length * (1 - validationSplit));
  return [data.slice(0, numTrain), data.slice(numTrain)];
}

export function formatLoss(loss: number, decimals: number = 6): string {
  return loss.toFixed(decimals);
}

export function formatProgressBar(current: number, total: number, width: number = 30): string {
  const ratio = Math.max(0, Math.min(1, current / total));
  const filled = Math.round(width * ratio);
  const empty = width - filled;
  return `[${"=".repeat(filled)}>${" ".repeat(empty)}] ${(ratio * 100).toFixed(1)}%`;
}
