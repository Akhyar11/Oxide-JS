
export class Profiler {
    private static markers: Map<string, number> = new Map();
    private static stats: Map<string, { total: number, count: number }> = new Map();

    static start(label: string) {
        this.markers.set(label, performance.now());
    }

    static end(label: string) {
        const start = this.markers.get(label);
        if (start === undefined) return;
        const duration = performance.now() - start;
        
        let s = this.stats.get(label) || { total: 0, count: 0 };
        s.total += duration;
        s.count += 1;
        this.stats.set(label, s);
    }

    static report() {
        console.log("\n=== PERFORMANCE PROFILE ===");
        const sorted = Array.from(this.stats.entries()).sort((a, b) => b[1].total - a[1].total);
        console.table(sorted.map(([label, stat]) => ({
            "Operation": label,
            "Total (ms)": stat.total.toFixed(2),
            "Avg (ms)": (stat.total / stat.count).toFixed(3),
            "Count": stat.count
        })));
        this.stats.clear();
    }
}
