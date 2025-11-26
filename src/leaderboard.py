import argparse
import matplotlib.pyplot as plt
from src.utils import metrics_leaderboard

def plot_all_metrics(records):
    models = [r.get("model", f"Model_{i}") for i, r in enumerate(records)]
    accuracy = [r.get("test_metrics", {}).get("accuracy", 0) for r in records]
    roc_auc = [r.get("test_metrics", {}).get("roc_auc", 0) for r in records]
    pr_auc = [r.get("test_metrics", {}).get("average_precision", 0) for r in records]
    f1 = [r.get("test_metrics", {}).get("f1", 0) for r in records]

    import numpy as np
    x = np.arange(len(models))  # label locations
    width = 0.2  # bar width

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.cm.inferno
    colors = cmap(np.linspace(0.2, 0.8, 4))
   
    bars_acc = ax.bar(x - 1.5*width, accuracy, width, label="Accuracy", color=colors[0])
    bars_roc = ax.bar(x - 0.5*width, roc_auc, width, label="ROC-AUC", color=colors[1])
    bars_pr  = ax.bar(x + 0.5*width, pr_auc, width, label="PR-AUC", color=colors[2])
    bars_f1  = ax.bar(x + 1.5*width, f1, width, label="F1 Score", color=colors[3])

    def add_values(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01, f'{height:.2f}', 
                    ha='center', va='bottom', fontsize=8)
    for bars in [bars_acc, bars_roc, bars_pr, bars_f1]:
        add_values(bars)

    ax.set_xlabel("Models")
    ax.set_ylabel("Metric Value")
    ax.set_title("Comparison of Metrics Across Models")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Print a model leaderboard from models/metrics.jsonl")
    parser.add_argument(
        "--path", default="models/metrics.jsonl",
        help="Path to the metrics JSONL file (default: models/metrics.jsonl)"
    )
    parser.add_argument(
        "--sort",
        default="pr",
        choices=["pr", "roc", "acc", "f1", "cv", "time"],
        help="Sort by: pr (PR-AUC), roc (ROC-AUC), acc, f1, cv (CV acc), time (timestamp). Default: pr"
    )
    parser.add_argument("--top", type=int, default=10, help="Show top N rows (default: 10)")
    parser.add_argument("--ascending", action="store_true", help="Sort ascending (default: descending)")
    args = parser.parse_args()

    # Map short keys to dotted keys used in utils.metrics_leaderboard
    sort_map = {
        "pr":   "test_metrics.average_precision",
        "roc":  "test_metrics.roc_auc",
        "acc":  "test_metrics.accuracy",
        "f1":   "test_metrics.f1",
        "cv":   "cv_best_accuracy",
        "time": "timestamp",
    }
    sort_key = sort_map[args.sort]

    records = metrics_leaderboard(
        path=args.path,
        sort_key=sort_key,
        reverse=not args.ascending,
        top=args.top
    )

    if not records:
        print(f"No records found at {args.path}. Train a model first.")
        return

    print("\n=== Leaderboard ===")
    print(f"Sorted by: {sort_key} ({'asc' if args.ascending else 'desc'})\n")
    # Header
    print(f"{'timestamp':<26} | {'model':<12} | {'ACC':>6} | {'ROC-AUC':>7} | {'PR-AUC':>7} | {'F1':>6}")
    print("-" * 80)

    for r in records:
        tm = r.get("test_metrics", {})
        ts = r.get("timestamp", "")
        model = r.get("model", "")
        acc = tm.get("accuracy", 0.0) or 0.0
        roc = tm.get("roc_auc", 0.0) or 0.0
        pr  = tm.get("average_precision", 0.0) or 0.0
        f1  = tm.get("f1", 0.0) or 0.0
        print(
            f"{ts:<26} | {model:<12} | {acc:6.3f} | {roc:7.3f} | {pr:7.3f} | {f1:6.3f}"
        )
    plot_all_metrics(records)

if __name__ == "__main__":
    main()

