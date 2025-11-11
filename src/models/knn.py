import argparse
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from src.preprocess import preprocess_data
from src.utils import save_model, append_metrics_jsonl
import matplotlib.pyplot as plt

def main(args):
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(
        filepath=args.data,
        target=args.target
    )

    # Try multiple K values
    k_range = range(1, 15)
    scores = []

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        scores.append(metrics.accuracy_score(y_test, y_pred))

    # Plot accuracy vs K
    plt.plot(k_range, scores, marker='o')
    plt.xlabel('K')
    plt.ylabel('Test Set Accuracy')
    plt.title('KNN Accuracy vs K')
    plt.xticks(k_range)
    plt.show()

    # Pick best K
    best_k = k_range[scores.index(max(scores))]
    print(f"Best K is {best_k} with accuracy {max(scores):.4f}")

    # Train final model
    final_knn = KNeighborsClassifier(n_neighbors=best_k)
    final_knn.fit(X_train, y_train)
    y_test_pred = final_knn.predict(X_test)

    # Metrics
    acc = metrics.accuracy_score(y_test, y_test_pred)
    auc = metrics.roc_auc_score(y_test, y_test_pred)
    precision = metrics.precision_score(y_test, y_test_pred, zero_division=0)
    recall = metrics.recall_score(y_test, y_test_pred, zero_division=0)
    f1 = metrics.f1_score(y_test, y_test_pred, zero_division=0)
    avg_precision = metrics.average_precision_score(y_test, y_test_pred)

    print(f"Accuracy: {acc:.4%}")
    print(f"AUC: {auc:.4f}")
    print(f"Precision: {precision:.4%}")
    print(f"Recall: {recall:.4%}")
    print(f"F1 Score: {f1:.4%}")
    print(f"Average Precision: {avg_precision:.4f}")

    # Save model and metrics
    model_path = save_model(final_knn, "models/knn_best.joblib")
    metrics_record = {
        "timestamp": datetime.utcnow().isoformat(),
        "model": "knn",
        "best_k": best_k,
        "test_metrics": {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(auc),
            "average_precision": float(avg_precision)
        }
    }
    metrics_path = append_metrics_jsonl(metrics_record, "models/metrics.jsonl")
    print(f"Saved model to {model_path}")
    print(f"Appended metrics to {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to your dataset (Excel/CSV handled in preprocess)")
    parser.add_argument("--target", required=True, help="Target column name (e.g., fraud_reported)")
    args = parser.parse_args()
    main(args)
