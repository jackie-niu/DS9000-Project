# src/models/knn.py
import argparse
from datetime import datetime
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

from src.preprocess import preprocess_data
from src.utils import save_model, append_metrics_jsonl


def main(args):
    # Load preprocessed data
    X_train, X_test, y_train, y_test = preprocess_data(
        filepath=args.data,
        target=args.target
    )

    # Find best K
    k_range = range(1, 15)
    scores = []

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        scores.append(accuracy_score(y_test, y_pred))

    best_k = k_range[scores.index(max(scores))]
    print(f"Best K is {best_k} with accuracy of {max(scores):.4f}")

    # Train final model with best K
    final_knn = KNeighborsClassifier(n_neighbors=best_k)
    final_knn.fit(X_train, y_train)
    y_test_pred = final_knn.predict(X_test)

    # Evaluate
    acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred, zero_division=0)
    rec = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_test_pred) if y_test.nunique() == 2 else None
    avg_prec = average_precision_score(y_test, y_test_pred) if y_test.nunique() == 2 else None
    cm = confusion_matrix(y_test, y_test_pred)

    print("\n=== KNN (final) ===")
    print(f"Accuracy: {acc:.4f}")
    if roc_auc is not None:
        print(f"ROC-AUC : {roc_auc:.4f}")
    if avg_prec is not None:
        print(f"Average Precision (PR-AUC): {avg_prec:.4f}")
    print("Confusion matrix:\n", cm)
    print("\nClassification report:\n", classification_report(y_test, y_test_pred))

    # Save model
    model_path = save_model(final_knn, "models/knn_best.joblib")
    print(f"Saved model -> {model_path}")

    # Append metrics
    metrics_record = {
        "timestamp": datetime.utcnow().isoformat(),
        "model": "knn",
        "best_k": best_k,
        "test_metrics": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "roc_auc": float(roc_auc) if roc_auc is not None else None,
            "average_precision": float(avg_prec) if avg_prec is not None else None,
            "confusion_matrix": cm.tolist(),
        },
    }
    metrics_path = append_metrics_jsonl(metrics_record, "models/metrics.jsonl")
    print(f"Appended metrics to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to dataset (Excel/CSV)")
    parser.add_argument("--target", required=True, help="Target column name (e.g., fraud_reported)")
    args = parser.parse_args()
    main(args)
