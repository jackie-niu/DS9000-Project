import argparse
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
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
    # Get preprocessed splits
    X_train, X_test, y_train, y_test = preprocess_data(
        filepath=args.data,
        target=args.target
    )

    # Hyperparameter grid
    param_grid = {
        "n_estimators": [100, 200, 300],        # number of boosting rounds
        "max_depth": [3, 4, 5, 6],              # depth of each tree
        "learning_rate": [0.01, 0.05, 0.1],     # step size shrinkage
        "subsample": [0.8, 1.0],                # fraction of samples used per tree
        "colsample_bytree": [0.8, 1.0],         # fraction of features used per tree
        "gamma": [0, 0.5, 1],                   # minimum loss reduction to make a split
        "reg_lambda": [1, 5, 10],               # L2 regularization strength
    }

    # Grid search
    base_xgb = XGBClassifier(
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",  # neutral for CV
        tree_method="hist",      # faster
    )

    grid = GridSearchCV(
        estimator=base_xgb,
        param_grid=param_grid,
        scoring="accuracy",
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)
    print("\n=== GridSearchCV ===")
    print("Best CV score (accuracy):", grid.best_score_)
    print("Best params:", grid.best_params_)

    # Train final model on best params
    best_params = grid.best_params_.copy()
    final_model = XGBClassifier(
        **best_params,
        random_state=42,
        n_jobs=-1,
        eval_metric="aucpr",  # better default for fraud/imbalance
        tree_method="hist",
    )
    final_model.fit(X_train, y_train)

    # Evaluate
    y_pred = final_model.predict(X_test)
    try:
        y_prob = final_model.predict_proba(X_test)[:, 1]
    except Exception:
        y_prob = None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    roc_auc = roc_auc_score(y_test, y_prob) if (y_prob is not None and y_test.nunique() == 2) else None
    ap = average_precision_score(y_test, y_prob) if (y_prob is not None and y_test.nunique() == 2) else None

    cm = confusion_matrix(y_test, y_pred)

    print("\n=== XGBoost (final) ===")
    print(f"Accuracy: {acc:.4f}")
    if roc_auc is not None:
        print(f"ROC-AUC : {roc_auc:.4f}")
    if ap is not None:
        print(f"Avg Precision (PR-AUC): {ap:.4f}")
    print("Confusion matrix:\n", cm)
    print("\nClassification report:\n", classification_report(y_test, y_pred))


    # Save model
    model_path = save_model(final_model, "models/xgboost_best.joblib")
    print(f"Saved model -> {model_path}")

    # Append metrics
    metrics_record = {
        "timestamp": datetime.utcnow().isoformat(),
        "model": "xgboost",
        "best_params": best_params,
        "cv_best_accuracy": float(grid.best_score_),
        "test_metrics": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "roc_auc": float(roc_auc) if roc_auc is not None else None,
            "average_precision": float(ap) if ap is not None else None,
            "confusion_matrix": cm.tolist(),
        },
    }
    metrics_path = append_metrics_jsonl(metrics_record, "models/metrics.jsonl")
    print(f"Appended metrics to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to your dataset (Excel/CSV handled in preprocess)")
    parser.add_argument("--target", required=True, help="Target column name (e.g., fraud_reported)")
    args = parser.parse_args()
    main(args)