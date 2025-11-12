import argparse
from datetime import datetime
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
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
from src.preprocess_catboost import preprocess_data 
from src.utils import save_model, append_metrics_jsonl
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from catboost import CatBoostClassifier, Pool
from pathlib import Path



def main(args):
    # Load preprocessed data
    X_train, X_test, y_train, y_test = preprocess_data(
        filepath=args.data,
        target=args.target
    )

    # CatBoost Training:
    base = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="AUC",
    verbose=False,
    random_seed=42
    )

    cat_features = [
        i
        for i, c in enumerate(X_train.columns)
        if (X_train[c].dtype == "object" or str(X_train[c].dtype) == "category")
    ]

    param_grid = {
        "depth": [4, 6, 8],
        "learning_rate": [0.03, 0.06, 0.1],
        "l2_leaf_reg": [1, 3, 7],
        "iterations": [300, 600],
        "bagging_temperature": [0, 0.5, 1.0],
        "random_strength": [1, 2]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=base,
        param_grid=param_grid, 
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train, cat_features=cat_features)
    print("Best parameters:", grid.best_params_)

    # Predict
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    # Mapping YES/No to 1/0 for evalutation purpose:
    mapping = {"Y": 1, "N": 0, "y": 1, "n": 0}
    y_test = pd.Series(y_test).map(mapping).astype(int)
    y_pred = pd.Series(y_pred).map(mapping).astype(int)
    print("Accuracy:", accuracy_score(y_test, y_pred))

    acc_cb = accuracy_score(y_test, y_pred)
    auc_cb = roc_auc_score(y_test, y_pred)
    precision_cb = precision_score(y_test, y_pred, zero_division=0)
    recall_cb = recall_score(y_test, y_pred, zero_division=0)
    f1_score_cb = f1_score(y_test, y_pred, zero_division=0)
    average_precision_score_cb = average_precision_score(y_test, y_pred)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    print(f'Accuracy for CatBoost: {acc_cb:.2%}')
    print(f'AUC for CatBoost: {auc_cb:.2f}')
    print(f'Precision for CatBoost: {precision_cb:.2%}')
    print(f'Recall for CatBoost: {recall_cb:.2%}')
    print(f'F1 Score for CatBoost: {f1_score_cb:.2%}')
    print(f'Average Precision Score for CatBoost: {average_precision_score_cb:.2f}')

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob) if (y_prob is not None and y_test.nunique() == 2) else None
    ap = average_precision_score(y_test, y_prob) if (y_prob is not None and y_test.nunique() == 2) else None
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== CatBoost (final) ===")
    print(f"Accuracy: {acc:.4f}")
    if roc_auc is not None:
        print(f"ROC-AUC : {roc_auc:.4f}")
    if ap is not None:
        print(f"Avg Precision (PR-AUC): {ap:.4f}")
    print("Confusion matrix:\n", cm)
    print("\nClassification report:\n", classification_report(y_test, y_pred))

    # Save model
    model_path = save_model(best_model, "models/CatBoost_best.joblib")
    print(f"Saved model -> {model_path}")

    metrics_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "model": "CatBoost",
            "best_params": grid.best_params_,
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