import argparse
from datetime import datetime
import joblib
import numpy as np
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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

def main(args):
    # Load preprocessed data
    X_train, X_test, y_train, y_test = preprocess_data(
        filepath=args.data,
        target=args.target
    )

    params = {
        # Model capacity (depth & width)
        "mlp__hidden_layer_sizes": [
            (64,),
            (128,),
            (128, 64),
            (256, 128),
            (64, 32, 16),
        ],

        # Regularization
        "mlp__alpha": [
            1e-5, 1e-4, 1e-3, 1e-2
        ],

        # Learning rate
        "mlp__learning_rate_init": [
            1e-5, 1e-4, 3e-4, 1e-3
        ],

        # Activation
        "mlp__activation": [
            "relu",
            "tanh"
        ],

        # Optimizer
        "mlp__solver": [
            "adam",
            "lbfgs"
        ],

        # Stability parameters
        "mlp__learning_rate": [
            "adaptive"
        ]
    }

    clf = Pipeline([
        ("preprocess", "passthrough"),
        ("mlp", MLPClassifier(
            early_stopping=True,
            max_iter=500,
            random_state=9000
        ))
    ])

    grid = GridSearchCV(
        estimator=clf,
        param_grid=params,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=2
    )

    grid.fit(X_train, y_train)
    print("Best parameters:", grid.best_params_)

    # Predict
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

    acc_rf = accuracy_score(y_test, y_pred)
    auc_rf = roc_auc_score(y_test, y_pred)
    precision_rf = precision_score(y_test, y_pred, zero_division=0)
    recall_rf = recall_score(y_test, y_pred, zero_division=0)
    f1_score_rf = f1_score(y_test, y_pred, zero_division=0)
    average_precision_score_rf = average_precision_score(y_test, y_pred)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    print(f'Accuracy for Neural Network: {acc_rf:.2%}')
    print(f'AUC for Neural Network: {auc_rf:.2f}')
    print(f'Precision for Neural Network: {precision_rf:.2%}')
    print(f'Recall for Neural Network: {recall_rf:.2%}')
    print(f'F1 Score for Neural Network: {f1_score_rf:.2%}')
    print(f'Average Precision Score for Neural Network: {average_precision_score_rf:.2f}')

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob) if (y_prob is not None and y_test.nunique() == 2) else None
    ap = average_precision_score(y_test, y_prob) if (y_prob is not None and y_test.nunique() == 2) else None
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Neural Network (final) ===")
    print(f"Accuracy: {acc:.4f}")
    if roc_auc is not None:
        print(f"ROC-AUC : {roc_auc:.4f}")
    if ap is not None:
        print(f"Avg Precision (PR-AUC): {ap:.4f}")
    print("Confusion matrix:\n", cm)
    print("\nClassification report:\n", classification_report(y_test, y_pred))

    # Save model
    model_path = save_model(best_model, "models/Neural_Network_best.joblib")
    print(f"Saved model -> {model_path}")

    metrics_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "model": "Neural Network",
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

