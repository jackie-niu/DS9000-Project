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
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression


def main(args):
    # Load preprocessed data
    X_train, X_test, y_train, y_test = preprocess_data(
        filepath=args.data,
        target=args.target
    )
    
    num_cols = X_train.select_dtypes(include=['number','float','int','bool']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object','category']).columns.tolist()

    for c in ['insured_zip', 'policy_state', 'policy_csl']:
        if c in X_train.columns and c in num_cols:
            num_cols.remove(c)
            cat_cols.append(c)
    
    numeric_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),   
        ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
    ])

    pre = ColumnTransformer([
        ('num', numeric_pipe, num_cols),
        ('cat', categorical_pipe, cat_cols)
    ])
    
    clf = Pipeline([
        ('prepprocessor', pre),
        ('model', LogisticRegression(max_iter=4000, random_state=42))
    ])

    param_grid = {
        'model__penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'model__class_weight': [None, 'balanced'],
        'model__solver': ['lbfgs', 'newton-cg', 'sag', 'saga'],
        'model__C': [0.01, 0.1, 1, 10, 100],
        'model__l1_ratio': [0, 0.5, 1],
        'model__tol': [1e-4, 1e-5, 1e-6]
    }
    
    grid = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    proba_test = best_model.predict_proba(X_test)[:, 1]
    y_pred = best_model.predict(X_test)

    # Predict
    y_pred = best_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

    acc_rf = accuracy_score(y_test, y_pred)
    auc_rf = roc_auc_score(y_test, y_pred)
    precision_rf = precision_score(y_test, y_pred, zero_division=0)
    recall_rf = recall_score(y_test, y_pred, zero_division=0)
    f1_score_rf = f1_score(y_test, y_pred, zero_division=0)
    average_precision_score_rf = average_precision_score(y_test, y_pred)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    print(f'Accuracy for Logistic Regression: {acc_rf:.2%}')
    print(f'AUC for Logistic Regression: {auc_rf:.2f}')
    print(f'Precision for Logistic Regression: {precision_rf:.2%}')
    print(f'Recall for Logistic Regression: {recall_rf:.2%}')
    print(f'F1 Score for Logistic Regression: {f1_score_rf:.2%}')
    print(f'Average Precision Score for Logistic Regression: {average_precision_score_rf:.2f}')

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob) if (y_prob is not None and y_test.nunique() == 2) else None
    ap = average_precision_score(y_test, y_prob) if (y_prob is not None and y_test.nunique() == 2) else None
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Logistic Regression (final) ===")
    print(f"Accuracy: {acc:.4f}")
    if roc_auc is not None:
        print(f"ROC-AUC : {roc_auc:.4f}")
    if ap is not None:
        print(f"Avg Precision (PR-AUC): {ap:.4f}")
    print("Confusion matrix:\n", cm)
    print("\nClassification report:\n", classification_report(y_test, y_pred))

    # Save model
    model_path = save_model(best_model, "models/Logistic_Regression_best.joblib")
    print(f"Saved model -> {model_path}")

    metrics_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "model": "Logistic Regression",
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