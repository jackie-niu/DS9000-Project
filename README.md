# DS9000 Project – Machine Learning Model Repository for Fraud Detection

This repository contains a modular machine learning workflow for training and evaluating multiple models on the **Insurance Fraud Detection dataset**.  
Each model runs independently but shares a common **preprocessing** and **metrics tracking** system.

---

## 🧭 Project Structure

```
DS9000-Project/
│
├── archive/                   # Legacy code
│   └── Project_XGBoost.ipynb  # Original preprocessing and XGB implementation
│   └── Worksheet in Case Study question 2.xlsx
│
├── data/                      # Raw data files (Excel/CSV)
│   └── Worksheet in Case Study question 2.xlsx
│
├── models/                    # Trained models + metrics storage
│   ├── xgboost_best.joblib
│   └── metrics.jsonl
│
├── src/
│   ├── preprocess.py          # Shared preprocessing logic (load, clean, encode, split)
│   ├── utils.py               # Helper functions (save_model, append_metrics_jsonl, etc.)
│   ├── leaderboard.py         # Prints leaderboard of all model runs
│   └── models/
│       ├── xgboost_grid.py    # XGBoost model with GridSearchCV
|       └── knn.py             # KNN model with manual K loop
│
└── requirements.txt
```

---

## ⚙️ Setup

### Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running a Model

All models share the same preprocessing pipeline (`src/preprocess.py`) and save results to `models/metrics.jsonl`.

### Example: Run XGBoost with Grid Search
From the **project root**, run:
```bash
python -m src.models.xgboost_grid --data "data/Worksheet in Case Study question 2.xlsx" --target fraud_reported
```

You’ll see:
- Preprocessing output (train/test fraud rates)
- Grid search results (best params)
- Final test performance (accuracy, ROC-AUC, PR-AUC)
- Saved model at: `models/xgboost_best.joblib`
- Logged metrics at: `models/metrics.jsonl`

---

## 🧠 Viewing the Leaderboard

After running one or more models:
```bash
python -m src.leaderboard
```

Output example:
```
=== Leaderboard ===
Sorted by: test_metrics.average_precision (desc)

timestamp                 | model        |   ACC | ROC-AUC | PR-AUC |    F1
--------------------------------------------------------------------------------
2025-11-04T02:35:46.311530 | xgboost     | 0.765 |   0.809 |   0.500 |  0.434
```

You can also sort or limit:
```bash
python -m src.leaderboard --sort roc --top 5
```

---

## ➕ Adding a New Model

To add another model (e.g., Logistic Regression, Random Forest, SVM):

1. **Create a new file** in `src/models/`, e.g. `logistic.py`
2. **Import**:
   ```python
   from src.preprocess import preprocess_data
   from src.utils import save_model, append_metrics_jsonl
   ```
3. **Train** your model and compute metrics (accuracy, precision, recall, F1, ROC-AUC, PR-AUC).
4. **Save results**:
   ```python
   save_model(model, "models/logistic_best.joblib")
   append_metrics_jsonl(metrics_record, "models/metrics.jsonl")
   ```
5. **Run it**:
   ```bash
   python -m src.models.logistic --data "data/Worksheet in Case Study question 2.xlsx" --target fraud_reported
   ```

Your new model will automatically appear in the leaderboard!

---

## 📦 Model Artifacts

| File | Description |
|------|--------------|
| `models/*.joblib` | Saved trained models (load with `joblib.load`) |
| `models/metrics.jsonl` | JSON Lines file storing all model run metrics |

---

## 🧩 Reusing Trained Models

To make predictions later:
```python
import joblib
import pandas as pd
from src.preprocess import preprocess_data

# Load saved model
model = joblib.load("models/xgboost_best.joblib")

# Prepare new data (same format as training)
X_train, X_test, y_train, y_test = preprocess_data("data/new_claims.xlsx", target="fraud_reported")

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
```

---

## 🧹 Notes
- Always run from the **project root** (`python -m src.models...`) to ensure relative imports work.
- Every model automatically appends metrics to `models/metrics.jsonl`.
- You can visualize performance in notebooks or Power BI by importing that file.
