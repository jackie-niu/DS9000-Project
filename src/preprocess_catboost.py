import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def preprocess_data(filepath="Worksheet in Case Study question 2.xlsx", target="fraud_reported"):
    """Load, clean, encode, and split the dataset for ML."""

    # Load
    df = pd.read_excel(filepath, sheet_name=0)

    # Replace ? with NaN
    df = df.replace("?", np.nan)

    # Fill missing values
    df["collision_type"] = df["collision_type"].fillna("Unknown")
    df["property_damage"] = df["property_damage"].fillna("Unknown")
    df["police_report_available"] = df["police_report_available"].fillna("Unknown")
    df["authorities_contacted"] = df["authorities_contacted"].fillna("Unknown")

    # Drop irrelevant columns
    drop_cols = [
        "policy_number", "policy_bind_date", "incident_date",
        "incident_location", "insured_zip"
    ]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Split into features and target
    if f"{target}" in df.columns:  
        y = df[f"{target}"]
        X = df.drop(columns=[f"{target}"])
    else:
        y = df[target]
        X = df.drop(columns=[target])

    # Scale numeric features
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data()
