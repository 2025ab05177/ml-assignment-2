import argparse
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from model.utils import (
    ensure_artifacts_dir,
    compute_metrics_multiclass,
    save_json,
    safe_predict_proba,
)

from model.logistic_regression import build_model as build_lr
from model.decision_tree import build_model as build_dt
from model.knn import build_model as build_knn
from model.naive_bayes import build_model as build_nb
from model.random_forest import build_model as build_rf
from model.xgboost_model import build_model as build_xgb


MODEL_KEYS = [
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
    "Random Forest",
    "XGBoost",
]


def load_dataset(csv_path: str, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Columns: {list(df.columns)[:10]} ...")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def build_all_models(num_class: int, random_state: int = 42) -> Dict[str, object]:
    return {
        "Logistic Regression": build_lr(random_state=random_state),
        "Decision Tree": build_dt(random_state=random_state),
        "KNN": build_knn(),
        "Naive Bayes": build_nb(),
        "Random Forest": build_rf(random_state=random_state),
        "XGBoost": build_xgb(num_class=num_class, random_state=random_state),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="har_full.csv", help="Path to prepared HAR CSV (must include Activity column)")
    parser.add_argument("--target", type=str, default="Activity", help="Target column name")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--artifacts_dir", type=str, default="artifacts")
    args = parser.parse_args()

    ensure_artifacts_dir(args.artifacts_dir)

    X, y = load_dataset(args.csv, args.target)

    # Save feature column order (critical for Streamlit prediction)
    feature_cols = list(X.columns)

    # Encode labels to 0..K-1 for consistent training (especially XGBoost)
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_enc,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y_enc,
    )

    num_class = len(le.classes_)
    models = build_all_models(num_class=num_class, random_state=args.random_state)

    rows = []
    for name, model in models.items():
        print(f"\nTraining: {name}")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = safe_predict_proba(model, X_test)

        metrics = compute_metrics_multiclass(y_test, y_pred, y_proba, average="macro")
        rows.append({"ML Model Name": name, **metrics})

        # Save model
        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        model_path = f"{args.artifacts_dir}/{safe_name}.joblib"
        joblib.dump(model, model_path)
        print(f"Saved: {model_path}")
        print("Metrics:", metrics)

    metrics_df = pd.DataFrame(rows)[
        ["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    ].sort_values("ML Model Name")

    metrics_csv_path = f"{args.artifacts_dir}/metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\nSaved metrics table: {metrics_csv_path}")

    # Save label encoder + meta
    joblib.dump(le, f"{args.artifacts_dir}/label_encoder.joblib")
    save_json({"target_col": args.target, "feature_cols": feature_cols}, f"{args.artifacts_dir}/meta.json")

    print("\nDone. Commit the artifacts/ folder to GitHub for Streamlit deployment.")


if __name__ == "__main__":
    main()