import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline(numeric_features, categorical_features):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=None,
        random_state=42,
    )

    model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    return model


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, zero_division=0, output_dict=True
        ),
    }

    # ROC AUC if proba exists
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, proba))

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="ayush_ehr_synthetic.csv",
        help="Path to ayush_ehr_synthetic.csv",
    )
    parser.add_argument(
        "--target",
        default="diabetes_mellitus",
        help="Target column name (default: diabetes_mellitus)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test size fraction (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output-model",
        default="ayush_diabetes_model.pkl",
        help="Output model filename (joblib)",
    )
    parser.add_argument(
        "--output-metrics",
        default="ayush_diabetes_metrics.json",
        help="Output metrics filename (json)",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.data)

    if args.target not in df.columns:
        raise ValueError(
            f"Target column '{args.target}' not found in CSV. Available: {list(df.columns)}"
        )

    # Drop identifiers
    for col in ["patient_id"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    y = df[args.target]
    X = df.drop(columns=[args.target])

    # Ensure binary int labels 0/1
    y = pd.Series(y).astype(int)

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    model = build_pipeline(numeric_features, categorical_features)
    model.fit(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)

    joblib.dump(model, args.output_model)

    Path(args.output_metrics).write_text(json.dumps(metrics, indent=2))

    print("Saved model to:", args.output_model)
    print("Saved metrics to:", args.output_metrics)
    print("Metrics summary:")
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        if k in metrics:
            print(f"- {k}: {metrics[k]:.4f}")


if __name__ == "__main__":
    main()
