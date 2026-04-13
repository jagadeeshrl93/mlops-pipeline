import argparse
import logging
import os
import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
import joblib
import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURES = [
    "cpu_usage",
    "memory_usage",
    "error_rate",
    "latency_p99",
    "disk_usage",
    "network_in",
    "pod_restarts",
    "hour_of_day",
]
TARGET = "severity"


def load_data(train_path: str, val_path: str):
    """Load train and validation datasets."""
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    logger.info(f"Train: {len(train)} rows, Val: {len(val)} rows")
    return train, val


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_split: int = 5,
) -> dict:
    """
    Train a Random Forest classifier with MLflow tracking.
    Uses class_weight='balanced' to handle severity imbalance.
    """
    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_val = val_df[FEATURES]
    y_val = val_df[TARGET]

    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "features": FEATURES,
            "class_weight": "balanced",
        })

        # Train
        logger.info("Training Random Forest...")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight="balanced",  # handles imbalanced classes
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # Evaluate on validation set
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        f1_weighted = f1_score(y_val, y_pred, average="weighted")
        f1_macro = f1_score(y_val, y_pred, average="macro")

        # Log metrics
        mlflow.log_metrics({
            "val_accuracy": round(accuracy, 4),
            "val_f1_weighted": round(f1_weighted, 4),
            "val_f1_macro": round(f1_macro, 4),
        })

        # Log model
        mlflow.sklearn.log_model(model, "model")
        run_id = mlflow.active_run().info.run_id

        logger.info(f"Val accuracy: {accuracy:.4f}")
        logger.info(f"Val F1 weighted: {f1_weighted:.4f}")
        logger.info(f"Val F1 macro: {f1_macro:.4f}")
        logger.info(f"\n{classification_report(y_val, y_pred)}")

        # Feature importance
        importance = dict(zip(FEATURES, model.feature_importances_.round(4)))
        mlflow.log_dict(importance, "feature_importance.json")
        logger.info(f"Feature importance: {importance}")

    return {
        "model": model,
        "run_id": run_id,
        "metrics": {
            "val_accuracy": accuracy,
            "val_f1_weighted": f1_weighted,
            "val_f1_macro": f1_macro,
        },
    }


def save_model(model, output_dir: str = "models"):
    """Save model to disk."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(output_dir, "model.joblib")
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="data/train.csv")
    parser.add_argument("--val-data", default="data/val.csv")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--min-samples-split", type=int, default=5)
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    mlflow.set_experiment("incident-severity-classifier")

    train_df, val_df = load_data(args.train_data, args.val_data)

    result = train_model(
        train_df,
        val_df,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
    )

    model_path = save_model(result["model"], args.output_dir)

    print(f"\nTraining complete:")
    print(f"  Run ID:       {result['run_id']}")
    print(f"  Accuracy:     {result['metrics']['val_accuracy']:.4f}")
    print(f"  F1 weighted:  {result['metrics']['val_f1_weighted']:.4f}")
    print(f"  F1 macro:     {result['metrics']['val_f1_macro']:.4f}")
    print(f"  Model saved:  {model_path}")