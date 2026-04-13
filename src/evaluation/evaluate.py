import argparse
import json
import logging
import os
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURES = [
    "cpu_usage", "memory_usage", "error_rate", "latency_p99",
    "disk_usage", "network_in", "pod_restarts", "hour_of_day",
]
TARGET = "severity"

THRESHOLDS = {
    "accuracy": 0.80,
    "f1_weighted": 0.80,
    "f1_macro": 0.60,
}


def evaluate_model(model_path: str, test_data_path: str) -> dict:
    model = joblib.load(model_path)
    test_df = pd.read_csv(test_data_path)

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    f1_macro = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    passed = (
        accuracy >= THRESHOLDS["accuracy"] and
        f1_weighted >= THRESHOLDS["f1_weighted"] and
        f1_macro >= THRESHOLDS["f1_macro"]
    )

    metrics = {
        "test_accuracy": round(accuracy, 4),
        "test_f1_weighted": round(f1_weighted, 4),
        "test_f1_macro": round(f1_macro, 4),
        "confusion_matrix": cm,
        "passed_gate": passed,
    }

    logger.info(f"Test accuracy:    {accuracy:.4f} (threshold: {THRESHOLDS['accuracy']})")
    logger.info(f"Test F1 weighted: {f1_weighted:.4f} (threshold: {THRESHOLDS['f1_weighted']})")
    logger.info(f"Test F1 macro:    {f1_macro:.4f} (threshold: {THRESHOLDS['f1_macro']})")
    logger.info(f"Evaluation gate:  {'PASSED' if passed else 'FAILED'}")
    logger.info(f"\n{classification_report(y_test, y_pred)}")

    return metrics


def save_evaluation_report(metrics: dict, output_dir: str = "models"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    report_path = os.path.join(output_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Evaluation report saved to {report_path}")
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/model.joblib")
    parser.add_argument("--test-data", default="data/test.csv")
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
    )

    metrics = evaluate_model(args.model_path, args.test_data)
    report_path = save_evaluation_report(metrics, args.output_dir)

    print(f"\nEvaluation complete:")
    print(f"  Accuracy:     {metrics['test_accuracy']:.4f}")
    print(f"  F1 weighted:  {metrics['test_f1_weighted']:.4f}")
    print(f"  F1 macro:     {metrics['test_f1_macro']:.4f}")
    print(f"  Gate:         {'PASSED' if metrics['passed_gate'] else 'FAILED'}")
    print(f"  Report:       {report_path}")

    if not metrics["passed_gate"]:
        exit(1)
