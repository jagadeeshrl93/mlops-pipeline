import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURES = [
    "cpu_usage", "memory_usage", "error_rate", "latency_p99",
    "disk_usage", "network_in", "pod_restarts", "hour_of_day",
]

DRIFT_THRESHOLD = 0.05


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types that json can't serialize by default."""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def detect_drift(reference_data_path: str, current_data_path: str) -> dict:
    reference = pd.read_csv(reference_data_path)
    current = pd.read_csv(current_data_path)

    drift_results = {}
    drifted_features = []

    for feature in FEATURES:
        if feature not in reference.columns or feature not in current.columns:
            continue

        ks_stat, p_value = stats.ks_2samp(
            reference[feature].dropna(),
            current[feature].dropna(),
        )

        drifted = bool(p_value < DRIFT_THRESHOLD)

        drift_results[feature] = {
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_value), 4),
            "drifted": drifted,
        }

        if drifted:
            drifted_features.append(feature)
            logger.warning(f"DRIFT detected in '{feature}': KS={ks_stat:.4f}, p={p_value:.4f}")
        else:
            logger.info(f"No drift in '{feature}': KS={ks_stat:.4f}, p={p_value:.4f}")

    overall_drift = len(drifted_features) > 0

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "reference_samples": int(len(reference)),
        "current_samples": int(len(current)),
        "drift_threshold": DRIFT_THRESHOLD,
        "overall_drift_detected": bool(overall_drift),
        "drifted_features": drifted_features,
        "feature_results": drift_results,
    }

    logger.info(
        f"Drift detection complete. "
        f"{'DRIFT DETECTED' if overall_drift else 'No drift'} "
        f"in {len(drifted_features)}/{len(FEATURES)} features"
    )

    return report


def send_cloudwatch_metrics(report: dict, namespace: str = "MLOps/IncidentClassifier"):
    try:
        client = boto3.client(
            "cloudwatch",
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        )
        client.put_metric_data(
            Namespace=namespace,
            MetricData=[
                {
                    "MetricName": "DriftDetected",
                    "Value": 1 if report["overall_drift_detected"] else 0,
                    "Unit": "Count",
                },
                {
                    "MetricName": "DriftedFeatureCount",
                    "Value": len(report["drifted_features"]),
                    "Unit": "Count",
                },
            ],
        )
        logger.info(f"Metrics sent to CloudWatch namespace: {namespace}")
    except Exception as e:
        logger.warning(f"CloudWatch metrics failed (not critical): {e}")


def save_drift_report(report: dict, output_dir: str = "models"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    report_path = os.path.join(output_dir, "drift_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)
    logger.info(f"Drift report saved to {report_path}")
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-data", default="data/train.csv")
    parser.add_argument("--current-data", default="data/val.csv")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--send-cloudwatch", action="store_true")
    args = parser.parse_args()

    report = detect_drift(args.reference_data, args.current_data)

    if args.send_cloudwatch:
        send_cloudwatch_metrics(report)

    report_path = save_drift_report(report, args.output_dir)

    print(f"\nDrift monitoring complete:")
    print(f"  Reference samples: {report['reference_samples']}")
    print(f"  Current samples:   {report['current_samples']}")
    print(f"  Drift detected:    {report['overall_drift_detected']}")
    print(f"  Drifted features:  {report['drifted_features']}")
    print(f"  Report:            {report_path}")
