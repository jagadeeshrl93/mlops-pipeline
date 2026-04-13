"""
Local pipeline runner — executes all stages in sequence.
Mirrors what Step Functions will orchestrate in production.
"""
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.generate import generate_incident_data, save_dataset
from src.training.train import load_data, train_model, save_model
from src.evaluation.evaluate import evaluate_model, save_evaluation_report
from src.monitoring.monitor import detect_drift, save_drift_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_pipeline():
    start = datetime.utcnow()
    logger.info("=" * 50)
    logger.info("MLOps pipeline starting")
    logger.info("=" * 50)

    results = {}

    # Stage 1: Data generation
    logger.info("\n[1/4] Generating data...")
    df = generate_incident_data(n_samples=5000)
    paths = save_dataset(df, output_dir="data")
    results["data"] = {"status": "ok", "paths": paths}
    logger.info("Stage 1 complete")

    # Stage 2: Training
    logger.info("\n[2/4] Training model...")
    import mlflow
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
    )
    mlflow.set_experiment("incident-severity-classifier")

    train_df, val_df = load_data(paths["train"], paths["val"])
    train_result = train_model(train_df, val_df)
    model_path = save_model(train_result["model"])
    results["training"] = {
        "status": "ok",
        "run_id": train_result["run_id"],
        "metrics": train_result["metrics"],
    }
    logger.info("Stage 2 complete")

    # Stage 3: Evaluation
    logger.info("\n[3/4] Evaluating model...")
    eval_metrics = evaluate_model(model_path, paths["test"])
    save_evaluation_report(eval_metrics)
    results["evaluation"] = {
        "status": "ok",
        "metrics": eval_metrics,
        "passed_gate": eval_metrics["passed_gate"],
    }

    if not eval_metrics["passed_gate"]:
        logger.error("Evaluation gate FAILED — stopping pipeline")
        results["pipeline_status"] = "failed_evaluation_gate"
        save_pipeline_report(results, start)
        sys.exit(1)

    logger.info("Stage 3 complete — gate passed")

    # Stage 4: Drift monitoring
    logger.info("\n[4/4] Running drift detection...")
    drift_report = detect_drift(paths["train"], paths["val"])
    save_drift_report(drift_report)
    results["monitoring"] = {
        "status": "ok",
        "drift_detected": drift_report["overall_drift_detected"],
        "drifted_features": drift_report["drifted_features"],
    }
    logger.info("Stage 4 complete")

    # Summary
    duration = (datetime.utcnow() - start).total_seconds()
    results["pipeline_status"] = "success"
    results["duration_seconds"] = round(duration, 2)

    save_pipeline_report(results, start)

    logger.info("\n" + "=" * 50)
    logger.info("Pipeline completed successfully")
    logger.info(f"Duration: {duration:.1f}s")
    logger.info(f"Accuracy: {eval_metrics['test_accuracy']:.4f}")
    logger.info(f"Drift:    {drift_report['overall_drift_detected']}")
    logger.info("=" * 50)

    return results


def save_pipeline_report(results: dict, start: datetime):
    Path("models").mkdir(exist_ok=True)
    report = {
        "run_timestamp": start.isoformat(),
        **results,
    }
    with open("models/pipeline_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Pipeline report saved to models/pipeline_report.json")


if __name__ == "__main__":
    run_pipeline()
