import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_incident_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic infrastructure incident data.

    Features:
        cpu_usage       — CPU utilization % at time of incident
        memory_usage    — Memory utilization %
        error_rate      — Errors per minute
        latency_p99     — 99th percentile latency in ms
        disk_usage      — Disk utilization %
        network_in      — Inbound network MB/s
        pod_restarts    — Number of pod restarts in last hour
        hour_of_day     — Hour when incident occurred (0-23)

    Target:
        severity        — 0=low, 1=medium, 2=high
    """
    np.random.seed(seed)

    cpu = np.random.beta(2, 5, n_samples) * 100
    memory = np.random.beta(2, 3, n_samples) * 100
    error_rate = np.random.exponential(2, n_samples)
    latency_p99 = np.random.lognormal(4, 1, n_samples)
    disk_usage = np.random.beta(3, 4, n_samples) * 100
    network_in = np.random.exponential(50, n_samples)
    pod_restarts = np.random.poisson(1, n_samples)
    hour_of_day = np.random.randint(0, 24, n_samples)

    # Severity logic — mirrors real on-call thresholds
    severity = np.zeros(n_samples, dtype=int)

    # Medium severity conditions
    medium_mask = (
        (cpu > 70) |
        (memory > 80) |
        (error_rate > 5) |
        (latency_p99 > 500) |
        (pod_restarts > 3)
    )
    severity[medium_mask] = 1

    # High severity — multiple signals or extreme values
    high_mask = (
        ((cpu > 90) & (memory > 85)) |
        (error_rate > 15) |
        (latency_p99 > 2000) |
        (pod_restarts > 8) |
        ((cpu > 80) & (error_rate > 8) & (latency_p99 > 800))
    )
    severity[high_mask] = 2

    df = pd.DataFrame({
        "cpu_usage": cpu.round(2),
        "memory_usage": memory.round(2),
        "error_rate": error_rate.round(3),
        "latency_p99": latency_p99.round(1),
        "disk_usage": disk_usage.round(2),
        "network_in": network_in.round(2),
        "pod_restarts": pod_restarts,
        "hour_of_day": hour_of_day,
        "severity": severity,
    })

    logger.info(f"Generated {n_samples} samples")
    logger.info(f"Severity distribution:\n{df['severity'].value_counts().sort_index()}")
    return df


def save_dataset(df: pd.DataFrame, output_dir: str = "data") -> dict:
    """Split into train/val/test and save to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 70/15/15 split
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train = df_shuffled.iloc[:train_end]
    val = df_shuffled.iloc[train_end:val_end]
    test = df_shuffled.iloc[val_end:]

    train.to_csv(output_path / "train.csv", index=False)
    val.to_csv(output_path / "val.csv", index=False)
    test.to_csv(output_path / "test.csv", index=False)

    paths = {
        "train": str(output_path / "train.csv"),
        "val": str(output_path / "val.csv"),
        "test": str(output_path / "test.csv"),
    }

    logger.info(f"Saved: train={len(train)}, val={len(val)}, test={len(test)}")
    return paths


if __name__ == "__main__":
    df = generate_incident_data(n_samples=5000)
    paths = save_dataset(df, output_dir="data")
    print(f"\nDataset saved:")
    for split, path in paths.items():
        print(f"  {split}: {path}")
    print(f"\nSample rows:")
    print(df.head(3).to_string())