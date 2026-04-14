# MLOps Pipeline

> End-to-end MLOps pipeline for infrastructure incident severity classification —
> data generation, model training, evaluation gate, and drift monitoring.

---

## What this is

A production-grade MLOps pipeline that trains a model to predict infrastructure
incident severity (low / medium / high) from system metrics like CPU usage,
memory, error rate, and latency.

The pipeline runs four stages automatically — data generation, training,
evaluation gate, and drift monitoring. If the model fails the evaluation gate,
the pipeline stops. If drift is detected, an alert is raised.

---

## Architecture

| Stage               | What it does                                        | Technology           |
| ------------------- | --------------------------------------------------- | -------------------- |
| Data generation     | Synthetic incident dataset (5k samples, 8 features) | pandas, numpy        |
| Training            | Random Forest with class balancing                  | scikit-learn, MLflow |
| Evaluation          | Accuracy + F1 gate before deployment                | scikit-learn         |
| Drift detection     | KS test across all features                         | scipy, CloudWatch    |
| Experiment tracking | Log params, metrics, artifacts                      | MLflow               |

---

## Results

| Metric            | Score |
| ----------------- | ----- |
| Test accuracy     | 0.997 |
| F1 weighted       | 0.997 |
| F1 macro          | 0.664 |
| Pipeline duration | 3.8s  |

---

## Key design decisions

**Why Random Forest over deep learning?**
Infrastructure incident classification is a tabular problem with engineered
features. Random Forest gives comparable accuracy on tabular data, trains in
seconds, and produces interpretable feature importance — critical for on-call
engineers who need to understand why a severity was assigned.

**Why KS test for drift detection?**
The Kolmogorov-Smirnov test makes no assumptions about the underlying
distribution. For infrastructure metrics which follow non-normal distributions
(CPU spikes, latency tails), this is more robust than mean/variance comparisons.

**Why an evaluation gate?**
Deploying a worse model is worse than not deploying at all. The gate enforces
minimum accuracy of 0.80 and F1 macro of 0.60. Step Functions catches a
non-zero exit code and routes to the rollback branch automatically.

**Why class_weight='balanced'?**
High severity incidents are rare (3 in 5000 samples). Without balancing, the
model predicts "low" for everything and achieves 86% accuracy while being
useless for the cases that matter most.

---

## Getting started

### Prerequisites

- Python 3.11
- AWS account with mlops-dev IAM user configured

### Setup

```bash
git clone https://github.com/jagadeeshrl93/mlops-pipeline
cd mlops-pipeline

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
```

### Run the full pipeline

```bash
# Terminal 1 — start MLflow
mlflow server --host 127.0.0.1 --port 5050

# Terminal 2 — run pipeline
export MLFLOW_TRACKING_URI=http://localhost:5050
python pipelines/run_pipeline.py
```

Open http://localhost:5050 to see all runs, metrics, and artifacts in the MLflow UI.

---

## Project structure

src/
├── data/
│ └── generate.py synthetic incident data generator
├── training/
│ └── train.py Random Forest training with MLflow tracking
├── evaluation/
│ └── evaluate.py test set scoring and deployment gate
└── monitoring/
└── monitor.py KS drift detection and CloudWatch metrics
pipelines/
└── run_pipeline.py end-to-end pipeline runner

---

## What I'd build next

- [ ] AWS Step Functions state machine — full serverless orchestration
- [ ] SageMaker training jobs — cloud compute instead of local
- [ ] S3 data versioning — track dataset changes over time
- [ ] Automated retraining trigger — when drift exceeds threshold
- [ ] Model serving endpoint — REST API for real-time predictions
- [ ] GitHub Actions CI — run pipeline on every push

---

_Built by Jagadeesh Reddy — Senior Platform & Cloud AI Engineer_
_GitHub: [jagadeeshrl93](https://github.com/jagadeeshrl93)_
