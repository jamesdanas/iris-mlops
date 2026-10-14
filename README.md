# Iris MLOps

**A production-minded MLOps reference implementation on the Iris dataset.**

This is not a tutorial. It is a working system that trains, tracks, serves, monitors, and retrains a model — with every phase documented and every decision explained.

---

## Why This Exists

Most MLOps learning materials stop at "train a model and print accuracy." That is not what production looks like. Production is a pipeline: versioned data, tracked experiments, validated inputs, gated promotions, monitored drift, and automated retraining.

This repository is a complete implementation of that pipeline. It uses the Iris dataset because it is small enough to run on a laptop in seconds — but the structure is the same one used for any model where a wrong prediction has consequences.

---

## What It Does

- **Reproducible training** — MLflow tracks every run; DVC versions the data
- **Config management** — Hydra keeps hyperparameters in files, not code
- **Production serving** — FastAPI + Pydantic with automatic validation
- **Containerized** — separate Dockerfiles for training and serving
- **Automated quality** — GitHub Actions lints, tests, and builds on every push
- **Orchestrated pipelines** — Prefect flows with retries, state, and gating
- **Drift detection** — Evidently compares production data to training data
- **Continuous training** — drift signals trigger retraining automatically
- **Local infrastructure** — MinIO (S3-compatible) and Kubernetes via Kind

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  DATA LAYER                                             │
│  DVC → MinIO (S3-compatible object storage)            │
│  Pandera → schema validation                            │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  TRAINING LAYER                                         │
│  Hydra → configuration                                  │
│  scikit-learn → model                                   │
│  MLflow → tracking + registry                           │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  SERVING LAYER                                          │
│  FastAPI → HTTP endpoint                                │
│  Pydantic → request validation                          │
│  Docker → container image                               │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  ORCHESTRATION LAYER                                    │
│  Prefect → pipeline flows                               │
│  Evidently → drift detection                            │
│  Kubernetes (Kind) → orchestration                      │
└─────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# 1. Set up the environment
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2. Start MinIO (object storage)
docker start minio || docker run -d \
  --name minio --restart unless-stopped \
  -p 9000:9000 -p 9001:9001 \
  -v ~/minio-data:/data \
  -e "MINIO_ROOT_USER=minioadmin" \
  -e "MINIO_ROOT_PASSWORD=minioadmin" \
  cgr.dev/chainguard/minio:latest \
  server /data --console-address ":9001"

# 3. Set credentials
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export AWS_ENDPOINT_URL=http://localhost:9000

# 4. Pull data and train
dvc pull
dvc repro

# 5. Serve the model
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 6. Test
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

---

## Project Structure

```
iris-mlops/
├── .github/workflows/     # CI/CD pipelines
├── app/                   # FastAPI serving
│   ├── main.py
│   └── schemas.py
├── conf/                  # Hydra configuration
├── data/                  # DVC-tracked datasets
├── flows/                 # Prefect orchestration
│   ├── training_pipeline.py
│   └── drift_retraining.py
├── k8s/                   # Kubernetes manifests
├── monitoring/            # Drift detection
│   └── drift_detector.py
├── src/iris_mlops/        # Core Python package
│   ├── data.py
│   ├── train.py
│   ├── cli.py
│   ├── validation.py
│   └── gate.py
├── tests/                 # Pytest suite
├── Dockerfile             # Training container
├── Dockerfile.api         # Serving container
├── dvc.yaml               # DVC pipeline
├── params.yaml            # Pipeline parameters
├── pyproject.toml         # Package metadata
├── requirements.txt       # Runtime dependencies
└── requirements-dev.txt   # Dev dependencies
```

---

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Running Training

```bash
python -m iris_mlops.cli seed=42 model.C=1.0
```

Or via Hydra multirun for grid search:

```bash
python -m iris_mlops.cli --multirun seed=1,2 model.C=0.1,1.0
```

### Running the DVC Pipeline

```bash
dvc repro           # runs only changed stages
dvc metrics show    # displays tracked metrics
dvc dag             # shows the pipeline graph
```

### Running Prefect Flows

```bash
# Terminal 1: start the Prefect server
prefect server start

# Terminal 2: run a flow
python flows/training_pipeline.py
```

### Drift Detection

```bash
python -c "
from monitoring.drift_detector import check_drift
result = check_drift('data/processed.csv', 'data/production_recent.csv')
print(f'Drift detected: {result}')
"
```

---

## Deployment

### Docker

```bash
docker build -f Dockerfile.api -t iris-api:latest .
docker run --rm -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  iris-api:latest
```

### Kubernetes (Kind)

```bash
# Create a local cluster
kind create cluster --name iris-mlops

# Load the image into the cluster
kind load docker-image iris-api:latest --name iris-mlops

# Create a ConfigMap with the model
kubectl create configmap iris-model --from-file=model.joblib=models/model.joblib

# Apply manifests
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Port-forward and test
kubectl port-forward svc/iris-api 8080:80
curl http://127.0.0.1:8080/health/live
```

---

## Local Infrastructure

This project uses **MinIO** for S3-compatible object storage and **Kind** for local Kubernetes.

### MinIO

The console is at **http://localhost:9001** (minioadmin / minioadmin).

The DVC remote is configured to `s3://iris-mlops-data/dvc`. DVC stores its credentials in `.dvc/config.local` (gitignored) and reads environment variables at runtime.

### Kind

A Kind cluster named `iris-mlops` provides a real Kubernetes environment on your laptop. The API image is loaded via `kind load docker-image`.

### Autoscaling

The HPA scales the API between 1 and 5 replicas based on CPU. Metrics come from the metrics-server.

```bash
# Install metrics-server (Kind-specific patch for TLS)
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
kubectl patch deployment metrics-server -n kube-system --type=json \
  -p='[{"op": "add", "path": "/spec/template/spec/containers/0/args/-", "value": "--kubelet-insecure-tls"}]'

# Apply the HPA
kubectl apply -f k8s/hpa.yaml
kubectl get hpa
```
---

## Phase Progress

This project is being built through a structured MLOps curriculum.

| Phase | Focus | Status |
|-------|-------|--------|
| 0 | Foundations (Python, Git, Docker, ML) | Complete |
| 1 | Experiment tracking (MLflow, DVC, Hydra) | Complete |
| 2 | Serving (FastAPI, Pydantic, Docker) | Complete |
| 3 | CI/CD (GitHub Actions, validation, gates) | Complete |
| 4 | Orchestration (Prefect, DVC pipelines, drift detection, CT) | Complete |
| 5 | Cloud and infrastructure (MinIO, Kubernetes, Terraform) | In progress |
| 6 | Monitoring and production ops | ⏳ Pending |
| 7 | Specialization (LLMOps / ML Platform / Edge / Governance) | ⏳ Pending |
| 8 | Advanced operations and career mastery | ⏳ Pending |

---

## What This Is Not

- **Not a tutorial.** It is a working system, not an explainer. For learning materials, see the phase notes in the commit history.
- **Not production-ready.** No authentication, no rate limiting, no disaster recovery, no multi-region deployment.
- **Not a large-scale system.** Iris is a toy dataset. The patterns transfer; the scale does not.
- **Not a full platform.** No feature store, no model monitoring UI, no A/B testing framework — yet.

The limitations are deliberate. The purpose is to demonstrate the **structure** of a production MLOps system, not to be one.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| **Language** | Python 3.12 |
| **ML** | scikit-learn |
| **Experiment tracking** | MLflow |
| **Data versioning** | DVC + dvc-s3 |
| **Config** | Hydra |
| **Serving** | FastAPI + Uvicorn + Pydantic |
| **Validation** | Pandera |
| **Orchestration** | Prefect |
| **Drift detection** | Evidently |
| **Containers** | Docker |
| **Kubernetes** | Kind |
| **Object storage** | MinIO |
| **CI/CD** | GitHub Actions |
| **Linting** | Ruff + pre-commit |

---

## License

MIT — see [LICENSE](LICENSE) if present.

## Acknowledgments

Built with: Python, scikit-learn, MLflow, DVC, Hydra, FastAPI, Pydantic, Prefect, Evidently, Prometheus, MinIO, Kind, Kubernetes.
