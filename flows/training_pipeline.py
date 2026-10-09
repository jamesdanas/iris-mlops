"""
Prefect pipeline for training and validation.

WHY PREFECT:
    A Python script runs once, top to bottom. If step 3 fials,
    you rerun the whole thing. Prefect gives you:
      - Retries on individual tasks
      - State tracking (which tasks succeeded, which failed)
      - A UI to debug
      - Scheduling
"""

import logging

import pandas as pd
from prefect import flow, task

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@task(retries=3, retry_delay_seconds=10)
def validate_data(data_path: str) -> str:
    """
    Validate the data against the schema.

    WHY retries:
        Transient failures (network blips, locked files) often succeed
        on retry. Retries cost nothing and prevent suprious pipeline failures.
    """
    from iris_mlops.validation import iris_schema

    df = pd.read_csv(data_path)
    iris_schema.validate(df)
    logger.info("Validation passed for %s", data_path)
    return data_path


@task
def train_model(seed: int = 42, C: float = 1.0) -> float:
    """Train the model and return accuracy."""
    from iris_mlops.train import train

    acc = train(seed=seed, C=C)
    logger.info("Training with accuracy=%.4f", acc)
    return acc


@task
def evaluate_gate(accuracy: float, threshold: float = 0.90) -> bool:
    """
    Gate: fail the pipeline if accuracy is below threshold.

    WHY a threshold:
        Even with retries, a bad model should not proceed to deployment.
        The gate makes the quality bar explicit and enforceable.
    """
    if accuracy < threshold:
        raise ValueError(f"Accuracy {accuracy:.4f} below threshold {threshold}")
    return True


@flow(name="iris-training-pipeline")
def training_pipeline(
    data_path: str = "data/processed.csv", seed: int = 42, C: float = 1.0
):
    """
    The full training pipeline as a flow.

    WHY a flow:
        Flows compose tasks. Each task is a unit of work with its own
        retry policy and state. The flow orchestrates them.
    """
    validate_data(data_path)
    accuracy = train_model(seed=seed, C=C)
    evaluate_gate(accuracy)
    logger.info("Pipeline complete. Accuracy=%.4f", accuracy)


if __name__ == "__main__":
    print("=== Entering main block ===", flush=True)
    print("=== Calling training_pipeline ===", flush=True)
    result = training_pipeline()
    print(f"=== Flow completed, result: {result} ===", flush=True)
