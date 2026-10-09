"""
Drift-triggered retraining pipeline.

WHY THIS EXISTS:
    A model that never retrains degrades forever. This flow checks
    for drift and retrains only when the world has changed.

    Without this, retraining is manual and late. With this, retraining
    is automatic and early — the closed loop of Continuous Training.
"""

import logging

from prefect import flow, task

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@task
def check_drift(reference_path: str, current_path: str) -> bool:
    """Check PSI between reference and current data."""
    from monitoring.drift_detector import check_drift as detect

    drift = detect(reference_path, current_path)

    if drift:
        logger.warning("Drift detected. Retraining recommended.")
    else:
        logger.info("No drift detected. Skipping retraining.")

    return drift


@flow(name="drift-triggered-retraining")
def drift_retraining_flow(
    reference_path: str = "data/processed.csv",
    current_path: str = "data/production_recent.csv",
):
    """
    Full pipeline: check drift, retrain if needed.

    WHY A SEPARATE FLOW:
        This is the entry point for scheduled runs. It decides
        whether the expensive training pipeline should run.
    """
    drift = check_drift(reference_path, current_path)

    if drift:
        from flows.training_pipeline import training_pipeline

        logger.info("Drift detected. Starting retraining.")
        training_pipeline()
        logger.info("Retraining complete.")
    else:
        logger.info("No drift. Pipeline idle.")


if __name__ == "__main__":
    print("=== Entering main block ===", flush=True)
    drift_retraining_flow()
    print("=== Flow complete ===", flush=True)
