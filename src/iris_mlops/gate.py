"""
Model gate: only promote if the candidate beats the champion.

WHY GATES EXIST:
    Without a gate, every successfully-trained model gets deployed-
    even if it is worse than what is currently serving traffic.
    The gate makes "is this model better?" an automated decision.
"""

import logging

from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)


def evaluate_gate(
    candidate_model, champion_model, X_test, y_test, min_improvement: float = 0.01
) -> bool:
    """
    Compare candidate and champion on the same test set.

    RETURNS:
        True if candidate improves on champion by at least min_improvement.
        Raises ValueError otherwise.
    """
    candidate_acc = accuracy_score(y_test, candidate_model.predict(X_test))
    champion_acc = accuracy_score(y_test, champion_model.predict(X_test))
    improvement = candidate_acc - champion_acc

    logger.info(
        "candidate=%.4f champion=%.4f improvement=%.4f threshold=%.4f",
        candidate_acc,
        champion_acc,
        improvement,
        min_improvement,
    )

    if improvement < min_improvement:
        raise ValueError(
            f"Candidate ({candidate_acc:.4f}) does not beat champiom"
            f"({champion_acc:.4f}) by {min_improvement:.4f}"
        )
    return True
