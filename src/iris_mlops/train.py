"""
Training logic for the Iris classifier.

WHY THIS MODULE EXISTS:
    Training is a separate concern from CLI parsing and from serving.
    Later phases (MLflow, Prefect, CI) will import and call `train()`.
    directly without going through the CLI.
"""

import logging
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from iris_mlops.data import load_data, split

logger = logging.getLogger(__name__)


def train(seed: int = 42, C: float = 1.0, max_iter: int = 200) -> float:
    """
    Train a logistic regression model on iris and return accuracy.

    PARAMETERS:
        seed: random seed for reproducibility
        C: inverse regularization strength (smaller = stronger reg)
        max_iter: max iterations for the solver to converge

    RETURNS:
        Accuracy on the held-out test set (float between 0 and 1).

    WHY LOGISTIC REGRESSION:
        It is fast, interpretable, and adequate for a linearly separable
        dataset like Iris. In Phase 7, we could swap it for a larger model.
    """
    # Step 1: Load and split
    X, y = load_data()
    X_train, X_test, y_train, y_test = split(X, y, seed=seed)

    # Step 2: Create the model
    model = LogisticRegression(C=C, max_iter=max_iter, random_state=seed)

    # Step 3: Fit the model
    model.fit(X_train, y_train)

    # Step 4: Evaluate on the held-out test set
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    # Step 5: Log the result
    logger.info("accuracy=%.4f", acc)

    return acc



