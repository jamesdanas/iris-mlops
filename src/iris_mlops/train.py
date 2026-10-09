"""
Training logic for the Iris classifier.

WHY THIS MODULE EXISTS:
    Training is a separate concern from CLI parsing and from serving.
    Later phases (MLflow, Prefect, CI) will import and call `train()`.
    directly without going through the CLI.
"""

import json
import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from iris_mlops.data import load_data, split

logger = logging.getLogger(__name__)


def train(seed: int = 42, C: float = 1.0, max_iter: int = 200) -> float:
    # WHY set_experiment:
    #   MLflow groups runs by experiment. All runs from this project
    #   appear under "iris-classification" in the UI.
    mlflow.set_experiment("iris-classification")

    # WHY with mlflow.start_run():
    #   Everything inside this block is attached to ONE run record.
    #   When the block exits, the run i finalized.
    with mlflow.start_run():
        # Log parameters - the INPUTS to training.
        # WHY: so we can query "which C gave the best accuracy?"
        mlflow.log_param("seed", seed)
        mlflow.log_param("C", C) 
        mlflow.log_param("max_iter", max_iter)

        X, y = load_data()
        X_train, X_test, y_train, y_test = split(X, y, seed=seed)

        model = LogisticRegression(C=C, max_iter=max_iter, random_state=seed)
        model.fit(X_train, y_train)


        acc = accuracy_score(y_test, model.predict(X_test))

        Path("models").mkdir(exist_ok=True)
        joblib.dump(model, "models/model.joblib")

        # Write metrics to a JSON file so DVC can track it as an artifact.
        # WHY: DVC reads this file to show "dvc metrics show" output and 
        #  "dvc metrics diff" between commits. It is the ML-native way 
        #  to version model quality alongide code and data.
        Path("metrics.json").write_text(
            json.dumps({"accuracy": acc, "seed": seed, "C": C}, indent=2)
        )

   
        # Log metric - the OUTPUT of training.
        # WHY: metrics are queryable; you can sort and compare runs.
        mlflow.log_metric("accuracy", acc)

        # Log the model artifact.
        # WHY: storing the model means we can load it later
        #   register it, and promote it.
        mlflow.sklearn.log_model(model, name="model")

        logger.info("accuracy=%.4f", acc)
        return acc
