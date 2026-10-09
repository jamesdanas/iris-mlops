"""
FastAPI serving for the Iris classifier.

WHY THIS MODULE EXISTS:
    A trained model is useless to other software unless it has an
    interface. This module wraps the model in an HTTP API so that
    any client — a web app, a mobile app, another service — can call it.
"""

import json
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Response, status
from fastapi.responses import JSONResponse

from app.schemas import PredictionRequest, PredictionResponse


class PrettyJSONResponse(JSONResponse):
    """JSONResponse that pretty-prints with indentation and spaces."""

    def render(self, content) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            separators=(", ", ": "),
        ).encode("utf-8")


app = FastAPI(
    title="Iris Prediction API",
    version="1.0.0",
    description="Serves a logistic regression classifier trained on Iris",
    default_response_class=JSONResponse,
)

# Module-level variable — the loaded model lives here for the app's lifetime.
# WHY at module level: loading a model is expensive; do it once, not per request.
model = None

MODEL_PATH = Path("models/model.joblib")


@app.on_event("startup")
def load_model() -> None:
    """
    Load the model when the app starts.

    WHY at startup (not per request):
        Loading a 100 MB model on every request would add seconds of latency.
        Loading once at startup means requests are fast.
    """
    global model
    if not MODEL_PATH.exists():
        # Fail loudly at startup — do not serve predictions with no model.
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)


@app.get("/health/live")
def liveness():
    """
    Liveness: is the process running?

    WHY trivial:
        Kubernetes restarts containers whose liveness probe fails. If this
        checked the database and the DB was slow, we would restart healthy
        containers. Keep liveness trivial.
    """
    return {"status": "alive"}


@app.get("/health/ready")
def readiness(response: Response):
    """
    Readiness: can we serve predictions?

    WHY separate from liveness:
        A container can be alive (process running) but not ready (model not
        loaded). Kubernetes removes not-ready containers from load balancers
        without restarting them.
    """
    if model is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "not ready", "reason": "model not loaded"}
    return {"status": "ready"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Predict the Iris class for one sample.

    WHY response_model:
        FastAPI validates the return value against PredictionResponse.
        If we accidentally returned a string, FastAPI would catch it.
    """
    # Defensive check — should never fire if startup worked
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Reshape: sklearn expects 2D input for a single sample
    features = np.array([request.features])

    # Predict class and probabilities
    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0].tolist()

    return PredictionResponse(
        prediction=prediction,
        probabilities=probabilities,
        model_version="v1.0.0",
    )
