"""
Request and response schemas for the prediction API.

WHY PYDANTIC:
    FastAPI reads the type hints and automatically:
      - Validates incoming JSON
      - Returns 422 errors for invalid input
      - Generates OpenAPI documentation at /docs

    Your prediction code only ever sees VALID data.
"""

from pydantic import BaseModel, Field
from typing import List


class PredictionRequest(BaseModel):
    # WHY Field(...) with constraints:
    #   `...` means required. min_length/max_length enforce exactly 4.
    #   Without these, a request with 3 or 5 features would reach the model
    #   and crash with a cryptic sklearn error.
    features: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="[sepal_length, sepal_width, petal_length, petal_width]",
    )

    # WHY json_schema_extra:
    #   Provides a sample value in the /docs UI so you can click "Try it out".
    class Config:
        json_schema_extra = {
            "example": {"features": [5.1, 3.5, 1.4, 0.2]}
        }


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Class 0, 1, or 2")
    probabilities: List[float] = Field(..., description="Probability per class")
    model_version: str = Field(..., description="Model identifier")
