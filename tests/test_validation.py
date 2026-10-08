import pandas as pd
import pandera.pandas as pa
import pytest

from iris_mlops.validation import iris_schema


def test_schema_accepts_valid_data():
    """A well-performed row must pass validation."""
    good = pd.DataFrame({
        "f0": [5.1],
        "f1": [3.5],
        "f2": [1.4],
        "f3": [0.2],
        "target": [0]
    })
    iris_schema.validate(good)  # Should not raise an exception

def test_schema_rejects_negative_values():
    """WHY: a negative measuremnt is always a bug."""
    bad = pd.DataFrame({
        "f0": [-1.0],
        "f1": [3.5],
        "f2": [1.4],
        "f3": [0.2],
        "target": [0]
    })
    with pytest.raises(pa.errors.SchemaError):
        iris_schema.validate(bad)

def test_schema_rejects_unknown_target():
    """WHY: Iris has exactly 3 classes: A target of 7 means corruption."""
    bad = pd.DataFrame({
        "f0": [5.1],
        "f1": [3.5],
        "f2": [1.4],
        "f3": [0.2],
        "target": [7]
    })
    with pytest.raises(pa.errors.SchemaError):
        iris_schema.validate(bad)
