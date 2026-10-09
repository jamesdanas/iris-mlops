"""
Drift detection with Evidently 0.7.x.

WHY TWO DATAFRAMES:
    reference = what the model was trained on
    current   = recent production traffic
    If they diverge, the model is operating on unfamiliar data.

WHY THE 0.7.x API:
    Evidently 0.7 (April 2025) made a breaking change. The old imports
    (`evidently.report`, `evidently.metric_preset`) were removed. The
    new API uses `evidently.Report` and `evidently.presets`, and wraps
    DataFrames in `Dataset` objects before running.

    This module isolates the library-specific code. Migrating to a future
    version means rewriting this file, not the whole codebase.
"""

import pandas as pd
from evidently import Dataset, Report
from evidently.presets import DataDriftPreset


def _result_dict_to_bool(result_dict: dict) -> bool:
    """
    Return True if any columns drifted.

    WHY THIS STRUCTURE:
        Evidently 0.7.x's Report.dict() returns:
          {
            "metrics": [
              {"id": ..., "metric_name": "...", "config": ..., "value": ...},
              ...
            ],
            "tests": [...]
          }

        The drift summary metric has a `value` shaped like:
          {"count": <int>, "share": <float>}

        Its `metric_name` is "DriftedColumnsCount" (or a variation
        containing "DriftedColumns"). When count > 0, drift was detected.
    """
    for metric in result_dict.get("metrics", []):
        name = metric.get("metric_name", "")
        value = metric.get("value", {})

        # Look for the drift count metric
        if "DriftedColumns" in name and isinstance(value, dict):
            count = value.get("count", 0)
            if count > 0:
                return True

    return False


def check_drift(
    reference_path: str,
    current_path: str,
    min_drifted_columns: int = 1,
) -> bool:
    """
    Check whether enough columns have drifted to warrant action.

    PARAMETERS:
        min_drifted_columns:
            Minimum number of drifted columns required to return True.
            - 1 (default): any column drift is significant. Conservative.
            - majority (e.g., 3 of 5): dataset-level drift. Pragmatic.
            - share-based: use share * total_columns to set dynamically.

    WHY THIS PARAMETER EXISTS:
        Drift detection sensitivity is a design decision, not a fixed rule.
        The right value depends on the system: a fraud model in finance
        wants high sensitivity; a recommendation engine can tolerate more.
    """
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)

    ref_ds = Dataset.from_pandas(reference)
    cur_ds = Dataset.from_pandas(current)

    report = Report([DataDriftPreset()])
    result = report.run(cur_ds, ref_ds)

    for metric in result.dict().get("metrics", []):
        name = metric.get("metric_name", "")
        value = metric.get("value", {})

        if "DriftedColumns" in name and isinstance(value, dict):
            count = value.get("count", 0)
            return count >= min_drifted_columns

    return False


def generate_drift_report(
    reference_path: str,
    current_path: str,
    output: str = "reports/drift_report.html",
) -> None:
    """Generate an HTML report comparing reference and current data."""
    from pathlib import Path

    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)

    ref_ds = Dataset.from_pandas(reference)
    cur_ds = Dataset.from_pandas(current)

    report = Report([DataDriftPreset()])
    result = report.run(cur_ds, ref_ds)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    result.save_html(output)
    print(f"Report saved to {output}")
