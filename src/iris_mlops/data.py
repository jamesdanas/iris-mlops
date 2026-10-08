"""
Data loading and splitting for the Iris dataset.

WHY THIS MODULE EXISTS:
    Loading data is a separate concern from training. By isolating it here,
    we can later add validation, versioning, and swapping the source
    without touching the training code.
"""

from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Load the Iris dataset as (X, y).
    
    RETURNS:
        X: numpy array of shape (150, 4) - 150 samples, 4 features
        y: numpy array of shape (150,) - class labels 0, 1, 2
    
    WHY sklearn.datasets:
        It ships with the library, so no network call is needed.
        In production, this function would read from a database or S3.

    """
    # load_iris() returns a Bunch of object with .data and .target
    data = load_iris()

    # Return as plain numpy arrays for downstream compatibility
    return data.data, data.target


def split(
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """"
    Split data into train and test sets.
    
    WHY random_state=seed:
        Without it, every run produces a different split, making results
        non-reproducible. Fixing the seed means the same command always
        produces the same split.

    WHY test_size=0.2:
        80/20 is a common default. The test set is held out and only
        used for final evaluation - never for training or tuning.
    """
    # stratify=y guarantees identical class proportions between train and test sets
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

def save_processed(X: np.ndarray, y: np.ndarray, path: Path) -> None:
    """
    Save the datasets to CSV for DVC tracking.

    WHY CSV:
        DVC works best with files it can hash. CSV is human-readable and
        diff-friendly, unlike pickle or parquet for small datasets.
    """
    import pandas as pd

    # Ensure the parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert 1D target arrays safely if passed individually
    y_2d = np.atleast_1d(y)
    if y_2d.ndim == 1:
        y_2d = y_2d.reshape(-1, 1)

    # Handle both full datasets and split fragments safely
    if X.ndim == 1:
        X = X.reshape(1, -1)

    # Build a DataFrame with named columns
    # Makes validation schemas more readable.

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target"] = y_2d.flatten()

    # index=False: don't write the pandas row numbers as a column
    df.to_csv(path, index=False)
     
