"""
Tests for the data module.
WHY TESTS:
    Tests are executable claims. When someone refactors load_data(),
    the tests catch regressions (things that used to work and now don't).
"""

import numpy as np

from iris_mlops.data import load_data, split


def test_load_data_shapes():
    """The Iris dataset must always be 150 samples x 4 features."""
    X, y = load_data()
    assert X.shape == (150, 4), f"Expected (150, 4), got {X.shape}"
    assert y.shape == (150, ), f"Expected (150,), got {y.shape}"


def test_load_data_classes():
    "Iris must have exactly 3 classes: 0, 1, 2."
    _, y = load_data()
    assert set(np.unique(y)) == {0, 1, 2}


def test_split_deterministic():
    """
    WHY: Two calls with the same seed must produce identical splits.
    Without this, results are non-reproducible. 
    """
    X, y = load_data()
    a = split(X, y, seed=42)
    b = split(X, y, seed=42)
    np.testing.assert_array_equal(a[0], b[0])


def test_split_differs_by_seed():
    """Different seeds must produce different splits - proving seed matters."""
    X, y = load_data()
    a = split(X, y, seed=42)
    b = split(X, y, seed=7)
    assert not np.array_equal(a[0], b[0])


def test_split_sizes():
    """80/20 split on 150 samples = 120 train, 30 test."""
    X, y = load_data()
    X_train, X_test, _, _ = split(X, y, test_size=0.2)
    assert len(X_train) == 120
    assert len(X_test) == 30

def test_split_stratification():
    """Proves that stratified splitting preserves the exact class distributions.

    With 30 test samples and 3 balanced classes, each class must have exactly
    10 samples.
    """
    X, y = load_data()
    _, _, _, y_test = split(X, y, test_size=0.2, seed=42)
    _, counts = np.unique(y_test, return_counts=True)

    # Assert that all 3 classes have exactly 10 representation counts
    np.testing.assert_array_equal(counts, [10, 10, 10])