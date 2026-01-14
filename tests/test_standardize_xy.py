import numpy as np

from geopfa.extrapolation import standardize_xy


def test_standardize_xy_training_mean_zero():
    X_train = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    X_full = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=float)

    X_train_std, X_full_std, mean, std = standardize_xy(X_train, X_full)

    assert np.allclose(X_train_std.mean(axis=0), [0, 0], atol=1e-7)
    assert np.allclose(X_train_std.std(axis=0), [1, 1], atol=1e-7)


def test_standardize_xy_full_uses_train_stats():
    X_train = np.array([[0, 0], [10, 10]], dtype=float)
    X_full = np.array([[5, 5]], dtype=float)

    _, X_full_std, mean, std = standardize_xy(X_train, X_full)

    # Mean should be (5,5)
    assert np.allclose(mean, [5, 5])
    assert np.allclose(std, [5, 5])

    # Full should transform to (0,0)
    assert np.allclose(X_full_std, [[0, 0]])
