import numpy as np
import pytest

from geopfa.extrapolation import (
    build_rbf_kernel_global,
    build_matern32_kernel_global,
    build_combined_kernel,
)

from tests.fixtures.campbell2d import DEFAULT_THETA
from tests.fixtures.data_generators import generate_campbell2d_grid


@pytest.fixture
def small_training_data():
    """
    Returns standardized training data X_std and Y_std suitable for kernel testing.
    """
    _gdf, X, Y, Z_true, _Z_obs, _mask = generate_campbell2d_grid(
        nx=15, ny=15, theta=DEFAULT_THETA, missing_pattern="none"
    )

    xs = X.ravel()
    ys = Y.ravel()
    vals = Z_true.ravel()

    # Standardize manually for predictable tests
    X_train = np.column_stack([xs, ys])
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    X_std = (X_train - mean) / std
    Y_std = (vals - vals.mean()) / vals.std()

    return X_std, Y_std.reshape(-1, 1)


def test_rbf_kernel_bounds_and_init(small_training_data):
    X_std, Y_std = small_training_data

    kern, info = build_rbf_kernel_global(X_std, Y_std)

    # ARD lengthscales must be 2D
    assert kern.lengthscale.size == 2

    # Validate bounds
    lower = info["ls_lower"]
    upper = info["ls_upper"]
    init = info["ls_init"]

    assert lower > 0
    assert lower < init < upper

    # Variance init must reflect variance(Y_std)
    assert kern.variance.values[0] > 0

    # Ensure constraints applied
    assert kern.lengthscale.constraints is not None


def test_matern_kernel_bounds_and_init(small_training_data):
    X_std, Y_std = small_training_data

    kern, info = build_matern32_kernel_global(X_std, Y_std)

    assert kern.lengthscale.size == 2

    lower = info["ls_lower"]
    upper = info["ls_upper"]
    init = info["ls_init"]

    assert lower < init < upper


def test_combined_kernel_components(small_training_data):
    X_std, Y_std = small_training_data

    kernel, info = build_combined_kernel(
        X_std,
        Y_std,
        use_matern=True,
        use_rbf=True,
        bias=True,
        white=True,
        longscale=True,
    )

    # Should be a sum kernel containing multiple subkernels
    assert hasattr(kernel, "parts")
    assert len(kernel.parts) >= 3

    # Check long-scale kernel properties included
    assert "longscale" in info
    assert info["longscale"]["variance"] == 0.05
