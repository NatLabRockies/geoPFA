import numpy as np
import pytest

from geopfa.extrapolation import build_and_fit_gp
from geopfa.spatial_lkx import LkxModel
from tests.fixtures.data_generators import generate_campbell2d_grid
from tests.fixtures.campbell2d import DEFAULT_THETA


@pytest.fixture
def small_training_set():
    _gdf, X, Y, Z_true, _Z_obs, _mask = generate_campbell2d_grid(
        nx=15, ny=15, theta=DEFAULT_THETA, missing_pattern="center_block"
    )

    xs = X.ravel()
    ys = Y.ravel()
    vals = Z_true.ravel()

    X_train = np.column_stack([xs, ys])
    Y_train = vals.reshape(-1, 1)

    Xm = X_train.mean(axis=0)
    Xs = X_train.std(axis=0)
    Ym = Y_train.mean()
    Ys = Y_train.std()

    return (
        (X_train - Xm) / Xs,
        (Y_train - Ym) / Ys,
    )


def test_gp_noise_constraints_respected(small_training_set):
    X_train_std, Y_train_std = small_training_set
    model, constraints = build_and_fit_gp(
        X_train_stdized=X_train_std,
        Y_train_stdized=Y_train_std,
        verbose=False,
    )

    assert isinstance(model, LkxModel)
    lambda_fit = float(constraints["lambda_fit"])
    assert lambda_fit > 0, f"lambda_fit should be positive, got {lambda_fit}"
    assert np.isfinite(lambda_fit), f"lambda_fit should be finite, got {lambda_fit}"
    assert "lambda_bounds" in constraints
