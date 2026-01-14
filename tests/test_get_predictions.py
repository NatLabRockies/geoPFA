import numpy as np
import pytest

from geopfa.extrapolation import get_predictions, build_and_fit_gp
from tests.fixtures.data_generators import generate_campbell2d_grid
from tests.fixtures.campbell2d import DEFAULT_THETA


@pytest.fixture
def small_trained_model():
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

    X_train_std = (X_train - Xm) / Xs
    Y_train_std = (Y_train - Ym) / Ys

    model, _constraints = build_and_fit_gp(
        X_train_stdized=X_train_std,
        Y_train_stdized=Y_train_std,
        verbose=False,
        optimize_restarts=0,
        n_inducing=40,
    )

    X_full_std = (X_train - Xm) / Xs

    return model, X_train, Y_train, X_train_std, Y_train_std, X_full_std, X, Y


def test_get_predictions_shapes(small_trained_model):
    (
        model,
        _X_train,
        Y_train,
        _X_train_std,
        _Y_train_std,
        X_full_std,
        _X_grid,
        _Y_grid,
    ) = small_trained_model

    preds, vars_ = get_predictions(
        model=model,
        X=X_full_std,
        Y_mean=Y_train.mean(),
        Y_std=Y_train.std(),
    )

    assert preds.shape[0] == X_full_std.shape[0]
    assert vars_.shape[0] == X_full_std.shape[0]


def test_get_predictions_flat_correct(small_trained_model):
    (
        model,
        _X_train,
        Y_train,
        _X_train_std,
        _Y_train_std,
        X_full_std,
        _X_grid,
        _Y_grid,
    ) = small_trained_model

    preds, vars_ = get_predictions(
        model=model,
        X=X_full_std,
        Y_mean=Y_train.mean(),
        Y_std=Y_train.std(),
    )

    assert len(preds) == X_full_std.shape[0]


def test_get_predictions_variance_nonnegative(small_trained_model):
    model, *_, X_full_std, _X_grid, _Y_grid = small_trained_model

    _preds, vars_ = get_predictions(
        model=model,
        X=X_full_std,
        Y_mean=0.0,
        Y_std=1.0,
    )

    assert np.all(vars_ >= 0.0)
