"""Core tests for the LatticeKrigX field adapter (P1-S01).

Acceptance criteria:

- finite ``(mean, std)`` of correct shape on a synthetic smooth 2-D
  field;
- recovers a held-out field better than a mean baseline;
- 3-D inputs run.
"""

from __future__ import annotations

import numpy as np
import pytest
from latticekrigx.model.predict_se import predict_se

from geopfa.exceptions import GEOPFAValueError
from geopfa.spatial_lkx import (
    LkxConfig,
    LkxModel,
    fit_lkx_field,
    lkx_predict,
)


def _smooth_field_2d(xy: np.ndarray) -> np.ndarray:
    x = xy[:, 0]
    y = xy[:, 1]
    return np.sin(1.5 * x) * np.cos(1.2 * y) + 0.25 * x - 0.15 * y


def _smooth_field_3d(xyz: np.ndarray) -> np.ndarray:
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    return np.sin(1.2 * x) * np.cos(0.9 * y) + 0.5 * np.tanh(0.8 * z)


def _standardize(a: np.ndarray, ref: np.ndarray | None = None):
    ref = a if ref is None else ref
    mu = ref.mean(axis=0)
    sd = ref.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (a - mu) / sd, mu, sd


@pytest.fixture(scope="module")
def synthetic_2d_split():
    rng = np.random.default_rng(0)
    n = 220
    xy = rng.uniform(-2.0, 2.0, size=(n, 2))
    y = _smooth_field_2d(xy) + rng.normal(0.0, 0.05, size=n)

    x_train_raw, x_test_raw = xy[:170], xy[170:]
    y_train_raw, y_test_raw = y[:170], y[170:]

    x_train_std, x_mu, x_sd = _standardize(x_train_raw)
    x_test_std = (x_test_raw - x_mu) / x_sd
    y_train_std, y_mu, y_sd = _standardize(y_train_raw)
    y_test_std = (y_test_raw - y_mu) / y_sd
    return {
        "x_train": x_train_std,
        "x_test": x_test_std,
        "y_train": y_train_std,
        "y_test": y_test_std,
    }


def test_fit_and_predict_finite_shapes_2d(synthetic_2d_split):
    data = synthetic_2d_split
    model = fit_lkx_field(
        data["x_train"],
        data["y_train"],
        config=LkxConfig(nlevel=2, NC=6, lambda_=0.05),
    )
    assert isinstance(model, LkxModel)
    assert model.ndim == 2
    assert model.geometry == "LKRectangle"

    mean, std = lkx_predict(model, data["x_test"])
    assert mean.shape == (data["x_test"].shape[0],)
    assert std.shape == (data["x_test"].shape[0],)
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(std))
    assert np.all(std >= 0.0)


def test_predictive_std_includes_fitted_observation_nugget(synthetic_2d_split):
    data = synthetic_2d_split
    model = fit_lkx_field(
        data["x_train"],
        data["y_train"],
        config=LkxConfig(nlevel=2, NC=6, lambda_=0.05),
    )
    assert model.fit is not None
    _, predictive_std = lkx_predict(model, data["x_test"])
    latent_std = predict_se(model.fit, data["x_test"])
    sigma2 = float(np.asarray(model.fit.sigma2_MLE).ravel()[0])
    observation_variance = model.fit.lambda_ * sigma2
    np.testing.assert_allclose(
        predictive_std**2,
        latent_std**2 + observation_variance,
        rtol=1e-12,
        atol=1e-12,
    )


def test_recovers_holdout_better_than_mean_baseline_2d(synthetic_2d_split):
    data = synthetic_2d_split
    model = fit_lkx_field(
        data["x_train"],
        data["y_train"],
        config=LkxConfig(nlevel=2, NC=6, lambda_=0.05),
    )
    mean, _ = lkx_predict(model, data["x_test"])
    rmse_model = float(np.sqrt(np.mean((mean - data["y_test"]) ** 2)))
    baseline = float(np.mean(data["y_train"]))
    rmse_mean = float(np.sqrt(np.mean((baseline - data["y_test"]) ** 2)))
    assert rmse_model < 0.5 * rmse_mean, (
        f"LKX RMSE {rmse_model} not better than mean baseline {rmse_mean}"
    )


def test_fit_and_predict_runs_3d():
    rng = np.random.default_rng(1)
    n = 60
    xyz = rng.uniform(-1.5, 1.5, size=(n, 3))
    y = _smooth_field_3d(xyz) + rng.normal(0.0, 0.05, size=n)

    x_train_raw, x_test_raw = xyz[:48], xyz[48:]
    y_train_raw = y[:48]

    x_train_std, x_mu, x_sd = _standardize(x_train_raw)
    x_test_std = (x_test_raw - x_mu) / x_sd
    y_train_std, _, _ = _standardize(y_train_raw)

    model = fit_lkx_field(
        x_train_std,
        y_train_std,
        config=LkxConfig(nlevel=1, NC=3, lambda_=0.1),
    )
    assert model.ndim == 3
    assert model.geometry == "LKBox"

    mean, std = lkx_predict(model, x_test_std)
    assert mean.shape == (x_test_std.shape[0],)
    assert std.shape == (x_test_std.shape[0],)
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(std))
    assert np.all(std >= 0.0)


def test_accepts_y_column_vector(synthetic_2d_split):
    data = synthetic_2d_split
    y_col = data["y_train"].reshape(-1, 1)
    model = fit_lkx_field(
        data["x_train"],
        y_col,
        config=LkxConfig(nlevel=2, NC=6, lambda_=0.05),
    )
    mean, std = lkx_predict(model, data["x_test"])
    assert mean.shape == (data["x_test"].shape[0],)
    assert std.shape == (data["x_test"].shape[0],)


def test_accepts_dict_config(synthetic_2d_split):
    data = synthetic_2d_split
    model = fit_lkx_field(
        data["x_train"],
        data["y_train"],
        config={"nlevel": 2, "NC": 6, "lambda_": 0.05},
    )
    assert model.constraint_info["config"]["NC"] == 6


def test_rejects_shape_mismatch(synthetic_2d_split):
    data = synthetic_2d_split
    with pytest.raises(GEOPFAValueError):
        fit_lkx_field(
            data["x_train"],
            data["y_train"][:-1],
            config=LkxConfig(nlevel=2, NC=6, lambda_=0.05),
        )


def test_rejects_predict_ndim_mismatch(synthetic_2d_split):
    data = synthetic_2d_split
    model = fit_lkx_field(
        data["x_train"],
        data["y_train"],
        config=LkxConfig(nlevel=2, NC=6, lambda_=0.05),
    )
    with pytest.raises(GEOPFAValueError):
        lkx_predict(model, np.zeros((5, 3)))


def test_rejects_unsupported_ndim():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(30, 4))
    y = rng.normal(size=30)
    with pytest.raises(GEOPFAValueError):
        fit_lkx_field(x, y, config=LkxConfig())


def test_fallback_too_few_training_points_returns_zero_field():
    rng = np.random.default_rng(3)
    x_train = rng.uniform(-1.0, 1.0, size=(3, 2))
    y_train = rng.normal(size=3)
    grid = rng.uniform(-1.0, 1.0, size=(25, 2))

    model = fit_lkx_field(x_train, y_train)
    assert model.fit is None
    assert model.constraint_info["fallback"] == "too_few_training_points"

    mean, std = lkx_predict(model, grid)
    assert mean.shape == (grid.shape[0],)
    assert std.shape == (grid.shape[0],)
    assert np.all(mean == 0.0)
    assert np.all(std == 0.0)


def test_fallback_constant_y_returns_zero_field():
    rng = np.random.default_rng(4)
    x_train = rng.uniform(-1.0, 1.0, size=(30, 2))
    y_train = np.full(30, 0.75)
    grid = rng.uniform(-1.0, 1.0, size=(40, 2))

    model = fit_lkx_field(x_train, y_train)
    assert model.fit is None
    assert model.constraint_info["fallback"] == "constant_Y"

    mean, std = lkx_predict(model, grid)
    assert mean.shape == (grid.shape[0],)
    assert np.all(mean == 0.0)
    assert np.all(std == 0.0)


def test_mle_improves_holdout_rmse_vs_fixed_lambda(synthetic_2d_split):
    data = synthetic_2d_split
    bad_cfg = LkxConfig(nlevel=2, NC=6, lambda_=5.0)
    mle_cfg = LkxConfig(nlevel=2, NC=6, lambda_=5.0, find_lambda=True)

    fixed_model = fit_lkx_field(
        data["x_train"], data["y_train"], config=bad_cfg
    )
    mle_model = fit_lkx_field(data["x_train"], data["y_train"], config=mle_cfg)

    fixed_mean, _ = lkx_predict(fixed_model, data["x_test"])
    mle_mean, _ = lkx_predict(mle_model, data["x_test"])

    rmse_fixed = float(np.sqrt(np.mean((fixed_mean - data["y_test"]) ** 2)))
    rmse_mle = float(np.sqrt(np.mean((mle_mean - data["y_test"]) ** 2)))

    assert mle_model.constraint_info["mle"] == "lambda"
    lambda_fit = mle_model.constraint_info["lambda_fit"]
    assert lambda_fit != pytest.approx(5.0)
    assert rmse_mle < rmse_fixed, (
        f"MLE RMSE {rmse_mle} did not improve on fixed-lambda {rmse_fixed}"
    )


def test_streaming_prediction_matches_unbatched(synthetic_2d_split):
    data = synthetic_2d_split
    model = fit_lkx_field(
        data["x_train"],
        data["y_train"],
        config=LkxConfig(nlevel=2, NC=6, lambda_=0.05),
    )

    rng = np.random.default_rng(5)
    grid = rng.uniform(-2.0, 2.0, size=(5000, 2))

    mean_full, std_full = lkx_predict(model, grid, chunk_size=None)
    mean_chunked, std_chunked = lkx_predict(model, grid, chunk_size=512)

    assert mean_chunked.shape == mean_full.shape
    np.testing.assert_allclose(mean_chunked, mean_full, atol=1e-8, rtol=0)
    np.testing.assert_allclose(std_chunked, std_full, atol=1e-8, rtol=0)
