"""Held-out quality checks for geoPFA's LatticeKrigX spatial wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from geopfa.spatial_lkx import (
    LkxConfig,
    compute_lkx_config,
    fit_lkx_field,
    lkx_predict,
)


NOMINAL_2SIGMA_COVERAGE = 0.9544997361036416
COVERAGE_TOLERANCE = 0.1
MAX_STANDARDIZED_RMSE = 0.75


def _smooth_field(coords: np.ndarray) -> np.ndarray:
    x = coords[:, 0]
    y = coords[:, 1]
    values = np.sin(1.5 * x) * np.cos(1.2 * y) + 0.25 * x - 0.15 * y
    if coords.shape[1] == 3:  # noqa: PLR2004
        values = values + 0.5 * np.tanh(0.8 * coords[:, 2])
    return values


def _standardize(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    scale = np.std(reference, axis=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    return (values - np.mean(reference, axis=0)) / scale


def _make_case(
    dimension: int, seed: int, n_train: int, n_test: int
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    raw = rng.uniform(-2.0, 2.0, size=(n_train + n_test, dimension))
    response = _smooth_field(raw) + rng.normal(
        0.0, 0.05, size=n_train + n_test
    )
    train_raw, test_raw = raw[:n_train], raw[n_train:]
    y_train_raw, y_test_raw = response[:n_train], response[n_train:]
    return {
        "x_train": _standardize(train_raw, train_raw),
        "x_test": _standardize(test_raw, train_raw),
        "y_train": _standardize(y_train_raw, y_train_raw),
        "y_test": _standardize(y_test_raw, y_train_raw),
    }


@pytest.fixture(
    scope="module",
    params=[
        pytest.param((2, 20240724, 200, 60), id="smooth_2d"),
        pytest.param((3, 20240725, 190, 50), id="smooth_3d"),
    ],
)
def parity_case(request):
    fixture = _make_case(*request.param)
    cfg = compute_lkx_config(
        fixture["x_train"], n_train=fixture["x_train"].shape[0]
    )
    cfg = LkxConfig(
        nlevel=cfg.nlevel,
        NC=cfg.NC,
        lambda_=cfg.lambda_,
        lambda_bounds=cfg.lambda_bounds,
        find_lambda=True,
    )
    model = fit_lkx_field(fixture["x_train"], fixture["y_train"], config=cfg)
    mean, std = lkx_predict(model, fixture["x_test"])
    residual = mean - fixture["y_test"]
    return {
        "fixture": fixture,
        "mean": mean,
        "std": std,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "coverage": float(np.mean(np.abs(residual) <= 2.0 * std)),
    }


def test_lkx_predictions_are_finite(parity_case) -> None:
    y_test = parity_case["fixture"]["y_test"]
    assert parity_case["mean"].shape == y_test.shape
    assert parity_case["std"].shape == y_test.shape
    assert np.all(np.isfinite(parity_case["mean"]))
    assert np.all(np.isfinite(parity_case["std"]))
    assert np.all(parity_case["std"] >= 0.0)


def test_lkx_held_out_rmse_is_below_signal_scale(parity_case) -> None:
    assert parity_case["rmse"] < MAX_STANDARDIZED_RMSE


def test_lkx_uncertainty_has_nominal_held_out_coverage(parity_case) -> None:
    delta = abs(parity_case["coverage"] - NOMINAL_2SIGMA_COVERAGE)
    assert delta <= COVERAGE_TOLERANCE
