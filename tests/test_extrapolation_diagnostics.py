"""Tests for model-agnostic diagnostics over latticekrigx (P3-S02).

Verifies that ``assess_gp_model_fit``, ``bootstrap_assess_residuals_stats``,
and ``check_param_limits_hit_from_constraints`` work correctly when the
underlying spatial model is an :class:`~geopfa.spatial_lkx.LkxModel`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geopfa.extrapolation import (
    assess_gp_model_fit,
    bootstrap_assess_residuals_stats,
    check_param_limits_hit_from_constraints,
    build_and_fit_gp,
    get_predictions,
)
from geopfa.spatial_lkx import LkxModel, LkxConfig, fit_lkx_field


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_smooth_2d(n: int = 60, seed: int = 42) -> tuple:
    """Return standardized (X, Y) from a smooth 2-D sine field."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 2))
    Y = (np.sin(np.pi * X[:, 0]) * np.cos(np.pi * X[:, 1])).reshape(-1, 1)
    Y_std_val = float(Y.std()) or 1.0
    Y_stdized = ((Y - Y.mean()) / Y_std_val).ravel()
    return X.astype(np.float64), Y_stdized.astype(np.float64)


@pytest.fixture(scope="module")
def lkx_fit():
    """Fitted LkxModel + constraint_info on the standard 2D test field."""
    X, Y = _make_smooth_2d()
    model, ci = build_and_fit_gp(X, Y, backend="latticekrigx")
    return model, ci, X, Y


# ---------------------------------------------------------------------------
# assess_gp_model_fit on latticekrigx predictions
# ---------------------------------------------------------------------------


def test_assess_gp_model_fit_lkx_returns_dict(lkx_fit):
    model, ci, X, Y = lkx_fit
    Y_pred, Y_std = get_predictions(model, X, backend="latticekrigx")
    result = assess_gp_model_fit(model, Y, Y_pred, Y_std, ci)
    assert isinstance(result, dict)


def test_assess_gp_model_fit_lkx_expected_keys(lkx_fit):
    model, ci, X, Y = lkx_fit
    Y_pred, Y_std = get_predictions(model, X, backend="latticekrigx")
    result = assess_gp_model_fit(model, Y, Y_pred, Y_std, ci)
    for key in ("RMSE", "R2", "MAE", "Coverage_95", "LogLikelihood", "AIC", "BIC", "Params_at_bounds"):
        assert key in result, f"Missing key: {key}"


def test_assess_gp_model_fit_lkx_metrics_finite(lkx_fit):
    model, ci, X, Y = lkx_fit
    Y_pred, Y_std = get_predictions(model, X, backend="latticekrigx")
    result = assess_gp_model_fit(model, Y, Y_pred, Y_std, ci)
    for key in ("RMSE", "R2", "MAE", "Coverage_95", "LogLikelihood", "AIC", "BIC"):
        assert np.isfinite(result[key]), f"{key} is not finite: {result[key]}"


def test_assess_gp_model_fit_lkx_rmse_non_negative(lkx_fit):
    model, ci, X, Y = lkx_fit
    Y_pred, Y_std = get_predictions(model, X, backend="latticekrigx")
    result = assess_gp_model_fit(model, Y, Y_pred, Y_std, ci)
    assert result["RMSE"] >= 0.0


def test_assess_gp_model_fit_lkx_coverage_in_unit_interval(lkx_fit):
    model, ci, X, Y = lkx_fit
    Y_pred, Y_std = get_predictions(model, X, backend="latticekrigx")
    result = assess_gp_model_fit(model, Y, Y_pred, Y_std, ci)
    assert 0.0 <= result["Coverage_95"] <= 1.0


def test_assess_gp_model_fit_lkx_log_likelihood_matches_constraint_info(lkx_fit):
    model, ci, X, Y = lkx_fit
    Y_pred, Y_std = get_predictions(model, X, backend="latticekrigx")
    result = assess_gp_model_fit(model, Y, Y_pred, Y_std, ci)
    assert result["LogLikelihood"] == pytest.approx(ci["lnProfileLike"], rel=1e-6)


def test_assess_gp_model_fit_lkx_params_at_bounds_is_list(lkx_fit):
    model, ci, X, Y = lkx_fit
    Y_pred, Y_std = get_predictions(model, X, backend="latticekrigx")
    result = assess_gp_model_fit(model, Y, Y_pred, Y_std, ci)
    assert isinstance(result["Params_at_bounds"], list)


def test_assess_gp_model_fit_lkx_mle_lambda_k_equals_2(lkx_fit):
    """AIC/BIC use k=2 when lambda was MLE-optimised."""
    model, ci, X, Y = lkx_fit
    Y_pred, Y_std = get_predictions(model, X, backend="latticekrigx")
    assert ci.get("mle") == "lambda"
    result = assess_gp_model_fit(model, Y, Y_pred, Y_std, ci)
    logL = result["LogLikelihood"]
    assert result["AIC"] == pytest.approx(2 * 2 - 2 * logL, rel=1e-6)


# ---------------------------------------------------------------------------
# bootstrap_assess_residuals_stats — already model-agnostic; verify on lkx
# ---------------------------------------------------------------------------


def test_bootstrap_residuals_lkx_returns_dataframe(lkx_fit):
    model, ci, X, Y = lkx_fit
    Y_pred, _ = get_predictions(model, X, backend="latticekrigx")
    result = bootstrap_assess_residuals_stats(
        Y, Y_pred, n_boot=10, sample_size=30, random_state=0
    )
    assert isinstance(result, pd.DataFrame)


def test_bootstrap_residuals_lkx_expected_rows(lkx_fit):
    model, ci, X, Y = lkx_fit
    Y_pred, _ = get_predictions(model, X, backend="latticekrigx")
    result = bootstrap_assess_residuals_stats(
        Y, Y_pred, n_boot=10, sample_size=30, random_state=0
    )
    expected_tests = {"Shapiro-Wilk", "D'Agostino", "Jarque-Bera", "Levene", "Ljung-Box"}
    assert set(result["Test"]) == expected_tests


def test_bootstrap_residuals_lkx_columns_present(lkx_fit):
    model, ci, X, Y = lkx_fit
    Y_pred, _ = get_predictions(model, X, backend="latticekrigx")
    result = bootstrap_assess_residuals_stats(
        Y, Y_pred, n_boot=10, sample_size=30, random_state=0
    )
    for col in ("Mean p-value", "Median p-value", "Rejection Rate", "Warn"):
        assert col in result.columns


# ---------------------------------------------------------------------------
# check_param_limits_hit_from_constraints — LkxModel adapter
# ---------------------------------------------------------------------------


def test_check_param_limits_lkx_returns_list(lkx_fit):
    model, ci, X, Y = lkx_fit
    result = check_param_limits_hit_from_constraints(model, ci)
    assert isinstance(result, list)


def test_check_param_limits_lkx_lambda_bound_info_stored(lkx_fit):
    """constraint_info must carry lambda_bounds for the check to work."""
    _, ci, _, _ = lkx_fit
    assert "lambda_bounds" in ci
    lo, hi = ci["lambda_bounds"]
    assert lo < hi


def test_check_param_limits_lkx_at_lower_lambda_detected():
    """Manually construct a constraint_info where lambda is at its lower bound."""
    rng = np.random.default_rng(7)
    X = rng.uniform(-1.0, 1.0, size=(30, 2))
    Y = rng.standard_normal(30)

    lo = 1e-5
    hi = 0.10
    pinned_ci = {
        "geometry": "LKRectangle",
        "ndim": 2,
        "n_train": 30,
        "config": {"find_lambda": True, "find_a_wght": False},
        "mle": "lambda",
        "lambda_fit": lo,
        "lambda_bounds": (lo, hi),
        "a_wght_fit": None,
        "a_wght_lower_bound": 4.01,
        "sigma2_MLE": 1.0,
        "lnProfileLike": -50.0,
    }
    cfg = LkxConfig(nlevel=3, NC=4, lambda_=lo, lambda_bounds=(lo, hi))
    model = fit_lkx_field(X, Y, config=cfg)
    model.constraint_info.update(
        lambda_fit=lo,
        lambda_bounds=(lo, hi),
    )
    hits = check_param_limits_hit_from_constraints(model, model.constraint_info)
    names = [h[0] for h in hits]
    assert "lambda_" in names


def test_check_param_limits_lkx_no_hit_when_lambda_interior(lkx_fit):
    """When lambda is well inside bounds, no bound hit should be reported."""
    model, ci, X, Y = lkx_fit
    interior_ci = dict(ci)
    interior_ci["lambda_fit"] = 0.05
    interior_ci["lambda_bounds"] = (1e-5, 0.10)
    hits = check_param_limits_hit_from_constraints(model, interior_ci)
    lambda_hits = [h for h in hits if h[0] == "lambda_"]
    assert len(lambda_hits) == 0


def test_check_param_limits_lkx_a_wght_at_lower_bound_detected():
    """When a_wght_fit is near a_wght_lower_bound, it should be reported."""
    rng = np.random.default_rng(11)
    X = rng.uniform(-1.0, 1.0, size=(30, 2))
    Y = rng.standard_normal(30)
    cfg = LkxConfig(nlevel=3, NC=4, lambda_=0.01, find_a_wght=True)
    model = fit_lkx_field(X, Y, config=cfg)
    ci = dict(model.constraint_info)
    lb = ci.get("a_wght_lower_bound", 4.01)
    ci["a_wght_fit"] = [lb, lb, lb]
    hits = check_param_limits_hit_from_constraints(model, ci)
    names = [h[0] for h in hits]
    assert "a_wght" in names


def test_check_param_limits_lkx_a_wght_none_when_not_fitted():
    """When find_a_wght is False, a_wght_fit should be None → no a_wght hit."""
    rng = np.random.default_rng(13)
    X = rng.uniform(-1.0, 1.0, size=(30, 2))
    Y = rng.standard_normal(30)
    cfg = LkxConfig(nlevel=3, NC=4, lambda_=0.01, find_a_wght=False)
    model = fit_lkx_field(X, Y, config=cfg)
    hits = check_param_limits_hit_from_constraints(model, model.constraint_info)
    a_wght_hits = [h for h in hits if h[0] == "a_wght"]
    assert len(a_wght_hits) == 0


# ---------------------------------------------------------------------------
# constraint_info structure: lambda_bounds stored after fit
# ---------------------------------------------------------------------------


def test_fit_lkx_stores_lambda_bounds_in_constraint_info():
    rng = np.random.default_rng(99)
    X = rng.uniform(-1.0, 1.0, size=(40, 2))
    Y = rng.standard_normal(40)
    cfg = LkxConfig(nlevel=3, NC=4, lambda_bounds=(1e-4, 0.5), find_lambda=True)
    model = fit_lkx_field(X, Y, config=cfg)
    ci = model.constraint_info
    assert "lambda_bounds" in ci
    assert ci["lambda_bounds"] == pytest.approx((1e-4, 0.5), rel=1e-6)


def test_fit_lkx_stores_a_wght_fit_when_find_a_wght():
    rng = np.random.default_rng(77)
    X = rng.uniform(-1.0, 1.0, size=(40, 2))
    Y = rng.standard_normal(40)
    cfg = LkxConfig(nlevel=3, NC=4, find_a_wght=True)
    model = fit_lkx_field(X, Y, config=cfg)
    ci = model.constraint_info
    assert "a_wght_fit" in ci
    assert ci["a_wght_fit"] is not None
    assert len(ci["a_wght_fit"]) == cfg.nlevel


def test_fit_lkx_a_wght_fit_none_when_not_optimised():
    rng = np.random.default_rng(55)
    X = rng.uniform(-1.0, 1.0, size=(40, 2))
    Y = rng.standard_normal(40)
    cfg = LkxConfig(nlevel=3, NC=4, find_a_wght=False)
    model = fit_lkx_field(X, Y, config=cfg)
    assert model.constraint_info["a_wght_fit"] is None
