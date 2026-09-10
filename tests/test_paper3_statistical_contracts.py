"""Scientific regression contracts introduced for Paper 3."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize as scipy_minimize
from scipy.special import expit
from shapely.geometry import Point
from sklearn.linear_model import LogisticRegression

from geopfa.prob.config import (
    InferenceConfig,
    LabelsConfig,
    SiteSelectionConfig,
)
from geopfa.prob.cv import spatial_block_cv
from geopfa.prob import gblk_runner
from geopfa.prob import pu as pu_module
from geopfa.prob import site_selection as site_selection_module
from geopfa.prob.spatial_alignment import (
    grid_footprint_mask,
    sample_layer_at_points,
    snap_to_grid_indices,
)
from geopfa.prob.gblk_backend import build_lkinfo
from geopfa.prob.pu import fit_nnpu_logistic
from geopfa.prob.site_selection import (
    fit_site_selection_model,
    run_site_selection_analysis,
    site_selection_sensitivity,
)


def test_three_dimensional_snapping_uses_depth() -> None:
    grid = gpd.GeoDataFrame(
        {
            "value": [10.0, 90.0],
            "geometry": [
                Point(500_000.0, 4_800_000.0, 0.0),
                Point(500_000.0, 4_800_000.0, 100.0),
            ],
        },
        crs="EPSG:32611",
    )
    wells = gpd.GeoDataFrame(
        {"geometry": [Point(500_000.0, 4_800_000.0, 91.0)]},
        crs=grid.crs,
    )

    np.testing.assert_array_equal(
        snap_to_grid_indices(wells, grid), np.array([1])
    )
    np.testing.assert_allclose(
        sample_layer_at_points(wells, grid, "value"), np.array([90.0])
    )


def test_snapping_rejects_points_outside_the_grid_cell_footprint() -> None:
    grid = gpd.GeoDataFrame(
        {
            "value": [0.0, 1.0, 2.0, 3.0],
            "geometry": [
                Point(0.0, 0.0),
                Point(1.0, 0.0),
                Point(0.0, 1.0),
                Point(1.0, 1.0),
            ],
        },
        crs="EPSG:32611",
    )
    outside = gpd.GeoDataFrame({"geometry": [Point(1.51, 0.5)]}, crs=grid.crs)

    with pytest.raises(ValueError, match="outside the grid cell footprint"):
        snap_to_grid_indices(outside, grid)


def test_snapping_accepts_points_inside_the_outer_half_cell() -> None:
    grid = gpd.GeoDataFrame(
        {
            "geometry": [
                Point(0.0, 0.0),
                Point(1.0, 0.0),
                Point(0.0, 1.0),
                Point(1.0, 1.0),
            ]
        },
        crs="EPSG:32611",
    )
    inside = gpd.GeoDataFrame({"geometry": [Point(1.49, 0.5)]}, crs=grid.crs)

    np.testing.assert_array_equal(snap_to_grid_indices(inside, grid), [1])


def test_grid_footprint_mask_identifies_analysis_support_without_clipping() -> (
    None
):
    grid = gpd.GeoDataFrame(
        {
            "geometry": [
                Point(0.0, 0.0),
                Point(1.0, 0.0),
                Point(0.0, 1.0),
                Point(1.0, 1.0),
            ]
        },
        crs="EPSG:32611",
    )
    points = gpd.GeoDataFrame(
        {
            "geometry": [
                Point(0.25, 0.25),
                Point(1.49, 0.5),
                Point(1.51, 0.5),
            ]
        },
        crs=grid.crs,
    )

    np.testing.assert_array_equal(
        grid_footprint_mask(points, grid),
        np.array([True, True, False]),
    )


def test_volumetric_lattice_is_multiresolution() -> None:
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.5],
            [1.0, 1.0, 1.0],
        ]
    )

    lkinfo = build_lkinfo(coords, nc=3, nlevel=2, a_wght=6.5)

    assert lkinfo.geometry.geometry_type == "LKBox"
    assert lkinfo.nlevel == 2
    assert len(lkinfo.levels) == 2
    assert lkinfo.levels[1].NC > lkinfo.levels[0].NC
    assert lkinfo.levels[1].delta < lkinfo.levels[0].delta


def test_gblk_runner_uses_stable_dimension_specific_defaults_and_honors_override() -> (
    None
):
    assert gblk_runner._resolve_a_wght(2, None) == 4.5  # noqa: SLF001
    assert gblk_runner._resolve_a_wght(3, None) == 8.0  # noqa: SLF001
    assert gblk_runner._resolve_a_wght(3, 9.25) == 9.25  # noqa: SLF001


def test_nnpu_requires_an_explicit_class_prior() -> None:
    x = np.column_stack([np.ones(8), np.linspace(-1.0, 1.0, 8)])
    s = np.array([1, 0, 0, 1, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="class_prior"):
        fit_nnpu_logistic(x, s, class_prior=None)


@pytest.mark.parametrize("bad_prior", [True, "0.3", 0.3 + 0.0j])
def test_nnpu_rejects_coerced_class_priors(bad_prior: object) -> None:
    x = np.column_stack([np.ones(8), np.linspace(-1.0, 1.0, 8)])
    s = np.array([1, 0, 0, 1, 0, 0, 0, 0])

    with pytest.raises(TypeError, match="class_prior"):
        fit_nnpu_logistic(
            x,
            s,
            class_prior=bad_prior,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("features", "offsets", "message"),
    [
        (
            np.array([["1.0", "-1.0"], ["1.0", "1.0"]]),
            None,
            "features",
        ),
        (
            np.array([[1.0, -1.0], [1.0, 1.0]]),
            np.array(["0.0", "0.0"]),
            "offsets",
        ),
    ],
)
def test_nnpu_rejects_coerced_design_inputs(
    features: np.ndarray,
    offsets: np.ndarray | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        fit_nnpu_logistic(
            features,
            np.array([1, 0]),
            class_prior=0.3,
            offsets=offsets,
        )


@pytest.mark.parametrize(
    ("keyword", "bad_value", "message"),
    [
        ("penalty_weights", "0.1", "penalty_weights"),
        ("penalty_weights", np.array([0.0, "0.1"]), "penalty_weights"),
        ("prior_means", np.array([0.0, 0.0 + 1.0j]), "prior_means"),
    ],
)
def test_nnpu_rejects_coerced_penalty_inputs(
    keyword: str,
    bad_value: object,
    message: str,
) -> None:
    x = np.column_stack([np.ones(8), np.linspace(-1.0, 1.0, 8)])
    s = np.array([1, 0, 0, 1, 0, 0, 0, 0])

    with pytest.raises(ValueError, match=message):
        fit_nnpu_logistic(
            x,
            s,
            class_prior=0.3,
            **{keyword: bad_value},
        )


@pytest.mark.parametrize("bad_max_iter", [True, 2.5, "10"])
def test_nnpu_rejects_coerced_iteration_limits(bad_max_iter: object) -> None:
    x = np.column_stack([np.ones(8), np.linspace(-1.0, 1.0, 8)])
    s = np.array([1, 0, 0, 1, 0, 0, 0, 0])

    with pytest.raises(
        ValueError, match="max_iter must be a positive integer"
    ):
        fit_nnpu_logistic(
            x,
            s,
            class_prior=0.3,
            max_iter=bad_max_iter,  # type: ignore[arg-type]
        )


def test_nnpu_improves_probability_recovery_under_scar() -> None:
    rng = np.random.default_rng(20260904)
    n = 8_000
    x_raw = rng.normal(size=n)
    x = np.column_stack([np.ones(n), x_raw])
    p_true = expit(-0.8 + 1.7 * x_raw)
    y = rng.binomial(1, p_true)
    selected_positive = (y == 1) & (rng.random(n) < 0.30)
    s = selected_positive.astype(int)

    nnpu = fit_nnpu_logistic(
        x,
        s,
        class_prior=float(p_true.mean()),
        penalty_weights=np.array([0.0, 1e-4]),
    )
    naive = LogisticRegression(C=10_000.0).fit(x_raw[:, None], s)

    p_nnpu = expit(x @ nnpu.coef)
    p_naive = naive.predict_proba(x_raw[:, None])[:, 1]
    nnpu_brier = float(np.mean((p_nnpu - y) ** 2))
    naive_brier = float(np.mean((p_naive - y) ** 2))

    assert nnpu.success
    assert nnpu.negative_risk >= -1e-10
    assert nnpu_brier < naive_brier - 0.03


def test_nnpu_analytic_gradient_matches_empirical_risk(monkeypatch) -> None:
    """The optimized nnPU gradient must differentiate the disclosed objective."""
    rng = np.random.default_rng(992)
    x = np.column_stack([np.ones(300), rng.normal(size=(300, 2))])
    observed = np.zeros(300, dtype=int)
    observed[rng.choice(300, size=55, replace=False)] = 1

    def checked_minimize(fun, x0, *, jac, method, options):
        point = np.array([0.15, -0.3, 0.4])
        step = 1e-6
        numeric = np.array(
            [
                (
                    fun(point + step * np.eye(3)[index])
                    - fun(point - step * np.eye(3)[index])
                )
                / (2.0 * step)
                for index in range(3)
            ]
        )
        np.testing.assert_allclose(jac(point), numeric, rtol=2e-5, atol=2e-6)
        return scipy_minimize(fun, x0, jac=jac, method=method, options=options)

    monkeypatch.setattr(pu_module, "minimize", checked_minimize)
    result = fit_nnpu_logistic(
        x,
        observed,
        class_prior=0.35,
        penalty_weights=np.array([0.2, 0.4, 0.6]),
    )
    assert result.success


def test_labels_config_nnpu_requires_class_prior() -> None:
    raw = {
        "source": "labels.gpkg",
        "id_col": "site_id",
        "label_columns": {"heat": "heat_label"},
        "pu_mode": "nnpu",
    }
    with pytest.raises(ValueError, match="pu_class_prior"):
        LabelsConfig.from_dict(raw)


@pytest.mark.parametrize(
    "bad_prior",
    [True, "0.3", 0.3 + 0.0j, {"heat": "0.3"}],
)
def test_labels_config_rejects_coerced_nnpu_class_priors(
    bad_prior: object,
) -> None:
    raw = {
        "source": "labels.gpkg",
        "id_col": "site_id",
        "label_columns": {"heat": "heat_label"},
        "pu_mode": "nnpu",
        "pu_class_prior": bad_prior,
    }

    with pytest.raises(ValueError, match="pu_class_prior"):
        LabelsConfig.from_dict(raw)


def test_joint_site_selection_likelihood_reduces_preferential_sampling_bias() -> (
    None
):
    rng = np.random.default_rng(78)
    n = 6_000
    x_raw = rng.normal(size=n)
    q_raw = rng.normal(size=n)
    x = x_raw[:, None]
    q = q_raw[:, None]
    p_true = expit(-0.7 + 1.25 * x_raw)
    y = rng.binomial(1, p_true)
    delta = 1.8
    p_select = expit(-1.0 + 0.8 * q_raw + delta * y)
    selected = rng.binomial(1, p_select).astype(bool)
    observed_y = np.where(selected, y, np.nan)

    fitted = fit_site_selection_model(
        outcome_features=x,
        selection_features=q,
        selected=selected,
        observed_outcome=observed_y,
        outcome_selection_log_odds=delta,
        outcome_penalty=1e-4,
        selection_penalty=1e-4,
    )
    naive = LogisticRegression(C=10_000.0).fit(x[selected], y[selected])

    p_joint = fitted.predict_outcome(x)
    p_naive = naive.predict_proba(x)[:, 1]
    error_joint = float(np.mean((p_joint - p_true) ** 2))
    error_naive = float(np.mean((p_naive - p_true) ** 2))

    assert fitted.success
    assert error_joint < error_naive * 0.5


def test_site_selection_analytic_gradient_matches_marginal_likelihood(
    monkeypatch,
) -> None:
    """Selected and marginalized candidate terms must share one exact gradient."""
    rng = np.random.default_rng(993)
    x = rng.normal(size=(240, 2))
    q = rng.normal(size=(240, 1))
    y = rng.binomial(1, expit(-0.2 + 0.6 * x[:, 0]))
    selected = rng.random(240) < expit(-0.6 + 0.4 * q[:, 0] + 0.9 * y)
    observed = np.where(selected, y, np.nan)

    def checked_minimize(fun, x0, *, jac, method, options):
        point = np.array([0.1, -0.2, 0.25, -0.15, 0.35])
        step = 1e-6
        directions = np.eye(point.size)
        numeric = np.array(
            [
                (fun(point + step * direction) - fun(point - step * direction))
                / (2.0 * step)
                for direction in directions
            ]
        )
        np.testing.assert_allclose(jac(point), numeric, rtol=2e-5, atol=2e-6)
        return scipy_minimize(fun, x0, jac=jac, method=method, options=options)

    monkeypatch.setattr(site_selection_module, "minimize", checked_minimize)
    result = fit_site_selection_model(
        outcome_features=x,
        selection_features=q,
        selected=selected,
        observed_outcome=observed,
        outcome_selection_log_odds=0.9,
        outcome_penalty=0.2,
        selection_penalty=0.3,
    )
    assert result.success
    assert (
        result.log_likelihood > result.diagnostics["penalized_log_objective"]
    )


def test_site_selection_gradient_remains_exact_for_extreme_predictors(
    monkeypatch,
) -> None:
    """Numerical safeguards must not change the optimized likelihood target."""
    rng = np.random.default_rng(1993)
    x = 8.0 * rng.normal(size=(180, 2))
    q = 8.0 * rng.normal(size=(180, 1))
    y = rng.binomial(1, expit(-0.2 + 0.1 * x[:, 0]))
    selected = rng.random(180) < expit(-0.7 + 0.08 * q[:, 0] + 0.8 * y)
    observed = np.where(selected, y, np.nan)

    def checked_minimize(fun, x0, *, jac, method, options):
        point = np.array([35.0, -22.0, 18.0, -30.0, 25.0])
        step = 2e-5
        directions = np.eye(point.size)
        numeric = np.array(
            [
                (fun(point + step * direction) - fun(point - step * direction))
                / (2.0 * step)
                for direction in directions
            ]
        )
        analytic = jac(point)
        assert np.all(np.isfinite(numeric))
        assert np.all(np.isfinite(analytic))
        np.testing.assert_allclose(analytic, numeric, rtol=2e-5, atol=2e-5)
        return scipy_minimize(fun, x0, jac=jac, method=method, options=options)

    monkeypatch.setattr(site_selection_module, "minimize", checked_minimize)
    result = fit_site_selection_model(
        outcome_features=x,
        selection_features=q,
        selected=selected,
        observed_outcome=observed,
        outcome_selection_log_odds=0.8,
        outcome_penalty=0.2,
        selection_penalty=0.3,
    )
    assert result.success


def test_site_selection_requires_a_complete_candidate_frame() -> None:
    x = np.arange(6, dtype=float)[:, None]
    selected = np.array([True, True, False, False, True, False])
    outcome = np.array([1.0, 0.0, np.nan, np.nan, 1.0, np.nan])
    with pytest.raises(ValueError, match="selection_features"):
        fit_site_selection_model(
            outcome_features=x,
            selection_features=None,
            selected=selected,
            observed_outcome=outcome,
            outcome_selection_log_odds=1.0,
        )


@pytest.mark.parametrize(
    ("outcome_features", "selection_features", "message"),
    [
        (
            np.array([["0.0"], ["1.0"], ["2.0"], ["3.0"]]),
            np.arange(4, dtype=float)[:, None],
            "outcome_features",
        ),
        (
            np.arange(4, dtype=float)[:, None],
            np.array([["0.0"], ["1.0"], ["2.0"], ["3.0"]]),
            "selection_features",
        ),
    ],
)
def test_site_selection_rejects_coerced_design_inputs(
    outcome_features: np.ndarray,
    selection_features: np.ndarray,
    message: str,
) -> None:
    selected = np.array([True, True, False, False])
    outcome = np.array([1.0, 0.0, np.nan, np.nan])

    with pytest.raises(ValueError, match=message):
        fit_site_selection_model(
            outcome_features=outcome_features,
            selection_features=selection_features,
            selected=selected,
            observed_outcome=outcome,
            outcome_selection_log_odds=1.0,
        )


@pytest.mark.parametrize(
    ("selected", "observed_outcome", "message"),
    [
        (
            np.array([1.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
            np.array([1.0, 0.0, np.nan, np.nan]),
            "selected",
        ),
        (
            np.array([True, True, False, False]),
            np.array(["1", "0", "nan", "nan"]),
            "observed_outcome",
        ),
    ],
)
def test_site_selection_rejects_coerced_indicator_inputs(
    selected: np.ndarray,
    observed_outcome: np.ndarray,
    message: str,
) -> None:
    x = np.arange(4, dtype=float)[:, None]

    with pytest.raises(ValueError, match=message):
        fit_site_selection_model(
            outcome_features=x,
            selection_features=x,
            selected=selected,
            observed_outcome=observed_outcome,
            outcome_selection_log_odds=1.0,
        )


@pytest.mark.parametrize("bad_max_iter", [True, 2.5, "10"])
def test_site_selection_rejects_coerced_iteration_limits(
    bad_max_iter: object,
) -> None:
    x = np.arange(6, dtype=float)[:, None]
    selected = np.array([True, True, False, False, True, False])
    outcome = np.array([1.0, 0.0, np.nan, np.nan, 1.0, np.nan])

    with pytest.raises(
        ValueError, match="max_iter must be a positive integer"
    ):
        fit_site_selection_model(
            outcome_features=x,
            selection_features=x,
            selected=selected,
            observed_outcome=outcome,
            outcome_selection_log_odds=1.0,
            max_iter=bad_max_iter,  # type: ignore[arg-type]
        )


def test_site_selection_requires_missing_unselected_outcomes() -> None:
    x = np.arange(6, dtype=float)[:, None]
    selected = np.array([True, True, False, False, True, False])
    outcome = np.array([1.0, 0.0, np.inf, np.nan, 1.0, np.nan])

    with pytest.raises(ValueError, match="unselected.*missing"):
        fit_site_selection_model(
            outcome_features=x,
            selection_features=x,
            selected=selected,
            observed_outcome=outcome,
            outcome_selection_log_odds=1.0,
        )


@pytest.mark.parametrize("bad_delta", [True, "1.0"])
def test_site_selection_rejects_coerced_sensitivity_values(
    bad_delta: object,
) -> None:
    x = np.arange(6, dtype=float)[:, None]
    selected = np.array([True, True, False, False, True, False])
    outcome = np.array([1.0, 0.0, np.nan, np.nan, 1.0, np.nan])

    with pytest.raises(
        (TypeError, ValueError), match="outcome_selection_log_odds"
    ):
        fit_site_selection_model(
            outcome_features=x,
            selection_features=x,
            selected=selected,
            observed_outcome=outcome,
            outcome_selection_log_odds=bad_delta,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("bad_delta", [True, "1.0"])
def test_site_selection_sensitivity_grid_rejects_coerced_values(
    bad_delta: object,
) -> None:
    x = np.arange(6, dtype=float)[:, None]
    selected = np.array([True, True, False, False, True, False])
    outcome = np.array([1.0, 0.0, np.nan, np.nan, 1.0, np.nan])

    with pytest.raises(
        (TypeError, ValueError), match="outcome_selection_log_odds"
    ):
        site_selection_sensitivity(
            outcome_selection_log_odds=[bad_delta],  # type: ignore[list-item]
            outcome_features=x,
            selection_features=x,
            selected=selected,
            observed_outcome=outcome,
        )


def test_configured_site_selection_uses_all_candidates(tmp_path) -> None:
    rng = np.random.default_rng(812)
    n = 400
    x = rng.normal(size=n)
    q = rng.normal(size=n)
    y = rng.binomial(1, expit(-0.4 + x))
    selected = rng.binomial(1, expit(-0.8 + q + 1.0 * y)).astype(bool)
    outcome = np.where(selected, y, np.nan)
    source = tmp_path / "candidates.csv"
    pd.DataFrame(
        {
            "candidate_id": np.arange(n),
            "selected": selected.astype(int),
            "outcome": outcome,
            "x": x,
            "q": q,
        }
    ).to_csv(source, index=False)
    config = SiteSelectionConfig(
        mode="joint_binary",
        candidate_source=str(source),
        id_col="candidate_id",
        selected_col="selected",
        outcome_feature_columns=("x",),
        selection_feature_columns=("q",),
        outcome_selection_log_odds=(0.0, 1.0),
    )

    result = run_site_selection_analysis(
        config,
        outcome_column="outcome",
    )

    assert result.n_candidates == n
    assert result.n_selected == int(selected.sum())
    assert result.outcome_probability.shape == (n, 2)
    np.testing.assert_allclose(result.sensitivity_values, np.array([0.0, 1.0]))


def test_configured_site_selection_rejects_malformed_unobserved_outcomes(
    tmp_path,
) -> None:
    """Malformed values must not be silently reclassified as missing outcomes."""
    source = tmp_path / "candidates.csv"
    pd.DataFrame(
        {
            "candidate_id": [0, 1, 2, 3],
            "selected": [1, 1, 0, 0],
            "outcome": [1, 0, "not-observed", np.nan],
            "x": [-1.0, 1.0, 0.5, -0.5],
            "q": [0.2, -0.2, 1.0, -1.0],
        }
    ).to_csv(source, index=False)
    config = SiteSelectionConfig(
        mode="joint_binary",
        candidate_source=str(source),
        id_col="candidate_id",
        selected_col="selected",
        outcome_feature_columns=("x",),
        selection_feature_columns=("q",),
    )

    with pytest.raises(ValueError, match="outcome.*numeric"):
        run_site_selection_analysis(config, outcome_column="outcome")


@pytest.mark.parametrize(
    ("outcome_features", "selection_features"),
    [
        (("outcome",), ("q",)),
        (("selected",), ("q",)),
        (("candidate_id",), ("q",)),
        (("x",), ("outcome",)),
        (("x",), ("selected",)),
        (("x",), ("candidate_id",)),
    ],
)
def test_configured_site_selection_rejects_target_or_identifier_leakage(
    tmp_path,
    outcome_features: tuple[str, ...],
    selection_features: tuple[str, ...],
) -> None:
    source = tmp_path / "candidates.csv"
    pd.DataFrame(
        {
            "candidate_id": [0, 1, 2, 3],
            "selected": [1, 1, 0, 0],
            "outcome": [1.0, 0.0, np.nan, np.nan],
            "x": [-1.0, 1.0, 0.5, -0.5],
            "q": [0.2, -0.2, 1.0, -1.0],
        }
    ).to_csv(source, index=False)
    config = SiteSelectionConfig(
        mode="joint_binary",
        candidate_source=str(source),
        id_col="candidate_id",
        selected_col="selected",
        outcome_feature_columns=outcome_features,
        selection_feature_columns=selection_features,
    )

    with pytest.raises(ValueError, match="must not include.*reserved"):
        run_site_selection_analysis(config, outcome_column="outcome")


def test_configured_site_selection_fails_closed_without_candidate_source() -> (
    None
):
    config = SimpleNamespace(
        mode="joint_binary",
        candidate_source=None,
        id_col="candidate_id",
        outcome_feature_columns=("x",),
        selection_feature_columns=("q",),
    )

    with pytest.raises(ValueError, match="candidate_source"):
        run_site_selection_analysis(  # type: ignore[arg-type]
            config, outcome_column="outcome"
        )


@pytest.mark.parametrize(
    "config_kwargs",
    [
        {"outcome_selection_log_odds": (float("nan"),)},
        {"outcome_selection_log_odds": (True,)},
        {"outcome_selection_log_odds": [0.0]},
        {"outcome_penalty": float("nan")},
        {"outcome_penalty": True},
        {"selection_penalty": "0.1"},
    ],
)
def test_site_selection_config_rejects_nonfinite_or_coerced_numerics(
    config_kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="site_selection"):
        SiteSelectionConfig(**config_kwargs)  # type: ignore[arg-type]


def test_buffered_spatial_cv_has_no_nearby_training_rows() -> None:
    coords = np.column_stack([np.linspace(0.0, 10_000.0, 120), np.zeros(120)])
    folds = list(
        spatial_block_cv(
            coords,
            n_folds=4,
            block_type="grid",
            grid_size=8,
            buffer_distance=600.0,
        )
    )

    assert len(folds) == 4
    for train_mask, test_mask in folds:
        distances = np.abs(
            coords[train_mask, 0, None] - coords[test_mask, 0][None, :]
        )
        assert distances.size
        assert float(distances.min()) >= 600.0
        assert (~train_mask & ~test_mask).any()


def test_only_gblk_exposes_a_bayesian_backend() -> None:
    with pytest.raises(ValueError, match="inference.backend"):
        InferenceConfig.from_dict({"backend": "bayesian"})

    config = InferenceConfig.from_dict(
        {"backend": "gblk", "gblk_bayesian": {"enabled": True}}
    )
    assert config.backend == "gblk"
    assert config.gblk_bayesian.enabled is True


def test_gblk_runner_delegates_to_one_canonical_bayesian_fitter() -> None:
    source = inspect.getsource(gblk_runner._run_gblk_bayesian)  # noqa: SLF001
    backend_source = inspect.getsource(gblk_runner.fit_gblk_bayesian_joint)
    state_source = inspect.getsource(
        gblk_runner.fit_gblk_bayesian_posterior_state
    )

    assert "fit_gblk_bayesian_joint(" in source
    assert "optimize_hyperparameters" not in source
    assert "draw_posterior" not in source
    assert "fit_gblk_bayesian_posterior_state(" in backend_source
    assert 'inference="inla"' in state_source
    assert "latticekrigx.glk.bayes.marginal" not in backend_source
    assert "latticekrigx.glk.bayes.posterior" not in backend_source


def test_current_user_docs_describe_nnpu_and_one_bayesian_path() -> None:
    root = Path(__file__).resolve().parents[1]
    method = (root / "docs/probabilistic_method.md").read_text()
    migration = (root / "docs/migration_guide.md").read_text()
    readme = (root / "README.md").read_text()

    assert '`pu_mode` | `"off"`' in method
    assert "non-negative PU" in method
    assert "Bayesian GBLK offsets" not in method
    assert '`backend` | Bayesian path | `"bayesian"`' not in method
    assert "Bayesian MCMC hyperparameters" not in method
    assert "PU correction is opt-in" in migration
    assert '`pu_mode: "nnpu"`' in migration
    assert "Elkan-Noto PU correction is the **default**" not in migration
    for text in (readme, method, migration):
        assert "sole Bayesian" in text
        normalized = " ".join(text.split())
        assert "Paige/INLA" in normalized
        assert "non-negative PU" in normalized
    stale_active_instructions = (
        "full Bayesian hierarchical in PyMC",
        '"pu_mode": "elkan_noto"',
        "operates per depth slice internally",
        "ship two backends",
        "Status:** planning. No implementation yet",
    )
    for stale in stale_active_instructions:
        for text in (readme, method, migration):
            assert stale not in " ".join(text.split())
