"""Tests for the GBLK ProbabilisticResult runner (P5-S01).

Requires the ``gblk`` pixi environment (``latticekrigx`` + ``scikit-sparse``).
Skipped otherwise.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
from scipy.special import ndtr

pytest.importorskip("latticekrigx.glk.joint")

from geopfa.exceptions import GEOPFAValueError  # noqa: E402
from geopfa.prob.config import (  # noqa: E402
    AlphaModeConfig,
    CalibrationConfig,
    CombinationConfig,
    EvidenceConfig,
    GridConfig,
    GBLKBayesianConfig,
    InferenceConfig,
    KleiberProfileConfig,
    LabelsConfig,
    ObservationModelConfig,
    OutputsConfig,
    PredictiveStackingConfig,
    ProbabilisticConfig,
    RegularizationConfig,
    SpatialFieldConfig,
)
from geopfa.prob.fitting import ComponentProbability  # noqa: E402
from geopfa.prob import gblk_runner  # noqa: E402
from geopfa.prob.gblk_runner import (  # noqa: E402
    _prepare_joint_evidence_arrays,
    _prior_predictive_evidence_draws,
    run_gblk_probabilistic,
)
from geopfa.prob.alpha import build_alpha_c  # noqa: E402
from geopfa.prob.pfa_grid import PFAGridAdapter  # noqa: E402
from geopfa.prob.runner import ProbabilisticResult  # noqa: E402
from tests.fixtures.synthetic_prob import make_synthetic_pfa  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COMPONENTS = ("component_a", "component_b")
_NC_SMALL = 4
_OUTER_SMALL = 100
_IRLS_SMALL = 50


def _wells_path(tmp_path: Path) -> Path:
    fixture = make_synthetic_pfa(grid_n=10, n_wells=50, seed=7)
    path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(path, layer="wells", driver="GPKG")
    return path


def _cfg(wells_path: Path, output_dir: Path) -> ProbabilisticConfig:
    return ProbabilisticConfig(
        enabled=True,
        output_dir=output_dir,
        dimensions="2d",
        grid=GridConfig(),
        labels=LabelsConfig(
            source=str(wells_path),
            id_col="well_id",
            label_columns={
                "component_a": "heat_label",
                "component_b": "reservoir_label",
            },
            layer="wells",
        ),
        alpha={
            "component_a": AlphaModeConfig(
                mode="layer_logit",
                layer="prior_layer_a",
                scalar_fallback_pr0=0.55,
            ),
            "component_b": AlphaModeConfig(
                mode="layer_logit",
                layer="prior_layer_b",
                scalar_fallback_pr0=0.50,
            ),
        },
        evidence=EvidenceConfig(),
        spatial_field=SpatialFieldConfig(
            enabled=False,
            lattice_centers_per_dimension=_NC_SMALL,
        ),
        inference=InferenceConfig(backend="gblk"),
        calibration=CalibrationConfig(method="none"),
        cross_validation=__import__(
            "geopfa.prob.config", fromlist=["CrossValidationConfig"]
        ).CrossValidationConfig(),
        combination=CombinationConfig(rule="product"),
        scenarios=(),
        outputs=OutputsConfig(
            probability_rasters=False,
            uncertainty_rasters=False,
            calibration_artifacts=False,
            decision_artifacts=False,
            scenarios=False,
            format=(),
        ),
    )


def _mismatched_gaussian_prior_case(
    tmp_path: Path,
) -> tuple[dict, ProbabilisticConfig]:
    """Return a Gaussian prior on a coarse grid inside a canonical grid."""
    fixture = make_synthetic_pfa(grid_n=3, n_wells=8, seed=912)
    components = fixture.pfa["criteria"]["geologic"]["components"]
    coarse_component = components["component_b"]
    coarse_grid = coarse_component["pr_norm"]
    x = coarse_grid.geometry.x.to_numpy()
    y = coarse_grid.geometry.y.to_numpy()
    corners = np.isin(x, [x.min(), x.max()]) & np.isin(y, [y.min(), y.max()])
    coarse_grid = coarse_grid.loc[corners].reset_index(drop=True).copy()
    coarse_component["pr_norm"] = coarse_grid.copy()
    thermal_model = coarse_grid[["geometry"]].copy()
    thermal_model["value_interpolated"] = [0.0, 10.0, 0.0, 10.0]
    thermal_model["temperature_sd_c"] = [1.0, 10.0, 1.0, 10.0]
    coarse_component["layers"]["prior_layer_b"]["model"] = thermal_model

    base = _cfg(tmp_path / "unused.gpkg", tmp_path / "gaussian_resampling")
    cfg = replace(
        base,
        labels=replace(
            base.labels,
            observation_models={
                "component_b": ObservationModelConfig(family="gaussian")
            },
        ),
        alpha={
            "component_a": replace(
                base.alpha["component_a"], force_prior_predictive=True
            ),
            "component_b": AlphaModeConfig(
                mode="thermal_layer_exceedance",
                layer="prior_layer_b",
                threshold=5.0,
                uncertainty_column="temperature_sd_c",
                p_min=0.05,
                p_max=0.95,
                force_prior_predictive=True,
            ),
        },
    )
    return fixture.pfa, cfg


def _run(
    tmp_path: Path, *, spatial: bool = False
) -> tuple[ProbabilisticResult, dict]:
    fixture = make_synthetic_pfa(grid_n=10, n_wells=50, seed=7)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    cfg = _cfg(wells_path, tmp_path / "out")
    if spatial:
        cfg = replace(
            cfg,
            spatial_field=replace(cfg.spatial_field, enabled=True, n_levels=1),
        )
    result = run_gblk_probabilistic(
        fixture.pfa,
        cfg,
        max_outer_iter=_OUTER_SMALL,
        irls_max_iter=_IRLS_SMALL,
    )
    return result, {"fixture": fixture, "cfg": cfg}


def test_direct_gblk_runner_rejects_non_gblk_config(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path / "missing.gpkg", tmp_path / "out")
    cfg = replace(cfg, inference=InferenceConfig(backend="sequential"))
    with pytest.raises(GEOPFAValueError, match="backend.*gblk"):
        run_gblk_probabilistic({}, cfg)


def test_direct_gblk_runner_rejects_nc_that_conflicts_with_config(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path / "missing.gpkg", tmp_path / "out")

    with pytest.raises(GEOPFAValueError, match="nc.*conflicts.*spatial_field"):
        run_gblk_probabilistic({}, cfg, nc=_NC_SMALL + 1)


def test_direct_gblk_runner_derives_nc_from_config(tmp_path: Path) -> None:
    result, _ = _run(tmp_path, spatial=True)

    diagnostics = next(iter(result.components.values())).diagnostics
    assert diagnostics["nc"] == _NC_SMALL


@pytest.mark.parametrize(
    "distance_contract",
    ["predictive_stacking", "physical_isotropic"],
)
def test_gblk_metric_distance_controls_require_projected_metre_crs(
    tmp_path: Path,
    distance_contract: str,
) -> None:
    fixture = make_synthetic_pfa(grid_n=8, n_wells=30, seed=72)
    components = fixture.pfa["criteria"]["geologic"]["components"]
    for component in components.values():
        component["pr_norm"] = component["pr_norm"].set_crs(
            "EPSG:4326", allow_override=True
        )
    cfg = _cfg(tmp_path / "not-read.gpkg", tmp_path / "metric-crs")
    cfg = replace(
        cfg,
        spatial_field=replace(
            cfg.spatial_field,
            enabled=True,
            coordinate_scaling=(
                "physical_isotropic"
                if distance_contract == "physical_isotropic"
                else "axis_range"
            ),
        ),
        inference=replace(
            cfg.inference,
            gblk_bayesian=GBLKBayesianConfig(
                enabled=True,
                n_draws=4,
                cluster_effect=False,
                validate_inla=False,
                kleiber_profiles={
                    "bernoulli": KleiberProfileConfig(r0=0.25, r1=0.10)
                },
            ),
            predictive_stacking=PredictiveStackingConfig(
                enabled=distance_contract == "predictive_stacking"
            ),
        ),
        cross_validation=replace(
            cfg.cross_validation,
            buffer_km=(
                1.0 if distance_contract == "predictive_stacking" else 0.0
            ),
        ),
    )

    with pytest.raises(
        GEOPFAValueError,
        match="projected metre-based CRS",
    ):
        run_gblk_probabilistic(fixture.pfa, cfg)


# ---------------------------------------------------------------------------
# Smoke / basic structure
# ---------------------------------------------------------------------------


def test_p_gblk_runner_returns_probabilistic_result(tmp_path: Path) -> None:
    result, _ = _run(tmp_path)
    assert isinstance(result, ProbabilisticResult)
    assert not result.skipped
    assert result.config is not None


def test_p_gblk_runner_has_q_component_surfaces(tmp_path: Path) -> None:
    result, _ = _run(tmp_path)
    assert set(result.components.keys()) == set(_COMPONENTS)
    for name in _COMPONENTS:
        assert isinstance(result.components[name], ComponentProbability)


def test_p_gblk_runner_component_probability_column(tmp_path: Path) -> None:
    result, _ = _run(tmp_path)
    for name in _COMPONENTS:
        prob_gdf = result.components[name].probability
        assert isinstance(prob_gdf, gpd.GeoDataFrame)
        assert "probability" in prob_gdf.columns
        p = prob_gdf["probability"].to_numpy()
        assert p.ndim == 1
        assert np.all(p >= 0.0) and np.all(p <= 1.0)


def test_p_gblk_runner_jointly_fits_declared_evidence(tmp_path: Path) -> None:
    result, _ = _run(tmp_path, spatial=True)
    for name in _COMPONENTS:
        diagnostics = result.components[name].diagnostics
        assert (
            diagnostics["evidence_stage"]
            == "joint_fixed_effect_and_spatial_laplace"
        )
        beta = np.asarray(diagnostics["evidence_beta"], dtype=float)
        assert beta.shape == (len(result.components[name].feature_names),)
        assert np.any(np.abs(beta) > 1e-8)
        assert (
            diagnostics["evidence_coefficients_frozen_before_spatial_fit"]
            is False
        )


def test_p_gblk_evidence_only_ablation_uses_explicit_prior_means(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_synthetic_pfa(grid_n=8, n_wells=50, seed=27)
    wells_path = tmp_path / "wells_prior_means.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    cfg = _cfg(wells_path, tmp_path / "out_prior_means")
    cfg = replace(
        cfg,
        evidence=replace(
            cfg.evidence,
            regularization=RegularizationConfig(
                prior_means={"gradient": 0.75},
                prior_precisions={"gradient": 2.0},
            ),
        ),
    )
    observed_prior_means: list[np.ndarray | None] = []
    original = gblk_runner._fit_offset_logit  # noqa: SLF001

    def capture_prior_means(*args, **kwargs):
        prior_means = kwargs.get("prior_means")
        observed_prior_means.append(
            None
            if prior_means is None
            else np.asarray(prior_means, dtype=float)
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(gblk_runner, "_fit_offset_logit", capture_prior_means)
    run_gblk_probabilistic(fixture.pfa, cfg)

    assert observed_prior_means
    assert all(value is not None for value in observed_prior_means)
    assert any(np.any(np.abs(value) > 0.0) for value in observed_prior_means)


def test_p_gblk_joint_and_evidence_only_use_partial_evidence_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(Path("unused.gpkg"), tmp_path / "out")
    evidence = np.array(
        [
            [0.0, np.nan],
            [1.0, np.nan],
            [2.0, 2.0],
            [3.0, 3.0],
            [np.nan, 4.0],
            [np.nan, 5.0],
            [6.0, 6.0],
            [7.0, 7.0],
        ]
    )
    assembled = gblk_runner.AssembledInputs(
        component_names=("component_a",),
        y=np.array([[0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0]]),
        observed_mask=np.ones((8, 1), dtype=bool),
        observation_weights=np.ones((8, 1), dtype=float),
        labeled_mask=np.ones(8, dtype=bool),
        well_offsets=np.zeros((8, 1)),
        grid_offsets=np.zeros((2, 1)),
        well_coords=np.column_stack([np.arange(8), np.zeros(8)]),
        well_ids=np.array([f"well-{index}" for index in range(8)]),
        well_depths_m=None,
        grid_coords=np.array([[0.0, 0.0], [1.0, 0.0]]),
        evidence={"component_a": evidence},
        grid_evidence={"component_a": np.array([[0.0, 2.0], [np.nan, 3.0]])},
        layer_names={"component_a": ["feature_a", "feature_b"]},
    )

    prepared = gblk_runner._prepare_joint_evidence(assembled, cfg)  # noqa: SLF001

    assert prepared.train is not None
    assert prepared.train.shape == (8, 2, 1)
    assert np.all(np.isfinite(prepared.train))
    diagnostics = prepared.diagnostics["component_a"]
    assert diagnostics["evidence_n"] == 8
    assert diagnostics["evidence_complete_row_n"] == 4
    assert diagnostics["evidence_finite_per_feature"] == [6, 6]

    captured: dict[str, np.ndarray] = {}

    def capture_fit(x, y, *_args, **_kwargs):
        captured["x"] = np.asarray(x)
        captured["y"] = np.asarray(y)
        return SimpleNamespace(x=np.zeros(x.shape[1]), success=True)

    monkeypatch.setattr(gblk_runner, "_fit_offset_logit", capture_fit)
    gblk_runner._fit_evidence_only_offsets(assembled, cfg)  # noqa: SLF001

    assert captured["x"].shape == (8, 2)
    assert captured["y"].shape == (8,)
    np.testing.assert_allclose(captured["x"][[4, 5], 0], 0.0)
    np.testing.assert_allclose(captured["x"][[0, 1], 1], 0.0)

    unsupported = replace(
        assembled,
        evidence={
            "component_a": np.column_stack(
                [
                    evidence[:, 0],
                    [0.0, 1.0, 2.0, np.nan, np.nan, np.nan, np.nan, np.nan],
                ]
            )
        },
    )
    with pytest.raises(
        GEOPFAValueError,
        match="fewer than 4 finite observed wells.*feature_b=3 wells",
    ):
        gblk_runner._prepare_joint_evidence(unsupported, cfg)  # noqa: SLF001

    no_evidence_sparse = replace(
        assembled,
        observed_mask=np.array(
            [
                [True],
                [True],
                [True],
                [False],
                [False],
                [False],
                [False],
                [False],
            ]
        ),
        evidence={"component_a": np.zeros((8, 0))},
        grid_evidence={"component_a": np.zeros((2, 0))},
        layer_names={"component_a": []},
    )
    with pytest.raises(GEOPFAValueError, match="fewer than 4 observed wells"):
        gblk_runner._prepare_joint_evidence(no_evidence_sparse, cfg)  # noqa: SLF001

    train_mask = np.array([True] * 6 + [False] * 2)
    test_mask = ~train_mask
    gblk_runner._cv_evidence_only_offsets(  # noqa: SLF001
        assembled,
        np.arange(8, dtype=np.intp),
        (0,),
        train_mask,
        test_mask,
        cfg=cfg,
    )

    assert captured["x"].shape == (6, 2)
    assert captured["y"].shape == (6,)
    np.testing.assert_allclose(captured["x"][[4, 5], 0], 0.0)
    np.testing.assert_allclose(captured["x"][[0, 1], 1], 0.0)


def test_p_gblk_runner_combined_surface_exists(tmp_path: Path) -> None:
    result, _ = _run(tmp_path)
    assert isinstance(result.combined, gpd.GeoDataFrame)
    assert "probability" in result.combined.columns
    p = result.combined["probability"].to_numpy()
    assert np.all(p >= 0.0) and np.all(p <= 1.0)


# ---------------------------------------------------------------------------
# Grid alignment: component and combined surfaces use the same grid cells
# ---------------------------------------------------------------------------


def test_p_gblk_runner_grid_sizes_consistent(tmp_path: Path) -> None:
    result, _ = _run(tmp_path)
    G = len(result.combined)
    for name in _COMPONENTS:
        assert len(result.components[name].probability) == G


# ---------------------------------------------------------------------------
# Combined surface <= per-component minimum (product property)
# ---------------------------------------------------------------------------


def test_p_gblk_runner_combined_leq_component_min(tmp_path: Path) -> None:
    result, _ = _run(tmp_path)
    p_joint = result.combined["probability"].to_numpy()
    p_min = np.column_stack(
        [
            result.components[n].probability["probability"].to_numpy()
            for n in sorted(result.components)
        ]
    ).min(axis=1)
    assert np.all(p_joint <= p_min + 1e-8)


def test_p_gblk_runner_force_prior_predictive_bypasses_joint_fit(
    tmp_path: Path,
) -> None:
    fixture = make_synthetic_pfa(grid_n=8, n_wells=50, seed=31)
    wells_path = tmp_path / "wells_prior.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    cfg = _cfg(wells_path, tmp_path / "out_prior")
    cfg = replace(
        cfg,
        labels=replace(
            cfg.labels,
            label_columns={"component_a": "heat_label"},
        ),
        alpha={
            "component_a": cfg.alpha["component_a"],
            "component_b": AlphaModeConfig(
                mode="scalar",
                scalar_fallback_pr0=0.23,
                force_prior_predictive=True,
            ),
        },
    )

    result = run_gblk_probabilistic(
        fixture.pfa,
        cfg,
        nc=_NC_SMALL,
        max_outer_iter=_OUTER_SMALL,
        irls_max_iter=_IRLS_SMALL,
    )

    prior_component = result.components["component_b"]
    np.testing.assert_allclose(
        prior_component.probability["probability"].to_numpy(), 0.23
    )
    assert prior_component.model is None
    assert prior_component.diagnostics["inference_role"] == "prior_predictive"


def test_deterministic_prior_only_gaussian_reports_response_distribution(
    tmp_path: Path,
) -> None:
    fixture = make_synthetic_pfa(grid_n=8, n_wells=50, seed=311)
    heat = fixture.pfa["criteria"]["geologic"]["components"]["component_a"]
    thermal = heat["layers"]["prior_layer_a"]["model"]
    thermal_mean = np.linspace(150.0, 250.0, len(thermal))
    thermal_sd = np.linspace(15.0, 30.0, len(thermal))
    thermal["value_interpolated"] = thermal_mean
    thermal["temperature_sd_c"] = thermal_sd
    wells_path = tmp_path / "wells_gaussian_prior.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    base = _cfg(wells_path, tmp_path / "out_gaussian_prior")
    cfg = replace(
        base,
        labels=replace(
            base.labels,
            observation_models={
                "component_a": ObservationModelConfig(family="gaussian")
            },
        ),
        alpha={
            "component_a": AlphaModeConfig(
                mode="thermal_layer_exceedance",
                layer="prior_layer_a",
                threshold=200.0,
                uncertainty_column="temperature_sd_c",
                p_min=1e-12,
                p_max=1.0 - 1e-12,
                force_prior_predictive=True,
            ),
            "component_b": base.alpha["component_b"],
        },
    )

    result = run_gblk_probabilistic(fixture.pfa, cfg)

    component = result.components["component_a"]
    response = component.probability
    np.testing.assert_allclose(
        response["response_predictive_mean"], thermal_mean
    )
    expected_z = 1.6448536269514722
    np.testing.assert_allclose(
        response["response_predictive_lo"],
        thermal_mean - expected_z * thermal_sd,
    )
    np.testing.assert_allclose(
        response["response_predictive_hi"],
        thermal_mean + expected_z * thermal_sd,
    )
    np.testing.assert_allclose(
        response["probability"],
        ndtr((thermal_mean - 200.0) / thermal_sd),
    )
    assert component.diagnostics["observation_family"] == "gaussian"
    assert component.diagnostics["event_threshold"] == 200.0
    assert (
        component.diagnostics["response_summary_estimand"]
        == "prior_predictive_response"
    )
    assert component.diagnostics[
        "response_interval_includes_prior_uncertainty"
    ]


def test_deterministic_gaussian_prior_resampling_preserves_one_estimand(
    tmp_path: Path,
) -> None:
    pfa, cfg = _mismatched_gaussian_prior_case(tmp_path)

    result = run_gblk_probabilistic(pfa, cfg)

    response = result.components["component_b"].probability
    mean = response["response_predictive_mean"].to_numpy()
    sd = (
        response["response_predictive_hi"].to_numpy()
        - response["response_predictive_lo"].to_numpy()
    ) / (2.0 * 1.6448536269514722)
    expected = np.clip(ndtr((mean - 5.0) / sd), 0.05, 0.95)
    np.testing.assert_allclose(response["probability"], expected)
    assert mean[4] == pytest.approx(5.0)
    assert sd[4] == pytest.approx(5.5)
    assert response["probability"].iloc[4] == pytest.approx(0.5)
    component_a = (
        result.components["component_a"].probability["probability"].to_numpy()
    )
    np.testing.assert_allclose(
        result.combined["probability"], component_a * expected
    )


def test_deterministic_gblk_uses_product_for_prior_components(
    tmp_path: Path,
) -> None:
    fixture = make_synthetic_pfa(grid_n=8, n_wells=30, seed=45)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    base = _cfg(wells_path, tmp_path / "product")
    cfg = replace(
        base,
        alpha={
            name: replace(alpha, force_prior_predictive=True)
            for name, alpha in base.alpha.items()
        },
    )

    result = run_gblk_probabilistic(fixture.pfa, cfg)
    component_values = np.column_stack(
        [
            result.components[name].probability["probability"].to_numpy()
            for name in sorted(result.components)
        ]
    )

    np.testing.assert_allclose(
        result.combined["probability"],
        np.prod(component_values, axis=1),
    )


def test_combination_config_rejects_geometric_mean() -> None:
    with pytest.raises(ValueError, match="geometric_mean"):
        CombinationConfig(rule="geometric_mean")


def test_p_gblk_runner_force_prior_predictive_ignores_configured_labels(
    tmp_path: Path,
) -> None:
    """A sparse configured label must not override the explicit prior-only role."""
    fixture = make_synthetic_pfa(grid_n=8, n_wells=50, seed=32)
    fixture.wells.loc[:, "reservoir_label"] = np.nan
    fixture.wells.loc[fixture.wells.index[0], "reservoir_label"] = 1.0
    wells_path = tmp_path / "wells_sparse_prior.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    cfg = _cfg(wells_path, tmp_path / "out_sparse_prior")
    cfg = replace(
        cfg,
        alpha={
            "component_a": cfg.alpha["component_a"],
            "component_b": AlphaModeConfig(
                mode="scalar",
                scalar_fallback_pr0=0.23,
                force_prior_predictive=True,
            ),
        },
    )

    result = run_gblk_probabilistic(
        fixture.pfa,
        cfg,
        nc=_NC_SMALL,
        max_outer_iter=_OUTER_SMALL,
        irls_max_iter=_IRLS_SMALL,
    )

    prior_component = result.components["component_b"]
    np.testing.assert_allclose(
        prior_component.probability["probability"].to_numpy(), 0.23
    )
    assert prior_component.model is None
    assert prior_component.diagnostics["inference_role"] == "prior_predictive"


def test_prior_predictive_evidence_uses_named_priors_not_legacy_weights(
    tmp_path: Path,
) -> None:
    fixture = make_synthetic_pfa(grid_n=8, n_wells=50, seed=33)
    wells_path = tmp_path / "wells_evidence_prior.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    cfg = _cfg(wells_path, tmp_path / "out_evidence_prior")
    cfg = replace(
        cfg,
        alpha={
            "component_a": cfg.alpha["component_a"],
            "component_b": AlphaModeConfig(
                mode="scalar",
                scalar_fallback_pr0=0.23,
                force_prior_predictive=True,
                use_evidence_prior=True,
            ),
        },
        evidence=EvidenceConfig(
            regularization=RegularizationConfig(
                prior_means={
                    "component_b:prior_layer_b": 0.5,
                    "component_b:gradient": 0.8,
                },
                prior_precisions={
                    "component_b:prior_layer_b": 4.0,
                    "component_b:gradient": 2.0,
                },
            )
        ),
        inference=replace(
            cfg.inference,
            gblk_bayesian=replace(
                cfg.inference.gblk_bayesian,
                enabled=True,
                n_draws=32,
                seed=91,
            ),
        ),
    )
    adapter = PFAGridAdapter(fixture.pfa, criteria="geologic", dimensions="2d")
    component_data = adapter.component_data("component_b")
    alpha = build_alpha_c(
        component_data,
        cfg.alpha["component_b"],
        grid_gdf=adapter.pr_norm("component_b"),
    )

    first = _prior_predictive_evidence_draws(
        adapter, "component_b", alpha, cfg, seed=91
    )
    for index, layer in enumerate(component_data["layers"].values(), start=1):
        layer["weight"] = 10_000.0 * index
    second = _prior_predictive_evidence_draws(
        adapter, "component_b", alpha, cfg, seed=91
    )

    np.testing.assert_array_equal(
        first.probability_draws, second.probability_draws
    )
    assert first.feature_names == ("prior_layer_b", "gradient")
    assert first.diagnostics["coefficient_prior_mean"] == [0.5, 0.8]
    assert first.diagnostics["coefficient_prior_sd"] == pytest.approx(
        [0.5, np.sqrt(0.5)]
    )
    assert not any("weight" in key for key in first.diagnostics)


def test_prior_predictive_evidence_requires_named_prior_for_every_layer(
    tmp_path: Path,
) -> None:
    fixture = make_synthetic_pfa(grid_n=8, n_wells=50, seed=34)
    cfg = _cfg(tmp_path / "wells.gpkg", tmp_path / "out_missing_prior")
    cfg = replace(
        cfg,
        alpha={
            "component_a": cfg.alpha["component_a"],
            "component_b": AlphaModeConfig(
                mode="scalar",
                scalar_fallback_pr0=0.23,
                force_prior_predictive=True,
                use_evidence_prior=True,
            ),
        },
        evidence=EvidenceConfig(
            regularization=RegularizationConfig(
                prior_means={"component_b:gradient": 0.8},
                prior_precisions={"component_b:gradient": 2.0},
            )
        ),
        inference=replace(
            cfg.inference,
            gblk_bayesian=replace(
                cfg.inference.gblk_bayesian, enabled=True, n_draws=8
            ),
        ),
    )
    adapter = PFAGridAdapter(fixture.pfa, criteria="geologic", dimensions="2d")
    alpha = build_alpha_c(
        adapter.component_data("component_b"),
        cfg.alpha["component_b"],
        grid_gdf=adapter.pr_norm("component_b"),
    )

    with pytest.raises(GEOPFAValueError, match="explicit Gaussian prior"):
        _prior_predictive_evidence_draws(
            adapter, "component_b", alpha, cfg, seed=0
        )


def test_bayesian_mixed_outcome_and_prior_components_combine_within_draw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = make_synthetic_pfa(grid_n=8, n_wells=50, seed=35)
    # Exercise the one-class identifiability boundary: a proper Bayesian
    # posterior can update from one observed class, while a frequentist fit
    # cannot. No opposite-class pseudo-observations are manufactured.
    fixture.wells.loc[:, "heat_label"] = np.nan
    fixture.wells.loc[fixture.wells.index[0], "heat_label"] = 1.0
    wells_path = tmp_path / "wells_mixed_bayesian.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    cfg = _cfg(wells_path, tmp_path / "out_mixed_bayesian")
    cfg = replace(
        cfg,
        labels=replace(
            cfg.labels,
            label_columns={"component_a": "heat_label"},
            min_wells_for_fit=1,
        ),
        alpha={
            "component_a": cfg.alpha["component_a"],
            "component_b": AlphaModeConfig(
                mode="scalar",
                scalar_fallback_pr0=0.5,
                force_prior_predictive=True,
                use_evidence_prior=True,
            ),
        },
        evidence=EvidenceConfig(
            regularization=RegularizationConfig(
                prior_means={
                    "component_b:prior_layer_b": 0.5,
                    "component_b:gradient": 0.8,
                },
                prior_precisions={
                    "component_b:prior_layer_b": 4.0,
                    "component_b:gradient": 4.0,
                },
            )
        ),
        spatial_field=replace(cfg.spatial_field, enabled=True, n_levels=1),
        inference=replace(
            cfg.inference,
            gblk_bayesian=replace(
                cfg.inference.gblk_bayesian,
                enabled=True,
                n_draws=4,
                seed=123,
                ci_level=0.5,
            ),
        ),
    )
    fitted_draws_seen: dict[str, np.ndarray] = {}

    def fake_bayesian_fit(assembled, grid_gdf, bayes_cfg, **_kwargs):
        assert assembled.component_names == ("component_a",)
        assert int(assembled.observed_mask.sum()) == 1
        fitted = np.broadcast_to(
            np.array([0.2, 0.4, 0.6, 0.8])[:, np.newaxis],
            (bayes_cfg.n_draws, len(grid_gdf)),
        ).copy()
        fitted_draws_seen["component_a"] = fitted
        interval = np.quantile(fitted, [0.25, 0.75], axis=0)
        probability = (
            grid_gdf[["geometry"]]
            .copy()
            .assign(
                probability=fitted.mean(axis=0),
                probability_lo=interval[0],
                probability_hi=interval[1],
            )
        )
        component = ComponentProbability(
            probability=probability,
            model=SimpleNamespace(extra={"mean_draws": fitted[:, :, None]}),
            feature_names=tuple(assembled.layer_names["component_a"]),
            diagnostics={"outcome_update": True},
        )
        return (
            {"component_a": component},
            probability.copy(),
            {"component_a": fitted},
        )

    monkeypatch.setattr(gblk_runner, "_run_gblk_bayesian", fake_bayesian_fit)

    result = run_gblk_probabilistic(fixture.pfa, cfg, nc=_NC_SMALL)

    assert set(result.components) == {"component_a", "component_b"}
    prior_draws = result.components["component_b"].model.probability_draws
    expected_joint_draws = fitted_draws_seen["component_a"] * prior_draws
    assert set(result.component_probability_draws) == {
        "component_a",
        "component_b",
    }
    np.testing.assert_allclose(
        result.component_probability_draws["component_a"],
        fitted_draws_seen["component_a"],
    )
    np.testing.assert_allclose(
        result.component_probability_draws["component_b"], prior_draws
    )
    np.testing.assert_allclose(
        result.combined_probability_draws, expected_joint_draws
    )
    np.testing.assert_allclose(
        result.combined["probability"].to_numpy(),
        expected_joint_draws.mean(axis=0),
    )
    np.testing.assert_allclose(
        result.combined[["probability_lo", "probability_hi"]].to_numpy().T,
        np.quantile(expected_joint_draws, [0.25, 0.75], axis=0),
    )


def test_prediction_support_standardization_is_identified_with_one_label(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path / "wells.gpkg", tmp_path / "out")
    cfg = replace(
        cfg,
        labels=replace(cfg.labels, min_wells_for_fit=1),
        evidence=replace(cfg.evidence, standardization="prediction_support"),
        inference=replace(
            cfg.inference,
            gblk_bayesian=replace(cfg.inference.gblk_bayesian, enabled=True),
        ),
    )
    prediction = np.array([[0.0], [2.0], [4.0]])

    design = _prepare_joint_evidence_arrays(
        component_names=("component_a",),
        train_evidence={"component_a": np.array([[2.0]])},
        prediction_evidence={"component_a": prediction},
        layer_names={"component_a": ["gradient"]},
        y_train=np.array([[1.0]]),
        observed_train=np.array([[True]]),
        well_ids=np.array(["well-0"]),
        cfg=cfg,
    )

    np.testing.assert_allclose(design.train[:, 0, 0], [0.0])
    np.testing.assert_allclose(
        design.prediction[:, 0, 0], [-np.sqrt(1.5), 0.0, np.sqrt(1.5)]
    )
    diagnostics = design.diagnostics["component_a"]
    assert diagnostics["evidence_standardization"] == "prediction_support"
    assert diagnostics["evidence_center"] == [2.0]


def test_calibration_cv_standardizes_evidence_on_canonical_grid_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _cfg(tmp_path / "unused.gpkg", tmp_path / "cv-support")
    cfg = replace(
        base,
        labels=replace(base.labels, min_wells_for_fit=2),
        evidence=replace(base.evidence, standardization="prediction_support"),
    )
    assembled = gblk_runner.AssembledInputs(
        component_names=("component_a",),
        y=np.array([[0.0], [1.0], [0.0], [1.0]]),
        observed_mask=np.ones((4, 1), dtype=bool),
        observation_weights=np.ones((4, 1), dtype=float),
        labeled_mask=np.ones(4, dtype=bool),
        well_offsets=np.zeros((4, 1)),
        grid_offsets=np.zeros((2, 1)),
        well_coords=np.column_stack([np.arange(4), np.zeros(4)]),
        grid_coords=np.array([[0.0, 0.0], [1.0, 0.0]]),
        well_ids=np.array(["a", "b", "c", "d"]),
        well_depths_m=None,
        evidence={"component_a": np.array([[0.0], [10.0], [100.0], [200.0]])},
        grid_evidence={"component_a": np.array([[0.0], [10.0]])},
        layer_names={"component_a": ["gradient"]},
    )
    train_mask = np.array([True, True, False, False])
    test_mask = ~train_mask

    design = gblk_runner._cv_joint_evidence(  # noqa: SLF001
        assembled,
        np.arange(4, dtype=np.intp),
        (0,),
        train_mask,
        test_mask,
        cfg=cfg,
    )
    assert design.diagnostics["component_a"]["evidence_center"] == [5.0]

    captured: dict[str, np.ndarray] = {}

    def fake_fit(x, *_args, **_kwargs):
        captured["train"] = np.asarray(x)
        return SimpleNamespace(x=np.ones(x.shape[1]), success=True)

    monkeypatch.setattr(gblk_runner, "_fit_offset_logit", fake_fit)
    _, test_offsets = gblk_runner._cv_evidence_only_offsets(  # noqa: SLF001
        assembled,
        np.arange(4, dtype=np.intp),
        (0,),
        train_mask,
        test_mask,
        cfg=cfg,
    )

    np.testing.assert_allclose(captured["train"][:, 0], [-1.0, 1.0])
    np.testing.assert_allclose(
        test_offsets[:, 0],
        [(100.0 - 5.0) / 5.0, (200.0 - 5.0) / 5.0],
    )


# ---------------------------------------------------------------------------
# Omega is non-trivial on correlated synthetic data
# ---------------------------------------------------------------------------


def test_p_gblk_runner_omega_nontrivial_on_coupled_data(
    tmp_path: Path,
) -> None:
    """Off-diagonal Omega entries should be non-zero on correlated labels."""
    fixture = make_synthetic_pfa(grid_n=10, n_wells=80, seed=42)
    wells_path = tmp_path / "wells_coupled.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    cfg = _cfg(wells_path, tmp_path / "out_coupled")
    cfg = replace(
        cfg,
        spatial_field=replace(cfg.spatial_field, enabled=True, n_levels=1),
    )
    result = run_gblk_probabilistic(
        fixture.pfa,
        cfg,
        nc=_NC_SMALL,
        max_outer_iter=_OUTER_SMALL,
        irls_max_iter=_IRLS_SMALL,
    )
    any_comp = next(iter(result.components.values()))
    omega = np.array(any_comp.diagnostics["omega"])
    assert omega.shape == (2, 2)
    off_diag = np.abs(omega[0, 1])
    assert off_diag > 1e-6, (
        f"Omega off-diagonal is {off_diag:.2e}; expected non-trivial coupling "
        "on shared-label synthetic data"
    )


# ---------------------------------------------------------------------------
# Disabled config short-circuits
# ---------------------------------------------------------------------------


def test_p_gblk_runner_skipped_when_disabled(tmp_path: Path) -> None:
    fixture = make_synthetic_pfa(grid_n=8, n_wells=20, seed=0)
    wells_path = tmp_path / "wells_skip.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    cfg = _cfg(wells_path, tmp_path / "out_skip")
    disabled_cfg = ProbabilisticConfig(
        enabled=False,
        output_dir=cfg.output_dir,
        dimensions=cfg.dimensions,
        grid=cfg.grid,
        labels=cfg.labels,
        alpha=cfg.alpha,
        evidence=cfg.evidence,
        spatial_field=cfg.spatial_field,
        inference=cfg.inference,
        calibration=cfg.calibration,
        cross_validation=cfg.cross_validation,
        combination=cfg.combination,
        scenarios=cfg.scenarios,
        outputs=cfg.outputs,
    )
    result = run_gblk_probabilistic(fixture.pfa, disabled_cfg)
    assert result.skipped
    assert result.components == {}
    assert isinstance(result.combined, gpd.GeoDataFrame)
    assert len(result.combined) == 0
