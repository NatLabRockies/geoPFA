"""Unit tests for runner.py private functions (coverage gaps).

These tests exercise defensive branches in _fit_component, _combine_components,
configured output routing, and _apply_scenario that are not reachable through
the public run_probabilistic API.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

import geopfa.prob.runner as probability_runner
from geopfa.exceptions import GEOPFAValueError
from geopfa.prob.alpha import AlphaCResult
from geopfa.prob.config import (
    AlphaModeConfig,
    CalibrationConfig,
    CombinationConfig,
    CrossValidationConfig,
    EvidenceConfig,
    GBLKBayesianConfig,
    GridConfig,
    InferenceConfig,
    KleiberProfileConfig,
    LabelsConfig,
    ObservationModelConfig,
    OutputsConfig,
    ProbabilisticConfig,
    ScenarioConfig,
    SiteSelectionConfig,
    SpatialFieldConfig,
)
from geopfa.prob.fitting import ComponentProbability
from geopfa.prob.labels import LoadedLabels
from geopfa.prob.pfa_grid import PFAGridAdapter
from geopfa.prob.runner import (
    ProbabilisticResult,
    _apply_scenario,  # noqa: PLC2701
    _combine_components,  # noqa: PLC2701
    _fit_component,  # noqa: PLC2701
    _run_site_selection,  # noqa: PLC2701
    _write_configured_probability_outputs,  # noqa: PLC2701
    run_probabilistic,
)
from tests.fixtures.synthetic_prob import make_synthetic_pfa


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_cfg(  # noqa: PLR0913
    wells_path: Path,
    output_dir: Path,
    *,
    label_columns: dict | None = None,
    alpha: dict | None = None,
    scenarios: tuple = (),
    formats: tuple = ("csv",),
) -> ProbabilisticConfig:
    lc = label_columns or {
        "component_a": "heat_label",
        "component_b": "reservoir_label",
    }
    alp = alpha or {
        "component_a": AlphaModeConfig(
            mode="scalar",
            scalar_fallback_pr0=0.5,
        ),
        "component_b": AlphaModeConfig(
            mode="scalar",
            scalar_fallback_pr0=0.5,
        ),
    }
    return ProbabilisticConfig(
        enabled=True,
        output_dir=output_dir,
        dimensions="2d",
        grid=GridConfig(),
        labels=LabelsConfig(
            source=str(wells_path),
            id_col="well_id",
            label_columns=lc,
            layer="wells",
        ),
        alpha=alp,
        evidence=EvidenceConfig(),
        spatial_field=SpatialFieldConfig(enabled=False),
        inference=InferenceConfig(backend="sequential"),
        calibration=CalibrationConfig(method="none"),
        cross_validation=CrossValidationConfig(),
        combination=CombinationConfig(rule="product"),
        scenarios=scenarios,
        outputs=OutputsConfig(
            probability_rasters=False,
            uncertainty_rasters=False,
            calibration_artifacts=False,
            decision_artifacts=False,
            scenarios=False,
            format=formats,
        ),
    )


def test_site_selection_runs_only_for_bernoulli_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_cfg(tmp_path / "wells.gpkg", tmp_path / "out")
    cfg = replace(
        cfg,
        labels=replace(
            cfg.labels,
            observation_models={
                "component_a": ObservationModelConfig(
                    family="gaussian", response_scale=50.0
                )
            },
        ),
        site_selection=SiteSelectionConfig(
            mode="joint_binary",
            candidate_source="candidates.csv",
            id_col="candidate_id",
            outcome_feature_columns=("temperature",),
            selection_feature_columns=("road_distance",),
        ),
    )
    received: list[str] = []

    def fake_analysis(_config, *, outcome_column):
        received.append(outcome_column)
        return {"outcome_column": outcome_column}

    monkeypatch.setattr(
        "geopfa.prob.site_selection.run_site_selection_analysis",
        fake_analysis,
    )

    result = _run_site_selection(cfg)

    assert received == ["reservoir_label"]
    assert set(result) == {"component_b"}


def _save_wells(tmp_path: Path) -> Path:
    fixture = make_synthetic_pfa(grid_n=8, n_wells=30, seed=42)
    path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(path, layer="wells", driver="GPKG")
    return path


def _minimal_alpha_result(n: int = 4) -> AlphaCResult:
    return AlphaCResult(
        grid_offset=np.zeros(n),
        scalar_fallback=0.0,
        excluded_layer_names=set(),
        provenance={},
    )


def _minimal_loaded_labels(fixture) -> LoadedLabels:
    lc = LabelsConfig(
        source="memory",
        id_col="well_id",
        label_columns={
            "component_a": "heat_label",
            "component_b": "reservoir_label",
        },
    )
    return LoadedLabels(gdf=fixture.wells, config=lc)


# ---------------------------------------------------------------------------
# _fit_component skip warnings
# ---------------------------------------------------------------------------


def test_fit_component_warns_if_component_not_in_label_columns(
    tmp_path: Path,
) -> None:
    """_fit_component should warn and return None when the component is not
    in cfg.labels.label_columns (defensive guard for direct callers)."""
    fixture = make_synthetic_pfa(grid_n=8, n_wells=30, seed=1)
    adapter = PFAGridAdapter(fixture.pfa, criteria="geologic", dimensions="2d")
    labels = _minimal_loaded_labels(fixture)

    # Config that has component_a but NOT component_b in label_columns
    cfg = ProbabilisticConfig(
        enabled=True,
        output_dir=tmp_path / "out",
        dimensions="2d",
        grid=GridConfig(),
        labels=LabelsConfig(
            source="memory",
            id_col="well_id",
            label_columns={"component_a": "heat_label"},  # component_b missing
        ),
        alpha={
            "component_a": AlphaModeConfig(
                mode="scalar", scalar_fallback_pr0=0.5
            ),
            "component_b": AlphaModeConfig(
                mode="scalar", scalar_fallback_pr0=0.5
            ),
        },
        evidence=EvidenceConfig(),
        spatial_field=SpatialFieldConfig(enabled=False),
        inference=InferenceConfig(backend="sequential"),
        calibration=CalibrationConfig(method="none"),
        cross_validation=CrossValidationConfig(),
        combination=CombinationConfig(rule="product"),
        scenarios=(),
        outputs=OutputsConfig(format=("csv",)),
    )

    alpha = _minimal_alpha_result(
        n=len(
            fixture.pfa["criteria"]["geologic"]["components"]["component_b"][
                "pr_norm"
            ]
        )
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _fit_component(adapter, "component_b", cfg, labels, alpha)

    assert result is None
    assert any("component_b" in str(w.message) for w in caught)
    assert any("label_columns" in str(w.message) for w in caught)


def test_fit_component_warns_if_component_not_in_alpha(
    tmp_path: Path,
) -> None:
    """_fit_component should warn and return None when the component is not
    in cfg.alpha (defensive guard for direct callers)."""
    fixture = make_synthetic_pfa(grid_n=8, n_wells=30, seed=2)
    adapter = PFAGridAdapter(fixture.pfa, criteria="geologic", dimensions="2d")
    labels = _minimal_loaded_labels(fixture)

    cfg = ProbabilisticConfig(
        enabled=True,
        output_dir=tmp_path / "out",
        dimensions="2d",
        grid=GridConfig(),
        labels=LabelsConfig(
            source="memory",
            id_col="well_id",
            label_columns={
                "component_a": "heat_label",
                "component_b": "reservoir_label",
            },
        ),
        alpha={
            "component_a": AlphaModeConfig(
                mode="scalar", scalar_fallback_pr0=0.5
            ),
            # component_b missing from alpha
        },
        evidence=EvidenceConfig(),
        spatial_field=SpatialFieldConfig(enabled=False),
        inference=InferenceConfig(backend="sequential"),
        calibration=CalibrationConfig(method="none"),
        cross_validation=CrossValidationConfig(),
        combination=CombinationConfig(rule="product"),
        scenarios=(),
        outputs=OutputsConfig(format=("csv",)),
    )

    alpha = _minimal_alpha_result(
        n=len(
            fixture.pfa["criteria"]["geologic"]["components"]["component_b"][
                "pr_norm"
            ]
        )
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _fit_component(adapter, "component_b", cfg, labels, alpha)

    assert result is None
    assert any("component_b" in str(w.message) for w in caught)
    assert any("alpha" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# _combine_components edge cases
# ---------------------------------------------------------------------------


def test_combine_components_empty_returns_empty_gdf() -> None:
    """_combine_components({}) must return an empty GeoDataFrame."""
    result = _combine_components({})
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 0


def test_combine_components_uses_product() -> None:
    """_combine_components returns the component-probability product."""
    geom = [Point(0, 0), Point(1, 0), Point(0, 1), Point(1, 1)]
    first = gpd.GeoDataFrame(
        {"probability": [0.2, 0.4, 0.6, 0.8]}, geometry=geom, crs="EPSG:4326"
    )
    second = gpd.GeoDataFrame(
        {"probability": [0.9, 0.7, 0.5, 0.3]}, geometry=geom, crs="EPSG:4326"
    )
    surfaces = {
        "first": ComponentProbability(first, None, ()),
        "second": ComponentProbability(second, None, ()),
    }

    result = _combine_components(surfaces)
    assert "probability" in result.columns
    assert len(result) == len(first)
    np.testing.assert_allclose(
        result["probability"].to_numpy(),
        first["probability"].to_numpy() * second["probability"].to_numpy(),
    )


def test_combine_components_rejects_different_grid_order() -> None:
    geometry = [Point(0, 0), Point(1, 0), Point(0, 1)]
    first = gpd.GeoDataFrame(
        {"probability": [0.2, 0.4, 0.6]},
        geometry=geometry,
        crs="EPSG:32611",
    )
    second = gpd.GeoDataFrame(
        {"probability": [0.8, 0.7, 0.5]},
        geometry=list(reversed(geometry)),
        crs="EPSG:32611",
    )
    surfaces = {
        "first": ComponentProbability(first, None, ()),
        "second": ComponentProbability(second, None, ()),
    }

    with pytest.raises(GEOPFAValueError, match="canonical grid order"):
        _combine_components(surfaces)


def test_requested_spatial_format_is_not_silently_omitted(
    tmp_path: Path,
) -> None:
    wells_path = _save_wells(tmp_path)
    base = _make_cfg(wells_path, tmp_path / "out", formats=("geotiff",))
    cfg = ProbabilisticConfig(
        enabled=base.enabled,
        output_dir=base.output_dir,
        dimensions=base.dimensions,
        grid=base.grid,
        labels=base.labels,
        alpha=base.alpha,
        evidence=base.evidence,
        spatial_field=base.spatial_field,
        inference=base.inference,
        calibration=base.calibration,
        cross_validation=base.cross_validation,
        combination=base.combination,
        scenarios=base.scenarios,
        outputs=OutputsConfig(
            probability_rasters=False,
            uncertainty_rasters=False,
            calibration_artifacts=False,
            decision_artifacts=False,
            scenarios=False,
            format=("geotiff",),
        ),
        site_selection=base.site_selection,
    )
    probability = gpd.GeoDataFrame(
        {"probability": [0.5]},
        geometry=[Point(0, 0)],
        crs="EPSG:32611",
    )
    component = ComponentProbability(probability, None, ())

    with pytest.raises(
        GEOPFAValueError,
        match="probability_rasters=false.*geotiff",
    ):
        _write_configured_probability_outputs(
            {"component_a": component},
            probability,
            cfg,
            cfg.output_dir,
        )


def test_configured_csv_outputs_route_through_canonical_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wells_path = _save_wells(tmp_path)
    cfg = _make_cfg(wells_path, tmp_path / "out", formats=("csv",))
    probability = gpd.GeoDataFrame(
        {"probability": [0.5]},
        geometry=[Point(0, 0)],
        crs="EPSG:4326",
    )
    component = ComponentProbability(probability, None, ())
    calls: list[tuple[dict[str, gpd.GeoDataFrame], Path, dict]] = []

    def capture_writer(surfaces, output_dir, **kwargs):
        calls.append((surfaces, output_dir, kwargs))
        return []

    monkeypatch.setattr(
        probability_runner, "write_probability_outputs", capture_writer
    )

    _write_configured_probability_outputs(
        {"heat": component}, probability, cfg, cfg.output_dir
    )

    assert len(calls) == 1
    surfaces, output_dir, kwargs = calls[0]
    assert list(surfaces) == ["heat", "combined"]
    assert output_dir == cfg.output_dir
    assert kwargs == {"formats": ("csv",), "include_uncertainty": False}


# ---------------------------------------------------------------------------
# _apply_scenario — include_priors=False path
# ---------------------------------------------------------------------------


def test_apply_scenario_disables_priors(tmp_path: Path) -> None:
    """_apply_scenario with include_priors=False must set all alpha modes
    to 'scalar' while preserving the scalar_fallback_pr0 value."""
    wells_path = _save_wells(tmp_path)
    cfg = _make_cfg(
        wells_path,
        tmp_path / "out",
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
    )
    scenario = ScenarioConfig(
        name="no_priors", include_priors=False, include_spatial=True
    )

    new_cfg = _apply_scenario(cfg, scenario)

    for name, alpha in new_cfg.alpha.items():
        assert alpha.mode == "scalar", (
            f"expected scalar mode for {name}, got {alpha.mode}"
        )
    assert {"prior_layer_a", "prior_layer_b"}.issubset(
        new_cfg.evidence.exclude_layers
    )


def test_apply_scenario_preserves_prior_predictive_inference_role(
    tmp_path: Path,
) -> None:
    """Removing alpha offsets must not turn prior prediction into a fit."""
    wells_path = _save_wells(tmp_path)
    cfg = _make_cfg(
        wells_path,
        tmp_path / "out",
        alpha={
            "component_a": AlphaModeConfig(
                mode="layer_logit",
                layer="prior_layer_a",
                scalar_fallback_pr0=0.61,
                force_prior_predictive=True,
                use_evidence_prior=True,
            )
        },
        label_columns={"component_a": "heat_label"},
    )

    new_cfg = _apply_scenario(
        cfg,
        ScenarioConfig(name="no_alpha", include_priors=False),
    )

    alpha = new_cfg.alpha["component_a"]
    assert alpha.mode == "scalar"
    assert alpha.scalar_fallback_pr0 == 0.5
    assert alpha.force_prior_predictive is True
    assert alpha.use_evidence_prior is True


def test_apply_scenario_rejects_undefined_gaussian_no_prior_ablation(
    tmp_path: Path,
) -> None:
    wells_path = _save_wells(tmp_path)
    base = _make_cfg(
        wells_path,
        tmp_path / "out",
        label_columns={"component_a": "heat_label"},
        alpha={"component_a": AlphaModeConfig()},
    )
    cfg = ProbabilisticConfig(
        enabled=base.enabled,
        output_dir=base.output_dir,
        dimensions=base.dimensions,
        grid=base.grid,
        labels=LabelsConfig(
            source=base.labels.source,
            id_col=base.labels.id_col,
            label_columns=base.labels.label_columns,
            observation_models={
                "component_a": ObservationModelConfig(
                    family="gaussian", response_scale=100.0
                )
            },
            layer=base.labels.layer,
        ),
        alpha=base.alpha,
        evidence=base.evidence,
        spatial_field=base.spatial_field,
        inference=base.inference,
        calibration=base.calibration,
        cross_validation=base.cross_validation,
        combination=base.combination,
        scenarios=base.scenarios,
        outputs=base.outputs,
        site_selection=base.site_selection,
    )

    with pytest.raises(
        ValueError,
        match="Gaussian.*include_priors=false.*undefined",
    ):
        _apply_scenario(
            cfg,
            ScenarioConfig(name="no_alpha", include_priors=False),
        )


def test_apply_scenario_disables_unrequested_posterior_persistence(
    tmp_path: Path,
) -> None:
    wells_path = _save_wells(tmp_path)
    base = _make_cfg(wells_path, tmp_path / "out")
    cfg = ProbabilisticConfig(
        enabled=base.enabled,
        output_dir=base.output_dir,
        dimensions=base.dimensions,
        grid=base.grid,
        labels=base.labels,
        alpha=base.alpha,
        evidence=base.evidence,
        spatial_field=base.spatial_field,
        inference=base.inference,
        calibration=base.calibration,
        cross_validation=base.cross_validation,
        combination=base.combination,
        scenarios=base.scenarios,
        outputs=OutputsConfig(
            probability_rasters=False,
            uncertainty_rasters=False,
            calibration_artifacts=False,
            decision_artifacts=False,
            scenarios=False,
            posterior_draw_blocks=True,
            format=(),
        ),
        site_selection=base.site_selection,
    )

    scenario_cfg = _apply_scenario(cfg, ScenarioConfig(name="in_memory"))

    assert scenario_cfg.outputs.scenarios is False
    assert scenario_cfg.outputs.posterior_draw_blocks is False


def test_gblk_scenario_stays_in_memory_when_outputs_are_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = make_synthetic_pfa(grid_n=5, n_wells=20, seed=52)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    base = _make_cfg(wells_path, tmp_path / "out")
    cfg = replace(
        base,
        inference=InferenceConfig(
            backend="gblk",
            gblk_bayesian=GBLKBayesianConfig(
                enabled=True,
                n_draws=2,
                cluster_effect=False,
                validate_inla=False,
                kleiber_profiles={
                    "bernoulli": KleiberProfileConfig(r0=0.25, r1=0.10)
                },
            ),
        ),
        spatial_field=SpatialFieldConfig(
            enabled=True,
            backend="latticekrigx",
        ),
        scenarios=(ScenarioConfig(name="in_memory"),),
        outputs=replace(
            base.outputs,
            format=(),
            scenarios=False,
            posterior_draw_blocks=True,
            posterior_draw_block_size=1,
        ),
    )
    received: list[ProbabilisticConfig] = []

    def fake_gblk(_pfa, sub_cfg, **_kwargs):
        received.append(sub_cfg)
        return ProbabilisticResult(config=sub_cfg)

    monkeypatch.setattr(
        "geopfa.prob.gblk_runner.run_gblk_probabilistic",
        fake_gblk,
    )

    result = run_probabilistic(fixture.pfa, cfg)

    assert len(received) == 2
    assert received[0].outputs.posterior_draw_blocks is True
    assert received[1].outputs.posterior_draw_blocks is False
    assert set(result.scenarios) == {"in_memory"}
    assert not (cfg.output_dir / "scenarios").exists()


def test_apply_scenario_preserves_priors_when_include_priors_true(
    tmp_path: Path,
) -> None:
    """_apply_scenario with include_priors=True must not change alpha configs."""
    wells_path = _save_wells(tmp_path)
    cfg = _make_cfg(
        wells_path,
        tmp_path / "out",
        alpha={
            "component_a": AlphaModeConfig(
                mode="layer_logit",
                layer="prior_layer_a",
                scalar_fallback_pr0=0.55,
            ),
        },
    )
    scenario = ScenarioConfig(
        name="full", include_priors=True, include_spatial=True
    )

    new_cfg = _apply_scenario(cfg, scenario)

    assert new_cfg.alpha["component_a"].mode == "layer_logit"
