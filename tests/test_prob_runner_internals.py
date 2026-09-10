"""Unit tests for runner.py private functions (coverage gaps).

These tests exercise defensive branches in _fit_component, _combine_components,
_write_one_csv, _write_csv_outputs, _write_calibrated_csv_outputs, and
_apply_scenario that are not reachable through the public run_probabilistic API.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from geopfa.prob.alpha import AlphaCResult
from geopfa.prob.config import (
    AlphaModeConfig,
    CalibrationConfig,
    CombinationConfig,
    CrossValidationConfig,
    EvidenceConfig,
    GridConfig,
    InferenceConfig,
    LabelsConfig,
    OutputsConfig,
    ProbabilisticConfig,
    ScenarioConfig,
    SpatialFieldConfig,
)
from geopfa.prob.fitting import ComponentProbability
from geopfa.prob.labels import LoadedLabels
from geopfa.prob.pfa_grid import PFAGridAdapter
from geopfa.prob.runner import (
    _apply_scenario,  # noqa: PLC2701
    _combine_components,  # noqa: PLC2701
    _fit_component,  # noqa: PLC2701
    _write_calibrated_csv_outputs,  # noqa: PLC2701
    _write_csv_outputs,  # noqa: PLC2701
    _write_one_csv,  # noqa: PLC2701
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
    combination_rule: str = "product",
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
        combination=CombinationConfig(rule=combination_rule),
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


def test_combine_components_empty_returns_empty_gdf(tmp_path: Path) -> None:
    """_combine_components({}, cfg) must return an empty GeoDataFrame."""
    wells_path = _save_wells(tmp_path)
    cfg = _make_cfg(wells_path, tmp_path / "out")
    result = _combine_components({}, cfg)
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 0


def test_combine_components_geometric_mean(tmp_path: Path) -> None:
    """_combine_components applies geometric_mean correctly."""
    wells_path = _save_wells(tmp_path)
    cfg = _make_cfg(
        wells_path, tmp_path / "out", combination_rule="geometric_mean"
    )

    geom = [Point(0, 0), Point(1, 0), Point(0, 1), Point(1, 1)]
    gdf = gpd.GeoDataFrame(
        {"probability": [0.2, 0.4, 0.6, 0.8]}, geometry=geom, crs="EPSG:4326"
    )
    surf = ComponentProbability(
        probability=gdf, model=None, feature_names=(), spatial_field=None
    )

    result = _combine_components({"comp": surf}, cfg)
    assert "probability" in result.columns
    assert len(result) == len(gdf)
    np.testing.assert_allclose(
        result["probability"].to_numpy(),
        gdf["probability"].to_numpy(),
        rtol=1e-5,
    )


# ---------------------------------------------------------------------------
# _write_one_csv — empty GDF short-circuits
# ---------------------------------------------------------------------------


def test_write_one_csv_skips_empty_gdf(tmp_path: Path) -> None:
    """_write_one_csv must not create the file when GDF has no rows."""
    out_path = tmp_path / "nothing.csv"
    empty = gpd.GeoDataFrame(columns=["geometry", "probability"])
    _write_one_csv(empty, out_path)
    assert not out_path.exists()


def test_write_one_csv_writes_non_empty_gdf(tmp_path: Path) -> None:
    """_write_one_csv writes a valid CSV for a non-empty GDF."""
    out_path = tmp_path / "out.csv"
    geom = [Point(0, 0), Point(1, 0)]
    gdf = gpd.GeoDataFrame(
        {"probability": [0.3, 0.7]}, geometry=geom, crs="EPSG:4326"
    )
    _write_one_csv(gdf, out_path)
    assert out_path.exists()
    df = pd.read_csv(out_path)
    assert "probability" in df.columns
    assert "x" in df.columns and "y" in df.columns


# ---------------------------------------------------------------------------
# _write_csv_outputs — empty combined surface
# ---------------------------------------------------------------------------


def test_write_csv_outputs_skips_empty_combined(tmp_path: Path) -> None:
    """_write_csv_outputs must not create combined_probability.csv when
    combined GDF is empty."""
    out_dir = tmp_path / "out"
    geom = [Point(0, 0)]
    prob_gdf = gpd.GeoDataFrame(
        {"probability": [0.5]}, geometry=geom, crs="EPSG:4326"
    )
    surf = ComponentProbability(
        probability=prob_gdf, model=None, feature_names=(), spatial_field=None
    )
    combined_empty = gpd.GeoDataFrame()

    _write_csv_outputs({"comp": surf}, combined_empty, out_dir)

    assert (out_dir / "comp_probability.csv").exists()
    assert not (out_dir / "combined_probability.csv").exists()


# ---------------------------------------------------------------------------
# _write_calibrated_csv_outputs — empty GDF entries are skipped
# ---------------------------------------------------------------------------


def test_write_calibrated_csv_outputs_skips_empty_entries(
    tmp_path: Path,
) -> None:
    """_write_calibrated_csv_outputs must skip GDFs with zero rows."""
    out_dir = tmp_path / "calibrated"
    empty = gpd.GeoDataFrame()
    _write_calibrated_csv_outputs({"comp": empty}, out_dir)
    # directory created but no CSV written for the empty GDF
    assert out_dir.exists()
    assert not (out_dir / "comp_probability.csv").exists()


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
