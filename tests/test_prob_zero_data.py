"""Tests for prior-predictive (zero-data) mode and side-by-side favorability."""

from __future__ import annotations

import warnings
from dataclasses import replace
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest

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
    SpatialFieldConfig,
)
from geopfa.prob.runner import run_probabilistic
from tests.fixtures.synthetic_prob import make_synthetic_pfa


def _base_cfg(
    wells_path: Path,
    output_dir: Path,
    *,
    alpha_mode: dict[str, AlphaModeConfig] | None = None,
    label_columns: dict[str, str] | None = None,
) -> ProbabilisticConfig:
    return ProbabilisticConfig(
        enabled=True,
        output_dir=output_dir,
        dimensions="2d",
        grid=GridConfig(),
        labels=LabelsConfig(
            source=str(wells_path),
            id_col="well_id",
            label_columns=label_columns
            or {
                "component_a": "heat_label",
                "component_b": "reservoir_label",
            },
            layer="wells",
            min_wells_for_fit=4,
        ),
        alpha=alpha_mode
        or {
            "component_a": AlphaModeConfig(
                mode="layer_logit",
                layer="prior_layer_a",
                scalar_fallback_pr0=0.55,
            ),
            "component_b": AlphaModeConfig(
                mode="layer_logit",
                layer="prior_layer_b",
                scalar_fallback_pr0=0.5,
            ),
        },
        evidence=EvidenceConfig(),
        spatial_field=SpatialFieldConfig(enabled=False),
        inference=InferenceConfig(backend="sequential"),
        calibration=CalibrationConfig(method="none"),
        cross_validation=CrossValidationConfig(),
        combination=CombinationConfig(rule="product"),
        scenarios=(),
        outputs=OutputsConfig(
            probability_rasters=False,
            uncertainty_rasters=False,
            calibration_artifacts=False,
            decision_artifacts=False,
            scenarios=False,
            format=("csv",),
        ),
    )


# ---------------------------------------------------------------------------
# Phase I — zero-data prior predictive
# ---------------------------------------------------------------------------


def test_zero_well_fixture_fails_closed_without_opt_in(tmp_path: Path) -> None:
    """Missing labels must not silently turn a fitted analysis into a prior map."""
    fixture = make_synthetic_pfa(grid_n=6, n_wells=10, seed=0)
    # Drop heat_label so component_a has no labels
    wells = fixture.wells.copy()
    wells["heat_label"] = np.nan
    wells_path = tmp_path / "wells.gpkg"
    wells.to_file(wells_path, layer="wells", driver="GPKG")
    cfg = _base_cfg(wells_path, tmp_path / "out")
    with pytest.raises(ValueError, match="fewer than 4 labelled wells"):
        run_probabilistic(fixture.pfa, cfg)


def test_force_prior_predictive_opt_in(tmp_path: Path) -> None:
    """A per-component opt-in should skip the regression even with labels present."""
    fixture = make_synthetic_pfa(grid_n=6, n_wells=15, seed=1)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    cfg = _base_cfg(
        wells_path,
        tmp_path / "out",
        alpha_mode={
            "component_a": AlphaModeConfig(
                mode="layer_logit",
                layer="prior_layer_a",
                scalar_fallback_pr0=0.55,
                force_prior_predictive=True,
            ),
            "component_b": AlphaModeConfig(
                mode="layer_logit",
                layer="prior_layer_b",
                scalar_fallback_pr0=0.5,
            ),
        },
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_probabilistic(fixture.pfa, cfg)
    # component_a uses prior-predictive (no regression coefficients)
    surface_a = result.components["component_a"].probability
    assert result.components["component_a"].model is None
    # component_b still gets the full fit
    assert result.components["component_b"].model is not None
    # component_a's surface should match sigma(alpha)
    probs_a = surface_a["probability"].to_numpy()
    assert probs_a.min() >= 0.20
    assert probs_a.max() <= 0.80


def test_prior_predictive_marked_in_result(tmp_path: Path) -> None:
    fixture = make_synthetic_pfa(grid_n=6, n_wells=10, seed=2)
    wells = fixture.wells.copy()
    wells["heat_label"] = np.nan
    wells["reservoir_label"] = np.nan
    wells_path = tmp_path / "wells.gpkg"
    wells.to_file(wells_path, layer="wells", driver="GPKG")
    cfg = _base_cfg(
        wells_path,
        tmp_path / "out",
        alpha_mode={
            "component_a": AlphaModeConfig(
                mode="layer_logit",
                layer="prior_layer_a",
                scalar_fallback_pr0=0.55,
                force_prior_predictive=True,
            ),
            "component_b": AlphaModeConfig(
                mode="layer_logit",
                layer="prior_layer_b",
                scalar_fallback_pr0=0.5,
                force_prior_predictive=True,
            ),
        },
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_probabilistic(fixture.pfa, cfg)
    # Both components should be flagged prior-predictive
    for name in ("component_a", "component_b"):
        # The component result's `model` field is None for prior-predictive
        assert result.components[name].model is None
        assert (
            result.components[name].diagnostics["inference_role"]
            == "prior_predictive"
        )


def test_all_prior_sequential_run_does_not_load_labels(tmp_path: Path) -> None:
    fixture = make_synthetic_pfa(grid_n=6, n_wells=10, seed=4)
    base = _base_cfg(tmp_path / "absent.gpkg", tmp_path / "out")
    cfg = replace(
        base,
        labels=replace(base.labels, source=None, id_col=None),
        alpha={
            name: replace(alpha, force_prior_predictive=True)
            for name, alpha in base.alpha.items()
        },
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_probabilistic(fixture.pfa, cfg)

    assert set(result.components) == {"component_a", "component_b"}
    assert all(
        component.diagnostics["inference_role"] == "prior_predictive"
        for component in result.components.values()
    )
